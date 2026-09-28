# Hybrid-lite late-fusion baseline trainer (from scratch)
import os
import sys
import json
import time
import math
import argparse
import inspect
import platform
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2Tokenizer
from tqdm import tqdm

sys.path.insert(0, "src")
from models import GrassmannGPTv4
from train_exp4_ddp import SmallTransformer, TextDataset


# -----------------------------------------------------------------------------
# Distributed helpers
# -----------------------------------------------------------------------------

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def configure_visible_devices(gpu_id: str) -> None:
    if gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def setup_distributed() -> Tuple[bool, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
            device = torch.device("cuda", local_rank)
        else:
            dist.init_process_group(backend="gloo")
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return distributed, local_rank, device


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def reduce_sum(values, device: torch.device):
    t = torch.tensor([float(v) for v in values], device=device, dtype=torch.float64)
    if is_dist_avail_and_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return [float(x.item()) for x in t]


# -----------------------------------------------------------------------------
# General utils
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def env_metadata() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "ALL"),
        "platform": platform.platform(),
        "rank": get_rank(),
        "world_size": get_world_size(),
    }


def amp_context(device: torch.device, use_amp: bool):
    if device.type == "cuda":
        return autocast(device_type="cuda", enabled=use_amp)
    return nullcontext()


def sigmoid_inverse(p: float, eps: float = 1e-6) -> float:
    p = max(min(float(p), 1.0 - eps), eps)
    return math.log(p / (1.0 - p))


def coerce_int_with_default(value, default):
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def coerce_float_with_default(value, default):
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)




def make_text_dataset(tokenizer, max_seq_len, dataset_name, dataset_path, split, text_field,
                      offline, max_lines, encode_chars_per_batch,
                      tinystories_val_frac, split_seed):
    """
    Compatibility wrapper: only pass kwargs that the repo's current TextDataset supports.
    This avoids crashes when train_exp4_ddp.py is older/newer than expected.
    """
    sig = inspect.signature(TextDataset.__init__)
    supported = set(sig.parameters.keys())

    kwargs = {
        "tokenizer": tokenizer,
        "max_seq_len": max_seq_len,
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "split": split,
        "text_field": text_field,
        "offline": offline,
        "max_lines": max_lines,
        "encode_chars_per_batch": encode_chars_per_batch,
        "tinystories_val_frac": tinystories_val_frac,
        "split_seed": split_seed,
    }
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    return TextDataset(**filtered)

# -----------------------------------------------------------------------------
# Model builders
# -----------------------------------------------------------------------------

def build_grassmann(vocab_size: int, cfg: Dict[str, Any]) -> GrassmannGPTv4:
    window_sizes = cfg.get("window_sizes", "1,2,4")
    if isinstance(window_sizes, str):
        window_sizes = [int(x) for x in window_sizes.split(",") if x.strip()]
    return GrassmannGPTv4(
        vocab_size=vocab_size,
        max_seq_len=cfg["max_seq_len"],
        model_dim=cfg["model_dim"],
        num_layers=cfg["num_layers"],
        reduced_dim=cfg.get("reduced_dim", 32),
        ff_dim=4 * cfg["model_dim"],
        window_sizes=window_sizes,
        dropout=cfg.get("dropout", 0.1),
    )


def build_transformer(vocab_size: int, cfg: Dict[str, Any]) -> SmallTransformer:
    return SmallTransformer(
        vocab_size=vocab_size,
        max_seq_len=cfg["max_seq_len"],
        model_dim=cfg["model_dim"],
        num_layers=cfg["num_layers"],
        num_heads=8,
        ff_dim=4 * cfg["model_dim"],
        dropout=cfg.get("dropout", 0.1),
    )


# -----------------------------------------------------------------------------
# Hybrid-lite model
# -----------------------------------------------------------------------------

class HybridLateFusionAlphaModel(nn.Module):
    """
    Two independent small branches. Fuse only the last-k layer logits.
    """
    def __init__(self, grassmann: nn.Module, transformer: nn.Module, init_alpha: float = 0.5, late_k: int = 1):
        super().__init__()
        self.grassmann = grassmann
        self.transformer = transformer
        self.num_layers = int(min(len(self.grassmann.blocks), len(self.transformer.blocks)))
        if self.num_layers <= 0:
            raise ValueError("No valid layers found in branches.")
        self.late_k = int(max(1, min(int(late_k), self.num_layers)))
        init_logit = sigmoid_inverse(init_alpha)
        self.logit_alpha = nn.Parameter(torch.full((self.late_k,), float(init_logit), dtype=torch.float32))

    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_alpha)

    @torch.no_grad()
    def get_alpha_value(self) -> float:
        return float(self.alpha().mean().item())

    @torch.no_grad()
    def get_alpha_vector(self):
        return [float(x) for x in self.alpha().detach().cpu().view(-1).tolist()]

    def _transformer_hidden_per_layer(self, input_ids: torch.Tensor):
        seq_len = input_ids.size(1)
        device = input_ids.device
        tok_emb = self.transformer.token_embedding(input_ids)
        pos_emb = self.transformer.position_embedding(torch.arange(seq_len, device=device))
        hidden = self.transformer.embedding_dropout(tok_emb + pos_emb)
        per_layer = []
        for block in self.transformer.blocks:
            hidden = block(hidden)
            per_layer.append(hidden)
        return per_layer

    def _grassmann_hidden_per_layer(self, input_ids: torch.Tensor):
        seq_len = input_ids.size(1)
        device = input_ids.device
        tok_emb = self.grassmann.token_embedding(input_ids)
        pos_emb = self.grassmann.position_embedding(torch.arange(seq_len, device=device))
        hidden = self.grassmann.embedding_dropout(tok_emb + pos_emb)
        per_layer = []
        for block in self.grassmann.blocks:
            hidden = block(hidden)
            per_layer.append(hidden)
        return per_layer

    def _transformer_logits_from_hidden(self, hidden: torch.Tensor):
        hidden = self.transformer.ln_f(hidden)
        return self.transformer.lm_head(hidden)

    def _grassmann_logits_from_hidden(self, hidden: torch.Tensor):
        hidden = self.grassmann.ln_f(hidden)
        return self.grassmann.lm_head(hidden)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        t_layers = self._transformer_hidden_per_layer(input_ids)
        g_layers = self._grassmann_hidden_per_layer(input_ids)

        alpha = self.alpha()
        selected_t = t_layers[-self.late_k:]
        selected_g = g_layers[-self.late_k:]

        fused_logits_list = []
        for i in range(self.late_k):
            logits_t = self._transformer_logits_from_hidden(selected_t[i])
            logits_g = self._grassmann_logits_from_hidden(selected_g[i])
            a = alpha[i]
            fused_logits_list.append(a * logits_t + (1.0 - a) * logits_g)

        logits = torch.stack(fused_logits_list, dim=0).mean(dim=0)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss, alpha

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_hybrid(model: nn.Module, dataloader: DataLoader, device: torch.device, use_amp: bool = False):
    model.eval()
    total_loss = 0.0
    total_count = 0
    last_alpha = None
    last_alpha_vec = None

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with amp_context(device, use_amp):
            _, loss, alpha = model(x, labels=y)
        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)
        last_alpha = float(alpha.detach().mean().item())
        last_alpha_vec = [float(v) for v in alpha.detach().cpu().view(-1).tolist()]

    loss_sum, count_sum = reduce_sum([total_loss, total_count], device)
    avg_loss = loss_sum / max(count_sum, 1.0)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl, last_alpha, last_alpha_vec


def train_one_epoch(model, dataloader, optimizer, scaler, device, use_amp, log_interval=50):
    model.train()
    total_loss = 0.0
    total_count = 0
    total_tokens = 0
    start_time = time.time()

    iterator = tqdm(dataloader, desc="train", disable=not is_main_process())
    for step, (x, y) in enumerate(iterator, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp_context(device, use_amp):
            _, loss, alpha = model(x, labels=y)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)
        total_tokens += x.numel()

        if is_main_process() and step % log_interval == 0:
            elapsed = time.time() - start_time
            tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
            iterator.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ppl": f"{math.exp(min(loss.item(), 20)):.2f}",
                "alpha": f"{float(alpha.detach().mean().item()):.4f}",
                "tok/s": f"{tok_per_sec:.0f}",
                "gnorm": f"{float(grad_norm):.2f}",
            })

    loss_sum, count_sum = reduce_sum([total_loss, total_count], device)
    return loss_sum / max(count_sum, 1.0)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a small hybrid-lite late-fusion baseline")
    parser.add_argument("--tokenizer-dir", type=str, default="./gpt2_local")
    parser.add_argument("--gpu-id", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs/hybrid_experiments")
    parser.add_argument("--experiment-name", type=str, default="hybrid_lite_baseline")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--tags", type=str, default="hybrid-lite,baseline,late-fusion")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)

    parser.add_argument("--init-alpha", type=float, default=0.5)
    parser.add_argument("--late-k", type=int, default=1)

    parser.add_argument("--model-dim", type=int, default=224)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--reduced-dim", type=int, default=56)
    parser.add_argument("--window-sizes", type=str, default="1,2,4")
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--dataset-name", type=str, default="ptb")
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--text-field", type=str, default="")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-lines", type=int, default=0)
    parser.add_argument("--encode-chars-per-batch", type=int, default=200000)
    parser.add_argument("--tinystories-val-frac", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true")

    args = parser.parse_args()
    configure_visible_devices(args.gpu_id)

    distributed, local_rank, device = setup_distributed()
    set_seed(args.seed + get_rank())
    use_amp = args.amp and device.type == "cuda"

    if is_main_process():
        print(f"Using device: {device}")
        print(f"Distributed: {distributed}, world_size={get_world_size()}, local_rank={local_rank}")
        print(f"Visible CUDA devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}")
        print(f"AMP enabled: {use_amp}")

    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    if is_main_process():
        print(f"Hybrid-lite dataset_name = {args.dataset_name}")
        print(f"Hybrid-lite dataset_path = {args.dataset_path or None}")
        print(f"Hybrid-lite text_field = {args.text_field or None}")
        print(f"Hybrid-lite max_seq_len = {args.max_seq_len}")

    text_field = args.text_field if args.text_field else None
    max_lines = None if args.max_lines <= 0 else args.max_lines

    dataset_path = args.dataset_path if args.dataset_path else None

    train_dataset = make_text_dataset(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        dataset_name=args.dataset_name,
        dataset_path=dataset_path,
        split="train",
        text_field=text_field,
        offline=args.offline,
        max_lines=max_lines,
        encode_chars_per_batch=args.encode_chars_per_batch,
        tinystories_val_frac=args.tinystories_val_frac,
        split_seed=args.split_seed,
    )
    val_dataset = make_text_dataset(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        dataset_name=args.dataset_name,
        dataset_path=dataset_path,
        split="validation",
        text_field=text_field,
        offline=args.offline,
        max_lines=max_lines,
        encode_chars_per_batch=args.encode_chars_per_batch,
        tinystories_val_frac=args.tinystories_val_frac,
        split_seed=args.split_seed,
    )
    test_dataset = make_text_dataset(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        dataset_name=args.dataset_name,
        dataset_path=dataset_path,
        split="test",
        text_field=text_field,
        offline=args.offline,
        max_lines=max_lines,
        encode_chars_per_batch=args.encode_chars_per_batch,
        tinystories_val_frac=args.tinystories_val_frac,
        split_seed=args.split_seed,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            sampler=val_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             sampler=test_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    cfg = {
        "max_seq_len": args.max_seq_len,
        "model_dim": args.model_dim,
        "num_layers": args.num_layers,
        "reduced_dim": args.reduced_dim,
        "window_sizes": args.window_sizes,
        "dropout": args.dropout,
    }

    grassmann = build_grassmann(vocab_size, cfg)
    transformer = build_transformer(vocab_size, cfg)
    hybrid = HybridLateFusionAlphaModel(grassmann, transformer, init_alpha=args.init_alpha, late_k=args.late_k).to(device)

    if distributed:
        model = DDP(hybrid, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=False)
    else:
        model = hybrid

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    timestamp = now_str()
    run_dir = Path(args.output_dir) / f"{timestamp}_{args.experiment_name}"
    ckpt_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "hybrid_metrics.jsonl"

    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_json({
            "script": Path(__file__).name,
            "created_at": timestamp,
            "experiment_name": args.experiment_name,
            "notes": args.notes,
            "tags": args.tags,
            "config": vars(args),
            "env": env_metadata(),
        }, run_dir / "config.json")

        print(f"Hybrid-lite total params: {unwrap_model(model).get_num_params():,}")
        print(f"Hybrid-lite trainable params at start: {unwrap_model(model).get_num_trainable_params():,}")
        print(f"Initial alpha mean: {unwrap_model(model).get_alpha_value():.6f}")
        print(f"Initial alpha vector: {[round(x, 4) for x in unwrap_model(model).get_alpha_vector()]}")

    init_val_loss, init_val_ppl, init_alpha, init_alpha_vec = evaluate_hybrid(model, val_loader, device, use_amp=use_amp)
    if is_main_process():
        print(f"Initial hybrid-lite val loss={init_val_loss:.4f}, val ppl={init_val_ppl:.4f}, alpha_mean={init_alpha:.6f}")
        print(f"Initial eval alpha vector: {[round(x, 4) for x in init_alpha_vec]}")

    best_val_loss = float("inf")
    best_val_ppl = float("inf")
    best_epoch = -1
    best_alpha = None
    best_alpha_vector = None

    train_losses, val_losses, alpha_history, alpha_vector_history = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp, log_interval=args.log_interval)
        val_loss, val_ppl, alpha_value, alpha_vec = evaluate_hybrid(model, val_loader, device, use_amp=use_amp)

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        alpha_history.append(float(alpha_value))
        alpha_vector_history.append(alpha_vec)

        if is_main_process():
            epoch_record = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_ppl": float(val_ppl),
                "alpha": float(alpha_value),
                "alpha_vector": alpha_vec,
                "time_sec": time.time() - epoch_start,
            }
            append_jsonl(epoch_record, metrics_path)
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_ppl={val_ppl:.4f}, alpha_mean={alpha_value:.6f}, time={time.time()-epoch_start:.1f}s"
            )

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_val_ppl = float(val_ppl)
            best_epoch = epoch
            best_alpha = float(alpha_value)
            best_alpha_vector = list(alpha_vec)
            if is_main_process():
                torch.save(unwrap_model(model).state_dict(), ckpt_dir / "hybrid_best.pt")

    if distributed:
        dist.barrier()

    state = torch.load(ckpt_dir / "hybrid_best.pt", map_location=device, weights_only=True)
    unwrap_model(model).load_state_dict(state)

    test_loss, test_ppl, final_alpha, final_alpha_vec = evaluate_hybrid(model, test_loader, device, use_amp=use_amp)

    summary = {
        "hybrid": {
            "num_params": unwrap_model(model).get_num_params(),
            "num_trainable_params": unwrap_model(model).get_num_trainable_params(),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_ppl": best_val_ppl,
            "test_loss": float(test_loss),
            "test_ppl": float(test_ppl),
            "initial_alpha": float(args.init_alpha),
            "best_alpha": best_alpha,
            "best_alpha_vector": best_alpha_vector,
            "final_alpha": float(final_alpha),
            "final_alpha_vector": final_alpha_vec,
            "late_k": int(args.late_k),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "alpha_history": alpha_history,
            "alpha_vector_history": alpha_vector_history,
            "checkpoint_path": str((ckpt_dir / "hybrid_best.pt").resolve()),
            "metrics_path": str(metrics_path.resolve()),
        }
    }

    if is_main_process():
        save_json(summary, run_dir / "summary.json")
        report = [
            f"# Hybrid-lite Baseline Report: {args.experiment_name}",
            "",
            f"- Best Epoch: {best_epoch}",
            f"- Best Val PPL: {best_val_ppl:.4f}",
            f"- Test PPL: {float(test_ppl):.4f}",
            f"- Best Alpha Mean: {best_alpha:.6f}",
            f"- Best Alpha Vector: {best_alpha_vector}",
            f"- late_k: {args.late_k}",
        ]
        (run_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
        print("\nFinal Hybrid-lite Results:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nSaved hybrid-lite artifacts to: {run_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
