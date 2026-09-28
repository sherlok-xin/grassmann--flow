import os
import sys
import json
import time
import math
import argparse
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


def reduce_sum_pair(sum_value: float, count_value: float, device: torch.device) -> Tuple[float, float]:
    pair = torch.tensor([float(sum_value), float(count_value)], device=device, dtype=torch.float64)
    if is_dist_avail_and_initialized():
        dist.all_reduce(pair, op=dist.ReduceOp.SUM)
    return float(pair[0].item()), float(pair[1].item())


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


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_summary(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    return load_json(summary_path)


def load_config(run_dir: Path) -> Dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found: {config_path}")
    return load_json(config_path)


def get_env_info(device: torch.device = None) -> Dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    device_name = None
    if cuda_available:
        idx = device.index if isinstance(device, torch.device) and device.index is not None else 0
        device_name = torch.cuda.get_device_name(idx)
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": device_name,
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


def resolve_checkpoint_path(run_dir: Path, ckpt_value: str) -> Path:
    ckpt_path = Path(ckpt_value)
    if not ckpt_path.is_absolute():
        ckpt_path = run_dir / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def build_grassmann(vocab_size: int, cfg: Dict[str, Any]) -> GrassmannGPTv4:
    window_sizes = cfg.get("window_sizes", "1,2,4,8,12,16")
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


def compare_config_fields(g_cfg: Dict[str, Any], t_cfg: Dict[str, Any], fields) -> Dict[str, Dict[str, Any]]:
    mismatches = {}
    for field in fields:
        gv = g_cfg.get(field)
        tv = t_cfg.get(field)
        if gv != tv:
            mismatches[field] = {"grassmann": gv, "transformer": tv}
    return mismatches


def choose_dataset_value(args_value, g_cfg_value, t_cfg_value, default=None):
    if args_value is not None and args_value != "":
        return args_value
    if g_cfg_value == t_cfg_value:
        return g_cfg_value
    return t_cfg_value if t_cfg_value not in [None, ""] else (g_cfg_value if g_cfg_value not in [None, ""] else default)


# -----------------------------------------------------------------------------
# Hybrid model
# -----------------------------------------------------------------------------

class HybridLogitsAlphaModel(nn.Module):
    """
    Minimal hybrid teacher: fuse branch logits with one learnable scalar alpha.

    logits = alpha * transformer_logits + (1 - alpha) * grassmann_logits
    alpha = sigmoid(logit_alpha)
    """

    def __init__(self, grassmann: nn.Module, transformer: nn.Module, init_alpha: float = 0.5):
        super().__init__()
        self.grassmann = grassmann
        self.transformer = transformer
        self.logit_alpha = nn.Parameter(torch.tensor(sigmoid_inverse(init_alpha), dtype=torch.float32))

    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_alpha)

    def set_branch_trainable(self, trainable: bool) -> None:
        for p in self.grassmann.parameters():
            p.requires_grad = trainable
        for p in self.transformer.parameters():
            p.requires_grad = trainable

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        logits_g, _ = self.grassmann(input_ids, labels=None)
        logits_t, _ = self.transformer(input_ids, labels=None)

        alpha = self.alpha()
        logits = alpha * logits_t + (1.0 - alpha) * logits_g

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

    @torch.no_grad()
    def get_alpha_value(self) -> float:
        return float(self.alpha().item())

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

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with amp_context(device, use_amp):
            _, loss, alpha = model(x, labels=y)
        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)
        last_alpha = float(alpha.item())

    loss_sum, count_sum = reduce_sum_pair(total_loss, total_count, device)
    avg_loss = loss_sum / max(count_sum, 1.0)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl, last_alpha


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool = False,
    log_interval: int = 50,
):
    model.train()
    total_loss = 0.0
    total_count = 0
    total_tokens = 0
    start_time = time.time()

    iterator = dataloader
    if is_main_process():
        iterator = tqdm(dataloader, desc=f"Epoch {epoch}")

    for step, (x, y) in enumerate(iterator):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp_context(device, use_amp):
            _, loss, alpha = model(x, labels=y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)
        total_tokens += x.numel()

        if is_main_process() and step % log_interval == 0:
            elapsed = time.time() - start_time
            tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
            iterator.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ppl": f"{loss.exp().item():.2f}",
                "alpha": f"{float(alpha.item()):.4f}",
                "tok/s": f"{tok_per_sec:.0f}",
                "gnorm": f"{float(grad_norm):.2f}",
            })

    loss_sum, count_sum = reduce_sum_pair(total_loss, total_count, device)
    return loss_sum / max(count_sum, 1.0)


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def make_run_dir(output_root: Path, experiment_name: str):
    run_id = now_str()
    safe_name = experiment_name.replace(" ", "_")
    run_dir = output_root / f"{run_id}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def write_markdown_report(run_dir: Path, run_meta: Dict[str, Any], summary: Dict[str, Any]) -> None:
    lines = []
    lines.append(f"# Hybrid Alpha Training Report: {run_meta['experiment_name']}")
    lines.append("")
    lines.append(f"- Run ID: `{run_meta['run_id']}`")
    lines.append(f"- Time: `{run_meta['start_time']}`")
    lines.append(f"- Notes: {run_meta.get('notes', '')}")
    lines.append(f"- Tags: {', '.join(run_meta.get('tags', []))}")
    lines.append("")
    lines.append("## Learned Alpha")
    lines.append("")
    lines.append(f"- Initial alpha: **{summary['hybrid']['initial_alpha']:.6f}**")
    lines.append(f"- Best alpha: **{summary['hybrid']['best_alpha']:.6f}**")
    lines.append(f"- Final alpha: **{summary['hybrid']['final_alpha']:.6f}**")
    lines.append("")
    lines.append("## Final Results")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Best epoch | {summary['hybrid']['best_epoch']} |")
    lines.append(f"| Best val ppl | {summary['hybrid']['best_val_ppl']:.4f} |")
    lines.append(f"| Test ppl | {summary['hybrid']['test_ppl']:.4f} |")
    lines.append(f"| Total params | {summary['hybrid']['num_params']} |")
    lines.append(f"| Trainable params | {summary['hybrid']['num_trainable_params']} |")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(run_meta["config"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(run_meta["dataset_stats"], indent=2, ensure_ascii=False))
    lines.append("```")
    with open(run_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a learnable-alpha hybrid of Grassmann + Transformer (DDP)")
    parser.add_argument("--grassmann-run-dir", type=str, required=True)
    parser.add_argument("--transformer-run-dir", type=str, required=True)
    parser.add_argument("--tokenizer-dir", type=str, default="./gpt2_local")
    parser.add_argument("--output-dir", type=str, default="outputs/hybrid_experiments")
    parser.add_argument("--experiment-name", type=str, default="hybrid_alpha_training")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--tags", type=str, default="hybrid,alpha,logits")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5, help="Branch learning rate for joint fine-tuning")
    parser.add_argument("--alpha-lr", type=float, default=1e-2, help="Learning rate for scalar alpha")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)

    parser.add_argument("--train-mode", type=str, default="alpha_only", choices=["alpha_only", "joint"])
    parser.add_argument("--freeze-branch-epochs", type=int, default=1,
                        help="For joint mode, keep both branches frozen for the first N epochs")
    parser.add_argument("--init-alpha", type=float, default=0.5)
    parser.add_argument("--allow-config-mismatch", action="store_true")

    parser.add_argument("--dataset-name", type=str, default="")
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--text-field", type=str, default="")
    parser.add_argument("--max-seq-len", type=int, default=0,
                        help="Override sequence length. 0 means infer from checkpoints")
    parser.add_argument("--max-lines", type=int, default=-1,
                        help="Override dataset line cap. -1 means infer from checkpoints")
    parser.add_argument("--encode-chars-per-batch", type=int, default=-1,
                        help="Override batched tokenizer char count. -1 means infer from checkpoints")
    parser.add_argument("--tinystories-val-frac", type=float, default=-1.0)
    parser.add_argument("--split-seed", type=int, default=-1)
    parser.add_argument("--offline", action="store_true")

    args = parser.parse_args()

    distributed, local_rank, device = setup_distributed()
    set_seed(args.seed + get_rank())
    use_amp = args.amp and device.type == "cuda"

    if is_main_process():
        print(f"Using device: {device}")
        print(f"Distributed: {distributed}, world_size={get_world_size()}, local_rank={local_rank}")
        print(f"Training mode: {args.train_mode}, use_amp={use_amp}")

    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_DATASETS_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_HUB_OFFLINE", None)

    g_run_dir = Path(args.grassmann_run_dir)
    t_run_dir = Path(args.transformer_run_dir)
    g_summary = load_summary(g_run_dir)
    t_summary = load_summary(t_run_dir)
    g_config = load_config(g_run_dir).get("config", {})
    t_config = load_config(t_run_dir).get("config", {})

    if "grassmann" not in g_summary:
        raise ValueError(f"{g_run_dir} does not contain grassmann results in summary.json")
    if "transformer" not in t_summary:
        raise ValueError(f"{t_run_dir} does not contain transformer results in summary.json")

    dataset_fields = [
        "dataset_name",
        "dataset_path",
        "text_field",
        "max_lines",
        "encode_chars_per_batch",
        "tinystories_val_frac",
        "split_seed",
    ]
    config_mismatches = compare_config_fields(g_config, t_config, dataset_fields)
    max_seq_len_mismatch = g_config.get("max_seq_len") != t_config.get("max_seq_len")

    if is_main_process() and config_mismatches:
        print("Dataset/config mismatches detected between branch run dirs:")
        print(json.dumps(config_mismatches, indent=2, ensure_ascii=False))
        if max_seq_len_mismatch:
            print({"max_seq_len": {"grassmann": g_config.get("max_seq_len"), "transformer": t_config.get("max_seq_len")}})

    if (config_mismatches or max_seq_len_mismatch) and not args.allow_config_mismatch:
        mismatch_text = json.dumps({
            "dataset_mismatches": config_mismatches,
            "max_seq_len": {
                "grassmann": g_config.get("max_seq_len"),
                "transformer": t_config.get("max_seq_len"),
            } if max_seq_len_mismatch else None,
        }, indent=2, ensure_ascii=False)
        raise ValueError(
            "Run dirs are not configuration-compatible. Pass --allow-config-mismatch only if you know the dataset regime is still aligned.\n"
            + mismatch_text
        )

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    vocab_size = len(tokenizer)

    max_seq_len = args.max_seq_len if args.max_seq_len > 0 else min(
        int(g_config.get("max_seq_len", 256)),
        int(t_config.get("max_seq_len", 256)),
    )
    dataset_name = choose_dataset_value(args.dataset_name, g_config.get("dataset_name"), t_config.get("dataset_name"), "wikitext2")
    dataset_path = choose_dataset_value(args.dataset_path, g_config.get("dataset_path"), t_config.get("dataset_path"), "")
    text_field = choose_dataset_value(args.text_field, g_config.get("text_field"), t_config.get("text_field"), "")

    max_lines = args.max_lines if args.max_lines >= 0 else choose_dataset_value(None, g_config.get("max_lines"), t_config.get("max_lines"), 0)
    encode_chars_per_batch = args.encode_chars_per_batch if args.encode_chars_per_batch >= 0 else choose_dataset_value(None, g_config.get("encode_chars_per_batch"), t_config.get("encode_chars_per_batch"), 200000)
    tinystories_val_frac = args.tinystories_val_frac if args.tinystories_val_frac >= 0 else choose_dataset_value(None, g_config.get("tinystories_val_frac"), t_config.get("tinystories_val_frac"), 0.02)
    split_seed = args.split_seed if args.split_seed >= 0 else choose_dataset_value(None, g_config.get("split_seed"), t_config.get("split_seed"), 42)

    if is_main_process():
        print(f"Hybrid dataset_name = {dataset_name}")
        print(f"Hybrid dataset_path = {dataset_path}")
        print(f"Hybrid text_field = {text_field}")
        print(f"Hybrid max_seq_len = {max_seq_len}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id, run_dir = make_run_dir(output_root, args.experiment_name)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = TextDataset(
        "train",
        tokenizer,
        max_seq_len=max_seq_len,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        text_field=text_field,
        max_lines=max_lines,
        encode_chars_per_batch=encode_chars_per_batch,
        tinystories_val_frac=tinystories_val_frac,
        split_seed=split_seed,
    )
    val_dataset = TextDataset(
        "validation",
        tokenizer,
        max_seq_len=max_seq_len,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        text_field=text_field,
        max_lines=max_lines,
        encode_chars_per_batch=encode_chars_per_batch,
        tinystories_val_frac=tinystories_val_frac,
        split_seed=split_seed,
    )
    test_dataset = TextDataset(
        "test",
        tokenizer,
        max_seq_len=max_seq_len,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        text_field=text_field,
        max_lines=max_lines,
        encode_chars_per_batch=encode_chars_per_batch,
        tinystories_val_frac=tinystories_val_frac,
        split_seed=split_seed,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    grassmann = build_grassmann(vocab_size, g_config)
    transformer = build_transformer(vocab_size, t_config)

    g_ckpt = resolve_checkpoint_path(g_run_dir, g_summary["grassmann"]["checkpoint_path"])
    t_ckpt = resolve_checkpoint_path(t_run_dir, t_summary["transformer"]["checkpoint_path"])

    if is_main_process():
        print(f"Loading grassmann checkpoint: {g_ckpt}")
        print(f"Loading transformer checkpoint: {t_ckpt}")

    grassmann.load_state_dict(torch.load(str(g_ckpt), map_location="cpu"))
    transformer.load_state_dict(torch.load(str(t_ckpt), map_location="cpu"))

    hybrid = HybridLogitsAlphaModel(grassmann, transformer, init_alpha=args.init_alpha).to(device)

    if args.train_mode == "alpha_only":
        hybrid.set_branch_trainable(False)
    else:
        hybrid.set_branch_trainable(False)

    num_params = hybrid.get_num_params()
    num_trainable_params = hybrid.get_num_trainable_params()

    if is_main_process():
        print(f"Hybrid total params: {num_params:,} ({num_params / 1e6:.2f}M)")
        print(f"Hybrid trainable params at start: {num_trainable_params:,}")
        print(f"Initial alpha: {hybrid.get_alpha_value():.6f}")

    model = DDP(hybrid, device_ids=[local_rank], output_device=local_rank) if distributed and device.type == "cuda" else hybrid

    alpha_params = [unwrap_model(model).logit_alpha]
    branch_params = [p for n, p in unwrap_model(model).named_parameters() if n != "logit_alpha"]
    optimizer = torch.optim.AdamW(
        [
            {"params": alpha_params, "lr": args.alpha_lr, "weight_decay": 0.0},
            {"params": branch_params, "lr": args.lr, "weight_decay": args.weight_decay},
        ]
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
    scaler = GradScaler(device.type, enabled=use_amp)

    run_meta = {
        "run_id": run_id,
        "experiment_name": args.experiment_name,
        "start_time": datetime.now().isoformat(),
        "notes": args.notes,
        "tags": [t.strip() for t in args.tags.split(",") if t.strip()],
        "config": vars(args),
        "env": get_env_info(device),
        "source_runs": {
            "grassmann_run_dir": str(g_run_dir.resolve()),
            "transformer_run_dir": str(t_run_dir.resolve()),
            "grassmann_checkpoint": str(g_ckpt.resolve()),
            "transformer_checkpoint": str(t_ckpt.resolve()),
        },
        "dataset_stats": {
            "train": train_dataset.stats,
            "validation": val_dataset.stats,
            "test": test_dataset.stats,
        },
        "compatibility": {
            "config_mismatches": config_mismatches,
            "max_seq_len_grassmann": g_config.get("max_seq_len"),
            "max_seq_len_transformer": t_config.get("max_seq_len"),
            "effective_max_seq_len": max_seq_len,
        },
    }
    if is_main_process():
        save_json(run_meta, run_dir / "config.json")

    metrics_path = run_dir / "hybrid_metrics.jsonl"

    if distributed:
        dist.barrier()

    if is_main_process():
        init_val_loss, init_val_ppl, init_alpha = evaluate_hybrid(model, val_loader, device, use_amp=use_amp)
        print(f"Initial hybrid val loss={init_val_loss:.4f}, val ppl={init_val_ppl:.4f}, alpha={init_alpha:.6f}")
    if distributed:
        dist.barrier()

    best_val_loss = float("inf")
    best_val_ppl = float("inf")
    best_epoch = -1
    alpha_history = []
    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if args.train_mode == "joint":
            branch_trainable = epoch > args.freeze_branch_epochs
            unwrap_model(model).set_branch_trainable(branch_trainable)
        else:
            unwrap_model(model).set_branch_trainable(False)

        current_trainable_params = unwrap_model(model).get_num_trainable_params()

        epoch_start = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            epoch,
            use_amp=use_amp,
            log_interval=args.log_interval,
        )
        epoch_time = time.time() - epoch_start

        val_loss, val_ppl, alpha_value = evaluate_hybrid(model, val_loader, device, use_amp=use_amp)
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        alpha_history.append(float(alpha_value))

        if is_main_process():
            epoch_record = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_ppl": float(torch.exp(torch.tensor(train_loss)).item()),
                "val_loss": float(val_loss),
                "val_ppl": float(val_ppl),
                "alpha": float(alpha_value),
                "epoch_time_sec": float(epoch_time),
                "lr_alpha": float(optimizer.param_groups[0]["lr"]),
                "lr_branch": float(optimizer.param_groups[1]["lr"]),
                "branch_trainable": bool(epoch > args.freeze_branch_epochs) if args.train_mode == "joint" else False,
                "num_trainable_params": int(current_trainable_params),
            }
            append_jsonl(epoch_record, metrics_path)
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_ppl={val_ppl:.4f}, alpha={alpha_value:.6f}, time={epoch_time:.1f}s, "
                f"trainable={current_trainable_params:,}"
            )

            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                best_val_ppl = float(val_ppl)
                best_epoch = epoch
                torch.save(unwrap_model(model).state_dict(), ckpt_dir / "hybrid_best.pt")

        if distributed:
            dist.barrier()

    if distributed:
        dist.barrier()

    summary = None
    if is_main_process():
        unwrap_model(model).load_state_dict(torch.load(ckpt_dir / "hybrid_best.pt", map_location=device))
        test_loss, test_ppl, best_alpha = evaluate_hybrid(model, test_loader, device, use_amp=use_amp)
        final_alpha = unwrap_model(model).get_alpha_value()

        summary = {
            "hybrid": {
                "num_params": int(num_params),
                "num_trainable_params": int(unwrap_model(model).get_num_trainable_params()),
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val_loss),
                "best_val_ppl": float(best_val_ppl),
                "test_loss": float(test_loss),
                "test_ppl": float(test_ppl),
                "initial_alpha": float(args.init_alpha),
                "best_alpha": float(best_alpha),
                "final_alpha": float(final_alpha),
                "train_mode": args.train_mode,
                "freeze_branch_epochs": int(args.freeze_branch_epochs),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "alpha_history": alpha_history,
                "checkpoint_path": str((ckpt_dir / "hybrid_best.pt").resolve()),
                "metrics_path": str(metrics_path.resolve()),
            }
        }
        save_json(summary, run_dir / "summary.json")
        write_markdown_report(run_dir, run_meta, summary)

        print("\nFinal Hybrid Results:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nSaved hybrid artifacts to: {run_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
