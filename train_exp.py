#WT2的训练
import os
import sys
import json
import time
import argparse
import platform
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm

sys.path.insert(0, "src")
from models import GrassmannGPTv4
# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(obj, path: Path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_env_info():
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
    }


# -----------------------------------------------------------------------------
# Small Transformer Baseline
# -----------------------------------------------------------------------------

class SmallTransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        normed = self.ln1(x)
        if attn_mask is None:
            seq_len = x.size(1)
            attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=x.device, dtype=x.dtype
            )
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask, is_causal=True)
        x = x + self.dropout(attn_out)

        normed = self.ln2(x)
        x = x + self.ffn(normed)
        return x


class SmallTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 256,
        model_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = None,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.model_dim = model_dim

        ff_dim = ff_dim or 4 * model_dim

        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(max_seq_len, model_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SmallTransformerBlock(model_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=device))
        hidden_states = self.embedding_dropout(tok_emb + pos_emb)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class Wikitext2Dataset(Dataset):
    def __init__(self, split: str, tokenizer, max_seq_len: int = 256):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.split = split

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        non_empty = [t for t in dataset["text"] if t.strip()]
        all_text = "\n".join(non_empty)

        print(f"[{split}] non-empty lines = {len(non_empty)}")
        print(f"[{split}] chars = {len(all_text)}")

        self.tokens = tokenizer.encode(all_text, add_special_tokens=False)

        print(f"[{split}] token count = {len(self.tokens)}")
        print(f"[{split}] first 20 tokens = {self.tokens[:20]}")

        self.num_chunks = len(self.tokens) // max_seq_len
        self.tokens = self.tokens[: self.num_chunks * max_seq_len]

        self.stats = {
            "split": split,
            "non_empty_lines": len(non_empty),
            "char_count": len(all_text),
            "token_count_before_trim": len(tokenizer.encode(all_text, add_special_tokens=False)),
            "num_chunks": self.num_chunks,
            "max_seq_len": max_seq_len,
        }

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = idx * self.max_seq_len
        chunk = self.tokens[start:start + self.max_seq_len]
        x = torch.tensor(chunk, dtype=torch.long)
        return x, x.clone()


# -----------------------------------------------------------------------------
# Training / Eval
# -----------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, log_interval=50):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        _, loss = model(x, labels=y)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * x.size(0)
        total_tokens += x.numel()

        if step % log_interval == 0:
            elapsed = time.time() - start_time
            tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ppl": f"{loss.exp().item():.2f}",
                "tok/s": f"{tok_per_sec:.0f}",
                "gnorm": f"{float(grad_norm):.2f}",
            })

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_count = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, labels=y)
        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)

    avg_loss = total_loss / total_count
    return avg_loss, torch.exp(torch.tensor(avg_loss)).item()


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def make_run_dir(output_root: Path, experiment_name: str):
    run_id = now_str()
    safe_name = experiment_name.replace(" ", "_")
    run_dir = output_root / f"{run_id}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def write_markdown_report(run_dir: Path, run_meta: dict, results: dict):
    lines = []
    lines.append(f"# Experiment Report: {run_meta['experiment_name']}")
    lines.append("")
    lines.append(f"- Run ID: `{run_meta['run_id']}`")
    lines.append(f"- Time: `{run_meta['start_time']}`")
    lines.append(f"- Notes: {run_meta.get('notes', '')}")
    lines.append(f"- Tags: {', '.join(run_meta.get('tags', []))}")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(run_meta["config"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(run_meta["env"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Dataset Stats")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(run_meta["dataset_stats"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Final Results")
    lines.append("")
    lines.append("| Model | Params | Best Val PPL | Test PPL |")
    lines.append("|---|---:|---:|---:|")
    for model_name, item in results.items():
        lines.append(
            f"| {model_name} | {item['num_params']} | {item['best_val_ppl']:.2f} | {item['test_ppl']:.2f} |"
        )

    if "grassmann" in results and "transformer" in results:
        g = results["grassmann"]
        t = results["transformer"]
        ppl_ratio = g["test_ppl"] / t["test_ppl"]
        gap_percent = (ppl_ratio - 1) * 100
        lines.append("")
        lines.append("## Comparison")
        lines.append("")
        lines.append(f"- Grassmann / Transformer Test PPL ratio: **{ppl_ratio:.3f}**")
        lines.append(f"- Gap: **{gap_percent:.2f}%**")

    with open(run_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Wikitext-2 Paper Reproduction with Experiment Tracking")
    parser.add_argument("--model", type=str, default="both", choices=["grassmann", "transformer", "both"])
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", type=str, default="outputs/experiments")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--experiment-name", type=str, default="wikitext2_reproduction")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--tags", type=str, default="baseline,wikitext2")

    parser.add_argument("--reduced-dim", type=int, default=32)
    parser.add_argument("--window-sizes", type=str, default="1,2,4,8,12,16")
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    run_id, run_dir = make_run_dir(output_root, args.experiment_name)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    window_sizes = [int(x) for x in args.window_sizes.split(",") if x.strip()]

    # force offline if needed
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_local", local_files_only=True)
    vocab_size = len(tokenizer)

    print("Loading Wikitext-2...")
    train_dataset = Wikitext2Dataset("train", tokenizer, args.max_seq_len)
    val_dataset = Wikitext2Dataset("validation", tokenizer, args.max_seq_len)
    test_dataset = Wikitext2Dataset("test", tokenizer, args.max_seq_len)

    print(f"Train: {len(train_dataset)} chunks, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    run_meta = {
        "run_id": run_id,
        "experiment_name": args.experiment_name,
        "start_time": datetime.now().isoformat(),
        "notes": args.notes,
        "tags": tags,
        "config": vars(args),
        "env": get_env_info(),
        "dataset_stats": {
            "train": train_dataset.stats,
            "validation": val_dataset.stats,
            "test": test_dataset.stats,
        },
    }
    save_json(run_meta, run_dir / "config.json")

    results = {}

    models_to_train = []
    if args.model in ["grassmann", "both"]:
        models_to_train.append("grassmann")
    if args.model in ["transformer", "both"]:
        models_to_train.append("transformer")

    for model_type in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training: {model_type.upper()}")
        print(f"{'='*60}")

        if model_type == "grassmann":
            model = GrassmannGPTv4(
                vocab_size=vocab_size,
                max_seq_len=args.max_seq_len,
                model_dim=args.model_dim,
                num_layers=args.num_layers,
                reduced_dim=args.reduced_dim,
                ff_dim=4 * args.model_dim,
                window_sizes=window_sizes,
                dropout=args.dropout,
            )
        else:
            model = SmallTransformer(
                vocab_size=vocab_size,
                max_seq_len=args.max_seq_len,
                model_dim=args.model_dim,
                num_layers=args.num_layers,
                num_heads=8,
                ff_dim=4 * args.model_dim,
                dropout=args.dropout,
            )

        model = model.to(device)
        num_params = model.get_num_params()
        print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        total_steps = len(train_loader) * args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

        best_val_loss = float("inf")
        best_val_ppl = float("inf")
        best_epoch = -1
        train_losses = []
        val_losses = []
        metrics_path = run_dir / f"{model_type}_metrics.jsonl"

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
            val_loss, val_ppl = evaluate(model, val_loader, device)
            epoch_time = time.time() - epoch_start

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            epoch_record = {
                "model": model_type,
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_ppl": float(torch.exp(torch.tensor(train_loss)).item()),
                "val_loss": float(val_loss),
                "val_ppl": float(val_ppl),
                "epoch_time_sec": float(epoch_time),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            append_jsonl(epoch_record, metrics_path)

            print(
                f"Epoch {epoch}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val PPL: {val_ppl:.2f}, "
                f"Time: {epoch_time:.1f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_ppl = val_ppl
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_dir / f"{model_type}_best.pt")

        model.load_state_dict(torch.load(ckpt_dir / f"{model_type}_best.pt", map_location=device))
        test_loss, test_ppl = evaluate(model, test_loader, device)

        print(f"\nFinal Results for {model_type.upper()}:")
        print(f"  Best Epoch: {best_epoch}")
        print(f"  Best Val Loss: {best_val_loss:.4f}, Best Val PPL: {best_val_ppl:.2f}")
        print(f"  Test Loss: {test_loss:.4f}, Test PPL: {test_ppl:.2f}")

        results[model_type] = {
            "num_params": int(num_params),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "best_val_ppl": float(best_val_ppl),
            "test_loss": float(test_loss),
            "test_ppl": float(test_ppl),
            "train_losses": [float(x) for x in train_losses],
            "val_losses": [float(x) for x in val_losses],
            "checkpoint_path": str((ckpt_dir / f"{model_type}_best.pt").resolve()),
            "metrics_path": str(metrics_path.resolve()),
        }

    save_json(results, run_dir / "summary.json")
    write_markdown_report(run_dir, run_meta, results)

    if len(results) == 2:
        print(f"\n{'='*60}")
        print("COMPARISON: Paper Reproduction on Wikitext-2")
        print(f"{'='*60}")

        g = results["grassmann"]
        t = results["transformer"]

        print(f"{'Model':<20} {'Params':<12} {'Val PPL':<12} {'Test PPL':<12}")
        print("-" * 56)
        print(f"{'Grassmann':<20} {g['num_params']/1e6:.2f}M{'':<6} {g['best_val_ppl']:<12.2f} {g['test_ppl']:<12.2f}")
        print(f"{'Transformer':<20} {t['num_params']/1e6:.2f}M{'':<6} {t['best_val_ppl']:<12.2f} {t['test_ppl']:<12.2f}")
        print("-" * 56)

        ppl_ratio = g["test_ppl"] / t["test_ppl"]
        gap_percent = (ppl_ratio - 1) * 100

        comparison = {
            "grassmann_test_ppl": g["test_ppl"],
            "transformer_test_ppl": t["test_ppl"],
            "ppl_ratio": float(ppl_ratio),
            "gap_percent": float(gap_percent),
        }
        save_json(comparison, run_dir / "comparison.json")

        print(f"\nGrassmann/Transformer PPL ratio: {ppl_ratio:.3f}")
        print(f"Gap: {gap_percent:.1f}% (Paper claims 10-15%)")

        if gap_percent <= 15:
            print("RESULT: Paper claim VERIFIED - within 15% gap")
        else:
            print(f"RESULT: Paper claim NOT verified - gap is {gap_percent:.1f}%")

    print(f"\nSaved experiment artifacts to: {run_dir}")


if __name__ == "__main__":
    main()