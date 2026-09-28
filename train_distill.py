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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm

sys.path.insert(0, "src")
from models import GrassmannGPTv4

from train_exp import SmallTransformer, Wikitext2Dataset


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(obj, path: Path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_env_info():
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
    }


def load_summary(run_dir: Path):
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(run_dir: Path):
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_teacher(vocab_size, cfg):
    return SmallTransformer(
        vocab_size=vocab_size,
        max_seq_len=cfg["max_seq_len"],
        model_dim=cfg["model_dim"],
        num_layers=cfg["num_layers"],
        num_heads=8,
        ff_dim=4 * cfg["model_dim"],
        dropout=cfg.get("dropout", 0.1),
    )


def build_student(vocab_size, args):
    return GrassmannGPTv4(
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        reduced_dim=args.reduced_dim,
        ff_dim=4 * args.model_dim,
        window_sizes=[int(x) for x in args.window_sizes.split(",") if x.strip()],
        dropout=args.dropout,
    )


def distill_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    # CE loss on ground truth
    shift_s = student_logits[:, :-1, :].contiguous()
    shift_t = teacher_logits[:, :-1, :].contiguous()
    shift_y = labels[:, 1:].contiguous()

    ce = F.cross_entropy(
        shift_s.view(-1, shift_s.size(-1)),
        shift_y.view(-1),
        ignore_index=-100,
    )

    # KL distillation
    s_log_probs = F.log_softmax(shift_s / temperature, dim=-1)
    t_probs = F.softmax(shift_t / temperature, dim=-1)
    kl = F.kl_div(
        s_log_probs,
        t_probs,
        reduction="batchmean",
    ) * (temperature ** 2)

    total = (1.0 - alpha) * ce + alpha * kl
    return total, ce.detach(), kl.detach()


@torch.no_grad()
def evaluate_student(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_count = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, labels=y)
        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)

    avg_loss = total_loss / total_count
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl


def train_epoch_distill(student, teacher, dataloader, optimizer, scheduler, device, epoch, temperature, alpha, log_interval=50):
    student.train()
    teacher.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    total_tokens = 0
    start_time = time.time()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits, _ = teacher(x, labels=None)

        student_logits, _ = student(x, labels=None)
        loss, ce_part, kl_part = distill_loss(
            student_logits, teacher_logits, y,
            temperature=temperature,
            alpha=alpha,
        )

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * x.size(0)
        total_ce += ce_part.item() * x.size(0)
        total_kl += kl_part.item() * x.size(0)
        total_tokens += x.numel()

        if step % log_interval == 0:
            elapsed = time.time() - start_time
            tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ce": f"{ce_part.item():.4f}",
                "kl": f"{kl_part.item():.4f}",
                "tok/s": f"{tok_per_sec:.0f}",
                "gnorm": f"{float(grad_norm):.2f}",
            })

    n = len(dataloader.dataset)
    return total_loss / n, total_ce / n, total_kl / n


def main():
    parser = argparse.ArgumentParser(description="Distill Transformer teacher into Grassmann student")
    parser.add_argument("--teacher-run-dir", type=str, required=True)
    parser.add_argument("--tokenizer-dir", type=str, default="./gpt2_local")
    parser.add_argument("--output-dir", type=str, default="outputs/distill_experiments")
    parser.add_argument("--experiment-name", type=str, default="wt2_distill_transformer_to_grassmann")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--tags", type=str, default="distill,wikitext2")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--reduced-dim", type=int, default=32)
    parser.add_argument("--window-sizes", type=str, default="1,2,4,8")
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--distill-alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)

    args = parser.parse_args()

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    teacher_run_dir = Path(args.teacher_run_dir)
    teacher_summary = load_summary(teacher_run_dir)
    teacher_config = load_config(teacher_run_dir)["config"]

    if "transformer" not in teacher_summary:
        raise ValueError(f"{teacher_run_dir} does not contain transformer results")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = now_str()
    run_dir = output_root / f"{run_id}_{args.experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    vocab_size = len(tokenizer)

    print("Loading Wikitext-2...")
    train_dataset = Wikitext2Dataset("train", tokenizer, args.max_seq_len)
    val_dataset = Wikitext2Dataset("validation", tokenizer, args.max_seq_len)
    test_dataset = Wikitext2Dataset("test", tokenizer, args.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    teacher = build_teacher(vocab_size, teacher_config)
    teacher_ckpt = teacher_summary["transformer"]["checkpoint_path"]
    print(f"Loading teacher checkpoint: {teacher_ckpt}")
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher.to(device)
    teacher.eval()

    student = build_student(vocab_size, args)
    student.to(device)

    num_params = student.get_num_params()
    print(f"Student parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    metrics_path = run_dir / "student_metrics.jsonl"

    run_meta = {
        "run_id": run_id,
        "experiment_name": args.experiment_name,
        "notes": args.notes,
        "tags": [x.strip() for x in args.tags.split(",") if x.strip()],
        "config": vars(args),
        "teacher_run_dir": str(teacher_run_dir.resolve()),
        "teacher_checkpoint": str(teacher_ckpt),
        "env": get_env_info(),
    }
    save_json(run_meta, run_dir / "config.json")

    best_val_loss = float("inf")
    best_val_ppl = float("inf")
    best_epoch = -1

    train_losses = []
    train_ce_losses = []
    train_kl_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_ce, train_kl = train_epoch_distill(
            student,
            teacher,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            temperature=args.temperature,
            alpha=args.distill_alpha,
        )
        val_loss, val_ppl = evaluate_student(student, val_loader, device)
        epoch_time = time.time() - epoch_start

        train_losses.append(float(train_loss))
        train_ce_losses.append(float(train_ce))
        train_kl_losses.append(float(train_kl))
        val_losses.append(float(val_loss))

        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_ce_loss": float(train_ce),
            "train_kl_loss": float(train_kl),
            "val_loss": float(val_loss),
            "val_ppl": float(val_ppl),
            "epoch_time_sec": float(epoch_time),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        append_jsonl(record, metrics_path)

        print(
            f"Epoch {epoch}: "
            f"Train Loss={train_loss:.4f}, "
            f"CE={train_ce:.4f}, "
            f"KL={train_kl:.4f}, "
            f"Val Loss={val_loss:.4f}, "
            f"Val PPL={val_ppl:.2f}, "
            f"Time={epoch_time:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_ppl = val_ppl
            best_epoch = epoch
            torch.save(student.state_dict(), ckpt_dir / "student_best.pt")

    student.load_state_dict(torch.load(ckpt_dir / "student_best.pt", map_location=device))
    test_loss, test_ppl = evaluate_student(student, test_loader, device)

    summary = {
        "student": {
            "num_params": int(num_params),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "best_val_ppl": float(best_val_ppl),
            "test_loss": float(test_loss),
            "test_ppl": float(test_ppl),
            "train_losses": train_losses,
            "train_ce_losses": train_ce_losses,
            "train_kl_losses": train_kl_losses,
            "val_losses": val_losses,
            "checkpoint_path": str((ckpt_dir / "student_best.pt").resolve()),
            "metrics_path": str(metrics_path.resolve()),
        }
    }
    save_json(summary, run_dir / "summary.json")

    print("\nFinal Distillation Results:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved distillation artifacts to: {run_dir}")


if __name__ == "__main__":
    main()