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
import torch.distributed as dist
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

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process():
    return get_rank() == 0


def setup_distributed():
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


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def reduce_sum_pair(a, b, c, n, device):
    t = torch.tensor([float(a), float(b), float(c), float(n)], device=device, dtype=torch.float64)
    if is_dist_avail_and_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return [float(x.item()) for x in t]


def get_env_info(device=None):
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


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

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
    shift_s = student_logits[:, :-1, :].contiguous()
    shift_t = teacher_logits[:, :-1, :].contiguous()
    shift_y = labels[:, 1:].contiguous()

    ce = F.cross_entropy(
        shift_s.view(-1, shift_s.size(-1)),
        shift_y.view(-1),
        ignore_index=-100,
    )

    s_log_probs = F.log_softmax(shift_s / temperature, dim=-1)
    t_probs = F.softmax(shift_t / temperature, dim=-1)
    kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temperature ** 2)

    total = (1.0 - alpha) * ce + alpha * kl
    return total, ce.detach(), kl.detach()


@torch.no_grad()
def evaluate_student(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_count = 0

    for x, y in dataloader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        _, loss = model(x, labels=y)
        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)

    avg_loss = total_loss / max(total_count, 1)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl


def train_epoch_distill(student, teacher, dataloader, optimizer, scheduler, device, epoch, temperature, alpha, log_interval=50):
    student.train()
    teacher.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    total_tokens = 0
    total_count = 0
    start_time = time.time()

    iterator = dataloader
    if is_main_process():
        iterator = tqdm(dataloader, desc=f"Epoch {epoch}")

    for step, (x, y) in enumerate(iterator):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_logits, _ = teacher(x, labels=None)

        student_logits, _ = student(x, labels=None)
        loss, ce_part, kl_part = distill_loss(student_logits, teacher_logits, y, temperature=temperature, alpha=alpha)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(unwrap_model(student).parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * x.size(0)
        total_ce += ce_part.item() * x.size(0)
        total_kl += kl_part.item() * x.size(0)
        total_tokens += x.numel()
        total_count += x.size(0)

        if is_main_process() and step % log_interval == 0:
            elapsed = time.time() - start_time
            tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            iterator.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ce": f"{ce_part.item():.4f}",
                "kl": f"{kl_part.item():.4f}",
                "tok/s": f"{tok_per_sec:.0f}",
                "gnorm": f"{float(grad_norm):.2f}",
            })

    loss_sum, ce_sum, kl_sum, count_sum = reduce_sum_pair(total_loss, total_ce, total_kl, total_count, device)
    return loss_sum / max(count_sum, 1.0), ce_sum / max(count_sum, 1.0), kl_sum / max(count_sum, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Distill Transformer teacher into Grassmann student (DDP)")
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
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--reduced-dim", type=int, default=32)
    parser.add_argument("--window-sizes", type=str, default="1,2,4,8")
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--distill-alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--dataset-name", type=str, default="",
                        choices=["", "wikitext2", "wikitext103", "ptb", "tinystories"])
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--text-field", type=str, default="")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-lines", type=int, default=0)
    parser.add_argument("--encode-chars-per-batch", type=int, default=200000)

    args = parser.parse_args()
    distributed, local_rank, device = setup_distributed()
    set_seed(args.seed + get_rank())

    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_DATASETS_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_HUB_OFFLINE", None)

    if is_main_process():
        print(f"Using device: {device}")
        print(f"Distributed: {distributed}, world_size={get_world_size()}, local_rank={local_rank}")

    teacher_run_dir = Path(args.teacher_run_dir)
    teacher_summary = load_summary(teacher_run_dir)
    teacher_config = load_config(teacher_run_dir)["config"]

    teacher_cfg = teacher_config
    dataset_name = args.dataset_name or teacher_cfg.get("dataset_name", "wikitext2")
    dataset_path = args.dataset_path or teacher_cfg.get("dataset_path", "")
    text_field = args.text_field or teacher_cfg.get("text_field", "")

    max_lines = teacher_cfg.get("max_lines", args.max_lines)
    encode_chars_per_batch = teacher_cfg.get("encode_chars_per_batch", args.encode_chars_per_batch)
    tinystories_val_frac = teacher_cfg.get("tinystories_val_frac", 0.02)
    split_seed = teacher_cfg.get("split_seed", 42)

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

    if is_main_process():
        print(f"Loading dataset: {dataset_name}...")

    train_dataset = TextDataset(
        "train",
        tokenizer,
        args.max_seq_len,
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
        args.max_seq_len,
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
        args.max_seq_len,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        text_field=text_field,
        max_lines=max_lines,
        encode_chars_per_batch=encode_chars_per_batch,
        tinystories_val_frac=tinystories_val_frac,
        split_seed=split_seed,
    )

    if is_main_process():
        print(f"Distill eval dataset_name = {dataset_name}")
        print(f"Distill eval dataset_path = {dataset_path}")
        print(f"Distill eval text_field = {text_field}")
        print(f"Train chunks = {len(train_dataset)}, Val chunks = {len(val_dataset)}, Test chunks = {len(test_dataset)}")

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

    teacher = build_teacher(vocab_size, teacher_config)
    teacher_ckpt = Path(teacher_summary["transformer"]["checkpoint_path"])

    # 兼容旧机器写死的绝对路径
    if not teacher_ckpt.exists():
        teacher_ckpt = teacher_run_dir / "checkpoints" / "transformer_best.pt"

    print(f"Loading teacher checkpoint: {teacher_ckpt}")
    teacher.load_state_dict(torch.load(str(teacher_ckpt), map_location=device))
    teacher.to(device)
    teacher.eval()

    student_base = build_student(vocab_size, args)
    student_base.to(device)
    num_params = student_base.get_num_params()
    if is_main_process():
        print(f"Student parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    student = DDP(student_base, device_ids=[local_rank], output_device=local_rank) if distributed and device.type == "cuda" else student_base

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
        "env": get_env_info(device),
        "resolved_dataset_name": dataset_name,
        "resolved_dataset_path": dataset_path,
        "resolved_text_field": text_field,
    }
    if is_main_process():
        save_json(run_meta, run_dir / "config.json")

    best_val_loss = float("inf")
    best_val_ppl = float("inf")
    best_epoch = -1

    train_losses = []
    train_ce_losses = []
    train_kl_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

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
        epoch_time = time.time() - epoch_start

        if is_main_process():
            val_loss, val_ppl = evaluate_student(student, val_loader, device)
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
                torch.save(unwrap_model(student).state_dict(), ckpt_dir / "student_best.pt")

        if distributed:
            dist.barrier()

    if is_main_process():
        unwrap_model(student).load_state_dict(torch.load(ckpt_dir / "student_best.pt", map_location=device))
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

    cleanup_distributed()


if __name__ == "__main__":
    main()
