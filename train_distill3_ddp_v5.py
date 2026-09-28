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

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process():
    return get_rank() == 0




def configure_visible_devices(gpu_id: str):
    """
    Restrict visible CUDA devices for single-process runs.
    For torchrun/DDP, expose as many GPUs as WORLD_SIZE requires.
    """
    if not gpu_id:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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


# -----------------------------------------------------------------------------
# Path migration helpers
# -----------------------------------------------------------------------------

def normalize_path_str(path_value) -> str:
    if path_value is None:
        return ""
    return str(path_value).strip()


def parse_path_remaps(remap_specs):
    pairs = []
    for spec in remap_specs or []:
        if "=" not in spec:
            raise ValueError(f"Invalid --path-remap value: {spec!r}. Expected OLD=NEW.")
        old, new = spec.split("=", 1)
        old = old.rstrip("/")
        new = new.rstrip("/")
        if old:
            pairs.append((old, new))
    return pairs


def default_project_root() -> Path:
    return Path(__file__).resolve().parent


def build_search_roots(extra_roots=None):
    project_root = default_project_root()
    workspace_root = project_root.parent
    raw_roots = [
        project_root,
        workspace_root,
        Path.cwd(),
        Path.cwd().resolve(),
        Path("/workspace/grassmannflows"),
        Path("/workspace"),
        Path.home(),
    ]
    for root in extra_roots or []:
        if root:
            raw_roots.append(Path(root).expanduser())

    roots = []
    seen = set()
    for root in raw_roots:
        try:
            resolved = root.resolve()
        except Exception:
            resolved = root
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            roots.append(resolved)
    return roots


def _join_parts(base: Path, parts):
    if not parts:
        return base
    return base.joinpath(*parts)


def _suffix_candidates(p: Path, project_root: Path, workspace_root: Path):
    candidates = []
    parts = list(p.parts)

    anchor_map = [
        ("outputs", project_root / "outputs"),
        ("datasets", workspace_root / "datasets"),
        ("gpt2_local", project_root / "gpt2_local"),
        ("grassmann-flows", project_root),
    ]
    for anchor, base in anchor_map:
        if anchor in parts:
            idx = parts.index(anchor)
            suffix = parts[idx + 1:]
            candidates.append(_join_parts(base, suffix))

    if "datasets" in parts:
        idx = parts.index("datasets")
        suffix = parts[idx + 1:]
        candidates.append(_join_parts(workspace_root, suffix))

    candidates.append(project_root / p.name)
    candidates.append(workspace_root / p.name)
    return candidates


def resolve_migrated_path(path_value, *, run_dir=None, search_roots=None, remaps=None, kind="path", must_exist=True):
    path_str = normalize_path_str(path_value)
    if not path_str:
        return Path(path_str)

    p = Path(path_str).expanduser()
    candidates = []

    def add(candidate):
        if candidate is None:
            return
        candidate = Path(candidate).expanduser()
        if candidate not in candidates:
            candidates.append(candidate)

    add(p)
    if run_dir is not None and not p.is_absolute():
        add(Path(run_dir) / p)

    try:
        if p.exists():
            return p.resolve()
    except (PermissionError, OSError):
        pass

    project_root = default_project_root()
    workspace_root = project_root.parent

    for old, new in remaps or []:
        old_clean = old.rstrip("/")
        if path_str == old_clean or path_str.startswith(old_clean + "/"):
            suffix = path_str[len(old_clean):].lstrip("/")
            add(Path(new) / suffix if suffix else Path(new))

    known_pairs = [
        ("/root/songxin/grassmann-flows", project_root),
        ("/root/songxin/grassmannflows/grassmann-flows", project_root),
        ("/root/songxin/datasets", workspace_root / "datasets"),
        ("/root/songxin", workspace_root),
    ]
    for old_base, new_base in known_pairs:
        if path_str == old_base or path_str.startswith(old_base + "/"):
            suffix = path_str[len(old_base):].lstrip("/")
            add(Path(new_base) / suffix if suffix else Path(new_base))

    for candidate in _suffix_candidates(p, project_root, workspace_root):
        add(candidate)

    parts = list(p.parts)
    for root in search_roots or []:
        root = Path(root)
        if not p.is_absolute():
            add(root / p)
        add(root / p.name)

        if "datasets" in parts:
            idx = parts.index("datasets")
            suffix = parts[idx + 1:]
            add(_join_parts(root / "datasets", suffix))
            add(_join_parts(root, suffix))

        if "outputs" in parts:
            idx = parts.index("outputs")
            suffix = parts[idx + 1:]
            add(_join_parts(root / "outputs", suffix))

        if "grassmann-flows" in parts:
            idx = parts.index("grassmann-flows")
            suffix = parts[idx + 1:]
            add(_join_parts(root, suffix))

    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.resolve()
        except (PermissionError, OSError):
            continue

    if must_exist:
        preview = "\n".join(str(x) for x in candidates[:30])
        raise FileNotFoundError(
            f"Could not resolve migrated {kind} path: {path_str}\n"
            f"Tried candidates:\n{preview}"
        )
    return candidates[0] if candidates else p


def resolve_and_log_path(path_value, *, label, run_dir=None, search_roots=None, remaps=None, must_exist=True, printer=print):
    original = normalize_path_str(path_value)
    resolved = resolve_migrated_path(
        original,
        run_dir=run_dir,
        search_roots=search_roots,
        remaps=remaps,
        kind=label,
        must_exist=must_exist,
    )
    if original:
        try:
            original_path = Path(original).expanduser()
        except Exception:
            original_path = Path(original)
        if str(original_path) != str(resolved):
            printer(f"[path-remap] {label}: {original} -> {resolved}")
    return str(resolved)



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
def evaluate_student(model, dataloader, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    total_count = 0

    for x, y in dataloader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp):
            _, loss = model(x, labels=y)
        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)

    avg_loss = total_loss / max(total_count, 1)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl


def train_epoch_distill(student, teacher, dataloader, optimizer, scheduler, device, epoch, temperature, alpha, use_amp=False, scaler=None, log_interval=50):
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
            with autocast("cuda", enabled=use_amp):
                teacher_logits, _ = teacher(x, labels=None)

        with autocast("cuda", enabled=use_amp):
            student_logits, _ = student(x, labels=None)
            loss, ce_part, kl_part = distill_loss(student_logits, teacher_logits, y, temperature=temperature, alpha=alpha)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(unwrap_model(student).parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
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
    parser.add_argument("--gpu-id", type=str, default="", help="Single GPU id or comma-separated visible GPU ids")
    parser.add_argument("--output-dir", type=str, default="outputs/distill_experiments")
    parser.add_argument("--experiment-name", type=str, default="wt2_distill_transformer_to_grassmann")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--tags", type=str, default="distill,wikitext2")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision distillation")

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
    parser.add_argument("--path-remap", action="append", default=[],
                        help="Manual old=new path remap for migrated servers; can be passed multiple times")
    parser.add_argument("--search-root", action="append", default=[],
                        help="Extra root to search when resolving migrated dataset/checkpoint paths")

    args = parser.parse_args()
    configure_visible_devices(args.gpu_id)
    distributed, local_rank, device = setup_distributed()
    set_seed(args.seed + get_rank())
    use_amp = args.amp and device.type == "cuda"
    remaps = parse_path_remaps(args.path_remap)
    search_roots = build_search_roots(args.search_root)

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
        print(f"Visible CUDA devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}")
        print(f"AMP enabled: {use_amp}")

    args.teacher_run_dir = resolve_and_log_path(
        args.teacher_run_dir,
        label="teacher_run_dir",
        search_roots=search_roots,
        remaps=remaps,
        must_exist=True,
        printer=print if is_main_process() else (lambda *a, **k: None),
    )
    teacher_run_dir = Path(args.teacher_run_dir)
    teacher_summary = load_summary(teacher_run_dir)
    teacher_config = load_config(teacher_run_dir)["config"]

    teacher_cfg = teacher_config
    dataset_name = args.dataset_name or teacher_cfg.get("dataset_name", "wikitext2")
    dataset_path = args.dataset_path or teacher_cfg.get("dataset_path", "")
    dataset_path = resolve_and_log_path(
        dataset_path,
        label="dataset_path",
        search_roots=search_roots,
        remaps=remaps,
        must_exist=True,
        printer=print if is_main_process() else (lambda *a, **k: None),
    ) if dataset_path else dataset_path
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

    args.tokenizer_dir = resolve_and_log_path(
        args.tokenizer_dir,
        label="tokenizer_dir",
        search_roots=search_roots,
        remaps=remaps,
        must_exist=True,
        printer=print if is_main_process() else (lambda *a, **k: None),
    )
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
    scaler = GradScaler("cuda", enabled=use_amp)

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
            use_amp=use_amp,
            scaler=scaler,
        )
        epoch_time = time.time() - epoch_start

        if is_main_process():
            val_loss, val_ppl = evaluate_student(student, val_loader, device, use_amp=use_amp)
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
        test_loss, test_ppl = evaluate_student(student, test_loader, device, use_amp=use_amp)

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
