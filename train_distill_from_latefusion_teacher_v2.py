
import os
import sys
import json
import time
import math
import argparse
import platform
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List

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

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process():
    return get_rank() == 0


def configure_visible_devices(gpu_id: str):
    if gpu_id:
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


def reduce_sum(values: List[float], device):
    t = torch.tensor([float(v) for v in values], device=device, dtype=torch.float64)
    if is_dist_avail_and_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return [float(x.item()) for x in t]


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


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def _safe_exists(path_obj: Path) -> bool:
    try:
        return path_obj.exists()
    except (PermissionError, OSError):
        return False


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
        c = Path(candidate).expanduser()
        candidates.append(c)

    if p.is_absolute():
        add(p)

    for old, new in remaps or []:
        if path_str.startswith(old):
            add(Path(new + path_str[len(old):]))

    if not p.is_absolute():
        if run_dir is not None:
            add(Path(run_dir) / p)
        add(Path.cwd() / p)
        add(default_project_root() / p)
        add(default_project_root().parent / p)

    project_root = default_project_root()
    workspace_root = project_root.parent
    for cand in _suffix_candidates(p, project_root, workspace_root):
        add(cand)

    for root in search_roots or []:
        try:
            add(Path(root) / p.name)
        except Exception:
            pass

    seen = set()
    deduped = []
    for cand in candidates:
        try:
            key = str(cand.resolve())
        except Exception:
            key = str(cand)
        if key not in seen:
            seen.add(key)
            deduped.append(cand)

    if must_exist:
        for cand in deduped:
            if _safe_exists(cand):
                return cand
        raise FileNotFoundError(
            f"Could not resolve existing {kind}: {path_value}\nTried:\n" + "\n".join(str(x) for x in deduped[:20])
        )
    return deduped[0] if deduped else p


def resolve_and_log_path(path_value, *, label, run_dir=None, search_roots=None, remaps=None, must_exist=True, printer=print):
    original = normalize_path_str(path_value)
    if not original:
        return original
    resolved = resolve_migrated_path(
        original,
        run_dir=run_dir,
        search_roots=search_roots,
        remaps=remaps,
        kind=label,
        must_exist=must_exist,
    )
    resolved_str = str(resolved)
    if resolved_str != original:
        printer(f"[path-remap] {label}: {original} -> {resolved_str}")
    return resolved_str


def choose_dataset_value(cli_value, *candidates, default=""):
    if cli_value not in ["", None, 0, -1, -1.0]:
        return cli_value
    for value in candidates:
        if value not in ["", None, 0, -1, -1.0]:
            return value
    return default


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class HybridLateFusionAlphaModel(nn.Module):
    """
    Same late-fusion teacher architecture used in training.
    """

    def __init__(self, grassmann: nn.Module, transformer: nn.Module, init_alpha: float = 0.5, late_k: int = 1):
        super().__init__()
        self.grassmann = grassmann
        self.transformer = transformer
        self.num_layers = int(min(len(self.grassmann.blocks), len(self.transformer.blocks)))
        if self.num_layers <= 0:
            raise ValueError("Could not infer a positive number of layers from branch models.")
        self.late_k = int(max(1, min(int(late_k), self.num_layers)))
        init_logit = math.log(init_alpha / max(1e-8, 1.0 - init_alpha))
        self.logit_alpha = nn.Parameter(torch.full((self.late_k,), float(init_logit), dtype=torch.float32))

    def alpha(self):
        return torch.sigmoid(self.logit_alpha)

    def _transformer_hidden_per_layer(self, input_ids):
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

    def _grassmann_hidden_per_layer(self, input_ids):
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

    def forward(self, input_ids, labels=None):
        alpha_vec = self.alpha()
        t_hidden_all = self._transformer_hidden_per_layer(input_ids)
        g_hidden_all = self._grassmann_hidden_per_layer(input_ids)
        t_hidden_sel = t_hidden_all[-self.late_k:]
        g_hidden_sel = g_hidden_all[-self.late_k:]

        fused_logits_all = []
        for alpha_i, h_t, h_g in zip(alpha_vec, t_hidden_sel, g_hidden_sel):
            logits_t = self.transformer.lm_head(self.transformer.ln_f(h_t))
            logits_g = self.grassmann.lm_head(self.grassmann.ln_f(h_g))
            fused_logits_all.append(alpha_i * logits_t + (1.0 - alpha_i) * logits_g)

        logits = torch.stack(fused_logits_all, dim=0).mean(dim=0)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss


def create_grassmann(vocab_size: int, max_seq_len: int, model_dim: int, num_layers: int,
                     reduced_dim: int, dropout: float, window_sizes: str):
    ws = [int(x) for x in str(window_sizes).split(",") if str(x).strip()]
    return GrassmannGPTv4(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        model_dim=model_dim,
        num_layers=num_layers,
        reduced_dim=reduced_dim,
        ff_dim=4 * model_dim,
        window_sizes=ws,
        dropout=dropout,
    )


def create_transformer(vocab_size: int, max_seq_len: int, model_dim: int, num_layers: int,
                       num_heads: int, dropout: float):
    return SmallTransformer(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        model_dim=model_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=4 * model_dim,
        dropout=dropout,
    )


def instantiate_from_config(model_type: str, vocab_size: int, max_seq_len: int, cfg: Dict[str, Any]):
    model_dim = int(cfg.get("model_dim", 256))
    num_layers = int(cfg.get("num_layers", 6))
    dropout = float(cfg.get("dropout", 0.1))
    if model_type == "grassmann":
        reduced_dim = int(cfg.get("reduced_dim", 32))
        window_sizes = cfg.get("window_sizes", "1,2,4,8")
        if isinstance(window_sizes, list):
            window_sizes = ",".join(str(x) for x in window_sizes)
        return create_grassmann(vocab_size, max_seq_len, model_dim, num_layers, reduced_dim, dropout, window_sizes)
    elif model_type == "transformer":
        num_heads = int(cfg.get("num_heads", 8))
        return create_transformer(vocab_size, max_seq_len, model_dim, num_layers, num_heads, dropout)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


def _safe_int(value, default=None):
    try:
        iv = int(value)
        return iv
    except Exception:
        return default


def infer_seq_len_from_state_dict(state_dict, fallback=256):
    """
    Infer max_seq_len from checkpoint weights instead of trusting old config.json.
    Works for both GrassmannGPTv4 and SmallTransformer.
    """
    for key in [
        "position_embedding.weight",  # nn.Embedding
        "position_embedding",         # raw tensor fallback
        "pos_embedding.weight",
        "pos_embedding",
    ]:
        if key in state_dict:
            tensor = state_dict[key]
            try:
                size0 = int(tensor.shape[0])
                if size0 > 0:
                    return size0
            except Exception:
                pass
    return int(fallback)


def choose_positive_seq_len(*candidates, default=256):
    vals = []
    for c in candidates:
        iv = _safe_int(c, default=None)
        if iv is not None and iv > 0:
            vals.append(iv)
    return int(vals[0] if vals else default)


# -----------------------------------------------------------------------------
# Loss / eval
# -----------------------------------------------------------------------------

def kd_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.7):
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
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp and device.type == "cuda"):
            _, loss = model(x, labels=y)
        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)
    loss_sum, count_sum = reduce_sum([total_loss, total_count], device)
    avg_loss = loss_sum / max(count_sum, 1.0)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl


def train_epoch(student, teacher, dataloader, optimizer, scheduler, scaler, device,
                epoch, temperature, alpha, use_amp=False, log_interval=50):
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
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            with autocast("cuda", enabled=use_amp and device.type == "cuda"):
                teacher_logits, _ = teacher(x, labels=None)

        with autocast("cuda", enabled=use_amp and device.type == "cuda"):
            student_logits, _ = student(x, labels=None)
            loss, ce_part, kl_part = kd_loss(student_logits, teacher_logits, y, temperature=temperature, alpha=alpha)

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
            tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
            iterator.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ce": f"{ce_part.item():.4f}",
                "kl": f"{kl_part.item():.4f}",
                "tok/s": f"{tok_per_sec:.0f}",
                "gnorm": f"{float(grad_norm):.2f}",
            })

    loss_sum, ce_sum, kl_sum, count_sum = reduce_sum([total_loss, total_ce, total_kl, total_count], device)
    return (
        loss_sum / max(count_sum, 1.0),
        ce_sum / max(count_sum, 1.0),
        kl_sum / max(count_sum, 1.0),
    )


# -----------------------------------------------------------------------------
# Teacher loading
# -----------------------------------------------------------------------------

def load_teacher_and_context(teacher_run_dir: Path, vocab_size: int, search_roots, remaps, device):
    teacher_meta = load_json(teacher_run_dir / "config.json")
    teacher_summary = load_json(teacher_run_dir / "summary.json")
    teacher_cfg = teacher_meta.get("config", {})
    source_runs = teacher_meta.get("source_runs", {})

    g_run_dir = resolve_migrated_path(
        source_runs.get("grassmann_run_dir", ""),
        search_roots=search_roots,
        remaps=remaps,
        kind="grassmann_run_dir",
        must_exist=True,
    )
    t_run_dir = resolve_migrated_path(
        source_runs.get("transformer_run_dir", ""),
        search_roots=search_roots,
        remaps=remaps,
        kind="transformer_run_dir",
        must_exist=True,
    )

    g_cfg = load_json(Path(g_run_dir) / "config.json").get("config", {})
    t_cfg = load_json(Path(t_run_dir) / "config.json").get("config", {})
    g_sum = load_json(Path(g_run_dir) / "summary.json")
    t_sum = load_json(Path(t_run_dir) / "summary.json")

    g_ckpt = source_runs.get("grassmann_checkpoint") or g_sum.get("grassmann", {}).get("checkpoint_path") or str(Path(g_run_dir) / "checkpoints" / "grassmann_best.pt")
    t_ckpt = source_runs.get("transformer_checkpoint") or t_sum.get("transformer", {}).get("checkpoint_path") or str(Path(t_run_dir) / "checkpoints" / "transformer_best.pt")
    hybrid_ckpt = teacher_summary.get("hybrid", {}).get("checkpoint_path") or str(teacher_run_dir / "checkpoints" / "hybrid_best.pt")

    g_ckpt = resolve_migrated_path(g_ckpt, run_dir=teacher_run_dir, search_roots=search_roots, remaps=remaps, kind="grassmann_checkpoint", must_exist=True)
    t_ckpt = resolve_migrated_path(t_ckpt, run_dir=teacher_run_dir, search_roots=search_roots, remaps=remaps, kind="transformer_checkpoint", must_exist=True)
    hybrid_ckpt = resolve_migrated_path(hybrid_ckpt, run_dir=teacher_run_dir, search_roots=search_roots, remaps=remaps, kind="hybrid_checkpoint", must_exist=True)

    # Load raw state dicts first, because old config.json may contain broken max_seq_len=0.
    g_state = torch.load(g_ckpt, map_location=device, weights_only=True)
    t_state = torch.load(t_ckpt, map_location=device, weights_only=True)

    seq_len_from_g_ckpt = infer_seq_len_from_state_dict(g_state, fallback=256)
    seq_len_from_t_ckpt = infer_seq_len_from_state_dict(t_state, fallback=256)

    max_seq_len = choose_positive_seq_len(
        teacher_cfg.get("max_seq_len"),
        g_cfg.get("max_seq_len"),
        t_cfg.get("max_seq_len"),
        seq_len_from_g_ckpt,
        seq_len_from_t_ckpt,
        default=max(seq_len_from_g_ckpt, seq_len_from_t_ckpt, 256),
    )

    late_k = int(teacher_summary.get("hybrid", {}).get("late_k", teacher_cfg.get("late_k", 1)))
    init_alpha = float(teacher_summary.get("hybrid", {}).get("final_alpha", teacher_cfg.get("init_alpha", 0.5)))

    grassmann = instantiate_from_config("grassmann", vocab_size, max_seq_len, g_cfg)
    transformer = instantiate_from_config("transformer", vocab_size, max_seq_len, t_cfg)

    if is_main_process():
        print(f"Loading source grassmann checkpoint: {g_ckpt}")
        print(f"Loading source transformer checkpoint: {t_ckpt}")
        print(f"Loading late-fusion teacher checkpoint: {hybrid_ckpt}")
        print(f"Inferred teacher max_seq_len={max_seq_len} (g_ckpt={seq_len_from_g_ckpt}, t_ckpt={seq_len_from_t_ckpt})")

    grassmann.load_state_dict(g_state)
    transformer.load_state_dict(t_state)

    teacher = HybridLateFusionAlphaModel(grassmann, transformer, init_alpha=init_alpha, late_k=late_k)
    teacher.load_state_dict(torch.load(hybrid_ckpt, map_location=device, weights_only=True))
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    context = {
        "teacher_meta": teacher_meta,
        "teacher_summary": teacher_summary,
        "teacher_cfg": teacher_cfg,
        "g_cfg": g_cfg,
        "t_cfg": t_cfg,
        "g_summary": g_sum,
        "t_summary": t_sum,
        "g_run_dir": str(Path(g_run_dir).resolve()),
        "t_run_dir": str(Path(t_run_dir).resolve()),
        "g_ckpt": str(Path(g_ckpt).resolve()),
        "t_ckpt": str(Path(t_ckpt).resolve()),
        "hybrid_ckpt": str(Path(hybrid_ckpt).resolve()),
        "max_seq_len": max_seq_len,
        "late_k": late_k,
    }
    return teacher, context


def write_markdown_report(run_dir: Path, run_meta: Dict[str, Any], summary: Dict[str, Any]) -> None:
    stu = summary["student"]
    lines = []
    lines.append("# Late-fusion Teacher Distillation Report")
    lines.append("")
    lines.append(f"- Experiment: `{run_meta['experiment_name']}`")
    lines.append(f"- Student type: `{run_meta['config']['student_type']}`")
    lines.append(f"- Student config: dim={run_meta['config']['model_dim']}, layers={run_meta['config']['num_layers']}, "
                 f"reduced_dim={run_meta['config'].get('reduced_dim')}, heads={run_meta['config'].get('num_heads')}")
    lines.append(f"- Teacher run: `{run_meta['source_teacher']['teacher_run_dir']}`")
    lines.append(f"- Teacher best test ppl: `{stu['teacher_test_ppl']:.4f}`")
    lines.append(f"- Grassmann baseline test ppl: `{stu.get('grassmann_baseline_test_ppl')}`")
    lines.append(f"- Transformer baseline test ppl: `{stu.get('transformer_baseline_test_ppl')}`")
    lines.append("")
    lines.append("## Final student metrics")
    lines.append("")
    lines.append(f"- Best epoch: `{stu['best_epoch']}`")
    lines.append(f"- Best val ppl: `{stu['best_val_ppl']:.4f}`")
    lines.append(f"- Test ppl: `{stu['test_ppl']:.4f}`")
    lines.append(f"- Params: `{stu['num_params']}`")
    lines.append(f"- Beats grassmann baseline: `{stu['beats_grassmann_baseline']}`")
    lines.append(f"- Beats transformer baseline: `{stu['beats_transformer_baseline']}`")
    lines.append(f"- Beats both baselines: `{stu['beats_both_baselines']}`")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Distill a late-fusion hybrid teacher into a lightweight student")
    parser.add_argument("--teacher-run-dir", type=str, required=True)
    parser.add_argument("--student-type", type=str, default="grassmann", choices=["grassmann", "transformer"])
    parser.add_argument("--tokenizer-dir", type=str, default="./gpt2_local")
    parser.add_argument("--gpu-id", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs/distill_experiments")
    parser.add_argument("--experiment-name", type=str, default="ptb_distill_latefusion_teacher")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--tags", type=str, default="distill,latefusion,ptb")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)

    parser.add_argument("--model-dim", type=int, default=224)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--reduced-dim", type=int, default=56)
    parser.add_argument("--window-sizes", type=str, default="1,2,4")
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--distill-alpha", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=2.0)

    parser.add_argument("--dataset-name", type=str, default="")
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--text-field", type=str, default="")
    parser.add_argument("--max-seq-len", type=int, default=0)
    parser.add_argument("--max-lines", type=int, default=-1)
    parser.add_argument("--encode-chars-per-batch", type=int, default=-1)
    parser.add_argument("--tinystories-val-frac", type=float, default=-1.0)
    parser.add_argument("--split-seed", type=int, default=-1)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--path-remap", action="append", default=[])
    parser.add_argument("--search-root", action="append", default=[])

    args = parser.parse_args()
    configure_visible_devices(args.gpu_id)
    distributed, local_rank, device = setup_distributed()
    set_seed(args.seed + get_rank())
    use_amp = args.amp and device.type == "cuda"
    remaps = parse_path_remaps(args.path_remap)
    search_roots = build_search_roots(args.search_root)

    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_DATASETS_OFFLINE", None)
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

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

    teacher, ctx = load_teacher_and_context(teacher_run_dir, vocab_size, search_roots, remaps, device)

    teacher_cfg = ctx["teacher_cfg"]
    dataset_name = choose_dataset_value(args.dataset_name, teacher_cfg.get("dataset_name"), default="ptb")
    dataset_path = choose_dataset_value(args.dataset_path, teacher_cfg.get("dataset_path"), default="")
    dataset_path = resolve_and_log_path(
        dataset_path,
        label="dataset_path",
        search_roots=search_roots,
        remaps=remaps,
        must_exist=True,
        printer=print if is_main_process() else (lambda *a, **k: None),
    ) if dataset_path else dataset_path
    text_field = choose_dataset_value(args.text_field, teacher_cfg.get("text_field"), default="")

    max_seq_len = int(choose_dataset_value(args.max_seq_len, teacher_cfg.get("max_seq_len"), ctx["max_seq_len"], default=256))
    max_lines = int(choose_dataset_value(args.max_lines, teacher_cfg.get("max_lines"), default=0))
    encode_chars_per_batch = int(choose_dataset_value(args.encode_chars_per_batch, teacher_cfg.get("encode_chars_per_batch"), default=200000))
    tinystories_val_frac = float(choose_dataset_value(args.tinystories_val_frac, teacher_cfg.get("tinystories_val_frac"), default=0.02))
    split_seed = int(choose_dataset_value(args.split_seed, teacher_cfg.get("split_seed"), default=42))

    if is_main_process():
        print(f"Distill dataset_name = {dataset_name}")
        print(f"Distill dataset_path = {dataset_path}")
        print(f"Distill text_field = {text_field}")
        print(f"Distill max_seq_len = {max_seq_len}")

    train_dataset = TextDataset(
        "train", tokenizer, max_seq_len,
        dataset_name=dataset_name, dataset_path=dataset_path, text_field=text_field,
        max_lines=max_lines, encode_chars_per_batch=encode_chars_per_batch,
        tinystories_val_frac=tinystories_val_frac, split_seed=split_seed,
    )
    val_dataset = TextDataset(
        "validation", tokenizer, max_seq_len,
        dataset_name=dataset_name, dataset_path=dataset_path, text_field=text_field,
        max_lines=max_lines, encode_chars_per_batch=encode_chars_per_batch,
        tinystories_val_frac=tinystories_val_frac, split_seed=split_seed,
    )
    test_dataset = TextDataset(
        "test", tokenizer, max_seq_len,
        dataset_name=dataset_name, dataset_path=dataset_path, text_field=text_field,
        max_lines=max_lines, encode_chars_per_batch=encode_chars_per_batch,
        tinystories_val_frac=tinystories_val_frac, split_seed=split_seed,
    )

    if is_main_process():
        print(f"Train chunks={len(train_dataset)}, Val chunks={len(val_dataset)}, Test chunks={len(test_dataset)}")

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            sampler=val_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             sampler=test_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    student_cfg = {
        "model_dim": args.model_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "num_heads": args.num_heads,
        "reduced_dim": args.reduced_dim,
        "window_sizes": args.window_sizes,
    }
    student = instantiate_from_config(args.student_type, vocab_size, max_seq_len, student_cfg).to(device)
    teacher = teacher.to(device)

    if distributed:
        student = DDP(student, device_ids=[local_rank] if device.type == "cuda" else None)

    if is_main_process():
        print(f"Teacher params: {get_num_params(teacher):,}")
        print(f"Student params: {get_num_params(unwrap_model(student)):,}")
        print(f"Teacher late_k: {ctx['late_k']}")
        print(f"Teacher source grassmann test ppl: {ctx['g_summary'].get('grassmann', {}).get('test_ppl')}")
        print(f"Teacher source transformer test ppl: {ctx['t_summary'].get('transformer', {}).get('test_ppl')}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"{now_str()}_{args.experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "distill_metrics.jsonl"

    run_meta = {
        "run_id": run_dir.name.split("_", 1)[0],
        "experiment_name": args.experiment_name,
        "start_time": datetime.now().isoformat(),
        "notes": args.notes,
        "tags": [t.strip() for t in args.tags.split(",") if t.strip()],
        "config": vars(args),
        "env": get_env_info(device),
        "source_teacher": {
            "teacher_run_dir": str(teacher_run_dir.resolve()),
            "teacher_checkpoint": ctx["hybrid_ckpt"],
            "grassmann_run_dir": ctx["g_run_dir"],
            "transformer_run_dir": ctx["t_run_dir"],
        },
        "dataset_stats": {
            "train": train_dataset.stats,
            "validation": val_dataset.stats,
            "test": test_dataset.stats,
        },
    }
    if is_main_process():
        save_json(run_meta, run_dir / "config.json")

    optimizer = torch.optim.AdamW(unwrap_model(student).parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler("cuda", enabled=use_amp)

    if distributed:
        dist.barrier()

    if is_main_process():
        init_val_loss, init_val_ppl = evaluate_student(student, val_loader, device, use_amp=use_amp)
        print(f"Initial student val loss={init_val_loss:.4f}, val ppl={init_val_ppl:.4f}")
    if distributed:
        dist.barrier()

    best_val_loss = float("inf")
    best_val_ppl = float("inf")
    best_epoch = -1
    train_losses = []
    train_ce = []
    train_kl = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_start = time.time()
        loss, ce_part, kl_part = train_epoch(
            student, teacher, train_loader, optimizer, scheduler, scaler, device, epoch,
            temperature=args.temperature, alpha=args.distill_alpha, use_amp=use_amp, log_interval=args.log_interval
        )
        epoch_time = time.time() - epoch_start

        val_loss, val_ppl = evaluate_student(student, val_loader, device, use_amp=use_amp)

        train_losses.append(float(loss))
        train_ce.append(float(ce_part))
        train_kl.append(float(kl_part))
        val_losses.append(float(val_loss))

        if is_main_process():
            record = {
                "epoch": epoch,
                "train_loss": float(loss),
                "train_ce": float(ce_part),
                "train_kl": float(kl_part),
                "val_loss": float(val_loss),
                "val_ppl": float(val_ppl),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "epoch_time_sec": float(epoch_time),
            }
            append_jsonl(record, metrics_path)
            print(
                f"Epoch {epoch}: train_loss={loss:.4f}, train_ce={ce_part:.4f}, train_kl={kl_part:.4f}, "
                f"val_loss={val_loss:.4f}, val_ppl={val_ppl:.4f}, time={epoch_time:.1f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                best_val_ppl = float(val_ppl)
                best_epoch = epoch
                torch.save(unwrap_model(student).state_dict(), ckpt_dir / "student_best.pt")

        if distributed:
            dist.barrier()

    if is_main_process():
        unwrap_model(student).load_state_dict(torch.load(ckpt_dir / "student_best.pt", map_location=device, weights_only=True))
    if distributed:
        dist.barrier()

    test_loss, test_ppl = evaluate_student(student, test_loader, device, use_amp=use_amp)

    if is_main_process():
        g_base = ctx["g_summary"].get("grassmann", {}).get("test_ppl")
        t_base = ctx["t_summary"].get("transformer", {}).get("test_ppl")
        teacher_test_ppl = ctx["teacher_summary"].get("hybrid", {}).get("test_ppl")
        beats_g = bool(g_base is not None and test_ppl < float(g_base))
        beats_t = bool(t_base is not None and test_ppl < float(t_base))
        summary = {
            "student": {
                "student_type": args.student_type,
                "num_params": int(get_num_params(unwrap_model(student))),
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val_loss),
                "best_val_ppl": float(best_val_ppl),
                "test_loss": float(test_loss),
                "test_ppl": float(test_ppl),
                "teacher_test_ppl": float(teacher_test_ppl) if teacher_test_ppl is not None else None,
                "grassmann_baseline_test_ppl": float(g_base) if g_base is not None else None,
                "transformer_baseline_test_ppl": float(t_base) if t_base is not None else None,
                "beats_grassmann_baseline": beats_g,
                "beats_transformer_baseline": beats_t,
                "beats_both_baselines": bool(beats_g and beats_t),
                "train_losses": train_losses,
                "train_ce": train_ce,
                "train_kl": train_kl,
                "val_losses": val_losses,
                "checkpoint_path": str((ckpt_dir / "student_best.pt").resolve()),
                "metrics_path": str(metrics_path.resolve()),
            }
        }
        save_json(summary, run_dir / "summary.json")
        write_markdown_report(run_dir, run_meta, summary)
        print("\nFinal Distillation Results:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nSaved distillation artifacts to: {run_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
