# Corrected layer-wise alpha trainer.
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




def configure_visible_devices(gpu_id: str) -> None:
    if not gpu_id:
        return
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


def resolve_checkpoint_path(run_dir: Path, ckpt_value: str, *, search_roots=None, remaps=None) -> Path:
    return resolve_migrated_path(
        ckpt_value,
        run_dir=run_dir,
        search_roots=search_roots,
        remaps=remaps,
        kind="checkpoint",
        must_exist=True,
    )


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


# -----------------------------------------------------------------------------
# Hybrid model
# -----------------------------------------------------------------------------

class HybridLayerwiseAlphaModel(nn.Module):
    """
    Layer-wise hybrid teacher.

    At each layer l, mix the Transformer and Grassmann hidden states with a learnable
    scalar alpha_l, then feed the mixed hidden state forward through both branches.

    h_l = alpha_l * h_l^trans + (1 - alpha_l) * h_l^grass
    alpha_l = sigmoid(logit_alpha_l)
    """

    def __init__(self, grassmann: nn.Module, transformer: nn.Module, init_alpha: float = 0.5):
        super().__init__()
        self.grassmann = grassmann
        self.transformer = transformer

        self.num_hybrid_layers = int(min(len(self.grassmann.blocks), len(self.transformer.blocks)))
        if self.num_hybrid_layers <= 0:
            raise ValueError("No compatible hybrid layers found between grassmann and transformer branches.")

        init_logit = float(sigmoid_inverse(init_alpha))
        self.logit_alpha = nn.Parameter(torch.full((self.num_hybrid_layers,), init_logit, dtype=torch.float32))

    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_alpha)

    def alpha_mean(self) -> torch.Tensor:
        return self.alpha().mean()

    def get_alpha_value(self) -> float:
        return float(self.alpha_mean().item())

    @torch.no_grad()
    def get_alpha_vector(self):
        return [float(x) for x in self.alpha().detach().cpu().view(-1).tolist()]

    def set_branch_trainable(self, trainable: bool) -> None:
        for p in self.grassmann.parameters():
            p.requires_grad = trainable
        for p in self.transformer.parameters():
            p.requires_grad = trainable

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Transformer branch embeddings
        tok_emb_t = self.transformer.token_embedding(input_ids)
        pos_emb_t = self.transformer.position_embedding(torch.arange(seq_len, device=device))
        hidden_t = self.transformer.embedding_dropout(tok_emb_t + pos_emb_t)

        # Grassmann branch embeddings
        tok_emb_g = self.grassmann.token_embedding(input_ids)
        pos_emb_g = self.grassmann.position_embedding(torch.arange(seq_len, device=device))
        hidden_g = self.grassmann.embedding_dropout(tok_emb_g + pos_emb_g)

        alpha_vec = self.alpha()

        # Coupled layer-wise mixing
        for layer_idx in range(self.num_hybrid_layers):
            hidden_t = self.transformer.blocks[layer_idx](hidden_t)
            hidden_g = self.grassmann.blocks[layer_idx](hidden_g)

            alpha_l = alpha_vec[layer_idx].view(1, 1, 1)
            mixed_hidden = alpha_l * hidden_t + (1.0 - alpha_l) * hidden_g

            # Feed the same mixed hidden state into both branches for the next layer
            hidden_t = mixed_hidden
            hidden_g = mixed_hidden

        # If one branch has extra layers, continue separately (usually both are aligned)
        for layer_idx in range(self.num_hybrid_layers, len(self.transformer.blocks)):
            hidden_t = self.transformer.blocks[layer_idx](hidden_t)
        for layer_idx in range(self.num_hybrid_layers, len(self.grassmann.blocks)):
            hidden_g = self.grassmann.blocks[layer_idx](hidden_g)

        hidden_t = self.transformer.ln_f(hidden_t)
        hidden_g = self.grassmann.ln_f(hidden_g)

        logits_t = self.transformer.lm_head(hidden_t)
        logits_g = self.grassmann.lm_head(hidden_g)

        alpha_mean = alpha_vec.mean()
        logits = alpha_mean * logits_t + (1.0 - alpha_mean) * logits_g

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss, alpha_vec

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
    last_alpha_mean = None
    last_alpha_vec = None

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with amp_context(device, use_amp):
            _, loss, alpha = model(x, labels=y)
        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)
        alpha_detached = alpha.detach().view(-1)
        last_alpha_mean = float(alpha_detached.mean().item())
        last_alpha_vec = [float(v) for v in alpha_detached.cpu().tolist()]

    loss_sum, count_sum = reduce_sum_pair(total_loss, total_count, device)
    avg_loss = loss_sum / max(count_sum, 1.0)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl, last_alpha_mean, last_alpha_vec


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
                "alpha": f"{float(alpha.detach().view(-1).mean().item()):.4f}",
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
    lines.append(f"# Hybrid Layer-wise Alpha Training Report: {run_meta['experiment_name']}")
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
    parser.add_argument("--gpu-id", type=str, default="", help="Single GPU id or comma-separated visible GPU ids")
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

    if is_main_process():
        print(f"Using device: {device}")
        print(f"Distributed: {distributed}, world_size={get_world_size()}, local_rank={local_rank}")
        print(f"Visible CUDA devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}")
        print(f"Training mode: {args.train_mode}, use_amp={use_amp}")

    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_DATASETS_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_HUB_OFFLINE", None)

    args.grassmann_run_dir = resolve_and_log_path(
        args.grassmann_run_dir,
        label="grassmann_run_dir",
        search_roots=search_roots,
        remaps=remaps,
        must_exist=True,
        printer=print if is_main_process() else (lambda *a, **k: None),
    )
    args.transformer_run_dir = resolve_and_log_path(
        args.transformer_run_dir,
        label="transformer_run_dir",
        search_roots=search_roots,
        remaps=remaps,
        must_exist=True,
        printer=print if is_main_process() else (lambda *a, **k: None),
    )
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

    max_seq_len = args.max_seq_len if args.max_seq_len > 0 else min(
        int(g_config.get("max_seq_len", 256)),
        int(t_config.get("max_seq_len", 256)),
    )
    dataset_name = choose_dataset_value(args.dataset_name, g_config.get("dataset_name"), t_config.get("dataset_name"), "wikitext2")
    dataset_path = choose_dataset_value(args.dataset_path, g_config.get("dataset_path"), t_config.get("dataset_path"), "")
    dataset_path = resolve_and_log_path(
        dataset_path,
        label="dataset_path",
        search_roots=search_roots,
        remaps=remaps,
        must_exist=True,
        printer=print if is_main_process() else (lambda *a, **k: None),
    ) if dataset_path else dataset_path
    text_field = choose_dataset_value(args.text_field, g_config.get("text_field"), t_config.get("text_field"), "")

    max_lines = args.max_lines if args.max_lines >= 0 else choose_dataset_value(None, g_config.get("max_lines"), t_config.get("max_lines"), 0)
    encode_chars_per_batch = args.encode_chars_per_batch if args.encode_chars_per_batch >= 0 else choose_dataset_value(None, g_config.get("encode_chars_per_batch"), t_config.get("encode_chars_per_batch"), 200000)
    tinystories_val_frac = args.tinystories_val_frac if args.tinystories_val_frac >= 0 else choose_dataset_value(None, g_config.get("tinystories_val_frac"), t_config.get("tinystories_val_frac"), 0.02)
    split_seed = args.split_seed if args.split_seed >= 0 else choose_dataset_value(None, g_config.get("split_seed"), t_config.get("split_seed"), 42)

    max_lines = coerce_int_with_default(max_lines, 0)
    encode_chars_per_batch = coerce_int_with_default(encode_chars_per_batch, 200000)
    tinystories_val_frac = coerce_float_with_default(tinystories_val_frac, 0.02)
    split_seed = coerce_int_with_default(split_seed, 42)

    if encode_chars_per_batch <= 0:
        encode_chars_per_batch = 200000
    if max_lines < 0:
        max_lines = 0
    if tinystories_val_frac <= 0:
        tinystories_val_frac = 0.02

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

    hybrid = HybridLayerwiseAlphaModel(grassmann, transformer, init_alpha=args.init_alpha).to(device)

    if args.train_mode == "alpha_only":
        hybrid.set_branch_trainable(False)
    else:
        hybrid.set_branch_trainable(False)

    num_params = hybrid.get_num_params()
    num_trainable_params = hybrid.get_num_trainable_params()

    if is_main_process():
        print(f"Hybrid total params: {num_params:,} ({num_params / 1e6:.2f}M)")
        print(f"Hybrid trainable params at start: {num_trainable_params:,}")
        print(f"Initial alpha mean: {hybrid.get_alpha_value():.6f}")
        print(f"Initial layerwise alpha: {[round(x, 4) for x in hybrid.get_alpha_vector()]}")

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
        init_val_loss, init_val_ppl, init_alpha, init_alpha_vec = evaluate_hybrid(model, val_loader, device, use_amp=use_amp)
        print(f"Initial hybrid val loss={init_val_loss:.4f}, val ppl={init_val_ppl:.4f}, alpha_mean={init_alpha:.6f}")
        print(f"Initial evaluated layerwise alpha: {[round(x, 4) for x in init_alpha_vec]}")
    if distributed:
        dist.barrier()

    best_val_loss = float("inf")
    best_val_ppl = float("inf")
    best_epoch = -1
    best_layerwise_alpha = None
    alpha_history = []
    layerwise_alpha_history = []
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

        val_loss, val_ppl, alpha_value, alpha_vec = evaluate_hybrid(model, val_loader, device, use_amp=use_amp)
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        alpha_history.append(float(alpha_value))
        layerwise_alpha_history.append(alpha_vec)

        if is_main_process():
            epoch_record = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_ppl": float(torch.exp(torch.tensor(train_loss)).item()),
                "val_loss": float(val_loss),
                "val_ppl": float(val_ppl),
                "alpha": float(alpha_value),
                "layerwise_alpha": alpha_vec,
                "epoch_time_sec": float(epoch_time),
                "lr_alpha": float(optimizer.param_groups[0]["lr"]),
                "lr_branch": float(optimizer.param_groups[1]["lr"]),
                "branch_trainable": bool(epoch > args.freeze_branch_epochs) if args.train_mode == "joint" else False,
                "num_trainable_params": int(current_trainable_params),
            }
            append_jsonl(epoch_record, metrics_path)
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_ppl={val_ppl:.4f}, alpha_mean={alpha_value:.6f}, time={epoch_time:.1f}s, "
                f"trainable={current_trainable_params:,}"
            )

            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                best_val_ppl = float(val_ppl)
                best_epoch = epoch
                best_layerwise_alpha = list(alpha_vec)
                torch.save(unwrap_model(model).state_dict(), ckpt_dir / "hybrid_best.pt")

        if distributed:
            dist.barrier()

    if distributed:
        dist.barrier()

    summary = None
    if is_main_process():
        unwrap_model(model).load_state_dict(torch.load(ckpt_dir / "hybrid_best.pt", map_location=device, weights_only=True))
        test_loss, test_ppl, best_alpha, _ = evaluate_hybrid(model, test_loader, device, use_amp=use_amp)
        final_alpha = unwrap_model(model).get_alpha_value()
        final_layerwise_alpha = unwrap_model(model).get_alpha_vector()

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
                "initial_layerwise_alpha": [float(args.init_alpha)] * int(unwrap_model(model).num_hybrid_layers),
                "best_alpha": float(best_alpha),
                "best_layerwise_alpha": best_layerwise_alpha,
                "final_alpha": float(final_alpha),
                "final_layerwise_alpha": final_layerwise_alpha,
                "train_mode": args.train_mode,
                "freeze_branch_epochs": int(args.freeze_branch_epochs),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "alpha_history": alpha_history,
                "layerwise_alpha_history": layerwise_alpha_history,
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
