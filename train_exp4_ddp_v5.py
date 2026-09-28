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
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset, load_from_disk
from transformers import GPT2Tokenizer
from tqdm import tqdm

sys.path.insert(0, "src")
from models import GrassmannGPTv4

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

    Examples:
        --gpu-id 0
        --gpu-id 1
        --gpu-id 0,1   (useful before torchrun with 2 processes)

    Note:
        For torchrun/DDP, --gpu-id should expose as many devices as WORLD_SIZE.
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


def reduce_mean(value, device):
    if not is_dist_avail_and_initialized():
        return float(value)
    t = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= get_world_size()
    return float(t.item())


def reduce_sum_pair(sum_value, count_value, device):
    pair = torch.tensor([float(sum_value), float(count_value)], device=device, dtype=torch.float64)
    if is_dist_avail_and_initialized():
        dist.all_reduce(pair, op=dist.ReduceOp.SUM)
    return float(pair[0].item()), float(pair[1].item())


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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



def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(obj, path: Path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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


def get_tinystories_split(ds_all, split, val_frac=0.02, split_seed=42):
    if "train" not in ds_all:
        raise ValueError(f"TinyStories dataset has no 'train' split. Available: {list(ds_all.keys())}")
    if "validation" not in ds_all:
        raise ValueError(f"TinyStories dataset has no 'validation' split. Available: {list(ds_all.keys())}")

    train_full = ds_all["train"]
    split_dict = train_full.train_test_split(test_size=val_frac, seed=split_seed, shuffle=True)

    if split == "train":
        return split_dict["train"]
    if split == "validation":
        return split_dict["test"]
    if split == "test":
        return ds_all["validation"]
    raise ValueError(f"Unknown split: {split}")


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

class TextDataset(Dataset):
    def __init__(
        self,
        split: str,
        tokenizer,
        max_seq_len: int = 256,
        dataset_name: str = "wikitext2",
        dataset_path: str = "",
        text_field: str = "",
        max_lines: int = 0,
        encode_chars_per_batch: int = 200000,
        tinystories_val_frac: float = 0.02,
        split_seed: int = 42,
    ):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.split = split

        dataset = None
        used_source = None

        offline = os.environ.get("HF_DATASETS_OFFLINE") == "1" or os.environ.get("HF_HUB_OFFLINE") == "1"

        if offline and dataset_path:
            ds_all = load_from_disk(dataset_path)

            if dataset_name == "tinystories":
                dataset = get_tinystories_split(
                    ds_all,
                    split,
                    val_frac=tinystories_val_frac,
                    split_seed=split_seed,
                )
                if not text_field:
                    text_field = "text"
            else:
                if split not in ds_all:
                    raise ValueError(
                        f"Cannot find split '{split}' in local dataset. Available: {list(ds_all.keys())}"
                    )
                dataset = ds_all[split]

                if not text_field:
                    if dataset_name in ["wikitext2", "wikitext103"]:
                        text_field = "text"
                    elif dataset_name == "ptb":
                        text_field = "sentence"

            used_source = "disk"
            if is_main_process():
                print(f"[{split}] loaded from local disk: {dataset_path}")

        else:
            try:
                if dataset_name == "wikitext2":
                    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
                    if not text_field:
                        text_field = "text"
                elif dataset_name == "wikitext103":
                    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
                    if not text_field:
                        text_field = "text"
                elif dataset_name == "ptb":
                    dataset = load_dataset("ptb_text_only", "penn_treebank", split=split)
                    if not text_field:
                        text_field = "sentence"
                elif dataset_name == "tinystories":
                    ds_all = load_dataset("roneneldan/TinyStories")
                    dataset = get_tinystories_split(
                        ds_all,
                        split,
                        val_frac=tinystories_val_frac,
                        split_seed=split_seed,
                    )
                    if not text_field:
                        text_field = "text"
                else:
                    raise ValueError(f"Unknown dataset_name: {dataset_name}")

                used_source = "online"
                if is_main_process():
                    print(f"[{split}] loaded from online dataset hub")

            except Exception as e:
                if is_main_process():
                    print(f"[{split}] online load failed: {repr(e)}")

                if dataset_path:
                    ds_all = load_from_disk(dataset_path)

                    if dataset_name == "tinystories":
                        dataset = get_tinystories_split(
                            ds_all,
                            split,
                            val_frac=tinystories_val_frac,
                            split_seed=split_seed,
                        )
                        if not text_field:
                            text_field = "text"
                    else:
                        if split not in ds_all:
                            raise ValueError(
                                f"Cannot find split '{split}' in local dataset. Available: {list(ds_all.keys())}"
                            )
                        dataset = ds_all[split]

                        if not text_field:
                            if dataset_name in ["wikitext2", "wikitext103"]:
                                text_field = "text"
                            elif dataset_name == "ptb":
                                text_field = "sentence"

                    used_source = "disk"
                    if is_main_process():
                        print(f"[{split}] loaded from local disk: {dataset_path}")
                else:
                    raise RuntimeError(
                        f"Failed to load dataset '{dataset_name}' online, and no --dataset-path provided."
                    )

        if max_lines is None:
            max_lines = 0
        if encode_chars_per_batch is None or int(encode_chars_per_batch) <= 0:
            encode_chars_per_batch = 200000
        if tinystories_val_frac is None or float(tinystories_val_frac) <= 0:
            tinystories_val_frac = 0.02
        if split_seed is None:
            split_seed = 42

        line_limit = None if max_lines == 0 else max_lines

        kept_lines = 0
        char_count = 0
        tokens = []
        batch_texts = []
        batch_chars = 0

        for item in dataset[text_field]:
            if item is None:
                continue

            s = str(item).strip()
            if not s:
                continue

            batch_texts.append(s)
            batch_chars += len(s) + 1
            char_count += len(s)
            kept_lines += 1

            if batch_chars >= encode_chars_per_batch:
                batch_text = "\n".join(batch_texts)
                batch_tokens = tokenizer.encode(batch_text, add_special_tokens=False)
                tokens.extend(batch_tokens)
                batch_texts = []
                batch_chars = 0

            if line_limit is not None and kept_lines >= line_limit:
                break

        if batch_texts:
            batch_text = "\n".join(batch_texts)
            batch_tokens = tokenizer.encode(batch_text, add_special_tokens=False)
            tokens.extend(batch_tokens)

        if is_main_process():
            print(f"[{split}] source = {used_source}")
            print(f"[{split}] non-empty lines = {kept_lines}")
            print(f"[{split}] chars = {char_count}")
            print(f"[{split}] token count = {len(tokens)}")
            print(f"[{split}] first 20 tokens = {tokens[:20]}")

        self.tokens = tokens
        self.num_chunks = len(self.tokens) // max_seq_len
        self.tokens = self.tokens[: self.num_chunks * max_seq_len]

        self.stats = {
            "split": split,
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "text_field": text_field,
            "source": used_source,
            "non_empty_lines": kept_lines,
            "char_count": char_count,
            "token_count_before_trim": len(tokens),
            "num_chunks": self.num_chunks,
            "max_seq_len": max_seq_len,
            "max_lines": max_lines,
            "encode_chars_per_batch": encode_chars_per_batch,
            "tinystories_val_frac": tinystories_val_frac,
            "split_seed": split_seed,
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

def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, epoch, use_amp=False, log_interval=50):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_count = 0
    start_time = time.time()

    iterator = dataloader
    if is_main_process():
        iterator = tqdm(dataloader, desc=f"Epoch {epoch}")

    for step, (x, y) in enumerate(iterator):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            _, loss = model(x, labels=y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * x.size(0)
        total_tokens += x.numel()
        total_count += x.size(0)

        if is_main_process() and step % log_interval == 0:
            elapsed = time.time() - start_time
            tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            iterator.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ppl": f"{loss.exp().item():.2f}",
                "tok/s": f"{tok_per_sec:.0f}",
                "gnorm": f"{float(grad_norm):.2f}",
            })

    loss_sum, count_sum = reduce_sum_pair(total_loss, total_count, device)
    avg_loss = loss_sum / max(count_sum, 1.0)
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, device, use_amp=False):
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
    parser = argparse.ArgumentParser(description="Wikitext-2 Paper Reproduction with Experiment Tracking (DDP)")
    parser.add_argument("--model", type=str, default="both", choices=["grassmann", "transformer", "both"])
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", type=str, default="outputs/experiments")
    parser.add_argument("--tokenizer-dir", type=str, default="./gpt2_local")
    parser.add_argument("--gpu-id", type=str, default="", help="Single GPU id or comma-separated visible GPU ids")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--experiment-name", type=str, default="wikitext2_reproduction")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--tags", type=str, default="baseline,wikitext2")

    parser.add_argument("--reduced-dim", type=int, default=32)
    parser.add_argument("--window-sizes", type=str, default="1,2,4,8,12,16")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dataset-name", type=str, default="wikitext2",
                        choices=["wikitext2", "wikitext103", "ptb", "tinystories"])
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--text-field", type=str, default="")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-lines", type=int, default=0,
                        help="Use only first N non-empty lines; 0 means all")
    parser.add_argument("--encode-chars-per-batch", type=int, default=200000,
                        help="Batch tokenization by total characters to avoid OOM")
    parser.add_argument("--tinystories-val-frac", type=float, default=0.02,
                        help="Fraction of TinyStories train split used as validation")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Random seed for deterministic dataset splitting")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--path-remap", action="append", default=[],
                        help="Manual old=new path remap for migrated servers; can be passed multiple times")
    parser.add_argument("--search-root", action="append", default=[],
                        help="Extra root to search when resolving migrated dataset/tokenizer paths")

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
    else:
        os.environ.pop("HF_DATASETS_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_HUB_OFFLINE", None)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    run_id, run_dir = make_run_dir(output_root, args.experiment_name)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    window_sizes = [int(x) for x in args.window_sizes.split(",") if x.strip()]

    args.tokenizer_dir = resolve_and_log_path(
        args.tokenizer_dir,
        label="tokenizer_dir",
        search_roots=search_roots,
        remaps=remaps,
        must_exist=True,
        printer=print if is_main_process() else (lambda *a, **k: None),
    )
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    if args.dataset_path:
        args.dataset_path = resolve_and_log_path(
            args.dataset_path,
            label="dataset_path",
            search_roots=search_roots,
            remaps=remaps,
            must_exist=True,
            printer=print if is_main_process() else (lambda *a, **k: None),
        )
    vocab_size = len(tokenizer)

    if is_main_process():
        print(f"Loading dataset: {args.dataset_name}...")
    train_dataset = TextDataset(
        "train", tokenizer, args.max_seq_len,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        text_field=args.text_field,
        max_lines=args.max_lines,
        encode_chars_per_batch=args.encode_chars_per_batch,
        tinystories_val_frac=args.tinystories_val_frac,
        split_seed=args.split_seed,
    )
    val_dataset = TextDataset(
        "validation", tokenizer, args.max_seq_len,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        text_field=args.text_field,
        max_lines=args.max_lines,
        encode_chars_per_batch=args.encode_chars_per_batch,
        tinystories_val_frac=args.tinystories_val_frac,
        split_seed=args.split_seed,
    )
    test_dataset = TextDataset(
        "test", tokenizer, args.max_seq_len,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        text_field=args.text_field,
        max_lines=args.max_lines,
        encode_chars_per_batch=args.encode_chars_per_batch,
        tinystories_val_frac=args.tinystories_val_frac,
        split_seed=args.split_seed,
    )

    if is_main_process():
        print(f"Train: {len(train_dataset)} chunks, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

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

    run_meta = {
        "run_id": run_id,
        "experiment_name": args.experiment_name,
        "start_time": datetime.now().isoformat(),
        "notes": args.notes,
        "tags": tags,
        "config": vars(args),
        "env": get_env_info(device),
        "dataset_stats": {
            "train": train_dataset.stats,
            "validation": val_dataset.stats,
            "test": test_dataset.stats,
        },
    }
    if is_main_process():
        save_json(run_meta, run_dir / "config.json")

    results = {}

    models_to_train = []
    if args.model in ["grassmann", "both"]:
        models_to_train.append("grassmann")
    if args.model in ["transformer", "both"]:
        models_to_train.append("transformer")

    for model_type in models_to_train:
        if is_main_process():
            print(f"\n{'='*60}")
            print(f"Training: {model_type.upper()}")
            print(f"{'='*60}")

        if model_type == "grassmann":
            base_model = GrassmannGPTv4(
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
            base_model = SmallTransformer(
                vocab_size=vocab_size,
                max_seq_len=args.max_seq_len,
                model_dim=args.model_dim,
                num_layers=args.num_layers,
                num_heads=8,
                ff_dim=4 * args.model_dim,
                dropout=args.dropout,
            )

        base_model = base_model.to(device)
        num_params = base_model.get_num_params()
        if is_main_process():
            print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

        model = DDP(base_model, device_ids=[local_rank], output_device=local_rank) if distributed and device.type == "cuda" else base_model

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        total_steps = len(train_loader) * args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

        scaler = GradScaler("cuda", enabled=use_amp)
        
        best_val_loss = float("inf")
        best_val_ppl = float("inf")
        best_epoch = -1
        train_losses = []
        val_losses = []
        metrics_path = run_dir / f"{model_type}_metrics.jsonl"

        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            epoch_start = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, use_amp=use_amp)
            train_loss = reduce_mean(train_loss, device)
            epoch_time = time.time() - epoch_start

            if is_main_process():
                val_loss, val_ppl = evaluate(model, val_loader, device, use_amp=use_amp)
                train_losses.append(float(train_loss))
                val_losses.append(float(val_loss))

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
                    torch.save(unwrap_model(model).state_dict(), ckpt_dir / f"{model_type}_best.pt")

            if distributed:
                dist.barrier()

        if is_main_process():
            unwrap_model(model).load_state_dict(torch.load(ckpt_dir / f"{model_type}_best.pt", map_location=device))
            test_loss, test_ppl = evaluate(model, test_loader, device, use_amp=use_amp)

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
                "checkpoint_path": f"checkpoints/{model_type}_best.pt",
                "metrics_path": f"{model_type}_metrics.jsonl",
            }

        if distributed:
            dist.barrier()

    if is_main_process():
        save_json(results, run_dir / "summary.json")
        write_markdown_report(run_dir, run_meta, results)

        if len(results) == 2:
            print(f"\n{'='*60}")
            print(f"COMPARISON: {args.dataset_name.upper()} Experiment")
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

    cleanup_distributed()


if __name__ == "__main__":
    main()
