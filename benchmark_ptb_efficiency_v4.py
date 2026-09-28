import os
import sys
import json
import time
import math
import argparse
import inspect
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

sys.path.insert(0, "src")
from models import GrassmannGPTv4
from train_exp4_ddp import SmallTransformer, TextDataset


def configure_visible_devices(gpu_id: str):
    if gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_cfg(run_dir: Path):
    cfg = load_json(run_dir / "config.json")
    return cfg.get("config", cfg)


def get_summary(run_dir: Path):
    return load_json(run_dir / "summary.json")


def _safe_int(value, default=None):
    try:
        iv = int(value)
        return iv
    except Exception:
        return default


def infer_seq_len_from_state_dict(state_dict, fallback=256):
    for key in [
        "position_embedding.weight",
        "position_embedding",
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


def normalize_path_str(path_value) -> str:
    if path_value is None:
        return ""
    return str(path_value).strip()


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

    roots, seen = [], set()
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


def _safe_exists(p: Path) -> bool:
    try:
        return p.exists()
    except (PermissionError, OSError):
        return False


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


def resolve_migrated_path(path_value, *, search_roots=None, remaps=None, kind="path", must_exist=True):
    path_str = normalize_path_str(path_value)
    if not path_str:
        return Path(path_str)

    p = Path(path_str).expanduser()
    candidates = []

    def add(candidate):
        if candidate is None:
            return
        try:
            c = candidate.expanduser()
        except Exception:
            c = Path(candidate)
        candidates.append(c)

    add(p)

    for old, new in remaps or []:
        if path_str == old or path_str.startswith(old + "/"):
            replaced = new + path_str[len(old):]
            add(Path(replaced))

    project_root = default_project_root()
    workspace_root = project_root.parent
    for c in _suffix_candidates(p, project_root, workspace_root):
        add(c)

    if not p.is_absolute():
        for root in search_roots or []:
            add(Path(root) / p)

    unique = []
    seen = set()
    for c in candidates:
        try:
            key = str(c.resolve())
        except Exception:
            key = str(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)

    for c in unique:
        if not must_exist or _safe_exists(c):
            if str(c) != path_str:
                print(f"[path-remap] {kind}: {path_str} -> {c}")
            return c

    if must_exist:
        raise FileNotFoundError(f"Could not resolve {kind}: {path_str}")
    return unique[0] if unique else p


def make_text_dataset(tokenizer, max_seq_len, dataset_name, dataset_path, split, text_field,
                      offline, max_lines, encode_chars_per_batch,
                      tinystories_val_frac, split_seed):
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


def parse_window_sizes(v):
    if isinstance(v, str):
        return [int(x) for x in v.split(",") if x.strip()]
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    return [1, 2, 4]


def build_grassmann(vocab_size: int, cfg: dict):
    return GrassmannGPTv4(
        vocab_size=vocab_size,
        max_seq_len=int(cfg.get("max_seq_len", 256)),
        model_dim=int(cfg.get("model_dim", 256)),
        num_layers=int(cfg.get("num_layers", 6)),
        reduced_dim=int(cfg.get("reduced_dim", 32)),
        ff_dim=4 * int(cfg.get("model_dim", 256)),
        window_sizes=parse_window_sizes(cfg.get("window_sizes", "1,2,4")),
        dropout=float(cfg.get("dropout", 0.1)),
    )


def build_transformer(vocab_size: int, cfg: dict):
    return SmallTransformer(
        vocab_size=vocab_size,
        max_seq_len=int(cfg.get("max_seq_len", 256)),
        model_dim=int(cfg.get("model_dim", 256)),
        num_layers=int(cfg.get("num_layers", 6)),
        num_heads=8,
        ff_dim=4 * int(cfg.get("model_dim", 256)),
        dropout=float(cfg.get("dropout", 0.1)),
    )


class HybridLateFusionAlphaModel(nn.Module):
    def __init__(self, grassmann: nn.Module, transformer: nn.Module, init_alpha: float = 0.5, late_k: int = 1):
        super().__init__()
        self.grassmann = grassmann
        self.transformer = transformer
        self.num_layers = int(min(len(self.grassmann.blocks), len(self.transformer.blocks)))
        self.late_k = int(max(1, min(int(late_k), self.num_layers)))
        p = max(min(float(init_alpha), 1.0 - 1e-6), 1e-6)
        init_logit = math.log(p / (1.0 - p))
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

    def _transformer_logits_from_hidden(self, hidden):
        hidden = self.transformer.ln_f(hidden)
        return self.transformer.lm_head(hidden)

    def _grassmann_logits_from_hidden(self, hidden):
        hidden = self.grassmann.ln_f(hidden)
        return self.grassmann.lm_head(hidden)

    def forward(self, input_ids):
        t_layers = self._transformer_hidden_per_layer(input_ids)
        g_layers = self._grassmann_hidden_per_layer(input_ids)
        alpha = self.alpha()
        selected_t = t_layers[-self.late_k:]
        selected_g = g_layers[-self.late_k:]
        fused_logits = []
        for i in range(self.late_k):
            logits_t = self._transformer_logits_from_hidden(selected_t[i])
            logits_g = self._grassmann_logits_from_hidden(selected_g[i])
            a = alpha[i]
            fused_logits.append(a * logits_t + (1.0 - a) * logits_g)
        return torch.stack(fused_logits, dim=0).mean(dim=0)


def load_single_model(run_dir: Path, kind: str, vocab_size: int, device: torch.device, search_roots, remaps):
    run_dir = resolve_migrated_path(run_dir, search_roots=search_roots, remaps=remaps, kind=f"{kind}_run_dir", must_exist=True)
    cfg = get_cfg(run_dir)
    summary = get_summary(run_dir)

    ckpt = run_dir / "checkpoints" / f"{kind}_best.pt"
    if not _safe_exists(ckpt):
        ckpt = resolve_migrated_path(summary[kind]["checkpoint_path"], search_roots=search_roots, remaps=remaps, kind=f"{kind}_checkpoint", must_exist=True)

    state = torch.load(ckpt, map_location=device, weights_only=True)
    inferred_seq_len = infer_seq_len_from_state_dict(state, fallback=256)
    cfg = dict(cfg)
    cfg["max_seq_len"] = choose_positive_seq_len(cfg.get("max_seq_len"), inferred_seq_len, default=inferred_seq_len)

    model = build_grassmann(vocab_size, cfg) if kind == "grassmann" else build_transformer(vocab_size, cfg)
    model.load_state_dict(state)
    return model.to(device).eval(), cfg, summary


def load_hybrid_lite(run_dir: Path, vocab_size: int, device: torch.device, search_roots, remaps):
    run_dir = resolve_migrated_path(run_dir, search_roots=search_roots, remaps=remaps, kind="hybrid_lite_run_dir", must_exist=True)
    cfg_all = load_json(run_dir / "config.json")
    cfg = cfg_all.get("config", cfg_all)
    summary = get_summary(run_dir)

    # Support both baseline-hybrid summaries {"hybrid": ...}
    # and distillation summaries {"student": ...}
    hybrid_block = None
    if isinstance(summary, dict):
        if "hybrid" in summary and isinstance(summary["hybrid"], dict):
            hybrid_block = summary["hybrid"]
        elif "student" in summary and isinstance(summary["student"], dict):
            hybrid_block = summary["student"]

    if hybrid_block is None:
        raise KeyError("Neither 'hybrid' nor 'student' block found in summary.json")

    late_k = int(hybrid_block.get("late_k", cfg.get("late_k", 1)))
    init_alpha = float(
        hybrid_block.get(
            "best_alpha",
            hybrid_block.get("final_alpha", cfg.get("init_alpha", 0.5))
        )
    )

    # Try common checkpoint locations first
    candidate_ckpts = [
        run_dir / "checkpoints" / "hybrid_best.pt",
        run_dir / "checkpoints" / "student_best.pt",
        run_dir / "checkpoints" / "best.pt",
    ]
    ckpt = None
    for c in candidate_ckpts:
        if _safe_exists(c):
            ckpt = c
            break

    if ckpt is None:
        ckpt_path = hybrid_block.get("checkpoint_path")
        if not ckpt_path:
            raise FileNotFoundError("No checkpoint_path found for hybrid-lite run")
        ckpt = resolve_migrated_path(ckpt_path, search_roots=search_roots, remaps=remaps, kind="hybrid_lite_checkpoint", must_exist=True)

    state = torch.load(ckpt, map_location=device, weights_only=True)
    inferred_seq_len = infer_seq_len_from_state_dict(state, fallback=256)
    cfg = dict(cfg)
    cfg["max_seq_len"] = choose_positive_seq_len(cfg.get("max_seq_len"), inferred_seq_len, default=inferred_seq_len)

    g_cfg = {
        "max_seq_len": cfg["max_seq_len"],
        "model_dim": int(cfg.get("model_dim", 224)),
        "num_layers": int(cfg.get("num_layers", 6)),
        "reduced_dim": int(cfg.get("reduced_dim", 56)),
        "window_sizes": cfg.get("window_sizes", "1,2,4"),
        "dropout": float(cfg.get("dropout", 0.1)),
    }
    t_cfg = {
        "max_seq_len": cfg["max_seq_len"],
        "model_dim": int(cfg.get("model_dim", 224)),
        "num_layers": int(cfg.get("num_layers", 6)),
        "dropout": float(cfg.get("dropout", 0.1)),
    }
    model = HybridLateFusionAlphaModel(
        build_grassmann(vocab_size, g_cfg),
        build_transformer(vocab_size, t_cfg),
        init_alpha=init_alpha,
        late_k=late_k,
    )
    model.load_state_dict(state)
    return model.to(device).eval(), cfg, summary


def load_hybrid_teacher(run_dir: Path, vocab_size: int, device: torch.device, search_roots, remaps):
    run_dir = resolve_migrated_path(run_dir, search_roots=search_roots, remaps=remaps, kind="hybrid_teacher_run_dir", must_exist=True)
    cfg_all = load_json(run_dir / "config.json")
    cfg = cfg_all.get("config", {})
    summary = get_summary(run_dir)
    source_runs = cfg_all.get("source_runs", {})

    g_run = resolve_migrated_path(source_runs["grassmann_run_dir"], search_roots=search_roots, remaps=remaps, kind="teacher_grassmann_run_dir", must_exist=True)
    t_run = resolve_migrated_path(source_runs["transformer_run_dir"], search_roots=search_roots, remaps=remaps, kind="teacher_transformer_run_dir", must_exist=True)
    g_cfg = get_cfg(g_run)
    t_cfg = get_cfg(t_run)
    g_sum = get_summary(g_run)
    t_sum = get_summary(t_run)

    g_ckpt = g_run / "checkpoints" / "grassmann_best.pt"
    t_ckpt = t_run / "checkpoints" / "transformer_best.pt"
    hybrid_ckpt = run_dir / "checkpoints" / "hybrid_best.pt"

    if not _safe_exists(g_ckpt):
        g_ckpt = resolve_migrated_path(source_runs.get("grassmann_checkpoint", g_sum["grassmann"]["checkpoint_path"]), search_roots=search_roots, remaps=remaps, kind="teacher_grassmann_checkpoint", must_exist=True)
    if not _safe_exists(t_ckpt):
        t_ckpt = resolve_migrated_path(source_runs.get("transformer_checkpoint", t_sum["transformer"]["checkpoint_path"]), search_roots=search_roots, remaps=remaps, kind="teacher_transformer_checkpoint", must_exist=True)
    if not _safe_exists(hybrid_ckpt):
        hybrid_ckpt = resolve_migrated_path(summary["hybrid"]["checkpoint_path"], search_roots=search_roots, remaps=remaps, kind="hybrid_teacher_checkpoint", must_exist=True)

    g_state = torch.load(g_ckpt, map_location=device, weights_only=True)
    t_state = torch.load(t_ckpt, map_location=device, weights_only=True)
    h_state = torch.load(hybrid_ckpt, map_location=device, weights_only=True)

    inferred_seq_len = choose_positive_seq_len(
        cfg.get("max_seq_len"),
        g_cfg.get("max_seq_len"),
        t_cfg.get("max_seq_len"),
        infer_seq_len_from_state_dict(g_state, fallback=256),
        infer_seq_len_from_state_dict(t_state, fallback=256),
        infer_seq_len_from_state_dict(h_state, fallback=256),
        default=256,
    )
    g_cfg = dict(g_cfg); g_cfg["max_seq_len"] = inferred_seq_len
    t_cfg = dict(t_cfg); t_cfg["max_seq_len"] = inferred_seq_len
    cfg = dict(cfg); cfg["max_seq_len"] = inferred_seq_len

    late_k = int(summary.get("hybrid", {}).get("late_k", cfg.get("late_k", 1)))
    init_alpha = float(summary.get("hybrid", {}).get("best_alpha", cfg.get("init_alpha", 0.5)))

    grassmann = build_grassmann(vocab_size, g_cfg)
    transformer = build_transformer(vocab_size, t_cfg)
    grassmann.load_state_dict(g_state)
    transformer.load_state_dict(t_state)

    model = HybridLateFusionAlphaModel(grassmann, transformer, init_alpha=init_alpha, late_k=late_k)
    model.load_state_dict(h_state)
    return model.to(device).eval(), cfg, summary


@torch.no_grad()
def benchmark_model(model, loader, device, use_amp, warmup_batches, measure_batches):
    latencies = []
    total_tokens = 0
    total_batches = 0

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for idx, (x, y) in enumerate(loader):
        if idx >= warmup_batches:
            break
        x = x.to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=use_amp) if device.type == "cuda" else torch.no_grad():
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for idx, (x, y) in enumerate(loader):
        if idx >= measure_batches:
            break
        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with autocast(device_type="cuda", enabled=use_amp) if device.type == "cuda" else torch.no_grad():
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)
        total_tokens += int(x.numel())
        total_batches += 1

    total_time = sum(latencies)
    return {
        "measured_batches": total_batches,
        "avg_batch_latency_ms": (total_time / max(total_batches, 1)) * 1000.0,
        "tokens_per_sec": total_tokens / max(total_time, 1e-9),
        "peak_memory_gb": (torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark PTB models for runtime/efficiency")
    parser.add_argument("--model-kind", type=str, required=True, choices=["grassmann", "transformer", "hybrid_teacher", "hybrid_lite"])
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="ptb")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--text-field", type=str, default="")
    parser.add_argument("--tokenizer-dir", type=str, default="./gpt2_local")
    parser.add_argument("--gpu-id", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--measure-batches", type=int, default=50)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--path-remap", action="append", default=[])
    parser.add_argument("--search-root", action="append", default=[])
    args = parser.parse_args()

    configure_visible_devices(args.gpu_id)
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    remaps = parse_path_remaps(args.path_remap)
    if not remaps:
        remaps = [("/root/songxin", "/workspace/grassmannflows")]
    search_roots = build_search_roots(args.search_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    run_dir = Path(args.run_dir)
    if args.model_kind in {"grassmann", "transformer"}:
        model, cfg, summary = load_single_model(run_dir, args.model_kind, vocab_size, device, search_roots, remaps)
    elif args.model_kind == "hybrid_teacher":
        model, cfg, summary = load_hybrid_teacher(run_dir, vocab_size, device, search_roots, remaps)
    else:
        model, cfg, summary = load_hybrid_lite(run_dir, vocab_size, device, search_roots, remaps)

    dataset = make_text_dataset(
        tokenizer=tokenizer,
        max_seq_len=choose_positive_seq_len(cfg.get("max_seq_len"), args.max_seq_len, default=256),
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        split="test",
        text_field=args.text_field or None,
        offline=args.offline,
        max_lines=None,
        encode_chars_per_batch=200000,
        tinystories_val_frac=0.1,
        split_seed=42,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    metrics = benchmark_model(model, loader, device, use_amp=args.amp and device.type == "cuda",
                              warmup_batches=args.warmup_batches, measure_batches=args.measure_batches)

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    if args.model_kind in {"grassmann", "transformer"}:
        test_ppl = summary.get(args.model_kind, {}).get("test_ppl")
    else:
        if "hybrid" in summary and isinstance(summary["hybrid"], dict):
            test_ppl = summary["hybrid"].get("test_ppl")
        elif "student" in summary and isinstance(summary["student"], dict):
            test_ppl = summary["student"].get("test_ppl")
        else:
            test_ppl = None

    out = {"model_kind": args.model_kind, "run_dir": str(resolve_migrated_path(run_dir, search_roots=search_roots, remaps=remaps, must_exist=False)),
           "test_ppl": test_ppl, "params_m": params_m, **metrics}
    print(json.dumps(out, indent=2, ensure_ascii=False))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
