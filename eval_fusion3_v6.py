import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

sys.path.insert(0, "src")
from models import GrassmannGPTv4

from train_exp4_ddp import TextDataset, SmallTransformer




def configure_visible_devices(gpu_id: str):
    if not gpu_id:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def compare_config_fields(g_cfg, t_cfg, fields):
    mismatches = {}
    for field in fields:
        gv = g_cfg.get(field)
        tv = t_cfg.get(field)
        if gv != tv:
            mismatches[field] = {"grassmann": gv, "transformer": tv}
    return mismatches

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



def build_grassmann(vocab_size, cfg):
    return GrassmannGPTv4(
        vocab_size=vocab_size,
        max_seq_len=cfg["max_seq_len"],
        model_dim=cfg["model_dim"],
        num_layers=cfg["num_layers"],
        reduced_dim=cfg.get("reduced_dim", 32),
        ff_dim=4 * cfg["model_dim"],
        window_sizes=[int(x) for x in str(cfg.get("window_sizes", "1,2,4,8,12,16")).split(",") if x.strip()],
        dropout=cfg.get("dropout", 0.1),
    )


def build_transformer(vocab_size, cfg):
    return SmallTransformer(
        vocab_size=vocab_size,
        max_seq_len=cfg["max_seq_len"],
        model_dim=cfg["model_dim"],
        num_layers=cfg["num_layers"],
        num_heads=8,
        ff_dim=4 * cfg["model_dim"],
        dropout=cfg.get("dropout", 0.1),
    )


@torch.no_grad()
def evaluate_fusion(model_g, model_t, dataloader, device, alpha=0.5):
    model_g.eval()
    model_t.eval()

    total_loss = 0.0
    total_count = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        logits_g, _ = model_g(x, labels=None)
        logits_t, _ = model_t(x, labels=None)

        logits = alpha * logits_t + (1.0 - alpha) * logits_g

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = y[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)

    avg_loss = total_loss / total_count
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser(description="Evaluate logits fusion of Grassmann and Transformer")
    parser.add_argument("--grassmann-run-dir", type=str, required=True)
    parser.add_argument("--transformer-run-dir", type=str, required=True)
    parser.add_argument("--tokenizer-dir", type=str, default="./gpt2_local")
    parser.add_argument("--dataset-name", type=str, default="", help="Override dataset name instead of reading from transformer run config")
    parser.add_argument("--dataset-path", type=str, default="", help="Override dataset path instead of reading from transformer run config")
    parser.add_argument("--text-field", type=str, default="", help="Override text field instead of reading from transformer run config")
    parser.add_argument("--max-seq-len", type=int, default=0, help="Override max sequence length instead of reading from transformer run config")
    parser.add_argument("--gpu-id", type=str, default="", help="Single GPU id or comma-separated visible GPU ids")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-config-mismatch", action="store_true")
    parser.add_argument("--path-remap", action="append", default=[],
                        help="Manual old=new path remap for migrated servers; can be passed multiple times")
    parser.add_argument("--search-root", action="append", default=[],
                        help="Extra root to search when resolving migrated dataset/checkpoint paths")
    args = parser.parse_args()
    configure_visible_devices(args.gpu_id)

    if args.offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    else:
        os.environ.pop("HF_DATASETS_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_HUB_OFFLINE", None)

    remaps = parse_path_remaps(args.path_remap)
    search_roots = build_search_roots(args.search_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Visible CUDA devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}")

    args.grassmann_run_dir = resolve_and_log_path(args.grassmann_run_dir, label="grassmann_run_dir", search_roots=search_roots, remaps=remaps)
    args.transformer_run_dir = resolve_and_log_path(args.transformer_run_dir, label="transformer_run_dir", search_roots=search_roots, remaps=remaps)
    g_run_dir = Path(args.grassmann_run_dir)
    t_run_dir = Path(args.transformer_run_dir)

    g_summary = load_summary(g_run_dir)
    t_summary = load_summary(t_run_dir)
    g_config = load_config(g_run_dir)["config"]
    t_config = load_config(t_run_dir)["config"]

    if "grassmann" not in g_summary:
        raise ValueError(f"{g_run_dir} does not contain grassmann results")
    if "transformer" not in t_summary:
        raise ValueError(f"{t_run_dir} does not contain transformer results")

    fields_to_check = [
        "dataset_name", "dataset_path", "text_field", "max_seq_len",
        "max_lines", "encode_chars_per_batch", "tinystories_val_frac", "split_seed"
    ]
    mismatches = compare_config_fields(g_config, t_config, fields_to_check)
    if mismatches and not args.allow_config_mismatch:
        raise ValueError("Run dirs are not configuration-compatible for fusion:\n" + json.dumps(mismatches, indent=2, ensure_ascii=False))
    if mismatches:
        print("WARNING: config mismatches detected, continuing because --allow-config-mismatch was set:")
        print(json.dumps(mismatches, indent=2, ensure_ascii=False))

    args.tokenizer_dir = resolve_and_log_path(args.tokenizer_dir, label="tokenizer_dir", search_roots=search_roots, remaps=remaps)
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    vocab_size = len(tokenizer)

    # use transformer config as the reference for dataset seq len, but allow explicit CLI overrides
    max_seq_len = args.max_seq_len or t_config["max_seq_len"]

    dataset_name = args.dataset_name or t_config.get("dataset_name", "wikitext2")
    dataset_path = args.dataset_path or t_config.get("dataset_path", "")
    dataset_path = resolve_and_log_path(dataset_path, label="dataset_path", search_roots=search_roots, remaps=remaps) if dataset_path else dataset_path
    text_field = args.text_field or t_config.get("text_field", "")

    max_lines = t_config.get("max_lines", 0)
    encode_chars_per_batch = t_config.get("encode_chars_per_batch", 200000)
    tinystories_val_frac = t_config.get("tinystories_val_frac", 0.02)
    split_seed = t_config.get("split_seed", 42)

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

    print(f"Fusion eval dataset_name = {dataset_name}")
    print(f"Fusion eval dataset_path = {dataset_path}")
    print(f"Fusion eval text_field = {text_field}")
    print(f"Val chunks = {len(val_dataset)}, Test chunks = {len(test_dataset)}")
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_g = build_grassmann(vocab_size, g_config)
    model_t = build_transformer(vocab_size, t_config)

    g_ckpt = resolve_migrated_path(g_summary["grassmann"]["checkpoint_path"], run_dir=g_run_dir, search_roots=search_roots, remaps=remaps, kind="checkpoint")
    t_ckpt = resolve_migrated_path(t_summary["transformer"]["checkpoint_path"], run_dir=t_run_dir, search_roots=search_roots, remaps=remaps, kind="checkpoint")

    print(f"Loading grassmann checkpoint: {g_ckpt}")
    print(f"Loading transformer checkpoint: {t_ckpt}")

    model_g.load_state_dict(torch.load(str(g_ckpt), map_location=device, weights_only=True))
    model_t.load_state_dict(torch.load(str(t_ckpt), map_location=device, weights_only=True))

    model_g.to(device)
    model_t.to(device)
    
    
    @torch.no_grad()
    def evaluate_single(model, dataloader, device):
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

    val_g_loss, val_g_ppl = evaluate_single(model_g, val_loader, device)
    val_t_loss, val_t_ppl = evaluate_single(model_t, val_loader, device)
    test_g_loss, test_g_ppl = evaluate_single(model_g, test_loader, device)
    test_t_loss, test_t_ppl = evaluate_single(model_t, test_loader, device)

    print(f"Grassmann  Val PPL: {val_g_ppl:.4f} | Test PPL: {test_g_ppl:.4f}")
    print(f"Transformer Val PPL: {val_t_ppl:.4f} | Test PPL: {test_t_ppl:.4f}")

    alphas = []
    alpha = 0.0
    while alpha <= 1.0 + 1e-8:
        alphas.append(round(alpha, 6))
        alpha += args.alpha_step

    best = None
    sweep = []

    for alpha in alphas:
        val_loss, val_ppl = evaluate_fusion(model_g, model_t, val_loader, device, alpha=alpha)
        record = {
            "alpha": float(alpha),
            "val_loss": float(val_loss),
            "val_ppl": float(val_ppl),
        }
        sweep.append(record)
        print(f"alpha={alpha:.2f} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

        if best is None or val_ppl < best["val_ppl"]:
            best = record.copy()

    test_loss, test_ppl = evaluate_fusion(model_g, model_t, test_loader, device, alpha=best["alpha"])
    best["test_loss"] = float(test_loss)
    best["test_ppl"] = float(test_ppl)

    print("\nBest fusion result:")
    print(json.dumps(best, indent=2, ensure_ascii=False))

    output = {
        "grassmann_run_dir": str(g_run_dir.resolve()),
        "transformer_run_dir": str(t_run_dir.resolve()),
        "best": best,
        "sweep": sweep,
    }

    if args.output_json:
        out_path = Path(args.output_json)
    else:
        out_path = Path("fusion_result.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved fusion results to: {out_path}")


if __name__ == "__main__":
    main()