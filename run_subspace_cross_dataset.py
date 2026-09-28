"""
Cross-Dataset Subspace Analysis

Loads existing checkpoints and measures Plucker effective rank
across all 4 datasets, without any training.

Hypothesis:
  Random init → low Plucker rank (W_red structure dominates)
  Warmstart    → high rank (semantic differentiation)
  KD           → highest rank on PTB/WT2, but not on TS

Usage: python run_subspace_cross_dataset.py
"""
import os, sys, json, math, argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

sys.path.insert(0, "src")
from models import GrassmannGPTv4
from train_exp4_ddp import TextDataset
from train_distill_hybrid_lite_from_latefusion_teacher_v2 import resolve_migrated_path, build_search_roots

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ---------------------------------------------------------------------------
# Hooks & metrics
# ---------------------------------------------------------------------------
plucker_snapshots = []

def plucker_hook(module, input, output):
    plucker_snapshots.append(output.detach().cpu())

def register_hooks(model):
    handles = []
    for block in model.blocks:
        if hasattr(block, 'grassmann') and hasattr(block.grassmann, 'plucker'):
            h = block.grassmann.plucker.register_forward_hook(plucker_hook)
            handles.append(h)
    return handles

def compute_metrics(plucker_list):
    if not plucker_list:
        return {"effective_rank": 0, "spectral_entropy": 0, "top5_ratio": 0}
    all_p = torch.cat([p.reshape(-1, p.shape[-1]) for p in plucker_list], dim=0).float()
    all_p = all_p - all_p.mean(dim=0, keepdim=True)
    try:
        _, S, _ = torch.linalg.svd(all_p, full_matrices=False)
    except:
        return {"effective_rank": 0, "spectral_entropy": 0, "top5_ratio": 0}
    S = S + 1e-10
    eff_rank = float(((S.sum())**2 / (S**2).sum()).item())
    p = S / S.sum()
    entropy = float((-(p * torch.log(p + 1e-10)).sum() / math.log(len(S))).item())
    top5 = float((S[:5].sum() / S.sum()).item())
    return {"effective_rank": eff_rank, "spectral_entropy": entropy, "top5_ratio": top5}

# ---------------------------------------------------------------------------
# Checkpoint definitions
# ---------------------------------------------------------------------------
DATASETS = {
    "PTB": {
        "dataset_path": "/workspace/grassmannflows/datasets/ptb_text_only_saved",
        "dataset_name": "ptb",
        "model_dim": 224, "reduced_dim": 56, "num_layers": 6,
        "window_sizes": "1,2,4", "max_seq_len": 256,
        "checkpoints": {
            "Random init (CE scratch)": None,  # use fresh model
            "KD scratch (random+KD)": None,    # use fresh model
            "Warmstart CE-only": "outputs/hybrid_experiments/20260328_142654_ptb_hybrid_lite_baseline_mild_e20",
            "Warmstart KD best": "outputs/distill_experiments/20260515_121432_ptb_warmstart_lambda_20",
        },
    },
    "WT2": {
        "dataset_path": "/workspace/grassmannflows/datasets/wikitext2_v1_saved",
        "dataset_name": "wikitext2",
        "model_dim": 224, "reduced_dim": 56, "num_layers": 6,
        "window_sizes": "1,2,4", "max_seq_len": 256,
        "checkpoints": {
            "Random init (CE scratch)": None,
            "KD scratch (random+KD)": None,
            "Warmstart CE-only": "outputs/hybrid_experiments/20260515_004119_wt2_hybrid_lite_baseline_224x56_l6_e20",
            "Warmstart KD best": "outputs/distill_experiments/20260515_022631_wt2_warmstart_kd002",
        },
    },
    "TinyStories": {
        "dataset_path": "/workspace/grassmannflows/datasets/tinystories_saved",
        "dataset_name": "tinystories",
        "model_dim": 224, "reduced_dim": 56, "num_layers": 6,
        "window_sizes": "1,2,4", "max_seq_len": 256,
        "checkpoints": {
            "Random init (CE scratch)": None,
            "KD scratch (random+KD)": None,
            "Warmstart CE-only": "outputs/hybrid_experiments/20260515_031101_ts_hybrid_lite_baseline_224x56_l6_e20",
            "Warmstart KD (λ=0.02)": "outputs/distill_experiments/20260516_011244_ts_warmstart_lambda_20",
        },
    },
    "Code": {
        "dataset_path": "/workspace/grassmannflows/datasets/codeparrot_saved",
        "dataset_name": "ptb",
        "text_field": "text",
        "model_dim": 224, "reduced_dim": 56, "num_layers": 6,
        "window_sizes": "1,2,4", "max_seq_len": 256,
        "checkpoints": {
            "Random init (CE scratch)": None,
            "KD scratch (random+KD)": None,
            "Warmstart CE-only": "outputs/hybrid_experiments/20260527_074518_code_hybrid_lite_baseline_ce_scratch",
            "Warmstart KD best": "outputs/distill_experiments/20260530_013210_code_ws_kd10",
        },
    },
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_local", local_files_only=True)
    V = len(tokenizer)

    all_results = {}

    for ds_name, ds_cfg in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"  {ds_name}")
        print(f"{'='*60}")

        # Load data (just 1 batch for spectral analysis)
        try:
            val_ds = TextDataset("validation", tokenizer, ds_cfg["max_seq_len"],
                                 dataset_name=ds_cfg["dataset_name"],
                                 dataset_path=ds_cfg["dataset_path"],
                                 text_field=ds_cfg.get("text_field", ""))
            val_ldr = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
            x_batch, _ = next(iter(val_ldr))
            x_batch = x_batch[:4].to(device)  # exactly 4 samples for consistent comparison
        except Exception as e:
            print(f"  SKIP: dataset load failed: {e}")
            continue

        ds_results = {}

        for label, ckpt_path in ds_cfg["checkpoints"].items():
            print(f"\n  --- {label} ---")

            # Build model
            ws = [int(w) for w in ds_cfg["window_sizes"].split(",")]
            model = GrassmannGPTv4(
                vocab_size=V, max_seq_len=ds_cfg["max_seq_len"],
                model_dim=ds_cfg["model_dim"], num_layers=ds_cfg["num_layers"],
                reduced_dim=ds_cfg["reduced_dim"], ff_dim=4*ds_cfg["model_dim"],
                window_sizes=ws, dropout=0.1,
            ).to(device)

            if ckpt_path is not None:
                # Resolve directory → actual checkpoint file
                p = Path(resolve_migrated_path(ckpt_path, search_roots=build_search_roots(), remaps=[], kind="ckpt", must_exist=True))
                if p.is_dir():
                    ckpt_file = p / "checkpoints" / "hybrid_best.pt"
                    if not ckpt_file.exists():
                        ckpt_file = p / "checkpoints" / "student_best.pt"
                    if not ckpt_file.exists():
                        # Try summary.json to find checkpoint_path
                        summary = json.loads((p / "summary.json").read_text())
                        for key in ["hybrid", "student"]:
                            if key in summary and "checkpoint_path" in summary[key]:
                                ckpt_file = Path(resolve_migrated_path(summary[key]["checkpoint_path"], search_roots=build_search_roots(), remaps=[], kind="ckpt"))
                                break
                    p = ckpt_file
                state = torch.load(p, map_location="cpu", weights_only=True)

                # Determine key prefix
                # Hybrid-lite has "grassmann." prefix, distill has "grassmann." prefix
                if any(k.startswith("grassmann.") for k in state):
                    grass_state = {k[len("grassmann."):]: v for k, v in state.items() if k.startswith("grassmann.")}
                elif any(k.startswith("student.grassmann.") for k in state):
                    grass_state = {k[len("student.grassmann."):]: v for k, v in state.items() if k.startswith("student.grassmann.")}
                else:
                    # Try loading directly (single-branch checkpoint)
                    grass_state = state

                # Filter to match model keys
                model_keys = set(model.state_dict().keys())
                grass_state = {k: v for k, v in grass_state.items() if k in model_keys}
                missing, unexpected = model.load_state_dict(grass_state, strict=False)
                print(f"    Loaded {len(grass_state)} keys from {p.parent.name if p.is_file() else p.name}")

            model.eval()

            # Register hooks and run forward pass
            global plucker_snapshots
            plucker_snapshots.clear()
            hooks = register_hooks(model)

            with torch.no_grad():
                _ = model(x_batch)

            for h in hooks:
                h.remove()

            metrics = compute_metrics(plucker_snapshots)
            ds_results[label] = metrics
            print(f"    eff_rank={metrics['effective_rank']:.1f}  entropy={metrics['spectral_entropy']:.4f}  top5={metrics['top5_ratio']:.4f}")

        all_results[ds_name] = ds_results

    # =====================================================================
    # Final summary table
    # =====================================================================
    print(f"\n\n{'='*85}")
    print(f"  CROSS-DATASET PLÜCKER SUBSPACE ANALYSIS")
    print(f"{'='*85}")
    print()
    print(f"  {'Condition':<30} {'PTB':>12} {'WT2':>12} {'TS':>12} {'Code':>12}")
    print(f"  {'─'*78}")

    conditions = ["Random init (CE scratch)", "KD scratch (random+KD)",
                  "Warmstart CE-only", "Warmstart KD best"]

    for cond in conditions:
        row = f"  {cond:<30}"
        for ds in ["PTB", "WT2", "TinyStories", "Code"]:
            if ds in all_results and cond in all_results[ds]:
                r = all_results[ds][cond]['effective_rank']
                row += f" {r:>12.1f}"
            else:
                row += f" {'─':>12}"
        print(row)

    # entropy
    print()
    print(f"  {'Spectral Entropy':<30} {'PTB':>12} {'WT2':>12} {'TS':>12} {'Code':>12}")
    print(f"  {'─'*78}")
    for cond in conditions:
        row = f"  {cond:<30}"
        for ds in ["PTB", "WT2", "TinyStories", "Code"]:
            if ds in all_results and cond in all_results[ds]:
                e = all_results[ds][cond]['spectral_entropy']
                row += f" {e:>12.4f}"
            else:
                row += f" {'─':>12}"
        print(row)

    # Key comparisons
    print(f"\n\n  {'='*60}")
    print(f"  MECHANISM VALIDATION")
    print(f"  {'='*60}")

    for ds in ["PTB", "WT2", "TinyStories", "Code"]:
        if ds not in all_results: continue
        r = all_results[ds]
        rand_rk = r.get("Random init (CE scratch)", {}).get("effective_rank", 0)
        warm_rk = r.get("Warmstart CE-only", {}).get("effective_rank", 0)
        kd_rk = r.get("Warmstart KD best", {}).get("effective_rank", 0)

        ratio_warm = warm_rk / max(rand_rk, 1)
        ratio_kd = kd_rk / max(warm_rk, 1)

        print(f"\n  {ds}:")
        print(f"    Random → Warmstart: rank {rand_rk:.0f} → {warm_rk:.0f} ({ratio_warm:.1f}x)")
        print(f"    Warmstart → KD:     rank {warm_rk:.0f} → {kd_rk:.0f} ({ratio_kd:.2f}x)")

        # Key test: does KD increase rank?
        if ratio_kd > 1.02:
            print(f"    KD INCREASES rank → Teacher signal expands geometric diversity ✅")
        elif ratio_kd < 0.98:
            print(f"    KD DECREASES rank → Teacher signal collapses geometry (bad)")
        else:
            print(f"    KD PRESERVES rank → Teacher signal maintains diversity (neutral)")

    # Prediction test
    print(f"\n  {'='*60}")
    print(f"  PREDICTION TEST")
    print(f"  {'='*60}")
    print(f"  PTB:  KD should INCREASE rank  (gap=8.4, large)  → KD is optimal")
    print(f"  WT2:  KD should INCREASE rank  (gap=3.3, medium) → KD is optimal")
    print(f"  TS:   KD should PRESERVE rank  (gap=0.1, tiny)   → CE-only is optimal")
    print(f"  Code: KD should INCREASE rank  (gap=1.99, small)  → KD is optimal (but weak)")

if __name__ == "__main__":
    main()
