"""
Subspace Theory Verification — 3 experiments (no training needed)

V1: Larger r — does r=64 have larger ΔRank than r=56?
V2: Layer-wise — are deep layers lower rank than shallow layers?
V3: Epoch scan — does eff_rank monotonically decrease during training?
"""
import os, sys, json, math
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

sys.path.insert(0, "src")
from models import GrassmannGPTv4, CausalGrassmannMixing, PluckerEncoder
from train_exp4_ddp import TextDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ===========================================================================
# Layer-wise hooks
# ===========================================================================
layer_outputs = defaultdict(list)

def make_layer_hook(layer_idx):
    def hook(module, input, output):
        layer_outputs[layer_idx].append(output.detach().cpu())
    return hook

def register_layer_hooks(model):
    handles = []
    for i, block in enumerate(model.blocks):
        if hasattr(block, 'grassmann') and hasattr(block.grassmann, 'plucker'):
            h = block.grassmann.plucker.register_forward_hook(make_layer_hook(i))
            handles.append(h)
    return handles

def compute_metrics(plucker_list):
    if not plucker_list: return {"effective_rank": 0}
    all_p = torch.cat([p.reshape(-1, p.shape[-1]) for p in plucker_list], dim=0).float()
    all_p = all_p - all_p.mean(dim=0, keepdim=True)
    try:
        _, S, _ = torch.linalg.svd(all_p, full_matrices=False)
    except:
        return {"effective_rank": 0}
    S = S + 1e-10
    eff = float(((S.sum())**2 / (S**2).sum()).item())
    return {"effective_rank": eff, "plucker_dim": plucker_list[0].shape[-1]}

# ===========================================================================
# Main
# ===========================================================================
def main():
    device = torch.device("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_local", local_files_only=True)
    V = len(tokenizer)

    # Load PTB validation data
    val_ds = TextDataset("validation", tokenizer, 256, dataset_name="ptb",
                         dataset_path="/workspace/grassmannflows/datasets/ptb_text_only_saved")
    val_ldr = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    x_batch, _ = next(iter(val_ldr))
    x_batch = x_batch[:4].to(device)
    del val_ldr

    print(f"Input: {x_batch.shape}")

    baseline_dir = Path("outputs/hybrid_experiments/20260328_142654_ptb_hybrid_lite_baseline_mild_e20")
    bl_summary = json.loads((baseline_dir / "summary.json").read_text())
    bl_ckpt = bl_summary["hybrid"]["checkpoint_path"]

    # =====================================================================
    # V1: r=56 vs r=64
    # =====================================================================
    print("\n" + "=" * 70)
    print("  V1: REDUCED DIMENSION r — r=56 vs r=64")
    print("=" * 70)

    def measure_model(model_dim, reduced_dim, num_layers, ckpt_path, label):
        model = GrassmannGPTv4(
            vocab_size=V, max_seq_len=256, model_dim=model_dim,
            num_layers=num_layers, reduced_dim=reduced_dim,
            ff_dim=4*model_dim, window_sizes=[1,2,4], dropout=0.1,
        ).to(device)

        # Load weights if provided
        if ckpt_path:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            grass_state = {k[len("grassmann."):]: v for k, v in state.items() if k.startswith("grassmann.")}
            model_keys = set(model.state_dict().keys())
            grass_state = {k: v for k, v in grass_state.items() if k in model_keys}
            model.load_state_dict(grass_state, strict=False)

        model.eval()

        global layer_outputs
        layer_outputs.clear()
        handles = register_layer_hooks(model)

        with torch.no_grad():
            _ = model(x_batch)

        for h in handles: h.remove()

        # Aggregate across layers
        all_p = torch.cat([p.reshape(-1, p.shape[-1]) for idx in layer_outputs for p in layer_outputs[idx]], dim=0)
        metrics = compute_metrics([all_p])

        plucker_dim = reduced_dim * (reduced_dim - 1) // 2
        gr_dim = 2 * (reduced_dim - 2)
        print(f"  {label}: r={reduced_dim}, Plucker_dim={plucker_dim}, Gr(2,{reduced_dim})_dim={gr_dim}")
        print(f"    eff_rank={metrics['effective_rank']:.1f} / {plucker_dim} ({metrics['effective_rank']/plucker_dim*100:.1f}%)")
        print(f"    Gr constraint: {gr_dim}/{plucker_dim} = {gr_dim/plucker_dim*100:.1f}% of ambient")
        return metrics

    # r=56 warmstart (existing)
    print()
    r56_warm = measure_model(224, 56, 6, bl_ckpt, "r=56 (warmstart)")

    # r=56 random
    r56_rand = measure_model(224, 56, 6, None, "r=56 (random)")

    # r=64 — random init comparison (ambient filling)
    r64_rand = measure_model(256, 64, 6, None, "r=64 (random)")
    r56_rand2 = measure_model(224, 56, 6, None, "r=56 (random, re-run)")

    # Theoretical comparison: ambient filling
    p56_fill = r56_rand["effective_rank"] / 1540 * 100
    p64_fill = r64_rand["effective_rank"] / 2016 * 100
    print(f"\n  Prediction: Both r values should fill ~90%+ of ambient (random init)")
    print(f"  r=56: {p56_fill:.1f}% of 1540 | Gr(2,56) dim = 108 (7.0% of ambient)")
    print(f"  r=64: {p64_fill:.1f}% of 2016 | Gr(2,64) dim = 124 (6.2% of ambient)")
    print(f"  Random init fills ambient: {'CONFIRMED ✅' if p56_fill > 85 and p64_fill > 85 else 'PARTIAL'}")
    print(f"  r=64 has MORE room to collapse: 2016-124 = 1892 vs 1540-108 = 1432")
    print(f"  → r=64 training should show LARGER ΔRank (more ambient to constrain)")
    print(f"  → r=56 ΔRank (measured): {r56_rand['effective_rank'] - r56_warm['effective_rank']:.0f}")

    # =====================================================================
    # V2: Layer-wise analysis
    # =====================================================================
    print("\n" + "=" * 70)
    print("  V2: LAYER-WISE PLÜCKER RANK (PTB warmstart, r=56)")
    print("=" * 70)

    model = GrassmannGPTv4(
        vocab_size=V, max_seq_len=256, model_dim=224, num_layers=6,
        reduced_dim=56, ff_dim=896, window_sizes=[1,2,4], dropout=0.1,
    ).to(device)

    state = torch.load(bl_ckpt, map_location="cpu", weights_only=True)
    grass_state = {k[len("grassmann."):]: v for k, v in state.items() if k.startswith("grassmann.")}
    model_keys = set(model.state_dict().keys())
    grass_state = {k: v for k, v in grass_state.items() if k in model_keys}
    model.load_state_dict(grass_state, strict=False)
    model.eval()

    layer_outputs.clear()
    handles = register_layer_hooks(model)
    with torch.no_grad():
        _ = model(x_batch)
    for h in handles: h.remove()

    print(f"\n  {'Layer':<8} {'eff_rank':>10} {'/ 1540':>10} {'top5_ratio':>12}")
    print(f"  {'─'*42}")
    for i in sorted(layer_outputs.keys()):
        p = layer_outputs[i]
        if not p: continue
        m = compute_metrics(p)
        top5_ratio = "n/a"
        all_p = torch.cat([x.reshape(-1, x.shape[-1]) for x in p], dim=0).float()
        all_p = all_p - all_p.mean(dim=0, keepdim=True)
        try:
            _, S, _ = torch.linalg.svd(all_p, full_matrices=False)
            top5_ratio = f"{float(S[:5].sum()/S.sum()):.4f}"
        except: pass
        depth = "shallow" if i < 2 else "deep" if i >= 4 else "middle"
        print(f"  Layer {i} ({depth:<7}) {m['effective_rank']:>10.1f} {m['effective_rank']/1540*100:>9.1f}% {top5_ratio:>12}")

    # Check monotonic trend
    ranks = [compute_metrics(layer_outputs[i])["effective_rank"] for i in sorted(layer_outputs.keys()) if layer_outputs[i]]
    decreasing = all(ranks[i] >= ranks[i+1] for i in range(len(ranks)-1))
    print(f"\n  Layer rank monotonic decreasing: {'YES ✅' if decreasing else 'NO ❌'}")
    shallow_avg = sum(ranks[:2])/2 if len(ranks)>=2 else ranks[0]
    deep_avg = sum(ranks[-2:])/2 if len(ranks)>=2 else ranks[-1]
    print(f"  Shallow (L0-1) avg rank: {shallow_avg:.0f} vs Deep (L4-5) avg rank: {deep_avg:.0f}")

    # =====================================================================
    # V3: Epoch scan — load intermediate checkpoints
    # =====================================================================
    print("\n" + "=" * 70)
    print("  V3: EPOCH SCAN — rank decreases monotonically during training")
    print("=" * 70)

    # Use subspace analysis checkpoints from earlier training run
    subspace_dir = Path("outputs/subspace_analysis")
    ce_dirs = sorted(subspace_dir.glob("*subspace_ce_scratch*"))

    # The spectral_history files from the training run had the hook bug (all zeros).
    # Use a more robust approach: load checkpoint at different training stages
    # and measure Plucker rank at each stage.
    #
    # We have checkpoints saved at each best epoch. We'll simulate by:
    # - Random init: fresh model before any training
    # - Mid-training: load a warmstart checkpoint (baseline, 20 epochs)
    # - Post-KD: load KD checkpoint (λ=0.02, 10 epochs)
    # These three points show the rank trajectory.

    print(f"\n  --- Epoch Scan (snapshot-based) ---")
    print(f"  {'Stage':<20} {'eff_rank':>10} {'/ ambient':>10}")
    print(f"  {'─'*42}")

    # Re-use the earlier measurements
    r56_rand_v3 = measure_model(224, 56, 6, None, "Random init")
    r56_warm_v3 = measure_model(224, 56, 6, bl_ckpt, "Warmstart (20ep CE)")

    # KD checkpoint
    kd_dir = sorted(Path("outputs/distill_experiments").glob("*ptb_warmstart_lambda_20*"))
    if kd_dir:
        kd_summary = json.loads((kd_dir[-1] / "summary.json").read_text())
        kd_ckpt = kd_summary["student"].get("checkpoint_path", str(kd_dir[-1] / "checkpoints" / "student_best.pt"))
        r56_kd = measure_model(224, 56, 6, kd_ckpt, "KD λ=0.02 (10ep)")
        print(f"\n  Trajectory: {r56_rand_v3['effective_rank']:.0f} → {r56_warm_v3['effective_rank']:.0f} → {r56_kd['effective_rank']:.0f}")

    decreasing = r56_rand_v3["effective_rank"] > r56_warm_v3["effective_rank"]
    print(f"  Random → Trained decreases: {'YES ✅' if decreasing else 'NO ❌'}")
    if kd_dir:
        kd_decreasing = r56_warm_v3["effective_rank"] > r56_kd["effective_rank"]
        print(f"  Trained → KD decreases: {'YES ✅' if kd_decreasing else 'NO ❌'}")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("  V1 (r=56 vs r=64): Larger r → more ambient dim to collapse → larger ΔRank")
    print("  V2 (layer-wise): Deep layers should have LOWER rank (more semantic)")
    print("  V3 (epoch scan): eff_rank should monotonically DECREASE during training")

if __name__ == "__main__":
    main()
