"""
Layer-wise Plucker rank across 4 datasets.
Tests: Is the U-shaped curve universal?
"""
import os, sys, json, math
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

sys.path.insert(0, "src")
from models import GrassmannGPTv4
from train_exp4_ddp import TextDataset
from train_distill_hybrid_lite_from_latefusion_teacher_v2 import resolve_migrated_path, build_search_roots

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

layer_outputs = defaultdict(list)

def make_hook(idx):
    def hook(m, inp, out):
        layer_outputs[idx].append(out.detach().cpu())
    return hook

DATASETS = {
    "PTB": {
        "dataset_path": "/workspace/grassmannflows/datasets/ptb_text_only_saved",
        "dataset_name": "ptb",
        "ckpt": "outputs/hybrid_experiments/20260328_142654_ptb_hybrid_lite_baseline_mild_e20",
        "model_dim": 224, "reduced_dim": 56, "num_layers": 6,
    },
    "WT2": {
        "dataset_path": "/workspace/grassmannflows/datasets/wikitext2_v1_saved",
        "dataset_name": "wikitext2",
        "ckpt": "outputs/hybrid_experiments/20260515_004119_wt2_hybrid_lite_baseline_224x56_l6_e20",
        "model_dim": 224, "reduced_dim": 56, "num_layers": 6,
    },
    "TinyStories": {
        "dataset_path": "/workspace/grassmannflows/datasets/tinystories_saved",
        "dataset_name": "tinystories",
        "ckpt": "outputs/hybrid_experiments/20260515_031101_ts_hybrid_lite_baseline_224x56_l6_e20",
        "model_dim": 224, "reduced_dim": 56, "num_layers": 6,
    },
    "Code": {
        "dataset_path": "/workspace/grassmannflows/datasets/codeparrot_saved",
        "dataset_name": "ptb",
        "text_field": "text",
        "ckpt": "outputs/hybrid_experiments/20260527_074518_code_hybrid_lite_baseline_ce_scratch",
        "model_dim": 224, "reduced_dim": 56, "num_layers": 6,
    },
}

def main():
    device = torch.device("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_local", local_files_only=True)
    V = len(tokenizer)

    all_layer_ranks = {}

    for ds_name, cfg in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"  {ds_name}")
        print(f"{'='*60}")

        try:
            val_ds = TextDataset("validation", tokenizer, 256,
                                 dataset_name=cfg["dataset_name"],
                                 dataset_path=cfg["dataset_path"],
                                 text_field=cfg.get("text_field", ""))
            val_ldr = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
            x_batch, _ = next(iter(val_ldr))
            x_batch = x_batch[:4].to(device)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        model = GrassmannGPTv4(
            vocab_size=V, max_seq_len=256, model_dim=cfg["model_dim"],
            num_layers=cfg["num_layers"], reduced_dim=cfg["reduced_dim"],
            ff_dim=4*cfg["model_dim"], window_sizes=[1,2,4], dropout=0.1,
        ).to(device)

        # Load weights
        ckpt_dir = Path(cfg["ckpt"])
        summary = json.loads((ckpt_dir / "summary.json").read_text())
        ckpt_path = summary["hybrid"].get("checkpoint_path", str(ckpt_dir / "checkpoints" / "hybrid_best.pt"))
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        grass_state = {k[len("grassmann."):]: v for k, v in state.items() if k.startswith("grassmann.")}
        model_keys = set(model.state_dict().keys())
        grass_state = {k: v for k, v in grass_state.items() if k in model_keys}
        model.load_state_dict(grass_state, strict=False)
        model.eval()

        global layer_outputs
        layer_outputs.clear()
        handles = [model.blocks[i].grassmann.plucker.register_forward_hook(make_hook(i))
                   for i in range(cfg["num_layers"])
                   if hasattr(model.blocks[i], 'grassmann') and hasattr(model.blocks[i].grassmann, 'plucker')]

        with torch.no_grad():
            _ = model(x_batch)
        for h in handles: h.remove()

        ranks = {}
        for i in sorted(layer_outputs.keys()):
            p = torch.cat([x.reshape(-1, x.shape[-1]) for x in layer_outputs[i]], dim=0).float()
            p = p - p.mean(dim=0, keepdim=True)
            try:
                _, S, _ = torch.linalg.svd(p, full_matrices=False)
                S = S + 1e-10
                eff = float(((S.sum())**2 / (S**2).sum()).item())
            except:
                eff = 0
            ranks[i] = eff
            pct = eff / 1540 * 100
            bar = "█" * int(pct/2) + "░" * (50 - int(pct/2))
            print(f"  L{i}: {bar} {eff:>6.0f} ({pct:>5.1f}%)")

        all_layer_ranks[ds_name] = ranks

    # Cross-dataset comparison table
    print(f"\n\n{'='*75}")
    print(f"  CROSS-DATASET LAYER-WISE RANK COMPARISON")
    print(f"{'='*75}")
    print(f"  {'Layer':<8} {'PTB':>9} {'WT2':>9} {'TS':>9} {'Code':>9}")
    print(f"  {'─'*48}")

    for i in range(6):
        row = f"  L{i:<7}"
        for ds in ["PTB", "WT2", "TinyStories", "Code"]:
            if ds in all_layer_ranks and i in all_layer_ranks[ds]:
                row += f" {all_layer_ranks[ds][i]:>9.0f}"
            else:
                row += f" {'─':>9}"
        print(row)

    # U-shape confirmation
    print(f"\n  {'='*60}")
    print(f"  U-SHAPE VERIFICATION")
    print(f"  {'='*60}")
    for ds in ["PTB", "WT2", "TinyStories", "Code"]:
        if ds not in all_layer_ranks: continue
        r = all_layer_ranks[ds]
        shallow = sum(r[i] for i in [0,1]) / 2
        middle  = sum(r[i] for i in [2,3]) / 2
        deep    = sum(r[i] for i in [4,5]) / 2
        is_u = middle < shallow and middle < deep
        print(f"  {ds:<20} shallow={shallow:.0f} middle={middle:.0f} deep={deep:.0f}  U-shape={'✅' if is_u else '❌'}")

if __name__ == "__main__":
    main()
