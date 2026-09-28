"""
Subspace Rank Collapse Analysis

3 training conditions, single GrassmannGPTv4 student:
  ce_scratch:   Random init + CE
  kd_scratch:   Random init + KD (hybrid teacher)
  kd_warmstart: Warm-start from baseline + KD

Measures Plucker SVD spectrum every epoch to test:
  "Random init KD collapses subspace rank; warmstart preserves it."
"""
import os, sys, json, time, math, argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm

sys.path.insert(0, "src")
from models import GrassmannGPTv4
from train_exp4_ddp import SmallTransformer, TextDataset
from train_distill_hybrid_lite_from_latefusion_teacher_v2 import (
    HybridLateFusionAlphaModel, load_teacher_and_context,
    resolve_migrated_path, build_search_roots, parse_path_remaps,
)

# ---------------------------------------------------------------------------
# Plucker hook + spectral metrics
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

def compute_spectral(plucker_list):
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
# Training
# ---------------------------------------------------------------------------
def train_with_spectral(student, teacher, train_loader, val_loader, optimizer,
                        scheduler, device, epochs, alpha, T, output_dir):
    history = []
    best_val_ppl = float("inf")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        student.train()
        if teacher: teacher.eval()

        total_loss = total_ce = total_kl = total_tok = 0.0

        pbar = tqdm(train_loader, desc=f"E{epoch}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            if teacher:
                with torch.no_grad():
                    t_logits, _ = teacher(x, labels=None)
                s_logits, _ = student(x, labels=None)
                # KD loss
                ss = s_logits[:, :-1].contiguous()
                tt = t_logits[:, :-1].contiguous()
                yy = y[:, 1:].contiguous()
                ce = F.cross_entropy(ss.view(-1, ss.size(-1)), yy.view(-1), ignore_index=-100)
                s_lp = F.log_softmax(ss / T, dim=-1)
                t_p  = F.softmax(tt / T, dim=-1)
                kl = F.kl_div(s_lp, t_p, reduction="batchmean") * (T ** 2)
                loss = (1 - alpha) * ce + alpha * kl
                ce_v, kl_v = float(ce.item()), float(kl.item())
            else:
                _, loss = student(x, labels=y)
                ce_v, kl_v = float(loss.item()), 0.0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * x.size(0)
            total_ce   += ce_v * x.size(0)
            total_kl   += kl_v * x.size(0)
            total_tok  += x.numel()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Spectral measurement: register hooks ONLY for this pass
        student.eval()
        global plucker_snapshots
        plucker_snapshots.clear()
        hooks = register_hooks(student)
        with torch.no_grad():
            for x_val, _ in val_loader:
                _ = student(x_val[:4].to(device))
                break
        metrics = compute_spectral(plucker_snapshots)
        for h in hooks: h.remove()

        # Validation PPL
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                _, lv = student(x_v, labels=y_v)
                val_loss += lv.item() * x_v.size(0)
                val_count += x_v.size(0)
        val_loss /= val_count
        val_ppl = math.exp(min(val_loss, 20))

        record = {"epoch": epoch, "train_loss": total_loss / len(train_loader.dataset),
                  "train_ce": total_ce / len(train_loader.dataset),
                  "train_kl": total_kl / len(train_loader.dataset),
                  "val_loss": val_loss, "val_ppl": val_ppl, **metrics}
        history.append(record)

        print(f"  epoch={epoch:2d} val_ppl={val_ppl:7.2f} eff_rank={metrics['effective_rank']:6.1f} "
              f"ent={metrics['spectral_entropy']:.4f} top5={metrics['top5_ratio']:.4f}")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(student.state_dict(), output_dir / "best.pt")

    return history

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True,
                        choices=["ce_scratch", "kd_scratch", "kd_warmstart"])
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--dataset-path", default="/workspace/grassmannflows/datasets/ptb_text_only_saved")
    parser.add_argument("--teacher-dir", default="outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint")
    parser.add_argument("--baseline-dir", default="outputs/hybrid_experiments/20260328_142654_ptb_hybrid_lite_baseline_mild_e20")
    parser.add_argument("--output-dir", default="outputs/subspace_analysis")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--distill-alpha", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_local", local_files_only=True)
    V = len(tokenizer)

    train_ds = TextDataset("train", tokenizer, 256, dataset_name="ptb",
                           dataset_path=args.dataset_path)
    val_ds   = TextDataset("validation", tokenizer, 256, dataset_name="ptb",
                           dataset_path=args.dataset_path)

    train_ldr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_ldr   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Student: single-branch Grassmann
    student = GrassmannGPTv4(
        vocab_size=V, max_seq_len=256, model_dim=224, num_layers=6,
        reduced_dim=56, ff_dim=896, window_sizes=[1, 2, 4], dropout=0.1,
    )

    # Warmstart: load Grassmann branch weights from hybrid-lite baseline
    if args.condition == "kd_warmstart":
        bl = Path(args.baseline_dir)
        bl_sum = json.loads((bl / "summary.json").read_text())
        bl_ckpt = bl_sum["hybrid"].get("checkpoint_path",
                                        str(bl / "checkpoints" / "hybrid_best.pt"))
        # Resolve path
        bl_ckpt = resolve_migrated_path(bl_ckpt, search_roots=build_search_roots(), remaps=[], kind="baseline_ckpt")
        state = torch.load(bl_ckpt, map_location="cpu", weights_only=True)
        # Extract grassmann.* weights
        grass_state = {k[len("grassmann."):]: v for k, v in state.items() if k.startswith("grassmann.")}
        missing, unexpected = student.load_state_dict(grass_state, strict=False)
        print(f"Warmstart loaded: {len(grass_state)} grassmann keys, {len(missing)} missing, {len(unexpected)} unexpected")

    student = student.to(device)
    n_p = sum(p.numel() for p in student.parameters())
    print(f"Student: {n_p:,} params ({n_p/1e6:.2f}M) | Condition: {args.condition}")

    # Teacher
    teacher = None
    if args.condition in ("kd_scratch", "kd_warmstart"):
        td = Path(args.teacher_dir)
        teacher, ctx = load_teacher_and_context(td, V, build_search_roots(), [], device)
        print(f"Teacher loaded: test_ppl={ctx['teacher_summary']['hybrid']['test_ppl']:.2f}")

    # Optimizer
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    tot = len(train_ldr) * args.epochs
    warm = int(tot * 0.05)
    def lr_fn(s):
        if s < warm: return s / max(1, warm)
        p = (s - warm) / max(1, tot - warm)
        return 0.5 * (1 + math.cos(math.pi * p))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    out = Path(args.output_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_subspace_{args.condition}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Training {args.epochs} epochs...")
    hist = train_with_spectral(
        student, teacher, train_ldr, val_ldr, opt, sch,
        device, args.epochs, args.distill_alpha, args.temperature, out,
    )

    json.dump(hist, open(out / "spectral_history.json", "w"), indent=2)
    print(f"\nSaved to {out / 'spectral_history.json'}")

    # Summary table
    print(f"\n{'Ep':>4s} {'val_ppl':>8s} {'eff_rank':>9s} {'entropy':>9s} {'top5':>7s}")
    for r in hist:
        print(f"{r['epoch']:4d} {r['val_ppl']:8.2f} {r['effective_rank']:9.1f} {r['spectral_entropy']:9.4f} {r['top5_ratio']:7.4f}")


if __name__ == "__main__":
    main()
