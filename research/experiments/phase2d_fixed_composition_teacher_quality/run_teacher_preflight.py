#!/usr/bin/env python3
"""Evaluate one frozen Phase 2D teacher and all three S0 students."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_SELECTION_HASH = "236633612db1e3144b8f3187a5bfb1795f558a50f26368252a8adf3f0b49413f"
STUDENTS = {
    42: (
        "outputs/hybrid_experiments/20260515_004119_wt2_hybrid_lite_baseline_224x56_l6_e20",
        "9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef",
    ),
    123: (
        "outputs/hybrid_experiments/20260922_071001_phase2c_wt2_s0_seed123",
        "a44b64ba5a84d3bd2708a96aad4eae829ff7ab0f00476f0e31146fc314da1f13",
    ),
    456: (
        "outputs/hybrid_experiments/20260922_071449_phase2c_wt2_s0_seed456",
        "3fbe985d6474c84d1c61fd7937ea906102cf460846829eb3af4c709593b01757",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-label", choices=["J", "A"], required=True)
    parser.add_argument("--teacher-run-dir", required=True)
    parser.add_argument("--dataset-path", default="/workspace/grassmannflows/datasets/wikitext2_v1_saved")
    parser.add_argument("--tokenizer-dir", default="./gpt2_local")
    parser.add_argument("--selection-seed", type=int, default=20260920)
    parser.add_argument("--max-chunks", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--token-block", type=int, default=32)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_concat(parts):
    return float(np.concatenate(parts).astype(np.float64).mean())


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    import torch.nn.functional as F
    from torch.amp import autocast
    from torch.utils.data import DataLoader, Subset
    from transformers import GPT2Tokenizer
    import train_distill_hybrid_lite_from_latefusion_teacher_v2 as train_module

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required on server 197")
    device = torch.device("cuda", 0)
    search_roots = train_module.build_search_roots([])
    teacher_dir = Path(train_module.resolve_migrated_path(
        args.teacher_run_dir, search_roots=search_roots, remaps=[], kind="teacher_run_dir", must_exist=True,
    ))
    tokenizer_dir = train_module.resolve_migrated_path(
        args.tokenizer_dir, search_roots=search_roots, remaps=[], kind="tokenizer_dir", must_exist=True,
    )
    dataset_path = train_module.resolve_migrated_path(
        args.dataset_path, search_roots=search_roots, remaps=[], kind="dataset_path", must_exist=True,
    )
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    teacher, context = train_module.load_teacher_and_context(
        teacher_dir, len(tokenizer), search_roots, [], device
    )
    checkpoint_alpha, effective_alpha = train_module.apply_teacher_alpha_override(teacher, 0.5)
    if effective_alpha != [0.5]:
        raise RuntimeError(f"Effective alpha is not exactly 0.5: {effective_alpha}")
    teacher.eval()

    dataset = train_module.TextDataset(
        "validation", tokenizer, int(context["max_seq_len"]),
        dataset_name="wikitext2", dataset_path=str(dataset_path), text_field="text",
        max_lines=0, encode_chars_per_batch=200000, tinystories_val_frac=0.02, split_seed=42,
    )
    rng = random.Random(args.selection_seed)
    indices = sorted(rng.sample(range(len(dataset)), args.max_chunks))
    selection_hash = hashlib.sha256(",".join(map(str, indices)).encode("utf-8")).hexdigest()
    if args.max_chunks == 512 and selection_hash != EXPECTED_SELECTION_HASH:
        raise RuntimeError(f"Validation selection hash mismatch: {selection_hash}")
    loader = DataLoader(
        Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=2,
        pin_memory=True, drop_last=False,
    )

    model_args = SimpleNamespace(
        model_dim=224, num_layers=6, num_heads=8, reduced_dim=56,
        window_sizes="1,2,4", dropout=0.1, student_late_k=1,
    )
    teacher_parts = None
    student_rows = []
    for student_index, (seed, (student_run, expected_hash)) in enumerate(STUDENTS.items()):
        student = train_module.instantiate_student_from_args(
            "hybrid_lite", len(tokenizer), int(context["max_seq_len"]), model_args,
        ).to(device)
        student_info = train_module.maybe_load_student_init(
            student, student_run, search_roots, [], device, expected_sha256=expected_hash,
        )
        student.eval()
        arrays = {key: [] for key in [
            "student_nll", "student_correct", "teacher_nll", "teacher_correct",
            "utility", "kl_t1", "kl_t2", "ce_grad_norm", "kd_grad_norm", "grad_cosine",
        ]}
        if teacher_parts is None:
            teacher_parts = {key: [] for key in [
                "fused_nll", "transformer_nll", "grassmann_nll", "branch_jsd",
                "branch_agreement", "fused_entropy", "fused_gold_prob",
                "transformer_gold_prob", "grassmann_gold_prob",
            ]}
        seen = 0
        with torch.inference_mode():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with autocast("cuda", enabled=True):
                    fused_logits, _, branches = teacher(x, return_branches=True)
                    student_logits, _ = student(x)
                labels = y[:, 1:].reshape(-1)
                fused_logits = fused_logits[:, :-1, :].reshape(-1, fused_logits.size(-1))
                t_logits = branches["transformer"][:, :-1, :].reshape(-1, fused_logits.size(-1))
                g_logits = branches["grassmann"][:, :-1, :].reshape(-1, fused_logits.size(-1))
                student_logits = student_logits[:, :-1, :].reshape(-1, fused_logits.size(-1))
                valid = labels.ne(-100)
                labels = labels[valid]
                fused_logits = fused_logits[valid]
                t_logits = t_logits[valid]
                g_logits = g_logits[valid]
                student_logits = student_logits[valid]
                for start in range(0, labels.numel(), args.token_block):
                    end = min(start + args.token_block, labels.numel())
                    yy = labels[start:end]
                    row = torch.arange(yy.numel(), device=device)
                    ff = fused_logits[start:end].float()
                    tt = t_logits[start:end].float()
                    gg = g_logits[start:end].float()
                    ss = student_logits[start:end].float()
                    log_f = F.log_softmax(ff, dim=-1)
                    log_t = F.log_softmax(tt, dim=-1)
                    log_g = F.log_softmax(gg, dim=-1)
                    log_s = F.log_softmax(ss, dim=-1)
                    p_f, p_t, p_g, p_s = log_f.exp(), log_t.exp(), log_g.exp(), log_s.exp()
                    s_nll = -log_s[row, yy]
                    f_nll = -log_f[row, yy]
                    arrays["student_nll"].append(s_nll.cpu().numpy())
                    arrays["student_correct"].append(ss.argmax(-1).eq(yy).cpu().numpy())
                    arrays["teacher_nll"].append(f_nll.cpu().numpy())
                    arrays["teacher_correct"].append(ff.argmax(-1).eq(yy).cpu().numpy())
                    arrays["utility"].append((s_nll - f_nll).cpu().numpy())
                    p_s_temp = F.softmax(ss / args.temperature, dim=-1)
                    p_f_temp = F.softmax(ff / args.temperature, dim=-1)
                    log_s_temp = F.log_softmax(ss / args.temperature, dim=-1)
                    log_f_temp = F.log_softmax(ff / args.temperature, dim=-1)
                    arrays["kl_t1"].append((p_f * (log_f - log_s)).sum(-1).clamp_min(0).cpu().numpy())
                    arrays["kl_t2"].append((p_f_temp * (log_f_temp - log_s_temp)).sum(-1).clamp_min(0).cpu().numpy())
                    g_ce = p_s.clone()
                    g_ce[row, yy] -= 1.0
                    g_kd = args.temperature * (p_s_temp - p_f_temp)
                    ce_norm = g_ce.square().sum(-1).sqrt()
                    kd_norm = g_kd.square().sum(-1).sqrt()
                    cosine = (g_ce * g_kd).sum(-1) / (ce_norm.clamp_min(1e-30) * kd_norm.clamp_min(1e-30))
                    arrays["ce_grad_norm"].append(ce_norm.cpu().numpy())
                    arrays["kd_grad_norm"].append(kd_norm.cpu().numpy())
                    arrays["grad_cosine"].append(cosine.clamp(-1, 1).cpu().numpy())
                    if student_index == 0:
                        mix = 0.5 * (p_t + p_g)
                        jsd = 0.5 * (p_t * (log_t - mix.clamp_min(1e-30).log())).sum(-1)
                        jsd += 0.5 * (p_g * (log_g - mix.clamp_min(1e-30).log())).sum(-1)
                        teacher_parts["fused_nll"].append(f_nll.cpu().numpy())
                        teacher_parts["transformer_nll"].append((-log_t[row, yy]).cpu().numpy())
                        teacher_parts["grassmann_nll"].append((-log_g[row, yy]).cpu().numpy())
                        teacher_parts["branch_jsd"].append(jsd.cpu().numpy())
                        teacher_parts["branch_agreement"].append(tt.argmax(-1).eq(gg.argmax(-1)).cpu().numpy())
                        teacher_parts["fused_entropy"].append((-(p_f * log_f).sum(-1)).cpu().numpy())
                        teacher_parts["fused_gold_prob"].append(p_f[row, yy].cpu().numpy())
                        teacher_parts["transformer_gold_prob"].append(p_t[row, yy].cpu().numpy())
                        teacher_parts["grassmann_gold_prob"].append(p_g[row, yy].cpu().numpy())
                seen += int(x.size(0))
                if seen % 64 == 0:
                    print(f"teacher={args.teacher_label} seed={seed} chunks={seen}/{len(indices)}", flush=True)

        packed = {key: np.concatenate(value) for key, value in arrays.items()}
        student_nll = packed["student_nll"].astype(np.float64)
        utility = packed["utility"].astype(np.float64)
        student_correct = packed["student_correct"].astype(bool)
        teacher_correct = packed["teacher_correct"].astype(bool)
        quintile_edge = np.quantile(student_nll, 0.8)
        hard = student_nll >= quintile_edge
        wrong = ~student_correct
        correct = student_correct
        student_rows.append({
            "teacher": args.teacher_label,
            "student_seed": seed,
            "student_checkpoint_sha256": student_info["student_init_checkpoint_sha256"],
            "valid_tokens": int(utility.size),
            "student_nll": float(student_nll.mean()),
            "teacher_nll": float(packed["teacher_nll"].mean()),
            "mean_gold_token_utility": float(utility.mean()),
            "positive_utility_fraction": float((utility > 0).mean()),
            "hardest_quintile_threshold": float(quintile_edge),
            "hardest_quintile_utility": float(utility[hard].mean()),
            "hardest_quintile_positive_fraction": float((utility[hard] > 0).mean()),
            "top1_rescue_rate": float(teacher_correct[wrong].mean()),
            "top1_harm_rate": float((~teacher_correct[correct]).mean()),
            "teacher_student_kl_t1": float(packed["kl_t1"].mean()),
            "teacher_student_kl_t2": float(packed["kl_t2"].mean()),
            "ce_logit_gradient_norm": float(packed["ce_grad_norm"].mean()),
            "kd_logit_gradient_norm": float(packed["kd_grad_norm"].mean()),
            "ce_kd_cosine": float(packed["grad_cosine"].mean()),
            "negative_cosine_fraction": float((packed["grad_cosine"] < 0).mean()),
        })
        del student, packed, arrays
        torch.cuda.empty_cache()

    teacher_values = {key: np.concatenate(value) for key, value in teacher_parts.items()}
    t_nll = float(teacher_values["transformer_nll"].mean())
    g_nll = float(teacher_values["grassmann_nll"].mean())
    f_nll = float(teacher_values["fused_nll"].mean())
    teacher_row = {
        "teacher": args.teacher_label,
        "effective_alpha": 0.5,
        "fused_nll": f_nll,
        "fused_ppl": math.exp(f_nll),
        "transformer_branch_nll": t_nll,
        "grassmann_branch_nll": g_nll,
        "fusion_gain_over_better_branch": min(t_nll, g_nll) - f_nll,
        "branch_jsd": float(teacher_values["branch_jsd"].mean()),
        "branch_top1_agreement": float(teacher_values["branch_agreement"].mean()),
        "teacher_entropy": float(teacher_values["fused_entropy"].mean()),
        "fused_gold_probability": float(teacher_values["fused_gold_prob"].mean()),
        "transformer_gold_probability": float(teacher_values["transformer_gold_prob"].mean()),
        "grassmann_gold_probability": float(teacher_values["grassmann_gold_prob"].mean()),
    }
    payload = {
        "manifest": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_only": True,
            "teacher": args.teacher_label,
            "teacher_run_dir": str(teacher_dir.resolve()),
            "teacher_checkpoint": context["hybrid_ckpt"],
            "teacher_checkpoint_sha256": sha256_file(context["hybrid_ckpt"]),
            "checkpoint_alpha": checkpoint_alpha,
            "effective_alpha": effective_alpha,
            "selection_seed": args.selection_seed,
            "selected_indices_sha256": selection_hash,
            "chunks": len(indices),
            "valid_tokens": int(teacher_values["fused_nll"].size),
            "dataset_stats": dataset.stats,
            "raw_logits_saved": False,
            "raw_text_saved": False,
        },
        "teacher_metrics": teacher_row,
        "student_utility": student_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "teacher_metrics": teacher_row, "student_utility": student_rows}, indent=2))


if __name__ == "__main__":
    main()
