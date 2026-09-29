#!/usr/bin/env python3
"""Evaluate F/T/G teacher utility and branch-complementarity groups for one S0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_TEACHER_HASH = "a5ad284f6895c0b4ed516c855c1b149e6636b945173aa2e842b8132299582694"
EXPECTED_SELECTION_HASH = "236633612db1e3144b8f3187a5bfb1795f558a50f26368252a8adf3f0b49413f"
CONDITIONS = {"F": 0.5, "T": 1.0, "G": 0.0}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-seed", type=int, required=True)
    parser.add_argument("--student-run-dir", required=True)
    parser.add_argument("--expected-student-sha256", required=True)
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


def concatenate(parts, dtype=None):
    array = np.concatenate(parts)
    return array.astype(dtype) if dtype is not None else array


def summarize_group(seed, group_type, group_value, mask, arrays):
    count = int(mask.sum())
    if count == 0:
        raise RuntimeError(f"Empty branch-utility group: {group_type}/{group_value}")
    total = int(mask.size)
    return {
        "student_seed": seed,
        "group_type": group_type,
        "group_value": group_value,
        "count": count,
        "fraction": count / total,
        "mean_student_loss": float(arrays["student_nll"][mask].mean()),
        "mean_u_T": float(arrays["u_T"][mask].mean()),
        "mean_u_G": float(arrays["u_G"][mask].mean()),
        "mean_u_F": float(arrays["u_F"][mask].mean()),
        "fused_improvement_over_transformer_gold_logprob": float(
            (arrays["u_F"][mask] - arrays["u_T"][mask]).mean()
        ),
        "fused_improvement_over_grassmann_gold_logprob": float(
            (arrays["u_F"][mask] - arrays["u_G"][mask]).mean()
        ),
    }


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
        args.teacher_run_dir, search_roots=search_roots, remaps=[],
        kind="teacher_run_dir", must_exist=True,
    ))
    tokenizer_dir = train_module.resolve_migrated_path(
        args.tokenizer_dir, search_roots=search_roots, remaps=[],
        kind="tokenizer_dir", must_exist=True,
    )
    dataset_path = train_module.resolve_migrated_path(
        args.dataset_path, search_roots=search_roots, remaps=[],
        kind="dataset_path", must_exist=True,
    )
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    teacher, context = train_module.load_teacher_and_context(
        teacher_dir, len(tokenizer), search_roots, [], device,
    )
    teacher_hash = sha256_file(context["hybrid_ckpt"])
    if teacher_hash != EXPECTED_TEACHER_HASH:
        raise RuntimeError(f"Teacher checkpoint hash mismatch: {teacher_hash}")
    teacher.eval()

    model_args = SimpleNamespace(
        model_dim=224, num_layers=6, num_heads=8, reduced_dim=56,
        window_sizes="1,2,4", dropout=0.1, student_late_k=1,
    )
    student = train_module.instantiate_student_from_args(
        "hybrid_lite", len(tokenizer), int(context["max_seq_len"]), model_args,
    ).to(device)
    student_info = train_module.maybe_load_student_init(
        student, args.student_run_dir, search_roots, [], device,
        expected_sha256=args.expected_student_sha256,
    )
    student.eval()

    dataset = train_module.TextDataset(
        "validation", tokenizer, int(context["max_seq_len"]),
        dataset_name="wikitext2", dataset_path=str(dataset_path), text_field="text",
        max_lines=0, encode_chars_per_batch=200000, tinystories_val_frac=0.02,
        split_seed=42,
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

    shared = defaultdict(list)
    condition_parts = {label: defaultdict(list) for label in CONDITIONS}
    eps = 1e-30
    seen = 0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast("cuda", enabled=True):
                _, _, branches = teacher(x, return_branches=True)
                student_logits, _ = student(x)
            t_logits = branches["transformer"][:, :-1, :]
            g_logits = branches["grassmann"][:, :-1, :]
            student_logits = student_logits[:, :-1, :]
            labels = y[:, 1:]
            valid = labels.ne(-100)
            labels = labels[valid]
            t_logits = t_logits[valid]
            g_logits = g_logits[valid]
            student_logits = student_logits[valid]
            for start in range(0, labels.numel(), args.token_block):
                end = min(start + args.token_block, labels.numel())
                yy = labels[start:end]
                row = torch.arange(yy.numel(), device=device)
                tt = t_logits[start:end].float()
                gg = g_logits[start:end].float()
                ss = student_logits[start:end].float()
                log_s = F.log_softmax(ss, dim=-1)
                p_s = log_s.exp()
                s_nll = -log_s[row, yy]
                s_correct = ss.argmax(-1).eq(yy)
                log_t = F.log_softmax(tt, dim=-1)
                log_g = F.log_softmax(gg, dim=-1)
                t_gold = log_t[row, yy]
                g_gold = log_g[row, yy]
                t_correct = tt.argmax(-1).eq(yy)
                g_correct = gg.argmax(-1).eq(yy)
                shared["student_nll"].append(s_nll.cpu().numpy())
                shared["student_correct"].append(s_correct.cpu().numpy())
                shared["t_gold"].append(t_gold.cpu().numpy())
                shared["g_gold"].append(g_gold.cpu().numpy())
                shared["t_correct"].append(t_correct.cpu().numpy())
                shared["g_correct"].append(g_correct.cpu().numpy())

                p_s_temp = F.softmax(ss / args.temperature, dim=-1)
                log_s_temp = F.log_softmax(ss / args.temperature, dim=-1)
                g_ce = p_s.clone()
                g_ce[row, yy] -= 1.0
                ce_norm = g_ce.square().sum(-1).sqrt()
                for label, alpha in CONDITIONS.items():
                    teacher_logits = alpha * tt + (1.0 - alpha) * gg
                    log_teacher = F.log_softmax(teacher_logits, dim=-1)
                    p_teacher = log_teacher.exp()
                    teacher_nll = -log_teacher[row, yy]
                    teacher_correct = teacher_logits.argmax(-1).eq(yy)
                    p_teacher_temp = F.softmax(teacher_logits / args.temperature, dim=-1)
                    log_teacher_temp = F.log_softmax(teacher_logits / args.temperature, dim=-1)
                    kl_t1 = (p_teacher * (log_teacher - log_s)).sum(-1).clamp_min(0.0)
                    kl_t2 = (p_teacher_temp * (log_teacher_temp - log_s_temp)).sum(-1).clamp_min(0.0)
                    g_kd = args.temperature * (p_s_temp - p_teacher_temp)
                    kd_norm = g_kd.square().sum(-1).sqrt()
                    cosine = (g_ce * g_kd).sum(-1) / (
                        ce_norm.clamp_min(eps) * kd_norm.clamp_min(eps)
                    )
                    store = condition_parts[label]
                    store["teacher_nll"].append(teacher_nll.cpu().numpy())
                    store["teacher_entropy"].append((-(p_teacher * log_teacher).sum(-1)).cpu().numpy())
                    store["teacher_correct"].append(teacher_correct.cpu().numpy())
                    store["utility"].append((s_nll - teacher_nll).cpu().numpy())
                    store["kl_t1"].append(kl_t1.cpu().numpy())
                    store["kl_t2"].append(kl_t2.cpu().numpy())
                    store["ce_grad_norm"].append(ce_norm.cpu().numpy())
                    store["kd_grad_norm"].append(kd_norm.cpu().numpy())
                    store["grad_cosine"].append(cosine.clamp(-1.0, 1.0).cpu().numpy())
                    if label == "F":
                        shared["f_gold"].append(log_teacher[row, yy].cpu().numpy())
            seen += int(x.size(0))
            if seen % 64 == 0:
                print(f"seed={args.student_seed} chunks={seen}/{len(indices)}", flush=True)

    arrays = {
        "student_nll": concatenate(shared["student_nll"], np.float64),
        "student_correct": concatenate(shared["student_correct"], bool),
        "t_gold": concatenate(shared["t_gold"], np.float64),
        "g_gold": concatenate(shared["g_gold"], np.float64),
        "f_gold": concatenate(shared["f_gold"], np.float64),
        "t_correct": concatenate(shared["t_correct"], bool),
        "g_correct": concatenate(shared["g_correct"], bool),
    }
    arrays["u_T"] = arrays["t_gold"] + arrays["student_nll"]
    arrays["u_G"] = arrays["g_gold"] + arrays["student_nll"]
    arrays["u_F"] = arrays["f_gold"] + arrays["student_nll"]
    edges = np.quantile(arrays["student_nll"], [0.2, 0.4, 0.6, 0.8])
    quintiles = np.searchsorted(edges, arrays["student_nll"], side="right") + 1

    overall_rows = []
    for label, alpha in CONDITIONS.items():
        packed = {key: concatenate(parts) for key, parts in condition_parts[label].items()}
        teacher_nll = packed["teacher_nll"].astype(np.float64)
        utility = packed["utility"].astype(np.float64)
        teacher_correct = packed["teacher_correct"].astype(bool)
        hard = quintiles == 5
        wrong = ~arrays["student_correct"]
        correct = arrays["student_correct"]
        overall_rows.append({
            "student_seed": args.student_seed,
            "condition": label,
            "effective_alpha": alpha,
            "chunks": len(indices),
            "valid_tokens": int(utility.size),
            "student_nll": float(arrays["student_nll"].mean()),
            "teacher_nll": float(teacher_nll.mean()),
            "teacher_ppl": math.exp(float(teacher_nll.mean())),
            "teacher_entropy": float(packed["teacher_entropy"].mean()),
            "delta_teacher": float(arrays["student_nll"].mean() - teacher_nll.mean()),
            "mean_gold_token_utility": float(utility.mean()),
            "positive_utility_fraction": float((utility > 0).mean()),
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

    group_rows = []
    group_masks = [
        ("gold_probability_order", "T_gt_G", arrays["t_gold"] > arrays["g_gold"]),
        ("gold_probability_order", "G_gt_T", arrays["g_gold"] > arrays["t_gold"]),
        ("branch_top1_pattern", "T_correct_G_wrong", arrays["t_correct"] & ~arrays["g_correct"]),
        ("branch_top1_pattern", "G_correct_T_wrong", arrays["g_correct"] & ~arrays["t_correct"]),
        ("branch_top1_pattern", "both_correct", arrays["t_correct"] & arrays["g_correct"]),
        ("branch_top1_pattern", "both_wrong", ~arrays["t_correct"] & ~arrays["g_correct"]),
    ]
    for group_type, group_value, mask in group_masks:
        group_rows.append(summarize_group(args.student_seed, group_type, group_value, mask, arrays))
    difficulty_rows = [
        summarize_group(args.student_seed, "student_loss_quintile", f"Q{quintile}", quintiles == quintile, arrays)
        for quintile in range(1, 6)
    ]

    payload = {
        "manifest": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_only": True,
            "uses_test_data": False,
            "teacher_run_dir": str(teacher_dir.resolve()),
            "teacher_checkpoint": context["hybrid_ckpt"],
            "teacher_checkpoint_sha256": teacher_hash,
            "student_seed": args.student_seed,
            "student_run_dir": student_info["student_init_run_dir"],
            "student_checkpoint": student_info["student_init_checkpoint"],
            "student_checkpoint_sha256": student_info["student_init_checkpoint_sha256"],
            "selection_seed": args.selection_seed,
            "selected_indices_sha256": selection_hash,
            "conditions": CONDITIONS,
            "student_loss_quintile_edges": [float(value) for value in edges],
            "dataset_stats": dataset.stats,
            "raw_text_saved": False,
            "raw_logits_saved": False,
        },
        "overall": overall_rows,
        "branch_groups": group_rows,
        "difficulty_groups": difficulty_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "overall": overall_rows}, indent=2))


if __name__ == "__main__":
    main()
