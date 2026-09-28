#!/usr/bin/env python3
"""Offline token-level teacher utility audit for a frozen student checkpoint."""

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

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-seed", type=int, required=True)
    parser.add_argument("--student-state", default="S0")
    parser.add_argument("--student-run-dir", required=True)
    parser.add_argument("--teacher-run-dir", required=True)
    parser.add_argument("--dataset-name", default="wikitext2")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokenizer-dir", default="./gpt2_local")
    parser.add_argument("--selection-seed", type=int, default=20260920)
    parser.add_argument("--max-chunks", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--token-block", type=int, default=32)
    parser.add_argument("--alphas", default="0.5,0.3,0.0")
    parser.add_argument("--model-dim", type=int, default=224)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--reduced-dim", type=int, default=56)
    parser.add_argument("--window-sizes", default="1,2,4")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--student-late-k", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--gpu-id", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_utility(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0, "mean": math.nan, "median": math.nan,
            "fraction_positive": math.nan, "positive_mean": math.nan,
            "negative_mean": math.nan, "positive_mass": 0.0,
            "negative_mass_signed": 0.0, "negative_mass_abs": 0.0,
        }
    positive = values[values > 0]
    negative = values[values < 0]
    result = {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "fraction_positive": float((values > 0).mean()),
        "positive_mean": float(positive.mean()) if positive.size else math.nan,
        "negative_mean": float(negative.mean()) if negative.size else math.nan,
        "positive_mass": float(positive.sum()) if positive.size else 0.0,
        "negative_mass_signed": float(negative.sum()) if negative.size else 0.0,
        "negative_mass_abs": float(-negative.sum()) if negative.size else 0.0,
    }
    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        result[f"q{int(q * 100):02d}"] = float(np.quantile(values, q))
    return result


def group_record(seed, alpha, group_type, group_value, utility, teacher_correct, student_correct):
    utility = np.asarray(utility, dtype=np.float64)
    teacher_correct = np.asarray(teacher_correct, dtype=bool)
    student_correct = np.asarray(student_correct, dtype=bool)
    summary = summarize_utility(utility)
    record = {
        "student_seed": int(seed),
        "alpha": float(alpha),
        "group_type": group_type,
        "group_value": group_value,
        "count": summary["count"],
        "fraction_teacher_higher_gold": summary["fraction_positive"],
        "mean_teacher_gold_advantage": summary["mean"],
        "positive_utility_mass": summary["positive_mass"],
        "negative_utility_mass_abs": summary["negative_mass_abs"],
        "teacher_accuracy": float(teacher_correct.mean()) if utility.size else math.nan,
        "student_accuracy": float(student_correct.mean()) if utility.size else math.nan,
        "rescue_rate": math.nan,
        "harm_rate": math.nan,
    }
    wrong = ~student_correct
    correct = student_correct
    if wrong.any():
        record["rescue_rate"] = float(teacher_correct[wrong].mean())
    if correct.any():
        record["harm_rate"] = float((~teacher_correct[correct]).mean())
    return record


def main():
    args = parse_args()
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.offline:
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
        raise RuntimeError("CUDA is required; run this evaluation on server 197")
    alphas = [float(value.strip()) for value in args.alphas.split(",") if value.strip()]
    if not alphas or any(alpha < 0 or alpha > 1 for alpha in alphas):
        raise ValueError("alphas must be in [0, 1]")
    if args.temperature <= 0 or args.token_block <= 0:
        raise ValueError("temperature and token-block must be positive")

    device = torch.device("cuda", 0)
    search_roots = train_module.build_search_roots([])
    teacher_dir = Path(train_module.resolve_migrated_path(
        args.teacher_run_dir, search_roots=search_roots, remaps=[],
        kind="teacher_run_dir", must_exist=True,
    ))
    student_dir = Path(train_module.resolve_migrated_path(
        args.student_run_dir, search_roots=search_roots, remaps=[],
        kind="student_run_dir", must_exist=True,
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
        teacher_dir, len(tokenizer), search_roots, [], device
    )
    max_seq_len = int(context["max_seq_len"])
    student = train_module.instantiate_student_from_args(
        "hybrid_lite", len(tokenizer), max_seq_len, args
    ).to(device)
    student_info = train_module.maybe_load_student_init(
        student, str(student_dir), search_roots, [], device
    )
    student.eval()
    teacher.eval()

    dataset = train_module.TextDataset(
        "validation", tokenizer, max_seq_len,
        dataset_name=args.dataset_name, dataset_path=str(dataset_path),
        text_field=args.text_field, max_lines=0,
        encode_chars_per_batch=200000, tinystories_val_frac=0.02,
        split_seed=args.split_seed,
    )
    if args.max_chunks <= 0 or args.max_chunks >= len(dataset):
        indices = list(range(len(dataset)))
        selection_mode = "full_validation"
    else:
        rng = random.Random(args.selection_seed)
        indices = sorted(rng.sample(range(len(dataset)), args.max_chunks))
        selection_mode = "fixed_random_subset"
    loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    student_nll_parts = []
    student_correct_parts = []
    by_alpha = {
        alpha: defaultdict(list) for alpha in alphas
    }
    eps = torch.finfo(torch.float32).tiny
    seen_chunks = 0

    with torch.inference_mode():
        for batch_index, (x, y) in enumerate(loader, start=1):
            seen_chunks += int(x.size(0))
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
                ss = student_logits[start:end].float()
                tt = t_logits[start:end].float()
                gg = g_logits[start:end].float()
                log_s = F.log_softmax(ss, dim=-1)
                p_s = log_s.exp()
                p_s_temp = F.softmax(ss / args.temperature, dim=-1)
                log_s_temp = F.log_softmax(ss / args.temperature, dim=-1)
                s_nll = -log_s[row, yy]
                s_correct = ss.argmax(dim=-1).eq(yy)
                student_nll_parts.append(s_nll.cpu().numpy())
                student_correct_parts.append(s_correct.cpu().numpy())

                g_ce = p_s.clone()
                g_ce[row, yy] -= 1.0
                ce_norm = g_ce.square().sum(dim=-1).sqrt()
                for alpha in alphas:
                    fused = alpha * tt + (1.0 - alpha) * gg
                    log_t = F.log_softmax(fused, dim=-1)
                    p_t = log_t.exp()
                    t_nll = -log_t[row, yy]
                    utility = s_nll - t_nll
                    t_correct = fused.argmax(dim=-1).eq(yy)

                    p_t_temp = F.softmax(fused / args.temperature, dim=-1)
                    log_t_temp = F.log_softmax(fused / args.temperature, dim=-1)
                    kl_t1 = (p_t * (log_t - log_s)).sum(dim=-1).clamp_min(0.0)
                    kl_t2 = (p_t_temp * (log_t_temp - log_s_temp)).sum(dim=-1).clamp_min(0.0)
                    g_kd = args.temperature * (p_s_temp - p_t_temp)
                    kd_norm = g_kd.square().sum(dim=-1).sqrt()
                    cosine = (g_ce * g_kd).sum(dim=-1) / (
                        ce_norm.clamp_min(eps) * kd_norm.clamp_min(eps)
                    )
                    store = by_alpha[alpha]
                    for key, value in {
                        "utility": utility,
                        "teacher_nll": t_nll,
                        "teacher_correct": t_correct,
                        "kl_t1": kl_t1,
                        "kl_t2": kl_t2,
                        "ce_grad_norm": ce_norm,
                        "kd_grad_norm": kd_norm,
                        "grad_cosine": cosine.clamp(-1.0, 1.0),
                    }.items():
                        store[key].append(value.cpu().numpy())

            print(
                f"seed={args.student_seed} state={args.student_state} "
                f"batch={batch_index}/{len(loader)} chunks={seen_chunks}/{len(indices)}",
                flush=True,
            )
            del branches, student_logits, t_logits, g_logits

    student_nll = np.concatenate(student_nll_parts).astype(np.float64)
    student_correct = np.concatenate(student_correct_parts).astype(bool)
    quintile_edges = np.quantile(student_nll, [0.2, 0.4, 0.6, 0.8])
    quintile_id = np.searchsorted(quintile_edges, student_nll, side="right") + 1
    overall_rows = []
    group_rows = []
    for alpha in alphas:
        arrays = {key: np.concatenate(parts) for key, parts in by_alpha[alpha].items()}
        utility = arrays["utility"].astype(np.float64)
        teacher_correct = arrays["teacher_correct"].astype(bool)
        summary = summarize_utility(utility)
        wrong = ~student_correct
        correct = student_correct
        overall_rows.append({
            "student_seed": args.student_seed,
            "student_state": args.student_state,
            "alpha": alpha,
            "chunks": len(indices),
            "valid_tokens": int(utility.size),
            "student_nll": float(student_nll.mean()),
            "student_top1_accuracy": float(student_correct.mean()),
            "teacher_nll": float(arrays["teacher_nll"].mean()),
            "delta_teacher": float(student_nll.mean() - arrays["teacher_nll"].mean()),
            "teacher_top1_accuracy": float(teacher_correct.mean()),
            "mean_utility": summary["mean"],
            "median_utility": summary["median"],
            "fraction_positive_utility": summary["fraction_positive"],
            "positive_utility_mean": summary["positive_mean"],
            "negative_utility_mean": summary["negative_mean"],
            "positive_utility_mass": summary["positive_mass"],
            "negative_utility_mass_signed": summary["negative_mass_signed"],
            "negative_utility_mass_abs": summary["negative_mass_abs"],
            "rescue_rate": float(teacher_correct[wrong].mean()),
            "harm_rate": float((~teacher_correct[correct]).mean()),
            "rescue_count": int((teacher_correct & wrong).sum()),
            "harm_count": int(((~teacher_correct) & correct).sum()),
            "teacher_student_kl_t1": float(arrays["kl_t1"].mean()),
            "teacher_student_kl_t2": float(arrays["kl_t2"].mean()),
            "ce_logit_grad_norm": float(arrays["ce_grad_norm"].mean()),
            "kd_logit_grad_norm": float(arrays["kd_grad_norm"].mean()),
            "ce_kd_cosine": float(arrays["grad_cosine"].mean()),
            "negative_cosine_fraction": float((arrays["grad_cosine"] < 0).mean()),
            **{key: value for key, value in summary.items() if key.startswith("q")},
        })
        for label, mask in [("correct", correct), ("incorrect", wrong)]:
            group_rows.append(group_record(
                args.student_seed, alpha, "student_top1", label,
                utility[mask], teacher_correct[mask], student_correct[mask],
            ))
        for quintile in range(1, 6):
            mask = quintile_id == quintile
            group_rows.append(group_record(
                args.student_seed, alpha, "student_loss_quintile", f"Q{quintile}",
                utility[mask], teacher_correct[mask], student_correct[mask],
            ))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True,
        "student_seed": args.student_seed,
        "student_state": args.student_state,
        "student_run_dir": str(student_dir.resolve()),
        "student_checkpoint": student_info["student_init_checkpoint"],
        "student_checkpoint_sha256": student_info["student_init_checkpoint_sha256"],
        "teacher_run_dir": str(teacher_dir.resolve()),
        "teacher_checkpoint": context["hybrid_ckpt"],
        "teacher_checkpoint_sha256": sha256_file(context["hybrid_ckpt"]),
        "selection_mode": selection_mode,
        "selection_seed": args.selection_seed,
        "selected_indices": indices,
        "selected_indices_sha256": hashlib.sha256(
            ",".join(map(str, indices)).encode("utf-8")
        ).hexdigest(),
        "dataset_stats": dataset.stats,
        "temperature": args.temperature,
        "alphas": alphas,
        "student_loss_quintile_edges": [float(value) for value in quintile_edges],
        "fusion_convention": "alpha * transformer_logits + (1-alpha) * grassmann_logits",
        "raw_text_saved": False,
        "raw_logits_saved": False,
        "arguments": vars(args),
    }
    payload = {"manifest": manifest, "overall": overall_rows, "groups": group_rows}
    (output_dir / "utility.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "output": str((output_dir / "utility.json").resolve()),
        "student_seed": args.student_seed,
        "student_state": args.student_state,
        "chunks": len(indices),
        "valid_tokens": int(student_nll.size),
        "overall": overall_rows,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
