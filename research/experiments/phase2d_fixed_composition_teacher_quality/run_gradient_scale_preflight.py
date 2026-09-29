#!/usr/bin/env python3
"""Measure Phase 2D KD scale on a frozen deterministic train prefix."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-label", choices=["J", "A"], required=True)
    parser.add_argument("--teacher-run-dir", required=True)
    parser.add_argument("--student-seed", type=int, required=True)
    parser.add_argument("--student-run-dir", required=True)
    parser.add_argument("--expected-student-sha256", required=True)
    parser.add_argument("--dataset-path", default="/workspace/grassmannflows/datasets/wikitext2_v1_saved")
    parser.add_argument("--tokenizer-dir", default="./gpt2_local")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--calibration-batches", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--kd-lambda", type=float, default=5.0)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def mean(values):
    return sum(values) / len(values)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import GPT2Tokenizer
    import train_distill_hybrid_lite_from_latefusion_teacher_v2 as train_module

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required on server 197")
    torch.manual_seed(args.student_seed)
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
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]

    dataset = train_module.TextDataset(
        "train", tokenizer, int(context["max_seq_len"]),
        dataset_name="wikitext2", dataset_path=str(dataset_path), text_field="text",
        max_lines=0, encode_chars_per_batch=200000, tinystories_val_frac=0.02, split_seed=42,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
        pin_memory=True, drop_last=False,
    )

    raw_kl = []
    weighted_kl = []
    ce_logit_norm = []
    kd_logit_norm = []
    cosine_sum = 0.0
    negative_cosine_count = 0
    valid_token_count = 0
    parameter_kd_norm = []
    for batch_index, (x, y) in enumerate(loader):
        if batch_index >= args.calibration_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.no_grad():
            teacher_logits, _ = teacher(x)
        student.zero_grad(set_to_none=True)
        student_logits, _ = student(x)
        kd = train_module.kd_loss(
            student_logits, teacher_logits, y,
            temperature=args.temperature, loss_mode="token_mean",
            kd_lambda=args.kd_lambda, chunk_tokens=1024,
        )
        raw_kl.append(float(kd.kl_token_mean.detach()))
        weighted_kl.append(float((args.kd_lambda * kd.kl_token_mean).detach()))

        shifted_student = student_logits[:, :-1, :].detach().float().reshape(-1, student_logits.size(-1))
        shifted_teacher = teacher_logits[:, :-1, :].detach().float().reshape(-1, teacher_logits.size(-1))
        labels = y[:, 1:].reshape(-1)
        valid = labels.ne(-100)
        shifted_student = shifted_student[valid]
        shifted_teacher = shifted_teacher[valid]
        labels = labels[valid]
        for start in range(0, labels.numel(), 32):
            end = min(start + 32, labels.numel())
            ss = shifted_student[start:end]
            tt = shifted_teacher[start:end]
            yy = labels[start:end]
            row = torch.arange(yy.numel(), device=device)
            p_s = F.softmax(ss, dim=-1)
            p_s_temp = F.softmax(ss / args.temperature, dim=-1)
            p_t_temp = F.softmax(tt / args.temperature, dim=-1)
            g_ce = p_s
            g_ce[row, yy] -= 1.0
            g_kd = args.temperature * (p_s_temp - p_t_temp)
            ce_norm = g_ce.square().sum(-1).sqrt()
            kd_norm = g_kd.square().sum(-1).sqrt()
            cosine = (g_ce * g_kd).sum(-1) / (ce_norm.clamp_min(1e-30) * kd_norm.clamp_min(1e-30))
            ce_logit_norm.extend(ce_norm.cpu().tolist())
            kd_logit_norm.extend(kd_norm.cpu().tolist())
            cosine_sum += float(cosine.sum())
            negative_cosine_count += int((cosine < 0).sum())
            valid_token_count += int(cosine.numel())

        gradients = torch.autograd.grad(
            kd.kl_token_mean, parameters, retain_graph=False, create_graph=False, allow_unused=True,
        )
        squared = torch.zeros((), device=device, dtype=torch.float64)
        for gradient in gradients:
            if gradient is not None:
                squared += gradient.detach().double().square().sum()
        parameter_kd_norm.append(float(squared.sqrt().cpu()))
        print(
            f"teacher={args.teacher_label} seed={args.student_seed} "
            f"batch={batch_index + 1}/{args.calibration_batches} "
            f"kl={raw_kl[-1]:.6f} param_grad={parameter_kd_norm[-1]:.6f}",
            flush=True,
        )
        del student_logits, teacher_logits, kd, gradients

    if len(raw_kl) != args.calibration_batches:
        raise RuntimeError(f"Expected {args.calibration_batches} calibration batches, found {len(raw_kl)}")
    row = {
        "teacher": args.teacher_label,
        "student_seed": args.student_seed,
        "effective_alpha": effective_alpha[0],
        "calibration_batches": args.calibration_batches,
        "batch_size": args.batch_size,
        "valid_tokens": valid_token_count,
        "unweighted_token_mean_kl": mean(raw_kl),
        "weighted_kl_contribution_lambda5": mean(weighted_kl),
        "kd_logit_gradient_norm": mean(kd_logit_norm),
        "ce_logit_gradient_norm": mean(ce_logit_norm),
        "ce_kd_cosine": cosine_sum / valid_token_count,
        "full_parameter_kd_gradient_norm": mean(parameter_kd_norm),
        "negative_cosine_fraction": negative_cosine_count / valid_token_count,
    }
    payload = {
        "manifest": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_only": True,
            "uses_test_data": False,
            "data_order": "first non-shuffled training batches",
            "teacher_run_dir": str(teacher_dir.resolve()),
            "teacher_checkpoint": context["hybrid_ckpt"],
            "checkpoint_alpha": checkpoint_alpha,
            "effective_alpha": effective_alpha,
            "student_run_dir": student_info["student_init_run_dir"],
            "student_checkpoint": student_info["student_init_checkpoint"],
            "student_checkpoint_sha256": student_info["student_init_checkpoint_sha256"],
            "dataset_stats": dataset.stats,
        },
        "metrics": row,
        "per_batch": {
            "unweighted_token_mean_kl": raw_kl,
            "weighted_kl_contribution": weighted_kl,
            "full_parameter_kd_gradient_norm": parameter_kd_norm,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "metrics": row}, indent=2))


if __name__ == "__main__":
    main()
