#!/usr/bin/env python3
"""Run full-parameter Phase 3A diagnostics without endpoint or test-data access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
PHASE_DIR = HERE.parent
PROJECT_ROOT = PHASE_DIR.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT))

from safe_lambda_core import candidate_safe_lambda, quadratic_delta, sign_label


CALIBRATION_SEEDS = (314159, 271828, 161803)
LAMBDA_GRID = (0.0, 0.5, 1.0, 2.5, 5.0, 10.0)
ETA = 1e-4
TEMPERATURE = 2.0
CLIP_NORM = 1.0
WEIGHT_DECAY = 0.01


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-ids", required=True, help="Comma-separated condition IDs")
    parser.add_argument("--inventory", default=str(PHASE_DIR / "condition_inventory.csv"))
    parser.add_argument("--tokenizer-dir", default="./gpt2_local")
    parser.add_argument("--output-dir", default=str(PHASE_DIR / "raw" / "diagnostics"))
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--train-probe-size", type=int, default=32)
    parser.add_argument("--train-microbatch-size", type=int, default=2)
    parser.add_argument("--calibration-examples", type=int, default=2)
    parser.add_argument("--calibration-seeds", default=",".join(str(x) for x in CALIBRATION_SEEDS))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_inventory(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["condition_id"]: row for row in csv.DictReader(handle)}


def zeros_like(parameters):
    import torch
    return [torch.zeros_like(parameter, memory_format=torch.preserve_format) for parameter in parameters]


def add_scaled_(destination, source, scale):
    for dst, src in zip(destination, source):
        if src is not None:
            dst.add_(src.detach(), alpha=scale)


def vector_dot(left, right):
    import torch
    total = torch.zeros((), device=left[0].device, dtype=torch.float64)
    for lhs, rhs in zip(left, right):
        if lhs is not None and rhs is not None:
            total = total + (lhs.reshape(-1).double() @ rhs.reshape(-1).double())
    return total


def vector_norm(vector):
    return vector_dot(vector, vector).clamp_min(0).sqrt()


def finite_vector(vector):
    import torch
    return all(item is None or bool(torch.isfinite(item).all()) for item in vector)


def choose_calibration_indices(length: int, seeds, examples_per_subset: int):
    used = set()
    subsets = {}
    for seed in seeds:
        candidates = list(range(length))
        random.Random(seed).shuffle(candidates)
        selected = [index for index in candidates if index not in used][:examples_per_subset]
        if len(selected) != examples_per_subset:
            raise RuntimeError("Not enough distinct validation chunks for calibration subsets")
        used.update(selected)
        subsets[str(seed)] = selected
    return subsets


def stack_examples(dataset, indices, device):
    import torch
    xs, ys = zip(*(dataset[index] for index in indices))
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def average_probe_gradients(student, teacher, dataset, indices, microbatch_size, parameters,
                            device, student_seed, train_module):
    import torch

    g_c = zeros_like(parameters)
    g_d = zeros_like(parameters)
    total_examples = 0
    ce_sum = 0.0
    kd_sum = 0.0
    student.train()
    teacher.eval()
    torch.manual_seed(900000 + int(student_seed))
    torch.cuda.manual_seed_all(900000 + int(student_seed))

    for start in range(0, len(indices), microbatch_size):
        batch_indices = indices[start:start + microbatch_size]
        x, y = stack_examples(dataset, batch_indices, device)
        with torch.no_grad():
            teacher_logits, _ = teacher(x, labels=None)
        student_logits, _ = student(x, labels=None)
        losses = train_module.kd_loss(
            student_logits, teacher_logits, y, temperature=TEMPERATURE,
            loss_mode="token_mean", kd_lambda=5.0, chunk_tokens=1024,
        )
        grad_c = torch.autograd.grad(
            losses.ce, parameters, retain_graph=True, create_graph=False, allow_unused=True,
        )
        grad_d = torch.autograd.grad(
            losses.kl_token_mean, parameters, create_graph=False, allow_unused=True,
        )
        weight = len(batch_indices)
        add_scaled_(g_c, grad_c, weight)
        add_scaled_(g_d, grad_d, weight)
        ce_sum += float(losses.ce.detach()) * weight
        kd_sum += float(losses.kl_token_mean.detach()) * weight
        total_examples += weight
        del x, y, teacher_logits, student_logits, losses, grad_c, grad_d

    if total_examples != len(indices):
        raise RuntimeError("Training-probe accumulation count mismatch")
    for item in g_c:
        item.div_(total_examples)
    for item in g_d:
        item.div_(total_examples)
    if not finite_vector(g_c) or not finite_vector(g_d):
        raise FloatingPointError("Non-finite training-probe gradient")
    norm_c = float(vector_norm(g_c).cpu())
    norm_d = float(vector_norm(g_d).cpu())
    dot_cd = float(vector_dot(g_c, g_d).cpu())
    cosine_cd = dot_cd / max(norm_c * norm_d, 1e-300)
    return {
        "g_c": g_c,
        "g_d": g_d,
        "train_ce": ce_sum / total_examples,
        "train_kl_token_mean": kd_sum / total_examples,
        "g_c_norm": norm_c,
        "g_d_norm": norm_d,
        "train_ce_kd_dot": dot_cd,
        "train_ce_kd_cosine": cosine_cd,
    }


def subset_second_order(student, teacher, dataset, indices, parameters, g_c, g_d, device):
    import torch

    student.eval()
    teacher.eval()
    g_v_sum = zeros_like(parameters)
    totals = {"student_nll": 0.0, "teacher_nll": 0.0, "a": 0.0, "b": 0.0, "c": 0.0}
    for index in indices:
        x, y = stack_examples(dataset, [index], device)
        with torch.no_grad():
            _, teacher_loss = teacher(x, labels=y)
        _, validation_loss = student(x, labels=y)
        g_v = torch.autograd.grad(
            validation_loss, parameters, create_graph=True, retain_graph=True, allow_unused=True,
        )
        g_v_dense = [torch.zeros_like(parameter) if grad is None else grad for parameter, grad in zip(parameters, g_v)]
        hvp_scalar = sum((grad_v * grad_d).sum() for grad_v, grad_d in zip(g_v_dense, g_d))
        hvp_d = torch.autograd.grad(
            hvp_scalar, parameters, create_graph=False, retain_graph=False, allow_unused=True,
        )
        hvp_d_dense = [torch.zeros_like(parameter) if hv is None else hv for parameter, hv in zip(parameters, hvp_d)]
        if not finite_vector(g_v_dense) or not finite_vector(hvp_d_dense):
            raise FloatingPointError(f"Non-finite validation gradient/HVP at index {index}")
        add_scaled_(g_v_sum, g_v_dense, 1.0)
        totals["student_nll"] += float(validation_loss.detach())
        totals["teacher_nll"] += float(teacher_loss.detach())
        totals["a"] += float(vector_dot(g_v_dense, g_d).cpu())
        totals["b"] += float(vector_dot(g_c, hvp_d_dense).cpu())
        totals["c"] += float(vector_dot(g_d, hvp_d_dense).cpu())
        del x, y, teacher_loss, validation_loss, g_v, g_v_dense, hvp_scalar, hvp_d, hvp_d_dense

    count = len(indices)
    for item in g_v_sum:
        item.div_(count)
    for key in totals:
        totals[key] /= count
    norm_v = float(vector_norm(g_v_sum).cpu())
    norm_d = float(vector_norm(g_d).cpu())
    dot_vd = float(vector_dot(g_v_sum, g_d).cpu())
    cosine_vd = dot_vd / max(norm_v * norm_d, 1e-300)
    safe = candidate_safe_lambda(eta=ETA, a=totals["a"], b=totals["b"], c=totals["c"])
    delta_quad = quadratic_delta(
        eta=ETA, lam=5.0, a=totals["a"], b=totals["b"], c=totals["c"],
    )
    return {
        "student_validation_nll": totals["student_nll"],
        "teacher_validation_nll": totals["teacher_nll"],
        "teacher_residual_advantage": totals["student_nll"] - totals["teacher_nll"],
        "validation_gradient_norm": norm_v,
        "validation_kd_dot_a": totals["a"],
        "validation_kd_cosine": cosine_vd,
        "mixed_curvature_b": totals["b"],
        "kd_curvature_c": totals["c"],
        "predicted_delta_v_quad_lambda5": delta_quad,
        "predicted_gain_quad_lambda5": -delta_quad,
        "predicted_quad_sign_lambda5": sign_label(-delta_quad),
        "lambda_safe": safe.value,
        "lambda_safe_status": safe.status,
    }


def evaluation_losses(student, dataset, calibration_subsets, device):
    student.eval()
    values = {}
    for seed, indices in calibration_subsets.items():
        total = 0.0
        for index in indices:
            x, y = stack_examples(dataset, [index], device)
            with __import__("torch").no_grad():
                _, loss = student(x, labels=y)
            total += float(loss)
        values[seed] = total / len(indices)
    return values


def virtual_curves(student, dataset, calibration_subsets, parameters, g_c, g_d, device,
                   total_training_steps, warmup_steps):
    import torch

    base_state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
    curves = {"literal_scheduler_step": [], "nominal_lr_sensitivity": []}
    for mode in curves:
        for lam in LAMBDA_GRID:
            student.load_state_dict(base_state, strict=True)
            optimizer = torch.optim.AdamW(student.parameters(), lr=ETA, weight_decay=WEIGHT_DECAY)
            scheduler = None
            if mode == "literal_scheduler_step":
                def lr_lambda(current_step):
                    if current_step < warmup_steps:
                        return float(current_step) / max(1, warmup_steps)
                    progress = float(current_step - warmup_steps) / max(1, total_training_steps - warmup_steps)
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            optimizer.zero_grad(set_to_none=True)
            for parameter, grad_c, grad_d in zip(parameters, g_c, g_d):
                parameter.grad = (grad_c + lam * grad_d).detach().clone()
            preclip_norm = float(torch.nn.utils.clip_grad_norm_(parameters, CLIP_NORM))
            lr_used = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses = evaluation_losses(student, dataset, calibration_subsets, device)
            curves[mode].append({
                "lambda": lam,
                "lr_used": lr_used,
                "preclip_grad_norm": preclip_norm,
                "clipped": bool(preclip_norm > CLIP_NORM),
                "calibration_nll": losses,
            })
            del optimizer, scheduler
    student.load_state_dict(base_state, strict=True)
    return curves


def run_condition(row, tokenizer, output_dir: Path, args, train_module, device, dataset_cache):
    import torch

    condition_id = row["condition_id"]
    output_path = output_dir / f"{condition_id}.json"
    if args.skip_existing and output_path.exists():
        print(f"[skip] {output_path}", flush=True)
        return

    search_roots = train_module.build_search_roots([])
    teacher_dir = Path(train_module.resolve_migrated_path(
        row["teacher_run_dir"], search_roots=search_roots, remaps=[],
        kind="teacher_run_dir", must_exist=True,
    ))
    teacher, context = train_module.load_teacher_and_context(
        teacher_dir, len(tokenizer), search_roots, [], device, teacher_type=row["teacher_type"],
    )
    checkpoint_alpha, effective_alpha = train_module.apply_teacher_alpha_override(
        teacher, float(row["teacher_alpha_override"]),
    )
    teacher_checkpoint = Path(context["hybrid_ckpt"])
    teacher_hash_before = sha256_file(teacher_checkpoint)

    model_args = SimpleNamespace(
        model_dim=224, num_layers=6, num_heads=8, reduced_dim=56,
        window_sizes="1,2,4", dropout=0.1, student_late_k=1,
    )
    student = train_module.instantiate_student_from_args(
        "hybrid_lite", len(tokenizer), int(context["max_seq_len"]), model_args,
    ).to(device)
    student_info = train_module.maybe_load_student_init(
        student, row["student_run_dir"], search_roots, [], device,
        expected_sha256=row["expected_student_sha256"],
    )
    student_checkpoint = Path(student_info["student_init_checkpoint"])
    student_hash_before = sha256_file(student_checkpoint)
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]

    dataset_key = (
        row["dataset"], row["dataset_path"], row["text_field"], int(row["max_lines"]),
        int(row["split_seed"]), int(context["max_seq_len"]),
    )
    if dataset_key not in dataset_cache:
        common = dict(
            tokenizer=tokenizer, max_seq_len=int(context["max_seq_len"]),
            dataset_name=row["dataset"], dataset_path=row["dataset_path"],
            text_field=row["text_field"], max_lines=int(row["max_lines"]),
            encode_chars_per_batch=200000, tinystories_val_frac=0.02,
            split_seed=int(row["split_seed"]),
        )
        dataset_cache[dataset_key] = (
            train_module.TextDataset("train", **common),
            train_module.TextDataset("validation", **common),
        )
    train_dataset, validation_dataset = dataset_cache[dataset_key]

    seeds = [int(value) for value in args.calibration_seeds.split(",") if value]
    calibration_subsets = choose_calibration_indices(
        len(validation_dataset), seeds, args.calibration_examples,
    )
    train_indices = list(range(args.train_probe_size))
    probe = average_probe_gradients(
        student, teacher, train_dataset, train_indices, args.train_microbatch_size,
        parameters, device, int(row["student_seed"]), train_module,
    )
    subset_results = {}
    for seed, indices in calibration_subsets.items():
        print(f"[{condition_id}] HVP calibration_seed={seed} indices={indices}", flush=True)
        subset_results[seed] = {
            "indices": indices,
            **subset_second_order(
                student, teacher, validation_dataset, indices, parameters,
                probe["g_c"], probe["g_d"], device,
            ),
        }

    steps_per_epoch = math.ceil(len(train_dataset) / 32)
    total_training_steps = steps_per_epoch * 10
    warmup_steps = int(total_training_steps * 0.05)
    curves = virtual_curves(
        student, validation_dataset, calibration_subsets, parameters,
        probe["g_c"], probe["g_d"], device, total_training_steps, warmup_steps,
    )

    student_hash_after = sha256_file(student_checkpoint)
    teacher_hash_after = sha256_file(teacher_checkpoint)
    if student_hash_after != student_hash_before or teacher_hash_after != teacher_hash_before:
        raise RuntimeError("A source checkpoint hash changed during an offline diagnostic")
    literal_lrs = {point["lr_used"] for point in curves["literal_scheduler_step"]}
    if literal_lrs != {0.0}:
        raise RuntimeError(f"Literal first-step LR audit changed unexpectedly: {literal_lrs}")

    payload = {
        "manifest": {
            "created_at_utc": utc_now(),
            "phase": "Phase 3A authorized post-freeze offline diagnostic",
            "condition_id": condition_id,
            "uses_test_data": False,
            "performs_persistent_training": False,
            "saves_virtual_states": False,
            "fp32": True,
            "amp": False,
            "full_parameter_gradients": True,
            "student_parameter_count": sum(parameter.numel() for parameter in parameters),
            "teacher_type": row["teacher_type"],
            "teacher_run_dir": str(teacher_dir.resolve()),
            "teacher_checkpoint": str(teacher_checkpoint.resolve()),
            "teacher_checkpoint_sha256_before": teacher_hash_before,
            "teacher_checkpoint_sha256_after": teacher_hash_after,
            "teacher_checkpoint_alpha": checkpoint_alpha,
            "teacher_effective_alpha": effective_alpha,
            "student_run_dir": student_info["student_init_run_dir"],
            "student_checkpoint": str(student_checkpoint.resolve()),
            "student_checkpoint_sha256_before": student_hash_before,
            "student_checkpoint_sha256_after": student_hash_after,
            "dataset": row["dataset"],
            "train_dataset_stats": train_dataset.stats,
            "validation_dataset_stats": validation_dataset.stats,
            "train_probe_indices": train_indices,
            "train_probe_size": args.train_probe_size,
            "train_microbatch_size": args.train_microbatch_size,
            "calibration_seeds": seeds,
            "calibration_examples_per_subset": args.calibration_examples,
            "eta": ETA,
            "temperature": TEMPERATURE,
            "clip_norm": CLIP_NORM,
            "weight_decay": WEIGHT_DECAY,
            "lambda_grid": LAMBDA_GRID,
            "formal_total_training_steps": total_training_steps,
            "formal_warmup_steps": warmup_steps,
            "literal_first_step_lr": 0.0,
        },
        "condition": {
            "condition_id": condition_id,
            "dataset": row["dataset"],
            "family": row["family"],
            "student_seed": int(row["student_seed"]),
        },
        "training_probe": {key: value for key, value in probe.items() if key not in {"g_c", "g_d"}},
        "calibration_subsets": subset_results,
        "virtual_step_curves": curves,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[complete] {output_path}", flush=True)
    del teacher, student, parameters, probe
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    if args.smoke:
        args.train_probe_size = min(args.train_probe_size, 2)
        args.train_microbatch_size = 1
        args.calibration_examples = 1
        args.calibration_seeds = str(CALIBRATION_SEEDS[0])
    if args.train_probe_size <= 0 or args.train_probe_size % args.train_microbatch_size:
        raise ValueError("train-probe-size must be positive and divisible by train-microbatch-size")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from transformers import GPT2Tokenizer
    import train_distill_hybrid_lite_from_latefusion_teacher_v2 as train_module

    if not torch.cuda.is_available():
        raise RuntimeError("Phase 3A model diagnostics must run on server 197 with CUDA")
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda", 0)
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    inventory = read_inventory(Path(args.inventory))
    requested = [value for value in args.condition_ids.split(",") if value]
    missing = sorted(set(requested) - set(inventory))
    if missing:
        raise KeyError(f"Unknown condition IDs: {missing}")
    dataset_cache = {}
    for condition_id in requested:
        run_condition(
            inventory[condition_id], tokenizer, Path(args.output_dir), args,
            train_module, device, dataset_cache,
        )


if __name__ == "__main__":
    main()
