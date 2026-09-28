#!/usr/bin/env python3
"""Validate and collect the three-seed Phase 2C replication."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
HYBRID = ROOT / "outputs" / "hybrid_experiments"
DISTILL = ROOT / "outputs" / "distill_experiments"
TEACHER_RUN = ROOT / "outputs" / "hybrid_experiments" / "20260328_080104_wt2_hybrid_alpha_joint"
SEEDS = [42, 123, 456]

S0_NAMES = {
    42: "20260515_004119_wt2_hybrid_lite_baseline_224x56_l6_e20",
    123: "phase2c_wt2_s0_seed123",
    456: "phase2c_wt2_s0_seed456",
}
ARM_NAMES = {
    42: {
        "C0": "phase2b_wt2_c0_ws_ce_a05",
        "C1": "phase2b_wt2_c1_good_a05",
        "C3": "phase2b_wt2_c3_bad_a00",
    },
    123: {
        "C0": "phase2c_wt2_c0_seed123",
        "C1": "phase2c_wt2_c1_a05_seed123",
        "C3": "phase2c_wt2_c3_a00_seed123",
    },
    456: {
        "C0": "phase2c_wt2_c0_seed456",
        "C1": "phase2c_wt2_c1_a05_seed456",
        "C3": "phase2c_wt2_c3_a00_seed456",
    },
}
ARM_SPEC = {
    "C0": {"condition": "WS+CE", "alpha": 0.5, "lambda": 0.0},
    "C1": {"condition": "T_a05", "alpha": 0.5, "lambda": 5.0},
    "C3": {"condition": "T_a00", "alpha": 0.0, "lambda": 5.0},
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_complete(root, name):
    exact = root / name
    candidates = [exact] if exact.is_dir() else sorted(root.glob(f"*_{name}"))
    complete = [path for path in candidates if (path / "config.json").exists() and (path / "summary.json").exists()]
    if len(complete) != 1:
        raise RuntimeError(f"expected one complete run for {name}, found {complete}")
    return complete[0]


def classify_residual(value):
    if value >= 0.05:
        return "good"
    if abs(value) <= 0.025:
        return "near"
    if value <= -0.05:
        return "bad"
    return "intermediate"


def sample_stats(values):
    values = [float(value) for value in values]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return {
        "mean": mean,
        "sample_std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
        "all_positive": all(value > 0 for value in values),
        "values": values,
    }


def main():
    raw_s0 = HERE / "raw" / "s0"
    raw_formal = HERE / "raw" / "formal"
    raw_s0.mkdir(parents=True, exist_ok=True)
    raw_formal.mkdir(parents=True, exist_ok=True)

    s0 = {}
    for seed in SEEDS:
        run_dir = find_complete(HYBRID, S0_NAMES[seed])
        summary = load_json(run_dir / "summary.json")["hybrid"]
        config = load_json(run_dir / "config.json")["config"]
        checkpoint = run_dir / "checkpoints" / "hybrid_best.pt"
        checkpoint_hash = sha256(checkpoint)
        if int(config["seed"]) != seed:
            raise RuntimeError(f"S0 seed mismatch for {seed}")
        for key, expected in {
            "batch_size": 32, "epochs": 20, "lr": 0.0002,
            "weight_decay": 0.01, "model_dim": 224, "num_layers": 6,
            "reduced_dim": 56, "window_sizes": "1,2,4", "dropout": 0.1,
            "late_k": 1, "max_seq_len": 256, "max_lines": 0,
            "split_seed": 42,
        }.items():
            if config.get(key) != expected:
                raise RuntimeError(f"S0 config mismatch seed={seed} key={key}: {config.get(key)}")
        destination = raw_s0 / f"seed{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ["config.json", "summary.json", "hybrid_metrics.jsonl", "report.md"]:
            shutil.copy2(run_dir / name, destination / name)
        s0[seed] = {
            "run_dir": run_dir,
            "config": config,
            "summary": summary,
            "hash": checkpoint_hash,
        }
    if len({entry["hash"] for entry in s0.values()}) != 3:
        raise RuntimeError("S0 checkpoints are not three distinct parameter files")

    utility_s0 = {}
    utility_c0 = {}
    selection_hashes = set()
    for seed in SEEDS:
        s0_payload = load_json(HERE / "raw" / f"utility_s0_seed{seed}" / "utility.json")
        c0_payload = load_json(HERE / "raw" / f"utility_c0_seed{seed}" / "utility.json")
        selection_hashes.add(s0_payload["manifest"]["selected_indices_sha256"])
        selection_hashes.add(c0_payload["manifest"]["selected_indices_sha256"])
        if s0_payload["manifest"]["student_checkpoint_sha256"] != s0[seed]["hash"]:
            raise RuntimeError(f"utility S0 hash mismatch for seed {seed}")
        utility_s0[seed] = {float(row["alpha"]): row for row in s0_payload["overall"]}
        utility_c0[seed] = {float(row["alpha"]): row for row in c0_payload["overall"]}
    if len(selection_hashes) != 1:
        raise RuntimeError(f"utility selection mismatch: {selection_hashes}")

    invariant_keys = [
        "batch_size", "epochs", "lr", "weight_decay", "warmup_ratio",
        "num_workers", "amp", "model_dim", "num_layers", "num_heads",
        "reduced_dim", "window_sizes", "dropout", "student_late_k",
        "temperature", "kd_loss_mode", "kd_chunk_tokens", "distill_strategy",
        "dataset_name", "dataset_path", "text_field", "max_seq_len", "max_lines",
        "encode_chars_per_batch", "tinystories_val_frac", "split_seed", "offline",
    ]
    invariant_reference = None
    runs = {}
    for seed in SEEDS:
        for arm, name in ARM_NAMES[seed].items():
            run_dir = find_complete(DISTILL, name)
            config_payload = load_json(run_dir / "config.json")
            summary = load_json(run_dir / "summary.json")["student"]
            config = config_payload["config"]
            spec = ARM_SPEC[arm]
            if int(config["seed"]) != seed:
                raise RuntimeError(f"continuation seed mismatch {seed}/{arm}")
            if config_payload["student_init"]["student_init_checkpoint_sha256"] != s0[seed]["hash"]:
                raise RuntimeError(f"S0 hash mismatch {seed}/{arm}")
            if summary["student_init_checkpoint_sha256"] != s0[seed]["hash"]:
                raise RuntimeError(f"summary S0 hash mismatch {seed}/{arm}")
            effective_alpha = float(config_payload["source_teacher"]["effective_alpha"][0])
            if abs(effective_alpha - spec["alpha"]) > 1e-6:
                raise RuntimeError(f"alpha mismatch {seed}/{arm}: {effective_alpha}")
            if abs(float(config["kd_lambda"]) - spec["lambda"]) > 1e-12:
                raise RuntimeError(f"lambda mismatch {seed}/{arm}")
            invariant = {key: config.get(key) for key in invariant_keys}
            if invariant_reference is None:
                invariant_reference = invariant
            elif invariant != invariant_reference:
                diff = {key: (invariant_reference[key], invariant[key]) for key in invariant_keys if invariant_reference[key] != invariant[key]}
                raise RuntimeError(f"matched-config violation {seed}/{arm}: {diff}")
            destination = raw_formal / f"seed{seed}" / arm
            destination.mkdir(parents=True, exist_ok=True)
            for file_name in ["config.json", "summary.json", "distill_metrics.jsonl", "report.md"]:
                shutil.copy2(run_dir / file_name, destination / file_name)
            runs[(seed, arm)] = {
                "run_dir": run_dir, "config": config, "summary": summary,
                "checkpoint_hash": sha256(run_dir / "checkpoints" / "student_best.pt"),
            }

    rows = []
    seed_metrics = {}
    for seed in SEEDS:
        c0_test = float(runs[(seed, "C0")]["summary"]["test_loss"])
        seed_rows = {}
        for arm in ["C0", "C1", "C3"]:
            spec = ARM_SPEC[arm]
            run = runs[(seed, arm)]
            summary = run["summary"]
            initial = utility_s0[seed][spec["alpha"]]
            residual = utility_c0[seed][spec["alpha"]]
            clip = float(summary["train_grad_clip_fraction"][0])
            overflow = float(summary["train_amp_overflow_fraction"][0])
            nonfinite = float(summary["train_grad_nonfinite_fraction"][0])
            status = ["COMPLETE"]
            seed42_norm = 0.1154844733496622 if spec["alpha"] == 0.5 else 0.11893116931076328
            ratio = float(initial["kd_logit_grad_norm"]) / seed42_norm
            if spec["lambda"] > 0 and (clip >= 0.95 or overflow >= 0.05 or nonfinite >= 0.05 or ratio > 2.0 or ratio < 0.5):
                status.append("OPTIMIZATION_MISMATCH")
            row = {
                "Seed": seed,
                "Arm": arm,
                "Teacher condition": spec["condition"],
                "Alpha": spec["alpha"],
                "Lambda": spec["lambda"],
                "Student S0 hash": s0[seed]["hash"],
                "Teacher validation NLL": float(residual["teacher_nll"]),
                "C0 validation-subset NLL": float(residual["student_nll"]),
                "Delta_teacher": float(residual["delta_teacher"]),
                "Residual classification": classify_residual(float(residual["delta_teacher"])),
                "Best epoch": int(summary["best_epoch"]),
                "Val NLL": float(summary["best_val_loss"]),
                "Test NLL": float(summary["test_loss"]),
                "Test PPL": float(summary["test_ppl"]),
                "Delta_KD": c0_test - float(summary["test_loss"]),
                "D": "",
                "CE loss epoch1": float(summary["train_ce"][0]),
                "KD loss epoch1": float(summary["train_kl_token_mean"][0]),
                "Weighted KD epoch1": spec["lambda"] * float(summary["train_kl_token_mean"][0]),
                "Initial teacher-student KL T1": float(initial["teacher_student_kl_t1"]),
                "Initial teacher-student KL T2": float(initial["teacher_student_kl_t2"]),
                "Initial CE logit-grad norm": float(initial["ce_logit_grad_norm"]),
                "Initial KD logit-grad norm": float(initial["kd_logit_grad_norm"]),
                "Initial CE-KD cosine": float(initial["ce_kd_cosine"]),
                "Initial negative cosine fraction": float(initial["negative_cosine_fraction"]),
                "Total parameter-grad norm epoch1": float(summary["train_grad_norm_mean"][0]),
                "Clip fraction epoch1": clip,
                "AMP overflow fraction epoch1": overflow,
                "Nonfinite fraction epoch1": nonfinite,
                "Run directory": str(run["run_dir"].relative_to(ROOT)),
                "Student checkpoint SHA256": run["checkpoint_hash"],
                "Status": "|".join(status),
            }
            rows.append(row)
            seed_rows[arm] = row
        d_value = float(seed_rows["C3"]["Test NLL"]) - float(seed_rows["C1"]["Test NLL"])
        seed_rows["C1"]["D"] = d_value
        seed_rows["C3"]["D"] = d_value
        seed_metrics[seed] = {
            "Delta_KD_a05": float(seed_rows["C1"]["Delta_KD"]),
            "Delta_KD_a00": float(seed_rows["C3"]["Delta_KD"]),
            "D": d_value,
        }

    with (HERE / "results_multiseed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    stats = {
        metric: sample_stats([seed_metrics[seed][metric] for seed in SEEDS])
        for metric in ["Delta_KD_a05", "Delta_KD_a00", "D"]
    }
    payload = {
        "selection_indices_sha256": next(iter(selection_hashes)),
        "s0": {str(seed): {
            "run_dir": str(s0[seed]["run_dir"].relative_to(ROOT)),
            "checkpoint_sha256": s0[seed]["hash"],
        } for seed in SEEDS},
        "matched_invariants": invariant_reference,
        "seed_metrics": seed_metrics,
        "statistics": stats,
        "rows": rows,
    }
    (HERE / "raw" / "multiseed_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"seed_metrics": seed_metrics, "statistics": stats}, indent=2))


if __name__ == "__main__":
    main()
