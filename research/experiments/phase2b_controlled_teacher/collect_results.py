#!/usr/bin/env python3
"""Collect and validate the four completed Phase 2B formal runs."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs" / "distill_experiments"
EXPECTED_S0 = "9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef"

ARMS = {
    "C0": {"name": "phase2b_wt2_c0_ws_ce_a05", "condition": "WS+CE", "alpha": 0.5, "lambda": 0.0, "composition": "heterogeneous fusion (KD inactive)"},
    "C1": {"name": "phase2b_wt2_c1_good_a05", "condition": "T_good", "alpha": 0.5, "lambda": 5.0, "composition": "heterogeneous fusion"},
    "C2": {"name": "phase2b_wt2_c2_near_a03", "condition": "T_near", "alpha": 0.3, "lambda": 5.0, "composition": "heterogeneous fusion"},
    "C3": {"name": "phase2b_wt2_c3_bad_a00", "condition": "T_bad", "alpha": 0.0, "lambda": 5.0, "composition": "pure Grassmann"},
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_complete_run(experiment_name):
    candidates = sorted(OUTPUTS.glob(f"*_{experiment_name}"))
    complete = [path for path in candidates if (path / "summary.json").exists() and (path / "config.json").exists()]
    if len(complete) != 1:
        raise RuntimeError(f"expected one complete run for {experiment_name}, found {complete}")
    return complete[0]


def main():
    preflight = load_json(HERE / "raw" / "preflight" / "preflight.json")
    preflight_rows = {
        (row["dataset"], float(row["alpha"])): row for row in preflight["rows"]
    }
    with (ROOT / "research" / "experiments" / "phase2_teacher_transfer_audit" / "gradient_compatibility.csv").open(encoding="utf-8") as handle:
        gradient_rows = list(csv.DictReader(handle))

    run_data = {}
    invariant_reference = None
    invariant_keys = [
        "seed", "batch_size", "epochs", "lr", "weight_decay", "warmup_ratio",
        "model_dim", "num_layers", "num_heads", "reduced_dim", "window_sizes",
        "dropout", "student_late_k", "temperature", "kd_loss_mode", "kd_chunk_tokens",
        "distill_strategy", "dataset_name", "dataset_path", "text_field", "max_seq_len",
        "max_lines", "encode_chars_per_batch", "tinystories_val_frac", "split_seed", "amp",
    ]
    raw_formal = HERE / "raw" / "formal"
    raw_formal.mkdir(parents=True, exist_ok=True)

    for arm, spec in ARMS.items():
        run_dir = find_complete_run(spec["name"])
        config_payload = load_json(run_dir / "config.json")
        summary_payload = load_json(run_dir / "summary.json")
        config = config_payload["config"]
        student = summary_payload["student"]
        init = config_payload["student_init"]
        source = config_payload["source_teacher"]

        if init["student_init_checkpoint_sha256"] != EXPECTED_S0:
            raise RuntimeError(f"{arm} S0 hash mismatch")
        if student["student_init_checkpoint_sha256"] != EXPECTED_S0:
            raise RuntimeError(f"{arm} summary S0 hash mismatch")
        effective_alpha = float(source["effective_alpha"][0])
        if abs(effective_alpha - spec["alpha"]) > 1e-6:
            raise RuntimeError(f"{arm} alpha mismatch: {effective_alpha}")
        if abs(float(config["kd_lambda"]) - spec["lambda"]) > 1e-12:
            raise RuntimeError(f"{arm} lambda mismatch")
        invariant = {key: config.get(key) for key in invariant_keys}
        if invariant_reference is None:
            invariant_reference = invariant
        elif invariant != invariant_reference:
            differing = {key: (invariant_reference[key], invariant[key]) for key in invariant_keys if invariant_reference[key] != invariant[key]}
            raise RuntimeError(f"{arm} matched-config violation: {differing}")

        destination = raw_formal / arm
        destination.mkdir(parents=True, exist_ok=True)
        for name in ["config.json", "summary.json", "distill_metrics.jsonl", "report.md"]:
            shutil.copy2(run_dir / name, destination / name)
        run_data[arm] = {
            "run_dir": run_dir,
            "config": config,
            "student": student,
            "source": source,
            "checkpoint_sha256": sha256(run_dir / "checkpoints" / "student_best.pt"),
        }

    c0_test = float(run_data["C0"]["student"]["test_loss"])
    rows = []
    for arm, spec in ARMS.items():
        data = run_data[arm]
        student = data["student"]
        pre = preflight_rows[("wikitext2", spec["alpha"])]
        grad = next(
            row for row in gradient_rows
            if row["domain"] == "wikitext2"
            and row["alpha_source"] == "grid"
            and row["student_state"] == "S0"
            and abs(float(row["alpha"]) - spec["alpha"]) < 1e-9
        )
        clip = float(student["train_grad_clip_fraction"][0])
        overflow = float(student["train_amp_overflow_fraction"][0])
        nonfinite = float(student["train_grad_nonfinite_fraction"][0])
        status_parts = ["COMPLETE"]
        if spec["lambda"] > 0 and clip >= 0.95:
            status_parts.append("LOSS_SCALE_CONFOUNDER")
        if overflow >= 0.05 or nonfinite >= 0.05:
            status_parts.append("INSTABILITY_WARNING")
        kl_epoch1 = float(student["train_kl_token_mean"][0])
        rows.append({
            "Arm": arm,
            "Dataset": "wikitext2",
            "Seed": int(data["config"]["seed"]),
            "Student S0 hash": EXPECTED_S0,
            "Teacher condition": spec["condition"],
            "Alpha": spec["alpha"],
            "Teacher val NLL": float(pre["teacher_val_nll"]),
            "Delta_teacher": float(pre["delta_teacher"]),
            "Lambda": spec["lambda"],
            "Val NLL": float(student["best_val_loss"]),
            "Test NLL": float(student["test_loss"]),
            "Test PPL": float(student["test_ppl"]),
            "Delta_KD": c0_test - float(student["test_loss"]),
            "CE loss epoch1": float(student["train_ce"][0]),
            "KD loss epoch1": kl_epoch1,
            "Weighted KD epoch1": spec["lambda"] * kl_epoch1,
            "Teacher-student KL initial T1": float(grad["mean_kl_t1"]),
            "Teacher-student KL initial T2": float(grad["mean_kl_t2"]),
            "CE logit-grad norm initial": float(grad["ce_grad_norm"]),
            "KD logit-grad norm initial": float(grad["kd_grad_norm"]),
            "CE-KD cosine initial": float(grad["grad_cosine"]),
            "Negative cosine fraction initial": float(grad["negative_grad_fraction"]),
            "Total parameter-grad norm epoch1": float(student["train_grad_norm_mean"][0]),
            "Clip fraction epoch1": clip,
            "AMP overflow fraction epoch1": overflow,
            "Nonfinite fraction epoch1": nonfinite,
            "Best epoch": int(student["best_epoch"]),
            "Teacher composition": spec["composition"],
            "Run directory": str(data["run_dir"].relative_to(ROOT)),
            "Student checkpoint SHA256": data["checkpoint_sha256"],
            "Status": "|".join(status_parts),
        })

    with (HERE / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    raw_summary = {
        "matched_invariants": invariant_reference,
        "expected_s0_sha256": EXPECTED_S0,
        "runs": {
            arm: {
                "run_dir": str(data["run_dir"].relative_to(ROOT)),
                "student_checkpoint_sha256": data["checkpoint_sha256"],
            }
            for arm, data in run_data.items()
        },
        "results": rows,
    }
    (HERE / "raw" / "formal_summary.json").write_text(
        json.dumps(raw_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(raw_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
