#!/usr/bin/env python3
"""Evaluate the frozen Phase 2D smoke gate and preserve a compact record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXPECTED_S0 = "9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    import torch

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    metric_lines = [line for line in (run_dir / "distill_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(metric_lines) != 1:
        raise RuntimeError(f"Expected one smoke epoch, found {len(metric_lines)}")
    metrics = json.loads(metric_lines[0])
    student = summary["student"]
    checkpoint = Path(student["checkpoint_path"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    serialization_ok = isinstance(state, dict) and bool(state)
    finite_fields = [
        metrics["train_loss"], metrics["train_ce"], metrics["train_kl_token_mean"],
        metrics["train_grad_norm_mean"], metrics["val_loss"], student["test_loss"],
    ]
    checks = {
        "all_losses_finite": all(math.isfinite(float(value)) for value in finite_fields),
        "checkpoint_serialization_and_load": serialization_ok,
        "effective_alpha_exact": student["teacher_effective_alpha"] == [0.5],
        "student_s0_hash_exact": student["student_init_checkpoint_sha256"] == EXPECTED_S0,
        "amp_overflow_fraction_below_0_05": float(metrics["train_amp_overflow_fraction"]) < 0.05,
        "nonfinite_fraction_below_0_05": float(metrics["train_grad_nonfinite_fraction"]) < 0.05,
        "clipping_fraction_below_0_95": float(metrics["train_grad_clip_fraction"]) < 0.95,
    }
    gradient = json.loads((HERE / "raw/gradient/seed42_A.json").read_text(encoding="utf-8"))["metrics"]
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "teacher": "A",
        "student_seed": 42,
        "effective_alpha": student["teacher_effective_alpha"],
        "student_s0_sha256": student["student_init_checkpoint_sha256"],
        "epochs": config["config"]["epochs"],
        "max_lines": config["config"]["max_lines"],
        "ce": metrics["train_ce"],
        "raw_kl": metrics["train_kl_token_mean"],
        "weighted_kd": 5.0 * float(metrics["train_kl_token_mean"]),
        "total_gradient_norm": metrics["train_grad_norm_mean"],
        "clipping_fraction": metrics["train_grad_clip_fraction"],
        "overflow_fraction": metrics["train_amp_overflow_fraction"],
        "nonfinite_fraction": metrics["train_grad_nonfinite_fraction"],
        "kd_logit_gradient_norm_preflight": gradient["kd_logit_gradient_norm"],
        "kd_parameter_gradient_norm_preflight": gradient["full_parameter_kd_gradient_norm"],
        "best_validation_nll": student["best_val_loss"],
        "test_nll_smoke_only": student["test_loss"],
        "checks": checks,
        "gate_passed": all(checks.values()),
        "formal_training_authorized": all(checks.values()),
        "interpretation": (
            "Smoke-only optimization integrity result. Reduced-data validation/test values are not "
            "comparable to the formal Phase 2C endpoints."
        ),
    }
    output_dir = HERE / "raw/smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "smoke_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["gate_passed"]:
        raise SystemExit(0)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
