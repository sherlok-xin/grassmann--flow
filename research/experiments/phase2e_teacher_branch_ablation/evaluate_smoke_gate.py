#!/usr/bin/env python3
"""Evaluate the preregistered reduced-data Phase 2E T smoke gate."""

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
EXPECTED_TEACHER = "a5ad284f6895c0b4ed516c855c1b149e6636b945173aa2e842b8132299582694"
TEACHER_SUFFIX = "outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    import torch

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    config_root = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    config = config_root["config"]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["student"]
    lines = [line for line in (run_dir / "distill_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 1:
        raise RuntimeError(f"Expected one smoke epoch, found {len(lines)}")
    metrics = json.loads(lines[0])
    checkpoint = Path(summary["checkpoint_path"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    teacher_checkpoint = ROOT / TEACHER_SUFFIX / "checkpoints/hybrid_best.pt"
    finite_values = [
        metrics["train_loss"], metrics["train_ce"], metrics["train_kl_token_mean"],
        metrics["train_grad_norm_mean"], metrics["val_loss"], summary["test_loss"],
    ]
    checks = {
        "all_losses_finite": all(math.isfinite(float(value)) for value in finite_values),
        "checkpoint_serialization_and_load": isinstance(state, dict) and bool(state),
        "teacher_identity_exact": str(config["teacher_run_dir"]).endswith(TEACHER_SUFFIX),
        "teacher_hash_exact": sha256_file(teacher_checkpoint) == EXPECTED_TEACHER,
        "effective_alpha_exact": summary["teacher_effective_alpha"] == [1.0],
        "student_s0_hash_exact": summary["student_init_checkpoint_sha256"] == EXPECTED_S0,
        "formal_objective_exact": (
            config["kd_loss_mode"] == "token_mean"
            and float(config["kd_lambda"]) == 5.0
            and float(config["temperature"]) == 2.0
        ),
        "reduced_data_one_epoch": int(config["max_lines"]) == 2000 and int(config["epochs"]) == 1,
        "amp_overflow_fraction_below_0_05": float(metrics["train_amp_overflow_fraction"]) < 0.05,
        "nonfinite_fraction_below_0_05": float(metrics["train_grad_nonfinite_fraction"]) < 0.05,
        "clipping_fraction_below_0_95": float(metrics["train_grad_clip_fraction"]) < 0.95,
    }
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "condition": "T",
        "student_seed": 42,
        "ce": float(metrics["train_ce"]),
        "raw_kl": float(metrics["train_kl_token_mean"]),
        "weighted_kd": 5.0 * float(metrics["train_kl_token_mean"]),
        "total_gradient_norm": float(metrics["train_grad_norm_mean"]),
        "clipping_fraction": float(metrics["train_grad_clip_fraction"]),
        "overflow_fraction": float(metrics["train_amp_overflow_fraction"]),
        "nonfinite_fraction": float(metrics["train_grad_nonfinite_fraction"]),
        "best_validation_nll": float(summary["best_val_loss"]),
        "test_nll_smoke_only": float(summary["test_loss"]),
        "checks": checks,
        "gate_passed": all(checks.values()),
        "formal_training_authorized": all(checks.values()),
        "interpretation": "Optimization-integrity result only; reduced-data endpoint NLL is not a formal comparison.",
    }
    output = HERE / "raw/smoke/reduced_smoke_summary.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    raise SystemExit(0 if payload["gate_passed"] else 2)


if __name__ == "__main__":
    main()
