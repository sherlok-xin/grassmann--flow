#!/usr/bin/env python3
"""Evaluate the authorized matched full-data one-epoch F/T stability gate."""

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


def load_arm(label: str, alpha: float, run_arg: str) -> dict:
    import torch

    run_dir = Path(run_arg)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    config_root = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    config = config_root["config"]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["student"]
    lines = [line for line in (run_dir / "distill_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 1:
        raise RuntimeError(f"{label}: expected one epoch, found {len(lines)}")
    metrics = json.loads(lines[0])
    checkpoint = Path(summary["checkpoint_path"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    finite_values = [
        metrics["train_loss"], metrics["train_ce"], metrics["train_kl_token_mean"],
        metrics["train_grad_norm_mean"], metrics["val_loss"], summary["test_loss"],
    ]
    checks = {
        "all_losses_finite": all(math.isfinite(float(value)) for value in finite_values),
        "checkpoint_serialization_and_load": isinstance(state, dict) and bool(state),
        "teacher_identity_exact": str(config["teacher_run_dir"]).endswith(TEACHER_SUFFIX),
        "effective_alpha_exact": summary["teacher_effective_alpha"] == [alpha],
        "student_s0_hash_exact": summary["student_init_checkpoint_sha256"] == EXPECTED_S0,
        "formal_objective_exact": (
            config["kd_loss_mode"] == "token_mean"
            and float(config["kd_lambda"]) == 5.0
            and float(config["temperature"]) == 2.0
        ),
        "full_data_one_epoch": int(config["max_lines"]) == 0 and int(config["epochs"]) == 1,
        "amp_overflow_fraction_below_0_05": float(metrics["train_amp_overflow_fraction"]) < 0.05,
        "nonfinite_fraction_below_0_05": float(metrics["train_grad_nonfinite_fraction"]) < 0.05,
        "clipping_fraction_below_0_95": float(metrics["train_grad_clip_fraction"]) < 0.95,
    }
    return {
        "condition": label,
        "alpha": alpha,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "train_valid_tokens": int(metrics["train_valid_tokens"]),
        "ce": float(metrics["train_ce"]),
        "raw_kl": float(metrics["train_kl_token_mean"]),
        "weighted_kd": 5.0 * float(metrics["train_kl_token_mean"]),
        "total_gradient_norm": float(metrics["train_grad_norm_mean"]),
        "clipping_fraction": float(metrics["train_grad_clip_fraction"]),
        "overflow_fraction": float(metrics["train_amp_overflow_fraction"]),
        "nonfinite_fraction": float(metrics["train_grad_nonfinite_fraction"]),
        "best_validation_nll": float(summary["best_val_loss"]),
        "test_nll_smoke_only": float(summary["test_loss"]),
        "student_s0_sha256": summary["student_init_checkpoint_sha256"],
        "checks": checks,
        "arm_passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-f", required=True)
    parser.add_argument("--run-t", required=True)
    args = parser.parse_args()
    teacher_checkpoint = ROOT / TEACHER_SUFFIX / "checkpoints/hybrid_best.pt"
    arms = {"F": load_arm("F", 0.5, args.run_f), "T": load_arm("T", 1.0, args.run_t)}
    matched_checks = {
        "teacher_hash_exact": sha256_file(teacher_checkpoint) == EXPECTED_TEACHER,
        "same_student_s0": arms["F"]["student_s0_sha256"] == arms["T"]["student_s0_sha256"],
        "same_training_token_count": arms["F"]["train_valid_tokens"] == arms["T"]["train_valid_tokens"],
        "both_arms_passed": arms["F"]["arm_passed"] and arms["T"]["arm_passed"],
    }
    gate_passed = all(matched_checks.values())
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_amendment_authorized_in_phase2e_plan": True,
        "purpose": "Matched optimization-integrity gate only; endpoint NLL is not used for selection.",
        "arms": arms,
        "matched_checks": matched_checks,
        "gate_passed": gate_passed,
        "formal_training_authorized": gate_passed,
    }
    output = HERE / "raw/smoke/matched_full_epoch_smoke_summary.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    raise SystemExit(0 if gate_passed else 2)


if __name__ == "__main__":
    main()
