#!/usr/bin/env python3
"""Build and hash the blinded Phase 3A table without reading endpoint files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
PHASE_DIR = HERE.parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default=str(PHASE_DIR / "condition_inventory.csv"))
    parser.add_argument("--raw-dir", default=str(PHASE_DIR / "raw" / "diagnostics"))
    parser.add_argument("--output", default=str(PHASE_DIR / "diagnostics_blinded.csv"))
    parser.add_argument("--freeze-record", default=str(PHASE_DIR / "raw" / "blinded_freeze.json"))
    return parser.parse_args()


def curve_lookup(curves, mode, lam, seed):
    for point in curves[mode]:
        if float(point["lambda"]) == float(lam):
            return float(point["calibration_nll"][str(seed)])
    raise KeyError((mode, lam, seed))


def main():
    args = parse_args()
    with Path(args.inventory).open(newline="", encoding="utf-8") as handle:
        inventory = list(csv.DictReader(handle))
    rows = []
    raw_hashes = {}
    for condition in inventory:
        raw_path = Path(args.raw_dir) / f"{condition['condition_id']}.json"
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        raw_hashes[condition["condition_id"]] = sha256_file(raw_path)
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        if payload["manifest"]["uses_test_data"] is not False:
            raise RuntimeError(f"Test-access flag is not false for {raw_path}")
        if payload["manifest"]["student_checkpoint_sha256_before"] != payload["manifest"]["student_checkpoint_sha256_after"]:
            raise RuntimeError(f"Student checkpoint mutation in {raw_path}")
        if payload["manifest"]["teacher_checkpoint_sha256_before"] != payload["manifest"]["teacher_checkpoint_sha256_after"]:
            raise RuntimeError(f"Teacher checkpoint mutation in {raw_path}")
        for calibration_seed, metrics in payload["calibration_subsets"].items():
            nominal_0 = curve_lookup(payload["virtual_step_curves"], "nominal_lr_sensitivity", 0, calibration_seed)
            nominal_5 = curve_lookup(payload["virtual_step_curves"], "nominal_lr_sensitivity", 5, calibration_seed)
            literal_0 = curve_lookup(payload["virtual_step_curves"], "literal_scheduler_step", 0, calibration_seed)
            literal_5 = curve_lookup(payload["virtual_step_curves"], "literal_scheduler_step", 5, calibration_seed)
            beneficial_grid = []
            contiguous_max = 0.0
            contiguous = True
            for lam in payload["manifest"]["lambda_grid"]:
                nll = curve_lookup(payload["virtual_step_curves"], "nominal_lr_sensitivity", lam, calibration_seed)
                if nll < nominal_0:
                    beneficial_grid.append(float(lam))
                if float(lam) == 0.0:
                    continue
                if contiguous and nll <= nominal_0:
                    contiguous_max = float(lam)
                else:
                    contiguous = False
            row = {
                "condition_id": condition["condition_id"],
                "dataset": condition["dataset"],
                "family": condition["family"],
                "student_seed": int(condition["student_seed"]),
                "calibration_seed": int(calibration_seed),
                "calibration_indices": ";".join(str(value) for value in metrics["indices"]),
                "teacher_validation_nll": metrics["teacher_validation_nll"],
                "student_validation_nll": metrics["student_validation_nll"],
                "teacher_residual_advantage": metrics["teacher_residual_advantage"],
                "student_teacher_kl": payload["training_probe"]["train_kl_token_mean"],
                "kd_gradient_norm": payload["training_probe"]["g_d_norm"],
                "training_ce_kd_cosine": payload["training_probe"]["train_ce_kd_cosine"],
                "validation_gradient_norm": metrics["validation_gradient_norm"],
                "validation_kd_dot_a": metrics["validation_kd_dot_a"],
                "validation_kd_cosine": metrics["validation_kd_cosine"],
                "mixed_curvature_b": metrics["mixed_curvature_b"],
                "kd_curvature_c": metrics["kd_curvature_c"],
                "predicted_delta_v_quad_lambda5": metrics["predicted_delta_v_quad_lambda5"],
                "predicted_gain_quad_lambda5": metrics["predicted_gain_quad_lambda5"],
                "predicted_quad_sign_lambda5": metrics["predicted_quad_sign_lambda5"],
                "lambda_safe": metrics["lambda_safe"],
                "lambda_safe_status": metrics["lambda_safe_status"],
                "virtual_gain_literal_lambda5": literal_0 - literal_5,
                "virtual_gain_nominal_lambda5": nominal_0 - nominal_5,
                "virtual_nominal_beneficial_grid": ";".join(str(value) for value in beneficial_grid),
                "virtual_nominal_safe_grid_max": contiguous_max,
                "literal_first_step_lr": payload["manifest"]["literal_first_step_lr"],
                "full_parameter_gradients": payload["manifest"]["full_parameter_gradients"],
                "uses_test_data": payload["manifest"]["uses_test_data"],
            }
            rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    forbidden = {field for field in fields if "endpoint" in field.lower() or field.lower() == "gain_kd"}
    if forbidden:
        raise RuntimeError(f"Blinded schema contains endpoint fields: {sorted(forbidden)}")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    freeze = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "blinded_table": str(output),
        "blinded_table_sha256": sha256_file(output),
        "row_count": len(rows),
        "condition_count": len(inventory),
        "contains_endpoint_gain": False,
        "raw_diagnostic_sha256": raw_hashes,
    }
    freeze_path = Path(args.freeze_record)
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
