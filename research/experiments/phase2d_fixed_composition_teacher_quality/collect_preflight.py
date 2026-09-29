#!/usr/bin/env python3
"""Collect Phase 2D teacher and gradient preflight artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    integrity = read_json(RAW / "teacher_integrity.json")
    if not integrity["gate_passed"]:
        raise SystemExit("Teacher-integrity gate failed")

    teacher_rows = []
    utility_rows = []
    manifests = {}
    for label in ["J", "A"]:
        payload = read_json(RAW / f"teacher_{label}_preflight.json")
        manifest = payload["manifest"]
        if manifest["effective_alpha"] != [0.5]:
            raise RuntimeError(f"Teacher {label} effective alpha mismatch")
        if manifest["selected_indices_sha256"] != "236633612db1e3144b8f3187a5bfb1795f558a50f26368252a8adf3f0b49413f":
            raise RuntimeError(f"Teacher {label} validation selection mismatch")
        teacher_rows.append(payload["teacher_metrics"])
        utility_rows.extend(payload["student_utility"])
        manifests[label] = manifest

    gradient_by_seed = {}
    for seed in [42, 123, 456]:
        pair = {}
        for label in ["J", "A"]:
            payload = read_json(RAW / "gradient" / f"seed{seed}_{label}.json")
            pair[label] = payload["metrics"]
        r_logit = pair["A"]["kd_logit_gradient_norm"] / pair["J"]["kd_logit_gradient_norm"]
        r_param = pair["A"]["full_parameter_kd_gradient_norm"] / pair["J"]["full_parameter_kd_gradient_norm"]
        mismatch = not (0.5 <= r_param <= 2.0)
        gradient_by_seed[seed] = {"R_logit": r_logit, "R_param": r_param, "KD_SCALE_MISMATCH": mismatch}

    gradient_rows = []
    for seed in [42, 123, 456]:
        for label in ["J", "A"]:
            row = read_json(RAW / "gradient" / f"seed{seed}_{label}.json")["metrics"]
            gradient_rows.append({
                **row,
                "R_logit_A_over_J": gradient_by_seed[seed]["R_logit"],
                "R_param_A_over_J": gradient_by_seed[seed]["R_param"],
                "KD_SCALE_MISMATCH": gradient_by_seed[seed]["KD_SCALE_MISMATCH"],
            })

    write_csv(HERE / "teacher_preflight.csv", teacher_rows)
    write_csv(HERE / "teacher_branch_comparison.csv", teacher_rows)
    write_csv(HERE / "teacher_utility_comparison.csv", utility_rows)
    write_csv(HERE / "gradient_scale_preflight.csv", gradient_rows)
    summary = {
        "teacher_integrity_gate": True,
        "teacher_manifests": manifests,
        "teacher_metrics": teacher_rows,
        "student_utility": utility_rows,
        "gradient_scale_by_seed": gradient_by_seed,
        "any_kd_scale_mismatch": any(item["KD_SCALE_MISMATCH"] for item in gradient_by_seed.values()),
    }
    (RAW / "preflight_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "teacher_integrity_gate": True,
        "gradient_scale_by_seed": gradient_by_seed,
        "any_kd_scale_mismatch": summary["any_kd_scale_mismatch"],
    }, indent=2))


if __name__ == "__main__":
    main()
