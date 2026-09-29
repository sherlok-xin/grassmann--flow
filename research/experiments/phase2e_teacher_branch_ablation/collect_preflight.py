#!/usr/bin/env python3
"""Validate and collect Phase 2E teacher, utility, and gradient preflights."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
SEEDS = [42, 123, 456]
CONDITIONS = {"F": 0.5, "T": 1.0, "G": 0.0}
EXPECTED_TEACHER = "a5ad284f6895c0b4ed516c855c1b149e6636b945173aa2e842b8132299582694"
EXPECTED_SELECTION = "236633612db1e3144b8f3187a5bfb1795f558a50f26368252a8adf3f0b49413f"
EXPECTED_S0 = {
    42: "9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef",
    123: "a44b64ba5a84d3bd2708a96aad4eae829ff7ab0f00476f0e31146fc314da1f13",
    456: "3fbe985d6474c84d1c61fd7937ea906102cf460846829eb3af4c709593b01757",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: list[float]) -> dict:
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    teacher_rows = []
    branch_rows = []
    difficulty_rows = []
    utility_payloads = {}
    for seed in SEEDS:
        payload = load_json(HERE / f"raw/utility_seed{seed}.json")
        manifest = payload["manifest"]
        if manifest["teacher_checkpoint_sha256"] != EXPECTED_TEACHER:
            raise RuntimeError(f"seed {seed}: teacher hash mismatch")
        if manifest["student_checkpoint_sha256"] != EXPECTED_S0[seed]:
            raise RuntimeError(f"seed {seed}: S0 hash mismatch")
        if manifest["selected_indices_sha256"] != EXPECTED_SELECTION:
            raise RuntimeError(f"seed {seed}: validation selection mismatch")
        if manifest["conditions"] != CONDITIONS:
            raise RuntimeError(f"seed {seed}: intervention conditions mismatch")
        rows = {row["condition"]: row for row in payload["overall"]}
        if set(rows) != set(CONDITIONS):
            raise RuntimeError(f"seed {seed}: missing teacher condition")
        for label, alpha in CONDITIONS.items():
            row = rows[label]
            if not math.isclose(float(row["effective_alpha"]), alpha, abs_tol=1e-12):
                raise RuntimeError(f"seed {seed}/{label}: effective alpha mismatch")
            teacher_rows.append(row)
        branch_rows.extend(payload["branch_groups"])
        difficulty_rows.extend(payload["difficulty_groups"])
        utility_payloads[seed] = rows

    gradient_rows = []
    gradients = {}
    for seed in SEEDS:
        gradients[seed] = {}
        for label, alpha in CONDITIONS.items():
            payload = load_json(HERE / f"raw/gradient/seed{seed}_{label}.json")
            manifest = payload["manifest"]
            row = dict(payload["metrics"])
            if manifest["teacher_checkpoint_sha256"] != EXPECTED_TEACHER:
                raise RuntimeError(f"gradient seed {seed}/{label}: teacher hash mismatch")
            if manifest["student_checkpoint_sha256"] != EXPECTED_S0[seed]:
                raise RuntimeError(f"gradient seed {seed}/{label}: S0 hash mismatch")
            if row["condition"] != label or not math.isclose(float(row["effective_alpha"]), alpha, abs_tol=1e-12):
                raise RuntimeError(f"gradient seed {seed}/{label}: condition mismatch")
            gradients[seed][label] = row
        f_param = float(gradients[seed]["F"]["full_parameter_kd_gradient_norm"])
        f_logit = float(gradients[seed]["F"]["kd_logit_gradient_norm"])
        for label in CONDITIONS:
            row = dict(gradients[seed][label])
            row["parameter_gradient_ratio_to_F"] = float(row["full_parameter_kd_gradient_norm"]) / f_param
            row["logit_gradient_ratio_to_F"] = float(row["kd_logit_gradient_norm"]) / f_logit
            row["optimization_mismatch_flag"] = (
                label == "T" and not 0.5 <= row["parameter_gradient_ratio_to_F"] <= 2.0
            )
            gradient_rows.append(row)

    write_csv(HERE / "teacher_preflight.csv", teacher_rows)
    write_csv(HERE / "gradient_preflight.csv", gradient_rows)
    write_csv(HERE / "token_branch_utility.csv", branch_rows)
    write_csv(HERE / "token_branch_utility_by_difficulty.csv", difficulty_rows)

    teacher_nll = {
        label: summarize([float(utility_payloads[seed][label]["teacher_nll"]) for seed in SEEDS])
        for label in CONDITIONS
    }
    residual = {
        label: summarize([float(utility_payloads[seed][label]["delta_teacher"]) for seed in SEEDS])
        for label in CONDITIONS
    }
    t_f_ratios = [
        float(gradients[seed]["T"]["full_parameter_kd_gradient_norm"])
        / float(gradients[seed]["F"]["full_parameter_kd_gradient_norm"])
        for seed in SEEDS
    ]
    t_g_ratios = [
        float(gradients[seed]["T"]["full_parameter_kd_gradient_norm"])
        / float(gradients[seed]["G"]["full_parameter_kd_gradient_norm"])
        for seed in SEEDS
    ]
    payload = {
        "complete": True,
        "teacher_checkpoint_sha256": EXPECTED_TEACHER,
        "validation_selection_sha256": EXPECTED_SELECTION,
        "conditions": CONDITIONS,
        "teacher_nll": teacher_nll,
        "residual_teacher_advantage": residual,
        "gradient_parameter_norm_ratio_T_over_F": summarize(t_f_ratios),
        "gradient_parameter_norm_ratio_T_over_G": summarize(t_g_ratios),
        "optimization_mismatch_rule": "flag only if T/F parameter-gradient ratio is outside [0.5, 2.0]",
        "optimization_mismatch": any(not 0.5 <= ratio <= 2.0 for ratio in t_f_ratios),
        "teacher_nll_seed_consistency_max_range": {
            label: teacher_nll[label]["max"] - teacher_nll[label]["min"] for label in CONDITIONS
        },
    }
    output = HERE / "raw/preflight_summary.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["optimization_mismatch"]:
        raise SystemExit("Phase 2E T/F gradient-scale gate failed")


if __name__ == "__main__":
    main()
