#!/usr/bin/env python3
"""Verify the blinded freeze and merge only frozen historical endpoint gains."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
PHASE_DIR = HERE.parent
PROJECT_ROOT = PHASE_DIR.parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def endpoint_map():
    gains = {}
    ptb_rows = read_rows(PROJECT_ROOT / "research/experiments/post_rejection_program/results.csv")
    for seed in (42, 123, 456):
        ce = next(row for row in ptb_rows if row["dataset"] == "ptb" and int(row["seed"]) == seed and row["arm"] == "ce")
        kd = next(row for row in ptb_rows if row["dataset"] == "ptb" and int(row["seed"]) == seed and row["arm"] == "kd")
        gains[f"PTB_F_S{seed}"] = math.log(float(ce["test_ppl"])) - math.log(float(kd["test_ppl"]))

    phase2d = {int(row["Seed"]): row for row in read_rows(PROJECT_ROOT / "research/experiments/phase2d_fixed_composition_teacher_quality/results_multiseed.csv")}
    phase2e = {int(row["Seed"]): row for row in read_rows(PROJECT_ROOT / "research/experiments/phase2e_teacher_branch_ablation/results_multiseed.csv")}
    phase2f = {int(row["Seed"]): row for row in read_rows(PROJECT_ROOT / "research/experiments/phase2f_homogeneous_ensemble_control/results_multiseed.csv")}
    phase2g = {int(row["Seed"]): row for row in read_rows(PROJECT_ROOT / "research/experiments/phase2g_tinystories_negative_transfer/results_multiseed.csv")}
    for seed in (42, 123, 456):
        gains[f"WT2_J_S{seed}"] = float(phase2e[seed]["Gain_F"])
        gains[f"WT2_A_S{seed}"] = float(phase2d[seed]["Gain_A"])
        gains[f"WT2_T_S{seed}"] = float(phase2e[seed]["Gain_T"])
        gains[f"WT2_G_S{seed}"] = float(phase2e[seed]["Gain_G"])
        gains[f"WT2_TT_S{seed}"] = float(phase2f[seed]["Gain_TT"])
        gains[f"TS_F_S{seed}"] = float(phase2g[seed]["Delta_KD"])
    return gains


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blinded", default=str(PHASE_DIR / "diagnostics_blinded.csv"))
    parser.add_argument("--freeze-record", default=str(PHASE_DIR / "raw" / "blinded_freeze.json"))
    parser.add_argument("--output", default=str(PHASE_DIR / "diagnostics_with_endpoints.csv"))
    args = parser.parse_args()

    blinded = Path(args.blinded)
    freeze = json.loads(Path(args.freeze_record).read_text(encoding="utf-8"))
    observed_hash = sha256_file(blinded)
    if observed_hash != freeze["blinded_table_sha256"]:
        raise RuntimeError("Blinded table hash does not match the frozen pre-endpoint record")
    rows = read_rows(blinded)
    gains = endpoint_map()
    if set(gains) != {row["condition_id"] for row in rows}:
        raise RuntimeError("Endpoint condition IDs do not match the blinded condition set")
    for row in rows:
        gain = gains[row["condition_id"]]
        row["Gain_KD"] = gain
        row["endpoint_sign"] = "beneficial" if gain > 0 else "harmful" if gain < 0 else "tie"
    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    merge_record = {
        "blinded_table_sha256_verified": observed_hash,
        "endpoint_sources": [
            "research/experiments/post_rejection_program/results.csv",
            "research/experiments/phase2d_fixed_composition_teacher_quality/results_multiseed.csv",
            "research/experiments/phase2e_teacher_branch_ablation/results_multiseed.csv",
            "research/experiments/phase2f_homogeneous_ensemble_control/results_multiseed.csv",
            "research/experiments/phase2g_tinystories_negative_transfer/results_multiseed.csv",
        ],
        "output_sha256": sha256_file(output),
        "row_count": len(rows),
    }
    (PHASE_DIR / "raw" / "endpoint_merge.json").write_text(
        json.dumps(merge_record, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(merge_record, indent=2))


if __name__ == "__main__":
    main()
