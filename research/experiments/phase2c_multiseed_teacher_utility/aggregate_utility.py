#!/usr/bin/env python3
"""Aggregate per-seed Phase 2C utility JSON into required CSV tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SEEDS = [42, 123, 456]


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    overall_rows = []
    group_rows = []
    selection_hash = None
    checkpoint_hashes = {}
    for seed in SEEDS:
        path = HERE / "raw" / f"utility_s0_seed{seed}" / "utility.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = payload["manifest"]
        current_hash = manifest["selected_indices_sha256"]
        if selection_hash is None:
            selection_hash = current_hash
        elif current_hash != selection_hash:
            raise RuntimeError(f"selection hash mismatch for seed {seed}")
        checkpoint_hashes[str(seed)] = manifest["student_checkpoint_sha256"]
        for row in payload["overall"]:
            overall_rows.append(row)
        for row in payload["groups"]:
            group_rows.append(row)

    overall_rows.sort(key=lambda row: (int(row["student_seed"]), -float(row["alpha"])))
    group_rows.sort(key=lambda row: (
        int(row["student_seed"]), -float(row["alpha"]),
        row["group_type"], row["group_value"],
    ))
    write_csv(HERE / "teacher_utility.csv", overall_rows)
    write_csv(HERE / "teacher_utility_by_student_difficulty.csv", group_rows)
    summary = {
        "selection_indices_sha256": selection_hash,
        "student_checkpoint_sha256": checkpoint_hashes,
        "overall_rows": len(overall_rows),
        "group_rows": len(group_rows),
    }
    (HERE / "raw" / "utility_aggregate_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
