#!/usr/bin/env python3
"""Build the Phase 2B teacher-quality preflight from frozen Phase 2A artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PHASE2A = ROOT / "research" / "experiments" / "phase2_teacher_transfer_audit"

STATE_SOURCES = {
    "ptb": PHASE2A / "raw" / "ptb_states" / "results.json",
    "wikitext2": HERE / "raw" / "preflight" / "wikitext2_states" / "results.json",
    "tinystories": PHASE2A / "raw" / "tinystories_states" / "results.json",
    "codeparrot_common5k": HERE / "raw" / "preflight" / "codeparrot_common5k_states" / "results.json",
}

S0_HASHES = {
    "ptb": "92e332c972f56d227c576f63ca8840e262830ec0addcc245ab906ae251d7b676",
    "wikitext2": "9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef",
    "tinystories": "e4d0959603bb39f0ad7b7a98025024975e2b87d117d36037e9f9aa9da966f9c2",
    "codeparrot_common5k": "177390c0c5bf56412def81aa404c6985c2c162ad1b1e7e3c00b4b41153e4cb0f",
}


def classify(delta):
    if delta >= 0.05:
        return "T_good"
    if abs(delta) <= 0.025:
        return "T_near"
    if delta <= -0.05:
        return "T_bad"
    return "transition_unclassified"


def main():
    with (PHASE2A / "alpha_landscape.csv").open(encoding="utf-8") as handle:
        alpha_rows = [row for row in csv.DictReader(handle) if row["alpha_source"] == "grid"]

    s1 = {}
    state_manifests = {}
    for domain, path in STATE_SOURCES.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        s1[domain] = float(payload["branch_metrics"]["s1_nll"])
        state_manifests[domain] = {
            "path": str(path.relative_to(ROOT)),
            "selection_mode": payload["manifest"]["selection_mode"],
            "selection_seed": payload["manifest"]["selection_seed"],
            "selected_indices_sha256": payload["manifest"]["selected_indices_sha256"],
            "chunks": payload["branch_metrics"]["chunks"],
            "valid_tokens": payload["branch_metrics"]["valid_tokens"],
            "s1_nll": s1[domain],
        }

    rows = []
    for row in alpha_rows:
        domain = row["domain"]
        delta = s1[domain] - float(row["teacher_nll"])
        rows.append({
            "dataset": domain,
            "alpha": float(row["alpha"]),
            "teacher_val_nll": float(row["teacher_nll"]),
            "ws_ce_val_nll": s1[domain],
            "delta_teacher": delta,
            "classification": classify(delta),
        })

    raw_dir = HERE / "raw" / "preflight"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with (raw_dir / "preflight_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    triplets = {}
    for domain in STATE_SOURCES:
        classes = {row["classification"] for row in rows if row["dataset"] == domain}
        triplets[domain] = all(label in classes for label in ["T_good", "T_near", "T_bad"])

    selected = {
        "route": "A",
        "dataset": "wikitext2",
        "conditions": {
            "C0": {"teacher_condition": "WS+CE", "alpha": 0.5, "kd_lambda": 0.0},
            "C1": {"teacher_condition": "T_good", "alpha": 0.5, "kd_lambda": 5.0},
            "C2": {"teacher_condition": "T_near", "alpha": 0.3, "kd_lambda": 5.0},
            "C3": {"teacher_condition": "T_bad", "alpha": 0.0, "kd_lambda": 5.0},
        },
        "student_s0_sha256": S0_HASHES["wikitext2"],
        "selection_reason": (
            "WikiText-2 contains a complete triplet on the existing grid, has exact S0/S1 checkpoints "
            "and reproducible data provenance, and is substantially cheaper than CodeParrot."
        ),
    }
    payload = {
        "thresholds": {"good_min": 0.05, "near_abs_max": 0.025, "bad_max": -0.05},
        "state_manifests": state_manifests,
        "s0_checkpoint_sha256": S0_HASHES,
        "complete_triplet_on_existing_grid": triplets,
        "selected_experiment": selected,
        "rows": rows,
    }
    (raw_dir / "preflight.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Phase 2B Preflight",
        "",
        "This preflight uses validation data only. Teacher NLL values come from the frozen Phase 2A alpha grid. "
        "Each WS+CE NLL was recomputed from the exact S1 checkpoint on the identical validation chunks used for its teacher landscape.",
        "",
        "Classification thresholds are: `T_good` when Delta_teacher >= +0.05, `T_near` when |Delta_teacher| <= 0.025, "
        "and `T_bad` when Delta_teacher <= -0.05. Remaining points are marked `transition_unclassified`.",
        "",
        "| Dataset | alpha | Teacher val NLL | WS+CE val NLL | Delta_teacher | classification |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['alpha']:.1f} | {row['teacher_val_nll']:.6f} | "
            f"{row['ws_ce_val_nll']:.6f} | {row['delta_teacher']:+.6f} | {row['classification']} |"
        )
    lines.extend([
        "",
        "## Triplet decision",
        "",
        "WikiText-2 provides a complete triplet on the existing frozen alpha grid:",
        "",
        "- `T_good`: alpha 0.5, Delta_teacher = +0.084900 nat/token.",
        "- `T_near`: alpha 0.3, Delta_teacher = +0.000329 nat/token.",
        "- `T_bad`: alpha 0.0, Delta_teacher = -0.266510 nat/token.",
        "",
        "PTB lacks T_bad, TinyStories lacks T_good, and the existing CodeParrot grid lacks T_near. "
        "No cross-dataset triplet and no fabricated condition is used.",
        "",
        "## Route selection",
        "",
        "Route A is selected on WikiText-2. Its separation is clean, its frozen S0 and S1 checkpoints are available, "
        "its dataset manifest is reproducible, and its 9,379 training chunks make the bounded four-arm pilot materially cheaper "
        "than CodeParrot common-5k. No additional alpha evaluation is required for route selection.",
        "",
        f"All four arms must load the same S0 checkpoint with SHA256 `{S0_HASHES['wikitext2']}`.",
    ])
    (HERE / "preflight.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
