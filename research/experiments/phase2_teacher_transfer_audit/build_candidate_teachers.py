#!/usr/bin/env python3
"""Select validation-only controlled alpha candidates for later training design."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOMAINS = ["ptb", "tinystories"]
NEAR_TOLERANCE = 0.025


def fmt(value):
    return f"{float(value):.6f}"


def main():
    with (ROOT / "alpha_landscape.csv").open(encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    lines = [
        "# Candidate Teachers from the Frozen Alpha Landscape",
        "",
        "Candidates use validation data only and the same frozen Transformer and Grassmann branches. "
        "A condition is operationally treated as near the WS+CE student when the absolute NLL difference is at most 0.025 nat/token; "
        "good and bad require margins larger than 0.025. This threshold is a descriptive selection rule, not a significance test.",
        "",
    ]
    for domain in DOMAINS:
        rows = [
            row for row in all_rows
            if row["domain"] == domain and row["alpha_source"] == "grid"
        ]
        for row in rows:
            row["delta"] = float(row["delta_teacher_validation_nll"])
        candidates = {
            "T_good": min((row for row in rows if row["delta"] > NEAR_TOLERANCE), key=lambda row: float(row["teacher_nll"]), default=None),
            "T_near": min((row for row in rows if abs(row["delta"]) <= NEAR_TOLERANCE), key=lambda row: abs(row["delta"]), default=None),
            "T_bad": max((row for row in rows if row["delta"] < -NEAR_TOLERANCE), key=lambda row: float(row["teacher_nll"]), default=None),
        }
        lines.extend([
            f"## {domain}",
            "",
            "| Condition | Alpha | Validation NLL | PPL | Delta teacher | KL to S0 (T=1) | CE–KD cosine S0 | Negative fraction S0 | Entropy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for label in ["T_good", "T_near", "T_bad"]:
            row = candidates[label]
            if row is None:
                lines.append(f"| {label} | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |")
            else:
                lines.append(
                    f"| {label} | {fmt(row['alpha'])} | {fmt(row['teacher_nll'])} | {fmt(row['teacher_ppl'])} | "
                    f"{fmt(row['delta_teacher_validation_nll'])} | {fmt(row['kl_teacher_to_s0_t1'])} | "
                    f"{fmt(row['grad_cosine_s0'])} | {fmt(row['negative_grad_fraction_s0'])} | {fmt(row['teacher_entropy'])} |"
                )
        missing = [label for label, row in candidates.items() if row is None]
        lines.append("")
        if missing:
            lines.append(
                "The frozen alpha grid does not provide " + ", ".join(missing) +
                " under the stated margin rule; no synthetic condition was invented."
            )
        else:
            lines.append("All three regimes are available under the stated descriptive margin rule.")
        lines.append("")
    (ROOT / "candidate_teachers.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
