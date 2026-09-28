#!/usr/bin/env python3
"""Audit the scale induced by the legacy batchmean KD implementation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


FIELDS = (
    "run_dir",
    "dataset_name",
    "experiment_name",
    "seed",
    "epoch",
    "max_seq_len",
    "distill_alpha",
    "train_ce",
    "train_kl_batchmean",
    "train_kl_per_token_approx",
    "weighted_ce",
    "weighted_kl",
    "kd_objective_fraction",
    "equivalent_token_lambda",
)


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--registry", type=Path, default=Path("docs/generated/experiment_registry.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/generated"))
    args = parser.parse_args()

    root = args.project_root.resolve()
    registry = args.registry if args.registry.is_absolute() else root / args.registry
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with registry.open(encoding="utf-8") as handle:
        registry_rows = list(csv.DictReader(handle))
    runs = {
        row["run_dir"]: row
        for row in registry_rows
        if row["family"] == "distill_experiments" and row["status"] == "complete"
    }

    rows = []
    for run_dir, meta in sorted(runs.items()):
        metrics_path = root / run_dir / "distill_metrics.jsonl"
        if not metrics_path.is_file():
            continue
        alpha = safe_float(meta["distill_alpha"])
        seq_len = safe_float(meta["max_seq_len"])
        if alpha is None or seq_len is None or seq_len <= 1:
            continue
        positions = seq_len - 1
        equivalent_lambda = alpha / max(1.0 - alpha, 1e-12) * positions
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            ce = safe_float(record.get("train_ce"))
            kl = safe_float(record.get("train_kl"))
            if ce is None or kl is None:
                continue
            weighted_ce = (1.0 - alpha) * ce
            weighted_kl = alpha * kl
            total = weighted_ce + weighted_kl
            rows.append(
                {
                    "run_dir": run_dir,
                    "dataset_name": meta["dataset_name"],
                    "experiment_name": meta["experiment_name"],
                    "seed": meta["seed"],
                    "epoch": record.get("epoch", ""),
                    "max_seq_len": int(seq_len),
                    "distill_alpha": alpha,
                    "train_ce": ce,
                    "train_kl_batchmean": kl,
                    "train_kl_per_token_approx": kl / positions,
                    "weighted_ce": weighted_ce,
                    "weighted_kl": weighted_kl,
                    "kd_objective_fraction": weighted_kl / total if total else 0.0,
                    "equivalent_token_lambda": equivalent_lambda,
                }
            )

    csv_path = output_dir / "kd_loss_scale_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    first_epoch = [
        row for row in rows if int(row["epoch"]) == 1 and row["distill_alpha"] > 0
    ]
    grouped = defaultdict(list)
    for row in first_epoch:
        grouped[row["dataset_name"]].append(row)

    lines = [
        "# Legacy KD Loss-Scale Audit",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "The current implementation applies PyTorch `batchmean` to `[batch, sequence, vocabulary]` logits. With fixed-length, unpadded chunks, the table approximates token-mean KL by dividing the logged KL by `max_seq_len - 1`. This is an accounting audit, not a replacement training result.",
        "",
        "The summary includes positive-alpha runs only. Coefficients still differ across runs, so these medians describe the historical sweep rather than a controlled domain comparison.",
        "",
        "| Dataset | Complete epoch-1 runs | Median alpha | Median CE | Median raw KL | Median approximate token KL | Median KD share of objective |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, items in sorted(grouped.items()):
        lines.append(
            "| {dataset} | {count} | {alpha:.3f} | {ce:.3f} | {kl:.3f} | {token:.4f} | {share:.1%} |".format(
                dataset=dataset,
                count=len(items),
                alpha=statistics.median(row["distill_alpha"] for row in items),
                ce=statistics.median(row["train_ce"] for row in items),
                kl=statistics.median(row["train_kl_batchmean"] for row in items),
                token=statistics.median(row["train_kl_per_token_approx"] for row in items),
                share=statistics.median(row["kd_objective_fraction"] for row in items),
            )
        )
    lines.extend(
        [
            "",
            "For sequence length 256, legacy alpha values 0.01, 0.02 and 0.05 correspond algebraically to approximate token-KL multipliers 2.58, 5.20 and 13.42 when the objective is rewritten as `CE + lambda * KL_token`. This conversion matches objective scale only; gradient behavior still requires a parity smoke test.",
        ]
    )
    report_path = output_dir / "kd_loss_scale_audit.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} epoch records to {csv_path}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
