#!/usr/bin/env python3
"""Create the preregistered Phase 2D endpoint figures after all D2 arms exist."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main():
    import matplotlib.pyplot as plt

    with (HERE / "results_multiseed.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row["Status"] != "COMPLETE" for row in rows):
        raise RuntimeError("Phase 2D endpoints are incomplete; refusing to create misleading endpoint figures.")

    seeds = [int(row["Seed"]) for row in rows]
    c0 = [float(row["C0 test NLL"]) for row in rows]
    a = [float(row["Teacher A test NLL"]) for row in rows]
    j = [float(row["Teacher J test NLL"]) for row in rows]
    gain_a = [float(row["Gain_A"]) for row in rows]
    gain_j = [float(row["Gain_J"]) for row in rows]
    figures = HERE / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = ["#0072B2", "#D55E00", "#009E73"]
    markers = ["o", "s", "^"]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for index, (seed, values) in enumerate(zip(seeds, zip(c0, a, j))):
        ax.plot(
            [0, 1, 2], values, marker=markers[index], color=colors[index],
            linewidth=1.5, markersize=5, label=f"seed {seed}",
        )
    ax.set_xticks([0, 1, 2], ["WS+CE", "Teacher A KD", "Teacher J KD"])
    ax.set_ylabel("Test NLL")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "paired_test_nll.pdf")
    fig.savefig(figures / "paired_test_nll.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for index, (seed, left, right) in enumerate(zip(seeds, gain_a, gain_j)):
        ax.plot(
            [0, 1], [left, right], marker=markers[index], color=colors[index],
            linewidth=1.5, markersize=5, label=f"seed {seed}",
        )
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.set_xticks([0, 1], ["Teacher A", "Teacher J"])
    ax.set_ylabel("KD gain in test NLL")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "paired_kd_gain.pdf")
    fig.savefig(figures / "paired_kd_gain.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for index, (seed, left, right) in enumerate(zip(seeds, gain_a, gain_j)):
        ax.plot(
            [6.376220, 4.300644], [left, right], marker=markers[index],
            color=colors[index], linewidth=1.5, markersize=5, label=f"seed {seed}",
        )
    ax.set_xlabel("Teacher validation NLL")
    ax.set_ylabel("Downstream KD gain")
    ax.invert_xaxis()
    ax.set_xticks([6.376220, 4.300644], ["A\n6.376", "J\n4.301"])
    ax.legend(frameon=False)
    ax.set_title("Two fixed-composition teachers", fontsize=9)
    ax.text(
        0.50, 0.03, "Descriptive pairs; no regression fit", transform=ax.transAxes,
        fontsize=7.5, color="0.35",
    )
    fig.tight_layout()
    fig.savefig(figures / "teacher_nll_vs_kd_gain.pdf")
    fig.savefig(figures / "teacher_nll_vs_kd_gain.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
