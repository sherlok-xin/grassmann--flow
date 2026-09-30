#!/usr/bin/env python3
"""Generate the three publication-quality Phase 2E endpoint figures."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
COLORS = {"G": "#009E73", "T": "#0072B2", "F": "#D55E00"}
LABELS = {"G": "Grassmann-only", "T": "Transformer-only*", "F": "Fused"}
SEED_MARKERS = {42: "o", 123: "s", 456: "^"}


def configure() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
        "lines.linewidth": 1.6,
        "lines.markersize": 5.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_rows() -> list[dict]:
    with (HERE / "results_multiseed.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300)
    plt.close(fig)


def paired_gain(rows: list[dict]) -> None:
    order = ["G", "T", "F"]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for row in rows:
        seed = int(row["Seed"])
        values = [float(row[f"Gain_{condition}"]) for condition in order]
        ax.plot(x, values, color="#8C8C8C", alpha=0.72, linewidth=1.2, zorder=1)
        for index, condition in enumerate(order):
            ax.scatter(
                x[index], values[index], marker=SEED_MARKERS[seed], s=38,
                color=COLORS[condition], edgecolor="white", linewidth=0.6, zorder=3,
                label=f"seed {seed}" if index == 0 else None,
            )
    means = [np.mean([float(row[f"Gain_{condition}"]) for row in rows]) for condition in order]
    ax.plot(x, means, color="#222222", marker="D", markersize=5.2, linewidth=2.0, zorder=2, label="mean")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, [LABELS[condition] for condition in order])
    ax.set_ylabel(r"KD gain, $\mathrm{NLL}_{C0}-\mathrm{NLL}$")
    ax.set_title("Paired teacher-source KD gain")
    ax.legend(ncol=2, loc="upper left")
    fig.text(0.98, 0.015, "* CLIP_SATURATION_WARNING", ha="right", va="bottom",
             fontsize=6.8, color="#555555")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig1_paired_kd_gain")


def teacher_nll_vs_gain(rows: list[dict]) -> None:
    with (HERE / "teacher_preflight.csv").open(encoding="utf-8") as handle:
        teacher_rows = list(csv.DictReader(handle))
    nll_by_condition = {}
    for row in teacher_rows:
        nll_by_condition[row["condition"]] = float(row["teacher_nll"])

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for condition in ["F", "T", "G"]:
        x = nll_by_condition[condition]
        for row in rows:
            seed = int(row["Seed"])
            ax.scatter(
                x, float(row[f"Gain_{condition}"]), marker=SEED_MARKERS[seed], s=42,
                color=COLORS[condition], edgecolor="white", linewidth=0.6, zorder=3,
            )
        mean_gain = np.mean([float(row[f"Gain_{condition}"]) for row in rows])
        y_offset = 10 if condition == "F" else 5
        ax.annotate(
            LABELS[condition], (x, mean_gain), xytext=(5, y_offset), textcoords="offset points",
            fontsize=7.5, color=COLORS[condition], fontweight="bold",
        )
    handles = [
        plt.Line2D([], [], linestyle="none", marker=SEED_MARKERS[seed], markersize=5.5,
                   markerfacecolor="#666666", markeredgecolor="white", label=f"seed {seed}")
        for seed in [42, 123, 456]
    ]
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Teacher validation NLL")
    ax.set_ylabel(r"KD gain, $\mathrm{NLL}_{C0}-\mathrm{NLL}$")
    ax.set_title("Teacher quality and endpoint transfer")
    ax.legend(handles=handles, ncol=3, loc="upper right")
    fig.text(0.98, 0.015, "No fitted regression; * clipping warning", ha="right", va="bottom",
             fontsize=6.8, color="#555555")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig2_teacher_nll_vs_kd_gain")


def hard_token_utility() -> None:
    with (HERE / "token_branch_utility_by_difficulty.csv").open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["group_value"] == "Q5"]
    by_seed = defaultdict(dict)
    for row in rows:
        seed = int(row["student_seed"])
        for condition in ["T", "G", "F"]:
            by_seed[seed][condition] = float(row[f"mean_u_{condition}"])

    order = ["T", "G", "F"]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for seed in [42, 123, 456]:
        values = [by_seed[seed][condition] for condition in order]
        ax.plot(x, values, color="#8C8C8C", alpha=0.72, linewidth=1.2, zorder=1)
        for index, condition in enumerate(order):
            ax.scatter(
                x[index], values[index], marker=SEED_MARKERS[seed], s=38,
                color=COLORS[condition], edgecolor="white", linewidth=0.6, zorder=3,
                label=f"seed {seed}" if index == 0 else None,
            )
    means = [np.mean([by_seed[seed][condition] for seed in by_seed]) for condition in order]
    ax.plot(x, means, color="#222222", marker="D", markersize=5.2, linewidth=2.0, zorder=2, label="mean")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, [LABELS[condition] for condition in order])
    ax.set_ylabel("Mean gold-token utility")
    ax.set_title("Hardest student-loss quintile")
    ax.legend(ncol=2, loc="lower right")
    fig.text(0.98, 0.015, "Validation-token observation", ha="right", va="bottom",
             fontsize=6.8, color="#555555")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig3_hard_token_utility")


def main() -> None:
    configure()
    rows = load_rows()
    paired_gain(rows)
    teacher_nll_vs_gain(rows)
    hard_token_utility()
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
