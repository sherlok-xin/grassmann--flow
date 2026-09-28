#!/usr/bin/env python3
"""Plot the controlled teacher intervention after results collection."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
COLORS = {"C0": "#8C8C8C", "C1": "#0072B2", "C2": "#E69F00", "C3": "#D55E00"}


def load_rows():
    with (HERE / "results.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4:
        raise RuntimeError(f"expected four formal rows, found {len(rows)}")
    return sorted(rows, key=lambda row: row["Arm"])


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = load_rows()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "savefig.dpi": 300,
    })

    kd_rows = [row for row in rows if row["Arm"] != "C0"]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    x = [float(row["Delta_teacher"]) for row in kd_rows]
    y = [float(row["Delta_KD"]) for row in kd_rows]
    for row, xx, yy in zip(kd_rows, x, y):
        ax.scatter(xx, yy, s=42, color=COLORS[row["Arm"]], label=f"{row['Arm']} {row['Teacher condition']}", zorder=3)
        ax.annotate(f"alpha={float(row['Alpha']):.1f}", (xx, yy), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.axvline(0, color="#666666", linewidth=0.8)
    ax.set_xlabel(r"Teacher residual advantage $\Delta_{teacher}$ (NLL)")
    ax.set_ylabel(r"KD gain $\Delta_{KD}$ (NLL)")
    ax.legend(frameon=False)
    save(fig, "teacher_advantage_vs_kd_gain")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
    labels = [row["Arm"] for row in rows]
    test_nll = [float(row["Test NLL"]) for row in rows]
    bars = axes[0].bar(labels, test_nll, color=[COLORS[label] for label in labels], width=0.65)
    axes[0].set_ylabel("Validation-selected test NLL")
    axes[0].set_xlabel("Training arm")
    lower = min(test_nll) - 0.03
    axes[0].set_ylim(lower, max(test_nll) + 0.03)
    for bar, value in zip(bars, test_nll):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.3f}", ha="center", va="bottom", fontsize=7)

    kd_labels = [row["Arm"] for row in kd_rows]
    weighted = [float(row["Weighted KD epoch1"]) for row in kd_rows]
    clip = [float(row["Clip fraction epoch1"]) for row in kd_rows]
    positions = np.arange(len(kd_rows))
    axes[1].bar(positions - 0.18, weighted, width=0.36, color="#56B4E9", label="Weighted KD")
    ax2 = axes[1].twinx()
    ax2.bar(positions + 0.18, clip, width=0.36, color="#D55E00", label="Clip fraction")
    axes[1].set_xticks(positions, kd_labels)
    axes[1].set_xlabel("KD arm")
    axes[1].set_ylabel("Epoch-1 weighted KD loss")
    ax2.set_ylabel("Epoch-1 clip fraction")
    ax2.set_ylim(0, 1.05)
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="best")
    fig.tight_layout()
    save(fig, "controlled_teacher_outcomes_and_scale")


if __name__ == "__main__":
    main()
