#!/usr/bin/env python3
"""Publication-quality plots for Phase 2C multiseed and utility results."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
SEED_COLORS = {42: "#0072B2", 123: "#E69F00", 456: "#009E73"}


def read_csv(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    results = read_csv(HERE / "results_multiseed.csv")
    groups = read_csv(HERE / "teacher_utility_by_student_difficulty.csv")
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

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    x = np.array([0.0, 1.0])
    for seed in [42, 123, 456]:
        rows = {row["Arm"]: row for row in results if int(row["Seed"]) == seed}
        y = [float(rows["C1"]["Delta_KD"]), float(rows["C3"]["Delta_KD"])]
        ax.plot(x, y, marker="o", color=SEED_COLORS[seed], label=f"seed {seed}", zorder=3)
        ax.text(1.025, y[1], f"D={y[0]-y[1]:.3f}", color=SEED_COLORS[seed], va="center", fontsize=7)
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_xticks(x, [r"T$_{a05}$", r"T$_{a00}$"])
    ax.set_xlim(-0.12, 1.42)
    ax.set_ylabel(r"KD gain $\Delta_{KD}$ (NLL)")
    ax.set_xlabel("Frozen teacher condition")
    ax.legend(frameon=False)
    save(fig, "paired_teacher_intervention")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharex=True)
    quintiles = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    qx = np.arange(1, 6)
    for seed in [42, 123, 456]:
        rows = {
            row["group_value"]: row for row in groups
            if int(row["student_seed"]) == seed
            and float(row["alpha"]) == 0.0
            and row["group_type"] == "student_loss_quintile"
        }
        means = [float(rows[q]["mean_teacher_gold_advantage"]) for q in quintiles]
        fractions = [float(rows[q]["fraction_teacher_higher_gold"]) for q in quintiles]
        axes[0].plot(qx, means, marker="o", color=SEED_COLORS[seed], label=f"seed {seed}")
        axes[1].plot(qx, fractions, marker="o", color=SEED_COLORS[seed], label=f"seed {seed}")
    axes[0].axhline(0, color="#666666", linewidth=0.8)
    axes[1].axhline(0.5, color="#666666", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel(r"Mean utility $u_i$")
    axes[1].set_ylabel(r"Fraction $u_i>0$")
    for ax in axes:
        ax.set_xticks(qx, quintiles)
        ax.set_xlabel("Student-loss quintile")
    axes[0].set_title(r"T$_{a00}$ gold-token utility")
    axes[1].set_title(r"T$_{a00}$ positive-utility fraction")
    axes[1].legend(frameon=False, loc="best")
    fig.tight_layout()
    save(fig, "conditional_utility_alpha00")


if __name__ == "__main__":
    main()
