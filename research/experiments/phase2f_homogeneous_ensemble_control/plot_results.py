#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linestyle": "-",
})

TG_COLOR = "#D55E00"
TT_COLOR = "#0072B2"
SEED_COLORS = ["#009E73", "#CC79A7", "#E69F00"]


def save(fig, stem):
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300)
    plt.close(fig)


def paired_gain():
    with (ROOT / "results_multiseed.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fig, ax = plt.subplots(figsize=(3.25, 2.55))
    x = np.arange(2)
    for color, row in zip(SEED_COLORS, rows):
        values = [float(row["Gain_TG"]), float(row["Gain_TT"])]
        ax.plot(x, values, color=color, marker="o", linewidth=1.35,
                markersize=4.5, alpha=0.9, label=f"Seed {row['Seed']}")
    means = [
        np.mean([float(row["Gain_TG"]) for row in rows]),
        np.mean([float(row["Gain_TT"]) for row in rows]),
    ]
    ax.plot(x, means, color="#222222", marker="D", linewidth=2.1,
            markersize=5.5, label="Mean", zorder=5)
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.set_xticks(x, ["TG", "TT"])
    ax.set_ylabel("KD gain in test NLL (higher is better)")
    ax.set_title("Paired transferable gain")
    ax.legend(ncol=2, loc="best")
    save(fig, "fig1_paired_kd_gain")


def teacher_comparison():
    payload = json.loads((ROOT / "raw/teacher_diagnostics.json").read_text(encoding="utf-8"))
    tg = payload["teachers"]["TG"]
    tt = payload["teachers"]["TT"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))

    ax = axes[0]
    positions = np.arange(2)
    width = 0.22
    tg_nll = [tg["branch_nll"]["transformer"], tg["branch_nll"]["grassmann"], tg["validation_nll"]]
    tt_nll = [tt["branch_nll"]["transformer1"], tt["branch_nll"]["transformer2"], tt["validation_nll"]]
    for offset, label, index, hatch in ((-width, "Branch 1", 0, ""), (0, "Branch 2", 1, "//"), (width, "Fused", 2, "xx")):
        bars = ax.bar(positions + offset, [tg_nll[index], tt_nll[index]], width * 0.9,
                      color=[TG_COLOR, TT_COLOR], edgecolor="white", linewidth=0.5,
                      hatch=hatch, label=label)
        for bar, value in zip(bars, [tg_nll[index], tt_nll[index]]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(positions, ["TG (37.75M)", "TT (35.34M)"])
    ax.set_ylim(4.24, 4.72)
    ax.set_ylabel("Validation NLL (lower is better)")
    ax.set_title("Teacher and branch quality", pad=27)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))

    ax = axes[1]
    metrics = ["Branch\nJSD", "Fusion\ngain", "Top-1\nagreement"]
    tg_values = [tg["branch_jsd"], tg["fusion_gain_over_better_branch"], tg["branch_top1_agreement"]]
    tt_values = [tt["branch_jsd"], tt["fusion_gain_over_better_branch"], tt["branch_top1_agreement"]]
    x = np.arange(len(metrics))
    width = 0.34
    bars_tg = ax.bar(x - width / 2, tg_values, width, color=TG_COLOR, label="TG")
    bars_tt = ax.bar(x + width / 2, tt_values, width, color=TT_COLOR, label="TT")
    for bars in (bars_tg, bars_tt):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 0.72)
    ax.set_title("Branch diversity and fusion")
    ax.legend()
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig2_teacher_comparison")


if __name__ == "__main__":
    paired_gain()
    teacher_comparison()
