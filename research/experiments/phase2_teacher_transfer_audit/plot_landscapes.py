#!/usr/bin/env python3
"""Generate publication-quality exploratory figures for the Phase 2A audit."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
DOMAIN_ORDER = ["ptb", "wikitext2", "tinystories", "codeparrot_common5k"]
DISPLAY = {
    "ptb": "PTB",
    "wikitext2": "WikiText-2",
    "tinystories": "TinyStories",
    "codeparrot_common5k": "CodeParrot common-5k",
}
COLORS = {
    "nll": "#0072B2",
    "cosine": "#D55E00",
    "kl": "#009E73",
    "student": "#666666",
}


def read_csv(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = [row for row in read_csv(ROOT / "alpha_landscape.csv") if row["alpha_source"] == "grid"]
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "lines.linewidth": 1.5,
        "lines.markersize": 3.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True)
    for ax, domain in zip(axes.flat, DOMAIN_ORDER):
        data = sorted((row for row in rows if row["domain"] == domain), key=lambda row: float(row["alpha"]))
        alpha = [float(row["alpha"]) for row in data]
        nll = [float(row["teacher_nll"]) for row in data]
        s1 = float(data[0]["s1_nll"])
        ax.plot(alpha, nll, marker="o", color=COLORS["nll"], label="Fused teacher")
        ax.axhline(s1, color=COLORS["student"], linestyle="--", label="WS+CE student")
        ax.set_title(DISPLAY[domain])
        ax.set_ylabel("Validation NLL")
        ax.set_xlim(0, 1)
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Fusion weight $\alpha$ (Transformer weight)")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    save(fig, "alpha_vs_validation_nll")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True)
    for ax, domain in zip(axes.flat, DOMAIN_ORDER):
        data = sorted((row for row in rows if row["domain"] == domain), key=lambda row: float(row["alpha"]))
        alpha = [float(row["alpha"]) for row in data]
        cosine = [float(row["grad_cosine_s0"]) for row in data]
        ax.plot(alpha, cosine, marker="o", color=COLORS["cosine"])
        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.set_title(DISPLAY[domain])
        ax.set_ylabel("Mean CE–KD cosine at S0")
        ax.set_xlim(0, 1)
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Fusion weight $\alpha$ (Transformer weight)")
    fig.tight_layout()
    save(fig, "alpha_vs_gradient_cosine")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True)
    for ax, domain in zip(axes.flat, DOMAIN_ORDER):
        data = sorted((row for row in rows if row["domain"] == domain), key=lambda row: float(row["alpha"]))
        alpha = [float(row["alpha"]) for row in data]
        kl0 = [float(row["kl_teacher_to_s0_t1"]) for row in data]
        kl1 = [float(row["kl_teacher_to_s1_t1"]) for row in data]
        ax.plot(alpha, kl0, marker="o", color=COLORS["kl"], label="Warm-start S0")
        ax.plot(alpha, kl1, marker="s", color="#CC79A7", label="WS+CE S1")
        ax.set_title(DISPLAY[domain])
        ax.set_ylabel("KL(teacher || student), T=1")
        ax.set_xlim(0, 1)
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Fusion weight $\alpha$ (Transformer weight)")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    save(fig, "alpha_vs_teacher_student_kl")

    predictor_rows = read_csv(ROOT / "predictor_comparison.csv")
    predictors = [
        ("P1_teacher_residual_advantage_test_nll", "Teacher residual advantage (NLL)"),
        ("P2_teacher_student_kl_s0_t1", "Teacher–student KL at S0"),
        ("P3_mean_gradient_cosine_s0", "Mean CE–KD cosine at S0"),
        ("P4_negative_gradient_fraction_s0", "Negative-cosine fraction at S0"),
        ("P5_teacher_entropy", "Teacher entropy"),
        ("P6_branch_jsd", "Transformer–Grassmann JSD"),
        ("P7_branch_top1_disagreement", "Branch top-1 disagreement"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(10.0, 4.8))
    for ax, (key, label) in zip(axes.flat, predictors):
        xs = [float(row[key]) for row in predictor_rows]
        ys = [float(row["delta_kd_test_nll"]) for row in predictor_rows]
        ax.scatter(xs, ys, color=COLORS["nll"], s=20)
        for x, y, row in zip(xs, ys, predictor_rows):
            ax.annotate(DISPLAY[row["domain"]], (x, y), xytext=(3, 2), textcoords="offset points", fontsize=6)
        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel(r"Existing $\Delta_{KD}$ (NLL)")
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    save(fig, "four_domain_predictor_overview")


if __name__ == "__main__":
    main()
