#!/usr/bin/env python3
"""Reproduce the figures used by the CAC manuscript.

All numerical values below are copied from repository ``summary.json`` and
``outputs/benchmark_reports`` artifacts.  The script writes vector PDF files
for LaTeX and 300-DPI PNG previews into this directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


OUT = Path(__file__).resolve().parent

BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
CHARCOAL = "#30343B"
GRAY = "#7A7F87"
LIGHT = "#F5F6F7"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "TeX Gyre Termes", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def rounded_box(ax, xy, width, height, text, *, face, edge=CHARCOAL,
                fontsize=8, weight="normal", hatch=None, zorder=2):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=0.9,
        facecolor=face,
        edgecolor=edge,
        hatch=hatch,
        zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=CHARCOAL,
        weight=weight,
        zorder=zorder + 1,
    )
    return box


def arrow(ax, start, end, *, color=CHARCOAL, style="-", width=1.0,
          mutation=9, connection="arc3,rad=0"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=width,
            linestyle=style,
            color=color,
            connectionstyle=connection,
            shrinkA=1,
            shrinkB=1,
            zorder=4,
        )
    )


def make_framework():
    """Figure 1: controlled factors and teacher-to-student information flow."""
    fig, ax = plt.subplots(figsize=(7.05, 3.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Domain column
    ax.text(0.105, 0.94, "DATA DOMAIN", ha="center", va="center",
            fontsize=8, weight="bold", color=CHARCOAL)
    for y, label in zip([0.76, 0.59, 0.42, 0.25],
                        ["PTB", "WikiText-2", "TinyStories", "Python"]):
        rounded_box(ax, (0.025, y - 0.055), 0.16, 0.11, label,
                    face="#FFFFFF", edge=GRAY, fontsize=7.5)
    ax.text(0.105, 0.095, "same tokenizer and\nsequence length",
            ha="center", va="center", fontsize=6.8, color=GRAY)

    # Teacher construction
    ax.text(0.415, 0.94, "HETEROGENEOUS TEACHER", ha="center", va="center",
            fontsize=8, weight="bold", color=CHARCOAL)
    rounded_box(ax, (0.255, 0.69), 0.19, 0.13,
                "Transformer branch\nself-attention", face="#DDECF7",
                edge=BLUE, fontsize=7.4)
    rounded_box(ax, (0.255, 0.49), 0.19, 0.13,
                "Grassmann branch\ncausal pairs $z_t,z_{t-\\Delta}$",
                face="#DFF2E8", edge=GREEN, fontsize=7.4)
    rounded_box(ax, (0.49, 0.59), 0.17, 0.14,
                "late logit fusion\n$z_T=a z^{(Tr)}+(1-a)z^{(G)}$",
                face="#FFF0CE", edge=ORANGE, fontsize=7.1, weight="bold")
    rounded_box(ax, (0.50, 0.80), 0.15, 0.095,
                "teacher $p_T$", face="#FFFFFF", edge=ORANGE,
                fontsize=7.5, weight="bold")
    arrow(ax, (0.445, 0.755), (0.49, 0.69), color=BLUE)
    arrow(ax, (0.445, 0.555), (0.49, 0.63), color=GREEN)
    arrow(ax, (0.575, 0.73), (0.575, 0.80), color=ORANGE)

    # Student initialization routes
    ax.text(0.415, 0.38, "STUDENT INITIALIZATION", ha="center", va="center",
            fontsize=8, weight="bold", color=CHARCOAL)
    rounded_box(ax, (0.255, 0.14), 0.19, 0.14,
                "random initialization\n$\\theta_0$", face="#FFFFFF",
                edge=VERMILION, fontsize=7.4, hatch="///")
    rounded_box(ax, (0.49, 0.14), 0.17, 0.14,
                "CE warm-start\n$\\theta_{CE}$", face="#E8F4F1",
                edge=GREEN, fontsize=7.4, weight="bold")
    arrow(ax, (0.445, 0.21), (0.49, 0.21), color=GRAY)
    ax.text(0.467, 0.30, "CE pretrain", ha="center", va="bottom",
            fontsize=6.2, color=GRAY)

    # Objective and analysis
    ax.text(0.825, 0.94, "CONTROLLED TRANSFER", ha="center", va="center",
            fontsize=8, weight="bold", color=CHARCOAL)
    rounded_box(ax, (0.715, 0.64), 0.22, 0.17,
                "$\\mathcal{L}=(1-\\lambda)\\mathcal{L}_{CE}$\n"
                "$+\\lambda T^2\\,\\mathrm{KL}(p_T\\Vert p_S)$",
                face="#F1E7F0", edge=PURPLE, fontsize=7.5, weight="bold")
    rounded_box(ax, (0.715, 0.39), 0.22, 0.13,
                "sweep $\\lambda\\in[0,0.04]$\n$T=2$; 10 epochs",
                face="#FFFFFF", edge=PURPLE, fontsize=7.2)
    rounded_box(ax, (0.715, 0.14), 0.22, 0.13,
                "measure test PPL\ninitialization $\\times$ domain",
                face=LIGHT, edge=CHARCOAL, fontsize=7.2, weight="bold")
    arrow(ax, (0.65, 0.847), (0.715, 0.755), color=ORANGE)
    arrow(ax, (0.575, 0.28), (0.715, 0.68), color=GREEN,
          connection="arc3,rad=-0.28")
    arrow(ax, (0.825, 0.64), (0.825, 0.52), color=PURPLE)
    arrow(ax, (0.825, 0.39), (0.825, 0.27), color=CHARCOAL)
    arrow(ax, (0.185, 0.50), (0.255, 0.755), color=GRAY,
          connection="arc3,rad=-0.14")
    arrow(ax, (0.185, 0.50), (0.255, 0.21), color=GRAY,
          connection="arc3,rad=0.14")

    ax.text(0.50, 0.02,
            "Research question: when does a fused teacher improve a hybrid student?",
            ha="center", va="bottom", fontsize=8, color=CHARCOAL,
            weight="bold")

    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"fig_framework.{suffix}", bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)


def make_lambda_sweep():
    """Figure 2: warm-start coefficient sweeps on the four datasets."""
    lambdas = np.array([0.00, 0.01, 0.02, 0.03, 0.04])
    panels = {
        "Penn Treebank": {
            "ppl": [59.0784, 51.9083, 51.2740, 51.3954, 51.6226],
            "scratch": 58.5325,
            "teacher": 50.1125,
        },
        "WikiText-2": {
            "ppl": [71.5748, 61.2148, 60.8349, 61.4270, 62.2060],
            "scratch": 70.1634,
            "teacher": 66.8663,
        },
        "TinyStories": {
            "ppl": [4.8611, 5.3127, 5.4201, 5.4623, 5.4857],
            "scratch": 5.0654,
            "teacher": 4.9691,
        },
        "Python code": {
            "ppl": [7.2454, 6.0018, 6.0986, 6.1510, 6.1826],
            "scratch": 7.6689,
            "teacher": 5.6821,
        },
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.15), sharex=True)
    axes = axes.ravel()
    for i, (ax, (name, vals)) in enumerate(zip(axes, panels.items())):
        y = np.asarray(vals["ppl"])
        ax.plot(lambdas, y, color=BLUE, marker="o", markersize=4.2,
                linewidth=1.7, label="warm-start student", zorder=3)
        ax.axhline(vals["scratch"], color=GRAY, linestyle=(0, (5, 2)),
                   linewidth=1.05, label="CE from scratch")
        ax.axhline(vals["teacher"], color=ORANGE, linestyle=(0, (1.5, 1.5)),
                   linewidth=1.2, label="teacher")
        best = int(np.argmin(y))
        ax.scatter([lambdas[best]], [y[best]], s=46, facecolors="none",
                   edgecolors=VERMILION, linewidths=1.3, zorder=4)
        ax.annotate(f"{y[best]:.2f}", (lambdas[best], y[best]),
                    xytext=(4, -11 if name != "TinyStories" else 7),
                    textcoords="offset points", fontsize=7,
                    color=VERMILION)
        ax.set_title(name, fontsize=8.5, weight="bold", pad=3)
        ax.grid(axis="y", color="#D5D7DA", linewidth=0.55, alpha=0.7)
        ax.margins(x=0.06, y=0.18)
        if i % 2 == 0:
            ax.set_ylabel("Test perplexity $\\downarrow$")
        if i >= 2:
            ax.set_xlabel("Distillation coefficient $\\lambda$")
        ax.set_xticks(lambdas)
        ax.set_xticklabels(["0", ".01", ".02", ".03", ".04"])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.015), handlelength=2.4)
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.985,
                        hspace=0.42, wspace=0.25)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"fig_lambda_sweep.{suffix}")
    plt.close(fig)


def make_latency():
    """Figure 3: batch-size sensitivity for the verified PTB operating points."""
    batch = np.array([1, 8, 32])
    series = {
        "Teacher (36.26M)": ([15.48, 19.15, 55.98], ORANGE, "o", "-"),
        "Accuracy student (31.43M)": ([16.09, 19.33, 48.46], BLUE, "s", "--"),
        "Efficiency student (23.57M)": ([11.50, 16.56, 34.96], GREEN, "^", "-."),
    }
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    for label, (lat, color, marker, linestyle) in series.items():
        ax.plot(batch, lat, color=color, marker=marker, linestyle=linestyle,
                linewidth=1.55, markersize=4.2, label=label)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Batch latency (ms) $\\downarrow$")
    ax.set_xticks(batch)
    ax.grid(axis="y", color="#D5D7DA", linewidth=0.55, alpha=0.75)
    ax.legend(loc="upper left", handlelength=2.3)
    ax.set_ylim(7, 60)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"fig_batch_latency.{suffix}")
    plt.close(fig)


if __name__ == "__main__":
    make_framework()
    make_lambda_sweep()
    make_latency()
    print(f"Wrote figures to {OUT}")
