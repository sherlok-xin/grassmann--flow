#!/usr/bin/env python3
"""Aggregate Phase 3A, compute preregistered metrics, and make compact figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score


HERE = Path(__file__).resolve().parent
PHASE_DIR = HERE.parent
FIG_DIR = PHASE_DIR / "figures"

COLORS = {
    "PTB_fused": "#0072B2",
    "WT2_J_fused_TG": "#56B4E9",
    "WT2_A_fixed_composition": "#D55E00",
    "WT2_Transformer_only": "#009E73",
    "WT2_Grassmann_only": "#E69F00",
    "WT2_TT_homogeneous": "#CC79A7",
    "TinyStories_fused": "#000000",
}
SHORT = {
    "PTB_fused": "PTB F",
    "WT2_J_fused_TG": "WT2 J/TG",
    "WT2_A_fixed_composition": "WT2 A",
    "WT2_Transformer_only": "WT2 T",
    "WT2_Grassmann_only": "WT2 G",
    "WT2_TT_homogeneous": "WT2 TT",
    "TinyStories_fused": "TinyStories F",
}

PREDICTORS = {
    "Teacher residual": ("teacher_residual_advantage", 1, True),
    "Teacher val NLL": ("teacher_validation_nll", -1, False),
    "KL magnitude": ("student_teacher_kl", -1, False),
    "KD grad norm": ("kd_gradient_norm", -1, False),
    "Train CE-KD cosine": ("training_ce_kd_cosine", 1, True),
    "Validation dot a": ("validation_kd_dot_a", 1, True),
    "Validation cosine": ("validation_kd_cosine", 1, True),
    "Quadratic gain": ("predicted_gain_quad_lambda5", 1, True),
    "Virtual nominal gain": ("virtual_gain_nominal_lambda5", 1, True),
}


def configure_plotting():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
    })


def signed_pattern(values):
    return "".join("+" if value > 0 else "-" if value < 0 else "0" for value in values)


def build_summaries(df):
    keys = ["condition_id", "dataset", "family", "student_seed"]
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [column for column in numeric if column not in {"calibration_seed", "student_seed"}]
    mean = df.groupby(keys, as_index=False)[numeric].mean()
    sd = df.groupby(keys, as_index=False)[numeric].std(ddof=1)
    sd = sd.rename(columns={column: f"{column}_calibration_sd" for column in numeric})
    summary = mean.merge(sd, on=keys, how="left")
    patterns = []
    for condition_id, group in df.groupby("condition_id", sort=False):
        patterns.append({
            "condition_id": condition_id,
            "first_order_subset_signs": signed_pattern(group.sort_values("calibration_seed")["validation_kd_dot_a"]),
            "quadratic_subset_signs": signed_pattern(group.sort_values("calibration_seed")["predicted_gain_quad_lambda5"]),
            "virtual_subset_signs": signed_pattern(group.sort_values("calibration_seed")["virtual_gain_nominal_lambda5"]),
            "first_order_subset_stable": group["validation_kd_dot_a"].gt(0).nunique() == 1,
            "quadratic_subset_stable": group["predicted_gain_quad_lambda5"].gt(0).nunique() == 1,
            "virtual_subset_stable": group["virtual_gain_nominal_lambda5"].gt(0).nunique() == 1,
            "finite_lambda_safe_fraction": group["lambda_safe"].notna().mean(),
        })
    summary = summary.merge(pd.DataFrame(patterns), on="condition_id", how="left")
    family_numeric = [column for column in numeric if column not in {"lambda_safe"}]
    family = summary.groupby(["dataset", "family"], as_index=False)[family_numeric].mean()
    return summary, family


def predictor_metrics(frame, level):
    actual = (frame["Gain_KD"] > 0).astype(int).to_numpy()
    rows = []
    for label, (column, orientation, natural_sign) in PREDICTORS.items():
        score = frame[column].to_numpy(dtype=float) * orientation
        finite = np.isfinite(score) & np.isfinite(frame["Gain_KD"].to_numpy(dtype=float))
        y = actual[finite]
        s = score[finite]
        gain = frame.loc[finite, "Gain_KD"].to_numpy(dtype=float)
        auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
        rho = spearmanr(s, gain).statistic if len(s) > 1 else np.nan
        row = {
            "level": level,
            "predictor": label,
            "n": len(s),
            "spearman_rho": rho,
            "auroc": auc,
            "sign_accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "mcc": np.nan,
        }
        if natural_sign:
            prediction = (s > 0).astype(int)
            row.update({
                "sign_accuracy": accuracy_score(y, prediction),
                "balanced_accuracy": balanced_accuracy_score(y, prediction),
                "mcc": matthews_corrcoef(y, prediction),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def save(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def plot_prediction(summary):
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for family, group in summary.groupby("family"):
        ax.scatter(
            group["predicted_gain_quad_lambda5"] * 1e5,
            group["Gain_KD"], s=38, color=COLORS[family], label=SHORT[family],
            edgecolor="white", linewidth=0.5, zorder=3,
        )
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.axvline(0, color="#666666", linewidth=0.8)
    ax.set_xlabel(r"Quadratic predicted gain at $\lambda=5$ ($\times 10^{-5}$)")
    ax.set_ylabel(r"Observed endpoint gain $NLL_{CE}-NLL_{KD}$")
    ax.set_title("Local quadratic score does not recover WT2 positive transfer")
    ax.legend(ncol=2, loc="best")
    save(fig, "fig1_predicted_vs_actual")


def plot_safe_lambda(summary):
    families = list(COLORS)
    y_base = {family: len(families) - 1 - index for index, family in enumerate(families)}
    offsets = {42: -0.16, 123: 0.0, 456: 0.16}
    fig, ax = plt.subplots(figsize=(5.5, 3.35))
    for _, row in summary.iterrows():
        y = y_base[row["family"]] + offsets[int(row["student_seed"])]
        if pd.notna(row["lambda_safe"]):
            ax.scatter(row["lambda_safe"], y, color=COLORS[row["family"]], s=34, zorder=3)
        else:
            ax.scatter(0.65, y, color=COLORS[row["family"]], marker="x", s=30, zorder=3)
    ax.axvline(5, color="#D55E00", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlim(0.5, max(10, np.nanmax(summary["lambda_safe"].to_numpy()) * 1.5))
    ax.set_xlabel(r"Candidate $\lambda_{safe}$ (log scale)")
    ax.set_yticks([y_base[family] for family in families])
    ax.set_yticklabels([SHORT[family] for family in families])
    ax.set_title("Finite safe bounds exist only under restrictive local conditions")
    ax.text(0.01, 0.03, "x: no positive local safe interval", transform=ax.transAxes, fontsize=7.5)
    ax.text(5.6, max(y_base.values()) - 0.35, r"fixed $\lambda=5$", color="#D55E00", fontsize=7.5)
    save(fig, "fig2_safe_lambda")


def confusion(actual, prediction):
    return np.array([
        [int(((actual == 0) & (prediction == 0)).sum()), int(((actual == 0) & (prediction == 1)).sum())],
        [int(((actual == 1) & (prediction == 0)).sum()), int(((actual == 1) & (prediction == 1)).sum())],
    ])


def plot_classification(summary):
    actual = (summary["Gain_KD"] > 0).astype(int).to_numpy()
    items = [
        ("First-order", summary["validation_kd_dot_a"].to_numpy() > 0),
        ("Second-order", summary["predicted_gain_quad_lambda5"].to_numpy() > 0),
        ("Virtual AdamW", summary["virtual_gain_nominal_lambda5"].to_numpy() > 0),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.35), constrained_layout=True)
    for ax, (title, prediction) in zip(axes, items):
        matrix = confusion(actual, prediction.astype(int))
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=11)
        ax.set_xticks([0, 1], ["Harm", "Benefit"])
        ax.set_yticks([0, 1], ["Harm", "Benefit"])
        ax.set_xlabel("Predicted")
        ax.set_title(title)
    axes[0].set_ylabel("Observed")
    fig.suptitle("Seed-level sign classification (21 retrospective rows)", fontsize=10)
    save(fig, "fig3_classification_comparison")


def plot_virtual_curves():
    records = []
    for path in sorted((PHASE_DIR / "raw" / "diagnostics").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        family = payload["condition"]["family"]
        for seed in payload["manifest"]["calibration_seeds"]:
            points = payload["virtual_step_curves"]["nominal_lr_sensitivity"]
            baseline = next(point for point in points if float(point["lambda"]) == 0.0)["calibration_nll"][str(seed)]
            for point in points:
                records.append({
                    "family": family,
                    "lambda": float(point["lambda"]),
                    "delta_nll": float(point["calibration_nll"][str(seed)]) - float(baseline),
                })
    frame = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for family, group in frame.groupby("family"):
        curve = group.groupby("lambda")["delta_nll"].agg(["mean", "std"]).reset_index()
        ax.plot(curve["lambda"], curve["mean"], marker="o", color=COLORS[family], label=SHORT[family])
        ax.fill_between(curve["lambda"], curve["mean"] - curve["std"], curve["mean"] + curve["std"], color=COLORS[family], alpha=0.10)
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.axvline(5, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"Virtual-step KD strength $\lambda$")
    ax.set_ylabel(r"Calibration NLL change vs. $\lambda=0$")
    ax.set_title("Nominal-LR one-step AdamW control")
    ax.legend(ncol=2)
    save(fig, "fig4_virtual_step_curves")


def main():
    configure_plotting()
    df = pd.read_csv(PHASE_DIR / "diagnostics_with_endpoints.csv")
    summary, family = build_summaries(df)
    summary.to_csv(PHASE_DIR / "condition_summary.csv", index=False)
    family.to_csv(PHASE_DIR / "family_summary.csv", index=False)
    seed_metrics = predictor_metrics(summary, "seed")
    family_metrics = predictor_metrics(family, "family")
    pd.concat([seed_metrics, family_metrics], ignore_index=True).to_csv(
        PHASE_DIR / "predictor_metrics.csv", index=False,
    )
    plot_prediction(summary)
    plot_safe_lambda(summary)
    plot_classification(summary)
    plot_virtual_curves()

    exact = {
        "seed_rows": len(summary),
        "family_rows": len(family),
        "harmful_families": family.loc[family["Gain_KD"] < 0, "family"].tolist(),
        "positive_families": family.loc[family["Gain_KD"] > 0, "family"].tolist(),
        "first_order_second_order_seed_sign_identity": bool(
            np.array_equal(summary["validation_kd_dot_a"] > 0, summary["predicted_gain_quad_lambda5"] > 0)
        ),
        "first_order_second_order_subset_sign_identity": bool(
            np.array_equal(df["validation_kd_dot_a"] > 0, df["predicted_gain_quad_lambda5"] > 0)
        ),
        "stable_seed_fraction": {
            "first_order": float(summary["first_order_subset_stable"].mean()),
            "second_order": float(summary["quadratic_subset_stable"].mean()),
            "virtual": float(summary["virtual_subset_stable"].mean()),
        },
    }
    (PHASE_DIR / "raw" / "analysis_summary.json").write_text(
        json.dumps(exact, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(exact, indent=2))


if __name__ == "__main__":
    main()
