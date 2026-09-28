#!/usr/bin/env python3
"""Analyze branch diagnostics with chunk-grouped CV and chunk bootstrap."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_FEATURES = ["teacher_entropy", "teacher_nll"]
EXTENDED_FEATURES = ["teacher_entropy", "teacher_nll", "branch_jsd_normalized"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def grouped_cv_metrics(frame: pd.DataFrame, folds: int):
    groups = frame["chunk_id"].to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(folds, len(unique_groups))
    if n_splits < 2:
        raise ValueError("At least two chunks are required for grouped cross-validation")
    splitter = GroupKFold(n_splits=n_splits)
    conflict = (frame["grad_cosine"].to_numpy() < 0).astype(np.int64)
    delta_nll = frame["delta_nll"].to_numpy()
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(frame, groups=groups), start=1):
        row = {"fold": fold, "test_chunks": int(len(np.unique(groups[test_idx])))}
        for name, features in [("baseline", BASE_FEATURES), ("extended", EXTENDED_FEATURES)]:
            x_train = frame.iloc[train_idx][features].to_numpy()
            x_test = frame.iloc[test_idx][features].to_numpy()
            y_train = conflict[train_idx]
            y_test = conflict[test_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                auc = float("nan")
            else:
                classifier = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=1000, random_state=0),
                )
                classifier.fit(x_train, y_train)
                auc = float(roc_auc_score(y_test, classifier.predict_proba(x_test)[:, 1]))

            regressor = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            regressor.fit(x_train, delta_nll[train_idx])
            prediction = regressor.predict(x_test)
            r2 = float(r2_score(delta_nll[test_idx], prediction))
            row[f"{name}_conflict_auc"] = auc
            row[f"{name}_delta_nll_r2"] = r2
        row["auc_delta"] = row["extended_conflict_auc"] - row["baseline_conflict_auc"]
        row["r2_delta"] = row["extended_delta_nll_r2"] - row["baseline_delta_nll_r2"]
        fold_rows.append(row)
    return pd.DataFrame(fold_rows)


def standardized_jsd_coefficients(chunk_frame: pd.DataFrame):
    x = chunk_frame[EXTENDED_FEATURES].to_numpy(dtype=np.float64)
    x = StandardScaler().fit_transform(x)
    grad_model = LinearRegression().fit(x, chunk_frame["grad_cosine"].to_numpy())
    harm_model = LinearRegression().fit(x, chunk_frame["delta_nll"].to_numpy())
    return float(grad_model.coef_[-1]), float(harm_model.coef_[-1])


def chunk_bootstrap(frame: pd.DataFrame, resamples: int, seed: int):
    numeric = EXTENDED_FEATURES + ["grad_cosine", "delta_nll"]
    chunk_frame = frame.groupby("chunk_id", as_index=False)[numeric].mean()
    rng = np.random.default_rng(seed)
    grad_coefficients = []
    harm_coefficients = []
    for _ in range(resamples):
        sampled = rng.integers(0, len(chunk_frame), size=len(chunk_frame))
        boot = chunk_frame.iloc[sampled]
        grad_coef, harm_coef = standardized_jsd_coefficients(boot)
        grad_coefficients.append(grad_coef)
        harm_coefficients.append(harm_coef)

    def interval(values):
        values = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(values)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }

    point_grad, point_harm = standardized_jsd_coefficients(chunk_frame)
    return {
        "chunk_count": int(len(chunk_frame)),
        "grad_cosine_jsd_coefficient": {
            "point": point_grad,
            **interval(grad_coefficients),
            "predicted_direction": "negative",
        },
        "delta_nll_jsd_coefficient": {
            "point": point_harm,
            **interval(harm_coefficients),
            "predicted_direction": "positive",
        },
    }


def disagreement_bins(frame: pd.DataFrame):
    output = frame.copy()
    output["jsd_bin"] = pd.qcut(
        output["branch_jsd_normalized"], q=10, duplicates="drop"
    )
    return (
        output.groupby("jsd_bin", observed=True)
        .agg(
            token_count=("branch_jsd_normalized", "size"),
            chunk_count=("chunk_id", "nunique"),
            jsd_mean=("branch_jsd_normalized", "mean"),
            entropy_mean=("teacher_entropy", "mean"),
            grad_cosine_mean=("grad_cosine", "mean"),
            negative_conflict_fraction=("grad_cosine", lambda x: float(np.mean(x < 0))),
            delta_nll_mean=("delta_nll", "mean"),
            harmful_endpoint_fraction=("delta_nll", lambda x: float(np.mean(x > 0))),
        )
        .reset_index()
        .assign(jsd_bin=lambda x: x["jsd_bin"].astype(str))
    )


def main():
    args = parse_args()
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    input_path = Path(args.input_csv)
    frame = pd.read_csv(input_path)
    required = set(EXTENDED_FEATURES + ["chunk_id", "grad_cosine", "delta_nll"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Diagnostic input contains missing values")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = grouped_cv_metrics(frame, args.folds)
    bins = disagreement_bins(frame)
    bootstrap = chunk_bootstrap(frame, args.bootstrap_resamples, args.seed)

    folds_path = output_dir / "grouped_cv_folds.csv"
    bins_path = output_dir / "jsd_bins.csv"
    summary_path = output_dir / "statistical_summary.json"
    folds.to_csv(folds_path, index=False)
    bins.to_csv(bins_path, index=False)

    summary = {
        "created_at": datetime.now().isoformat(),
        "source": str(input_path.resolve()),
        "dataset_values": sorted(str(value) for value in frame["dataset"].unique()),
        "token_count": int(len(frame)),
        "chunk_count": int(frame["chunk_id"].nunique()),
        "grouped_cross_validation": {
            "folds": int(len(folds)),
            "mean_baseline_conflict_auc": float(folds["baseline_conflict_auc"].mean()),
            "mean_extended_conflict_auc": float(folds["extended_conflict_auc"].mean()),
            "mean_auc_delta": float(folds["auc_delta"].mean()),
            "mean_baseline_delta_nll_r2": float(folds["baseline_delta_nll_r2"].mean()),
            "mean_extended_delta_nll_r2": float(folds["extended_delta_nll_r2"].mean()),
            "mean_r2_delta": float(folds["r2_delta"].mean()),
        },
        "chunk_bootstrap": {
            "resamples": args.bootstrap_resamples,
            "seed": args.seed,
            **bootstrap,
        },
        "outputs": {
            "folds": str(folds_path.resolve()),
            "bins": str(bins_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
