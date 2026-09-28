import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_branch_diagnostic_statistics import chunk_bootstrap, grouped_cv_metrics


def synthetic_frame():
    rng = np.random.default_rng(0)
    rows = []
    for chunk_id in range(20):
        for _ in range(20):
            jsd = rng.uniform()
            entropy = rng.normal()
            nll = rng.normal()
            grad_cosine = 0.5 - jsd + rng.normal(scale=0.05)
            delta_nll = jsd + rng.normal(scale=0.05)
            rows.append(
                {
                    "dataset": "synthetic",
                    "chunk_id": chunk_id,
                    "teacher_entropy": entropy,
                    "teacher_nll": nll,
                    "branch_jsd_normalized": jsd,
                    "grad_cosine": grad_cosine,
                    "delta_nll": delta_nll,
                }
            )
    return pd.DataFrame(rows)


def test_grouped_cv_detects_incremental_jsd_signal():
    folds = grouped_cv_metrics(synthetic_frame(), folds=5)
    assert folds["auc_delta"].mean() > 0
    assert folds["r2_delta"].mean() > 0


def test_chunk_bootstrap_recovers_predicted_directions():
    summary = chunk_bootstrap(synthetic_frame(), resamples=100, seed=1)
    assert summary["grad_cosine_jsd_coefficient"]["ci95_high"] < 0
    assert summary["delta_nll_jsd_coefficient"]["ci95_low"] > 0
