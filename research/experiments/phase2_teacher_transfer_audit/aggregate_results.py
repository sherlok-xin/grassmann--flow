#!/usr/bin/env python3
"""Aggregate Phase 2A raw JSON into analysis-ready CSV and JSON artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


OUT = Path(__file__).resolve().parent
DOMAINS = ["ptb", "wikitext2", "tinystories", "codeparrot_common5k"]
STATE_OVERRIDES = {"ptb": "ptb_states", "tinystories": "tinystories_states"}
S1_VAL_FALLBACKS = {
    "wikitext2": 4.383153787414674,
    "codeparrot_common5k": 1.9888231169760924,
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return sum(values) / len(values)


def pearson(xs, ys):
    mx, my = mean(xs), mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / den if den else float("nan")


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for pos in range(start, end):
            result[order[pos]] = rank
        start = end
    return result


def canonical(rows):
    return next(row for row in rows if row["alpha_source"] == "learned_canonical")


def grid_at(rows, alpha):
    return min(
        (row for row in rows if row["alpha_source"] == "grid"),
        key=lambda row: abs(float(row["alpha"]) - alpha),
    )


def main():
    raw = {domain: read_json(OUT / "raw" / domain / "results.json") for domain in DOMAINS}
    branch_rows = []
    alpha_rows = []
    gradient_rows = []
    grassmann_rows = []

    for domain in DOMAINS:
        result = raw[domain]
        branch = dict(result["branch_metrics"])
        override_name = STATE_OVERRIDES.get(domain)
        if override_name and (OUT / "raw" / override_name / "results.json").exists():
            override = read_json(OUT / "raw" / override_name / "results.json")["branch_metrics"]
            for key in ["s0_nll", "s0_ppl", "s0_top1_accuracy", "s1_nll", "s1_ppl", "s1_top1_accuracy"]:
                branch[key] = override[key]
            branch["student_state_metric_source"] = "same selected validation chunks"
        else:
            branch["s1_nll"] = S1_VAL_FALLBACKS[domain]
            branch["s1_ppl"] = math.exp(S1_VAL_FALLBACKS[domain])
            branch["student_state_metric_source"] = "full-validation training summary fallback"
        branch_rows.append(branch)

        for row in result["alpha_landscape"]:
            enriched = dict(row)
            enriched["s1_nll"] = branch["s1_nll"]
            enriched["s1_ppl"] = branch["s1_ppl"]
            enriched["delta_teacher_validation_nll"] = branch["s1_nll"] - float(row["teacher_nll"])
            alpha_rows.append(enriched)
            for state in ["s0", "s1"]:
                gradient_rows.append({
                    "domain": domain,
                    "alpha": row["alpha"],
                    "alpha_source": row["alpha_source"],
                    "student_state": state.upper(),
                    "chunks": row["chunks"],
                    "valid_tokens": row["valid_tokens"],
                    "ce_grad_norm": row[f"ce_grad_norm_{state}"],
                    "kd_grad_norm": row[f"kd_grad_norm_{state}"],
                    "grad_cosine": row[f"grad_cosine_{state}"],
                    "negative_grad_fraction": row[f"negative_grad_fraction_{state}"],
                    "mean_kl_t1": row[f"kl_teacher_to_{state}_t1"],
                    "mean_kl_t2": row[f"kl_teacher_to_{state}_t2"],
                    "teacher_entropy": row["teacher_entropy"],
                    "branch_jsd": row["branch_jsd"],
                })

        rows = result["alpha_landscape"]
        fused = canonical(rows)
        transformer = grid_at(rows, 1.0)
        grassmann = grid_at(rows, 0.0)
        best_branch = min([transformer, grassmann], key=lambda row: row["teacher_nll"])
        comp = result["complementarity"]
        grassmann_rows.append({
            "domain": domain,
            "best_quality_branch": "transformer" if best_branch is transformer else "grassmann",
            "best_branch_nll": best_branch["teacher_nll"],
            "fused_nll": fused["teacher_nll"],
            "fusion_nll_improvement": float(best_branch["teacher_nll"]) - float(fused["teacher_nll"]),
            "best_branch_kl_s0": best_branch["kl_teacher_to_s0_t1"],
            "fused_kl_s0": fused["kl_teacher_to_s0_t1"],
            "fusion_minus_best_branch_kl_s0": float(fused["kl_teacher_to_s0_t1"]) - float(best_branch["kl_teacher_to_s0_t1"]),
            "best_branch_cosine_s0": best_branch["grad_cosine_s0"],
            "fused_cosine_s0": fused["grad_cosine_s0"],
            "fusion_minus_best_branch_cosine_s0": float(fused["grad_cosine_s0"]) - float(best_branch["grad_cosine_s0"]),
            "transformer_errors": comp["transformer_errors"],
            "fused_corrections_with_higher_grassmann_gold": comp["fused_corrects_transformer_with_higher_g_gold"],
            "correction_fraction_of_transformer_errors": comp["correction_fraction_of_transformer_errors"],
            "transformer_correct": comp["transformer_correct"],
            "fused_harms_with_lower_grassmann_gold": comp["fused_harms_transformer_with_lower_g_gold"],
            "harm_fraction_of_transformer_correct": comp["harm_fraction_of_transformer_correct"],
        })

    write_csv(OUT / "teacher_branch_metrics.csv", branch_rows)
    write_csv(OUT / "alpha_landscape.csv", alpha_rows)
    write_csv(OUT / "gradient_compatibility.csv", gradient_rows)
    write_csv(OUT / "grassmann_specific.csv", grassmann_rows)

    transfer_rows = list(csv.DictReader((OUT / "transfer_table.csv").open(encoding="utf-8")))
    transfer = {}
    for domain in DOMAINS:
        candidates = [row for row in transfer_rows if row["domain"] == domain]
        transfer[domain] = next(
            (row for row in candidates if row["row_type"] == "aggregate_mean"),
            next(row for row in candidates if row["row_type"] == "seed"),
        )
    predictors = []
    for domain in DOMAINS:
        row = canonical(raw[domain]["alpha_landscape"])
        branch = next(value for value in branch_rows if value["domain"] == domain)
        predictors.append({
            "domain": domain,
            "delta_kd_test_nll": float(transfer[domain]["delta_kd_nll"]),
            "P1_teacher_residual_advantage_test_nll": float(transfer[domain]["delta_teacher_nll"]),
            "P2_teacher_student_kl_s0_t1": float(row["kl_teacher_to_s0_t1"]),
            "P3_mean_gradient_cosine_s0": float(row["grad_cosine_s0"]),
            "P4_negative_gradient_fraction_s0": float(row["negative_grad_fraction_s0"]),
            "P5_teacher_entropy": float(row["teacher_entropy"]),
            "P6_branch_jsd": float(row["branch_jsd"]),
            "P7_branch_top1_disagreement": 1.0 - float(branch["branch_top1_agreement"]),
            "evidence_note": transfer[domain]["evidence"],
        })
    write_csv(OUT / "predictor_comparison.csv", predictors)
    outcome = [row["delta_kd_test_nll"] for row in predictors]
    correlations = {}
    for key in [key for key in predictors[0] if key.startswith("P")]:
        values = [row[key] for row in predictors]
        correlations[key] = {
            "pearson_r": pearson(values, outcome),
            "spearman_rho": pearson(ranks(values), ranks(outcome)),
            "n": 4,
            "interpretation": "descriptive only; domains are not independent replicates",
        }
    summary = {
        "predictor_rows": predictors,
        "descriptive_correlations": correlations,
        "grassmann_specific": grassmann_rows,
    }
    (OUT / "raw" / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
