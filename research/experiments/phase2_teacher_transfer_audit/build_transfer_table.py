#!/usr/bin/env python3
"""Build the exact checkpoint-backed four-domain transfer table."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

EXPERIMENTS = {
    "ptb": {
        "teacher": "outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint",
        "seeds": {
            42: (
                "outputs/hybrid_experiments/20260328_142654_ptb_hybrid_lite_baseline_mild_e20",
                "outputs/distill_experiments/20260916_150712_h0_ptb_confirm_token_l0_seed42_retry50m_v2",
                "outputs/distill_experiments/20260916_150712_h0_ptb_confirm_token_l5_seed42_retry50m",
            ),
            123: (
                "outputs/hybrid_experiments/20260402_140126_ptb_hybrid_lite_baseline_224x56_l6_seed123_e20",
                "outputs/distill_experiments/20260916_152242_h0_ptb_confirm_token_l0_seed123",
                "outputs/distill_experiments/20260916_155134_h0_ptb_confirm_token_l5_seed123",
            ),
            456: (
                "outputs/hybrid_experiments/20260402_140326_ptb_hybrid_lite_baseline_224x56_l6_seed456_e20",
                "outputs/distill_experiments/20260916_153710_h0_ptb_confirm_token_l0_seed456",
                "outputs/distill_experiments/20260916_155348_h0_ptb_confirm_token_l5_seed456",
            ),
        },
        "evidence": "matched token-normalized KD; three seeds",
    },
    "tinystories": {
        "teacher": "outputs/hybrid_experiments/20260515_022710_ts_latefusion_last1_joint_e10",
        "seeds": {
            42: (
                "outputs/hybrid_experiments/20260515_031101_ts_hybrid_lite_baseline_224x56_l6_e20",
                "outputs/distill_experiments/20260916_073243_h0_ts_confirm_token_l0_seed42",
                "outputs/distill_experiments/20260916_073219_h0_ts_confirm_token_l5_seed42",
            ),
        },
        "evidence": "matched token-normalized KD; single seed",
    },
    "wikitext2": {
        "teacher": "outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint",
        "seeds": {
            42: (
                "outputs/hybrid_experiments/20260515_004119_wt2_hybrid_lite_baseline_224x56_l6_e20",
                "outputs/distill_experiments/20260515_014145_wt2_warmstart_ce_only",
                "outputs/distill_experiments/20260515_022631_wt2_warmstart_kd002",
            ),
        },
        "evidence": "legacy batch-normalized KD sweep winner; single seed",
    },
    "codeparrot_common5k": {
        "teacher": "outputs/hybrid_experiments/20260529_120718_code_teacher_last1_joint",
        "seeds": {
            42: (
                "outputs/hybrid_experiments/20260527_074518_code_hybrid_lite_baseline_ce_scratch",
                "outputs/distill_experiments/20260529_154901_code_ws_ce_only",
                "outputs/distill_experiments/20260530_013210_code_ws_kd10",
            ),
        },
        "evidence": "legacy batch-normalized KD sweep winner; common 5k; single seed",
    },
}


def read_run(rel):
    path = ROOT / rel
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))["config"]
    key = "hybrid" if "hybrid" in summary else "student"
    return path, summary[key], config


def main():
    rows = []
    for domain, spec in EXPERIMENTS.items():
        teacher_path, teacher, _ = read_run(spec["teacher"])
        domain_rows = []
        for seed, (scratch_rel, ce_rel, kd_rel) in spec["seeds"].items():
            scratch_path, scratch, _ = read_run(scratch_rel)
            ce_path, ce, ce_cfg = read_run(ce_rel)
            kd_path, kd, kd_cfg = read_run(kd_rel)
            configured_init = str(ce_cfg.get("student_init_run_dir", ""))
            if scratch_path.name not in configured_init:
                raise RuntimeError(f"CE run {ce_path} does not reference expected init {scratch_path}")
            if scratch_path.name not in str(kd_cfg.get("student_init_run_dir", "")):
                raise RuntimeError(f"KD run {kd_path} does not reference expected init {scratch_path}")
            kd_weight = kd_cfg.get("kd_lambda", kd_cfg.get("distill_alpha"))
            loss_mode = kd_cfg.get("kd_loss_mode") or (
                "token_mean" if kd_cfg.get("kd_lambda") is not None else "legacy_batchmean"
            )
            row = {
                "domain": domain,
                "seed": seed,
                "row_type": "seed",
                "evidence": spec["evidence"],
                "eval_split": "test",
                "teacher_nll": teacher["test_loss"],
                "teacher_ppl": teacher["test_ppl"],
                "ce_scratch_nll": scratch["test_loss"],
                "ce_scratch_ppl": scratch["test_ppl"],
                "warm_start_nll": scratch["test_loss"],
                "warm_start_ppl": scratch["test_ppl"],
                "ws_ce_nll": ce["test_loss"],
                "ws_ce_ppl": ce["test_ppl"],
                "ws_kd_nll": kd["test_loss"],
                "ws_kd_ppl": kd["test_ppl"],
                "delta_teacher_nll": ce["test_loss"] - teacher["test_loss"],
                "delta_kd_nll": ce["test_loss"] - kd["test_loss"],
                "delta_kd_ppl": ce["test_ppl"] - kd["test_ppl"],
                "kd_weight": kd_weight,
                "kd_loss_mode": loss_mode,
                "teacher_run": str(teacher_path.relative_to(ROOT)),
                "ce_scratch_run": str(scratch_path.relative_to(ROOT)),
                "warm_start_checkpoint": scratch.get("checkpoint_path", ""),
                "ws_ce_run": str(ce_path.relative_to(ROOT)),
                "ws_kd_run": str(kd_path.relative_to(ROOT)),
            }
            rows.append(row)
            domain_rows.append(row)
        if len(domain_rows) > 1:
            mean_row = {key: "" for key in domain_rows[0]}
            mean_row.update({
                "domain": domain,
                "seed": "mean",
                "row_type": "aggregate_mean",
                "evidence": spec["evidence"],
                "eval_split": "test",
                "teacher_run": domain_rows[0]["teacher_run"],
                "ce_scratch_run": "three seed-specific runs",
                "warm_start_checkpoint": "three seed-specific checkpoints",
                "ws_ce_run": "three seed-specific runs",
                "ws_kd_run": "three seed-specific runs",
                "kd_weight": domain_rows[0]["kd_weight"],
                "kd_loss_mode": domain_rows[0]["kd_loss_mode"],
            })
            numeric = [
                "teacher_nll", "teacher_ppl", "ce_scratch_nll", "ce_scratch_ppl",
                "warm_start_nll", "warm_start_ppl", "ws_ce_nll", "ws_ce_ppl",
                "ws_kd_nll", "ws_kd_ppl", "delta_teacher_nll", "delta_kd_nll",
                "delta_kd_ppl",
            ]
            for key in numeric:
                mean_row[key] = statistics.mean(float(row[key]) for row in domain_rows)
            rows.append(mean_row)
            std_row = dict(mean_row)
            std_row["seed"] = "std"
            std_row["row_type"] = "aggregate_sample_std"
            for key in numeric:
                std_row[key] = statistics.stdev(float(row[key]) for row in domain_rows)
            for key in ["teacher_run", "ce_scratch_run", "warm_start_checkpoint", "ws_ce_run", "ws_kd_run"]:
                std_row[key] = ""
            rows.append(std_row)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "transfer_table.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "raw" / "transfer_table.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "raw" / "transfer_table.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
