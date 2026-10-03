# Phase 3A GPT Handoff

Phase 3A is complete and terminal.

## Decision

`STOP_NEW_METHOD_DIAGNOSTIC_FAILED`

Do not launch Phase 3B or any new training. Do not turn the diagnostic into a method claim or modify the manuscript from this result.

## What was tested

The frozen primary set contains 21 seed-level rows across PTB fused, WT2 J/TG, WT2 A, WT2 Transformer-only, WT2 Grassmann-only, WT2 TT, and TinyStories fused. Three fixed validation calibration subsets were used. Diagnostics are full-parameter FP32 measurements at S0; no test data or persistent optimizer state was used.

The blinded table was frozen before endpoint merge at SHA256 `594ac6d441fbe19776280225a0d652eb38baa7402dbf7a7295e065f7530fd9ad`.

## Core result

The first-order validation dot and second-order quadratic score have identical signs on all 63 subset rows and identical primary metrics. Both miss established positive WT2 J/TG, Grassmann-only, and TT transfer. Seed-level quadratic sign accuracy/balanced accuracy/MCC/AUROC are 0.571/0.650/0.279/0.644. Only 11/21 seed rows are sign-stable across calibration subsets.

The nominal-LR virtual AdamW control detects all harmful seed rows but incorrectly predicts harm for 9/15 positive rows. The exact literal first scheduled AdamW step is a zero-LR no-op by source implementation.

## Artifact entry points

- `REPORT.md`: complete scientific judgment and gate audit.
- `diagnostics_blinded.csv`: pre-endpoint diagnostic table.
- `raw/blinded_freeze.json`: blind-table hash and raw hashes.
- `diagnostics_with_endpoints.csv`: verified post-freeze merge.
- `condition_summary.csv`: 21 seed-level averages and subset stability.
- `family_summary.csv`: seven family-level summaries.
- `predictor_metrics.csv`: seed- and family-level predictor metrics.
- `figures/`: four compact PDF/PNG figure pairs.
- `provenance.md`: execution, hash, and numerical-integrity record.

The frozen manuscript evidence under `research/final_evidence/` remains unchanged.
