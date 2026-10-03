# Phase 3A — Offline Safe-Distillation Diagnostic

## Executive decision

`DECISION = STOP_NEW_METHOD_DIAGNOSTIC_FAILED`

The validation-risk-aware local diagnostic does not reliably predict the frozen KD endpoints. The second-order score has exactly the same sign as first-order validation alignment for all 63 condition-by-calibration rows and all 21 seed-level averages. It correctly gives harmful family-level signs to TinyStories and WT2 Teacher A, but it incorrectly gives harmful signs to three established positive families: WT2 Teacher J/TG, Grassmann-only, and TT. Its seed-level sign accuracy is 0.571, balanced accuracy is 0.650, MCC is 0.279, and AUROC is 0.644. The first-order dot product has identical values for all four classification metrics and the same Spearman correlation with endpoint gain.

This is stronger than a first-order-overlap stop alone: neither first- nor second-order diagnostics satisfies the required transfer-family gate. Phase 3B is not authorized or scientifically justified from these results.

## Frozen scope and retrospective set

This phase performed no KD training, coefficient sweep training, endpoint modification, checkpoint write, test-set diagnostic, or manuscript change. It used 21 existing seed-level conditions from seven families: PTB fused, WT2 Teacher J/fused TG, WT2 Teacher A, Transformer-only, Grassmann-only, homogeneous TT, and TinyStories fused. Every family uses independent student seeds 42, 123, and 456. TG/J/F aliases are represented once. CodeParrot and the exploratory Phase 2B alpha-0.3 condition were excluded.

For each condition, the training probe was the first 32 training chunks, accumulated in FP32 microbatches of two. Three deterministic, disjoint calibration subsets used seeds 314159, 271828, and 161803, with two validation chunks per subset. All gradients and HVPs covered the complete 31.43M-parameter student. The nominal local step was `eta=1e-4`, the fixed target was `lambda=5`, and the teacher temperature was 2.

The blinded diagnostic table was generated before endpoint access and frozen at SHA256 `594ac6d441fbe19776280225a0d652eb38baa7402dbf7a7295e065f7530fd9ad`. The endpoint merger verified this hash before reading historical gains.

## Family-level exact results

Positive endpoint gain means KD helped. Positive predicted gain means the local diagnostic predicts help. Values below average three student seeds and three calibration subsets per family.

| Family | Endpoint gain | Teacher residual | Validation dot `a` | Quadratic gain at 5 | Nominal virtual gain at 5 | Quadratic sign correct? |
|---|---:|---:|---:|---:|---:|---|
| PTB fused | +0.144937 | +0.187186 | +0.061017 | +3.045e-5 | +0.007920 | yes |
| TinyStories fused | -0.107219 | +0.016068 | -0.006291 | -0.323e-5 | -0.004411 | yes at family mean; unstable |
| WT2 Teacher A | -0.027654 | -2.197901 | -0.152110 | -7.646e-5 | -0.011410 | yes |
| WT2 Grassmann-only | +0.040991 | -0.304887 | -0.191911 | -9.625e-5 | -0.015855 | no |
| WT2 Teacher J/TG | +0.155665 | +0.017191 | -0.071991 | -3.628e-5 | -0.006130 | no |
| WT2 TT | +0.168125 | +0.021102 | -0.023022 | -1.183e-5 | -0.001820 | no |
| WT2 Transformer-only | +0.134868 | -0.122761 | +0.030021 | +1.456e-5 | -0.000111 | yes for quadratic; virtual mean is wrong |

The TinyStories family mean is negative, but the per-seed first-/second-order signs are negative for only two of three students. Teacher residual advantage is positive on these six small calibration chunks even though the full historical endpoint comparison shows a worse teacher. This illustrates calibration-subset noise and prevents using this small-sample residual as a stable boundary.

## Predictor comparison

The primary seed-level analysis first averages the three calibration subsets, producing 21 rows. Family-level analysis then averages the three students, producing seven non-IID families.

| Predictor | Level | Spearman with gain | AUROC | Sign accuracy | Balanced accuracy | MCC |
|---|---|---:|---:|---:|---:|---:|
| Teacher residual | seed | 0.438 | 0.711 | 0.524 | 0.517 | 0.030 |
| KL magnitude | seed | 0.451 | 0.722 | n/a | n/a | n/a |
| KD gradient norm | seed | -0.118 | 0.500 | n/a | n/a | n/a |
| Training CE--KD cosine | seed | -0.025 | 0.478 | 0.714 | 0.500 | 0.000 |
| First-order validation dot `a` | seed | 0.249 | 0.644 | 0.571 | 0.650 | 0.279 |
| Validation cosine | seed | 0.286 | 0.644 | 0.571 | 0.650 | 0.279 |
| Quadratic predicted gain | seed | 0.249 | 0.644 | 0.571 | 0.650 | 0.279 |
| Nominal virtual AdamW gain | seed | 0.427 | 0.700 | 0.571 | 0.700 | 0.400 |
| First-order validation dot `a` | family | 0.179 | 0.600 | 0.571 | 0.700 | 0.400 |
| Quadratic predicted gain | family | 0.179 | 0.600 | 0.571 | 0.700 | 0.400 |
| Nominal virtual AdamW gain | family | 0.393 | 0.700 | 0.429 | 0.600 | 0.258 |

Absolute teacher validation NLL has seed/family AUROC 0.5 in the pooled cross-domain table and is not scale-comparable across corpora. KL magnitude has the strongest seed-level rank association among the listed scalar baselines, but it has no preregistered universal sign threshold, overlaps harmful and helpful families, and is evaluated on a small non-IID retrospective set. It cannot rescue the safe-strength hypothesis.

## Why curvature does not add information

Across all 63 rows, `a` ranges from -0.3434 to +0.1973, `b` from -0.0893 to +0.5724, and `c` from +0.3709 to +6.8887. Curvature is finite and positive in every row. At `eta=1e-4`, however, the absolute second-order correction has median `2.73e-7`, only 0.52% of the absolute first-order term at the median. Even its largest relative contribution, 20.9%, occurs near a small first-order term and never changes the predicted sign. First- and second-order signs are therefore identical in 63/63 calibration rows, with identical seed- and family-level classification results.

Only 22/63 rows permit a finite positive candidate `lambda_safe`; the other 41 have no positive local safe interval because the initial local slope is non-beneficial. The finite values range from 56.44 to 6429.64, far above the fixed strength 5. The bound consequently behaves as a restatement of first-order sign rather than a useful boundary near the deployed coefficient.

## Optimizer-aware virtual control

The exact source scheduler sets the first AdamW learning rate to zero. The literal protocol-matched curve is therefore a verified no-op for every lambda and condition: the maximum absolute lambda-5 versus lambda-0 difference is exactly zero. This is an implementation property, not evidence for safety.

The preregistered nominal-LR sensitivity applies one clipped AdamW step at `1e-4`. It identifies all six harmful seed rows with no harmful-to-beneficial false positive, but it predicts harm for nine of the fifteen beneficial seed rows. At family level it retains only PTB among five positive families. Thus it does not meet the `THEORY_NEEDS_REVISION` rule, which required the virtual control to work when the analytic theory did not.

## Calibration stability

First- and second-order signs are stable across all three calibration subsets for only 11/21 seed-level conditions (52.4%); the virtual sign is stable for 10/21 (47.6%). At family level, first/second order is stable for PTB, Teacher A, Grassmann-only, and Teacher J, but not for TinyStories, TT, or Transformer-only. Stability therefore fails the fourth novelty gate.

## Numerical and provenance safeguards

Four remote unit tests passed before the formal matrix. They verify the quadratic expression against an exact quadratic loss, candidate-bound failure cases, explicit-Hessian HVP equality, finite-difference directional consistency, Hessian symmetry, and absence of test-split construction in the runner. A real-model smoke then completed a full-parameter FP32 HVP without OOM.

All 21 formal condition files completed. Every recorded gradient and HVP is finite. No AMP, damping, finite-difference approximation, or last-layer restriction was used in the formal computation. Student and teacher checkpoint SHA256 values match before and after every diagnostic. Raw manifests explicitly record `uses_test_data=false`, `performs_persistent_training=false`, and `saves_virtual_states=false`.

## Novelty gate

1. Harmful families: partial pass. Teacher A is stable; TinyStories is correct only after family averaging and is calibration-unstable.
2. Positive families: fail. PTB and Transformer-only are retained, but Teacher J/TG, Grassmann-only, and TT are not.
3. Increment beyond baselines and first order: fail. Curvature changes no sign and no primary metric relative to `a`.
4. Calibration stability: fail. Only 52.4% of seed rows have stable first-/second-order signs.

The authorized extension therefore ends with `STOP_NEW_METHOD_DIAGNOSTIC_FAILED`. No Safe-KD method claim, Phase 3B, coefficient tuning, new training, Qwen experiment, or manuscript modification follows from Phase 3A.
