# Phase 3A Preregistered Gate

Protocol lock date: 2026-10-03. The protocol commit must precede all model-level Phase 3A results.

## Frozen question

Can a validation-risk-aware second-order diagnostic at S0 predict the sign and relative magnitude of fixed-strength KD transfer better than teacher quality, KL magnitude, KD-gradient norm, training CE--KD alignment, and first-order validation-gradient alignment?

## Primary retrospective set

The primary set contains 21 seed-level rows and seven condition families: PTB fused teacher; WikiText-2 Teacher J/fused TG, Teacher A at fixed alpha 0.5, Transformer-only, Grassmann-only, and TT; and TinyStories frozen fused teacher. Each family contains seeds 42, 123, and 456. Teacher J, F, and TG are aliases and appear once. Phase 2B alpha 0.3 and all CodeParrot rows are excluded from primary analysis.

The condition inventory is frozen in `condition_inventory.csv`. Existing downstream gains are not read by the diagnostic runner or blind-table builder.

## Data and numerical protocol

Only the frozen S0 checkpoint, frozen teacher, dataset training split, and dataset validation split are accessible to the diagnostic runner. No test dataset is constructed. Checkpoint SHA256 values are verified before computation.

The training probe is the first 32 training chunks in dataset order, matching the formal batch size. It is evaluated as FP32 microbatches of size 2 and accumulated into an exact batch-mean gradient. Student dropout is enabled for the training probe and deterministically seeded by student seed. The same student forward supplies CE and token-mean KD gradients. Temperature is 2.

Three fixed validation calibration subsets are generated without replacement from seeds `314159`, `271828`, and `161803`, in that order. Each contains two chunks. No subset is selected after seeing results. Student and teacher validation losses use evaluation mode. Full-parameter HVPs are computed in FP32 with calibration microbatch size 1 and then averaged within each subset. No AMP and no last-layer approximation are permitted.

The nominal quadratic step is `eta=1e-4`; the reported target is `lambda=5`. The analytic outputs are `a`, validation-gradient cosine, `b`, `c`, predicted `DeltaV_quad(5)`, and the conditionally defined `lambda_safe`. All gradients and HVPs must be finite.

## Optimizer-aware virtual control

The frozen lambda grid is `[0, 0.5, 1, 2.5, 5, 10]`. For each lambda, the combined full-batch gradient `gC + lambda*gD` is clipped to norm 1 and passed to a fresh AdamW optimizer with weight decay 0.01. S0 is restored before every lambda and virtual states are never saved.

Two curves are recorded because the source implementation initializes the formal `LambdaLR` at a zero learning rate before the first optimizer update:

1. `literal_scheduler_step`: exact scheduler initialization and therefore a protocol-faithful first-step no-op check;
2. `nominal_lr_sensitivity`: one AdamW step at `1e-4`, explicitly a sensitivity control rather than the literal first scheduled update.

Both curves use the same frozen gradients and calibration examples. Only the nominal sensitivity can be used as an informative optimizer-aware predictor; the literal curve audits source-protocol behavior.

## Blinding and freeze order

The blind builder reads only raw diagnostic JSON and `condition_inventory.csv`. It emits `diagnostics_blinded.csv`, which contains condition IDs and diagnostic values but no endpoint gain, then writes its SHA256 to `raw/blinded_freeze.json`. The endpoint merger refuses to run unless the current blind-table hash matches that freeze record. Only then may it read the frozen Phase 2 endpoint tables and create `diagnostics_with_endpoints.csv`.

## Fixed predictor directions

Positive endpoint gain denotes beneficial KD. Predictor scores are oriented as follows before classification/ranking: higher teacher residual advantage is beneficial; lower teacher validation NLL, KL, and KD-gradient norm are beneficial; higher training CE--KD cosine, higher validation cosine, and higher `a` are beneficial; higher `predicted_gain_quad=-DeltaV_quad(5)` is beneficial; and higher `virtual_gain_nominal=V_after(lambda=0)-V_after(lambda=5)` is beneficial. Absolute teacher NLL is reported with a cross-domain scale warning.

Sign predictions use the natural zero boundary for residual advantage, alignments, predicted quadratic gain, and virtual gain. KL, gradient norm, and absolute teacher NLL have no scientifically justified universal zero threshold; they are ranking baselines and do not receive post hoc classification thresholds.

## Evaluation

Primary seed-level evaluation first averages each diagnostic across the three calibration subsets, yielding 21 retrospective rows. Family-level evaluation then averages the three student seeds, yielding seven non-IID families. Repeated seeds, teachers, and domains are not treated as independent IID observations.

For predictors with a natural sign, report sign accuracy, balanced accuracy, MCC, and AUROC when both classes and finite scores exist. For all oriented scores, report Spearman correlation with `Gain_KD`. Report exact family cases, effect ordering, and 3/3 calibration-subset sign consistency rather than emphasizing p-values.

## Numerical gates

Before the formal matrix: unit tests must verify the quadratic expression, safe-bound cases, explicit-Hessian HVP equality, finite-difference directional consistency, and Hessian symmetry on a synthetic model. A real-model smoke must confirm finite full-parameter gradients/HVPs without checkpoint mutation. If memory fails, calibration batch size is already one; no last-layer fallback is allowed.

## Novelty decision gate

Continuation is allowed only if the diagnostic identifies both harmful families (TinyStories and WT2 Teacher A), retains PTB and the positive WT2 J, G, T, and TT families, adds meaningful information over all listed baselines including first-order validation alignment, and is stable across calibration subsets.

The terminal decisions are frozen:

- `STOP_NEW_METHOD_FIRST_ORDER_OVERLAP` if first-order validation alignment performs equally well and curvature adds no useful information;
- `STOP_NEW_METHOD_DIAGNOSTIC_FAILED` if neither first- nor second-order diagnostics predict transfer;
- `THEORY_NEEDS_REVISION` if only the informative nominal-LR virtual control works;
- `PROCEED_TO_PHASE3B` only if second-order information robustly improves prediction.

Every outcome stops for research-lead review. No Phase 3B, coefficient sweep, Safe-KD training, Qwen experiment, or manuscript rewrite is authorized.
