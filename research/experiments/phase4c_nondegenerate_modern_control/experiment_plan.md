# Phase4C preregistration — fixed-step disjoint continuation

Parent: 6506c79e560efd2bbdacce7b87e7164b21d56180. Explicit authorization
supersedes only the earlier STOP for this bounded seed42 control. No manuscript,
final_evidence, datasets/checkpoints, old Phase4A/B code or endpoints are changed.
No new method, clipping sweep, lambda5 or extra seed is authorized.

## Inputs and disjoint order

Reuse exact seed42 preparation step256, 2,097,152 targets, whose archived
selected.pt SHA256 is f497cf2310050b163ecb167f75ae4f89c2780d4165aa8f25ff77025715f4bd28.
This is a fixed time point, not a new validation-selected S0. It happens to
equal the old selected state; that coincidence is explicitly disclosed.
The same frozen adapted360M teacher, tokenizer, revisions and token splits
remain unchanged. input_audit.json records the source-state chain.

Freeze continuation_manifest.json before any continuation outcomes. Exclude
the first8192 chunk IDs of the seed42 preparation permutation from the full
seed4242 training permutation; take the first31250 remaining chunks. Each
chunk supervises256 distinct target positions. Exactly8M new targets, no
target position reused from early S0 or repeated during continuation; adjacent
chunks may share one context token. No validation/test chunk enters training.
All arms use this same order. No attempt to select a more favorable subset.

## CE-only gate

Only three formal CE runs: peakLR5e-6 / 1e-5 / 2e-5. A disposable two-update
CE-only smoke (no saved state or test forward) precedes formal training.
No KD model run/probe before gate. Exact shared settings: seed42, AdamW
wd.01, betas(.9,.95), eps1e-8, batch32, micro4, seq256, clip1, FP32 weights,
BF16 autocast, FP32 chunk losses, SDPA, no GradScaler. Fresh optimizer from
S0 for every continuation. Same Phase4A scheduler function, 5% linear warmup
(49 updates) plus cosine over977 updates. Last batch18 rows/4608 targets.
Full validation at steps0/256/512/768/977. Strict minimum NLL; ties choose
earliest checkpoint including S0. All full budgets run despite selection.
For exact equal best NLL across LRs choose smaller LR. No test NLL/forward
while selecting LR; checksum-only test-file integrity audit is not evaluation.

Gate requires S0_val minus best_CE_val >=.01 and best step>0. Otherwise
STOP_MODERN_CONTINUATION_NOT_ESTABLISHED, no KD and no more LR/budget tuning.

## Conditional KD freeze

Only if gate passes: write kd_protocol_frozen.json with selectedLR, S0 and
manifest hashes, schedule and settings, commit/push it BEFORE KD execution.
Reuse protocol-identical selected CE gate run. Run lambda.25 and1 from exact
same S0; T2, token_mean T-squared forward KL(teacher||student), same8M order,
optimizer and scheduler. Lambda.25 is a preregistered low-strength control
motivated by Phase4A initial KD/CE gradient ratio, not endpoints. No lambda5.
All selections include S0; archive all selected-state choices before test.
After choices freeze, test only the selected CE/KD/S0 states, not rejected LRs.
Initial gradient-ratio diagnostic uses the first manifest batch, separate CE
and T-squared KL gradients without updating or changing formal state/order;
run only after gate passes. Report raw KL/CE and lambda-scaled norm ratios.

## Decisions and optimization

Any numerical failure stops; no silent retry or parameter change. Persistent
clip saturation is operationalized BEFORE outcomes as whole-run clip fraction
>=.95; flag CLIP_SATURATION_WARNING for any such arm. If lambda.25 meets this
criterion or any numerical failure occurs, MODERN_OPTIMIZATION_CONFOUNDED
dominates endpoint interpretation. Frequent clipping below.95 is still reported.

If gate passes, evaluate each strength independently: require positive CE
improvement on validation and test, matching CE-KD directions, and absolute
CE-KD test difference>=.01 for a clear pilot. Negative gain yields
MODERN_CLEAN_NEGATIVE_PILOT; positive gain MODERN_KD_BENEFIT_PILOT. Below.01,
direction disagreement or no CE test headroom yields
MODERN_NONDEGENERATE_INCONCLUSIVE. If strengths disagree, report both bounded
per-strength decisions explicitly rather than forcing a common narrative.
Single seed is never robust replication; better-teacher/general-corpus/clipping
causal/universal modern KD claims remain forbidden. STOP after compact artifacts.
