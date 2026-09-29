# Phase 2E GPT Handoff

## Current status

Phase 2E formal training is authorized under the explicit 2026-09-29 clipping amendment. At amendment time, no Transformer-only formal endpoint existed. Do not create `results_multiseed.csv` from smoke values.

The source-state base is Git commit `a9642cf9393defa2f335811ea77a069becb1747a`. The fixed joint teacher checkpoint SHA256 is `a5ad284f6895c0b4ed516c855c1b149e6636b945173aa2e842b8132299582694`. The S0 checkpoint hashes are `9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef` for seed 42, `a44b64ba5a84d3bd2708a96aad4eae829ff7ab0f00476f0e31146fc314da1f13` for seed 123, and `3fbe985d6474c84d1c61fd7937ea906102cf460846829eb3af4c709593b01757` for seed 456.

## Completed evidence

The teacher validation NLL values on the frozen 512-chunk subset are `4.3006458619` for fused, `4.4456469292` for Transformer-only, and `4.6520635191` for Grassmann-only. Mean residual teacher advantage over the three S0 students is `+0.0745149494`, `-0.0704861179`, and `-0.2769027078`, respectively.

Transformer-only to fused full-parameter KD-gradient ratios are `1.3164257`, `1.2788220`, and `1.2323507`; no gradient-scale mismatch was triggered. All compact diagnostics are stored in `teacher_preflight.csv`, `gradient_preflight.csv`, `token_branch_utility.csv`, `token_branch_utility_by_difficulty.csv`, and `raw/`.

The reduced Transformer-only smoke failed because clipping was `1.00` and overflow/non-finite fractions were `0.16`. The authorized matched full-data gate then produced:

| Condition | Mean pre-clip grad norm | Clip fraction | Overflow | Non-finite | Gate |
|---|---:|---:|---:|---:|---|
| Fused | 1.0439 | 0.2347 | 0.0136 | 0.0136 | pass |
| Transformer-only | 1.4028 | 1.0000 | 0.0136 | 0.0136 | fail |

The formal launcher requires either smoke summary to contain `formal_training_authorized=true`; both currently contain `false`. This safeguard must not be bypassed silently.

## Authorized execution state

The research lead explicitly authorized continuation with `clip_fraction=1.0` because gradients were finite, T/F parameter-gradient ratios remained preregistered-scale compatible, and overflow/non-finite fractions were `0.0136`. Alpha, lambda, temperature, AMP, clip value, budget, data, S0, and endpoint definitions remain frozen.

`CLIP_SATURATION_WARNING`: Transformer-only gradients continuously triggered clipping in the gate. Any endpoint interpretation must acknowledge that optimization constraint. Stop without retuning if formal runs develop NaN, training failure, materially higher overflow/non-finite fractions, or anomalous endpoints.
