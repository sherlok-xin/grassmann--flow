# Phase 2E GPT Handoff

## Current status

Phase 2E is blocked before formal endpoints. Do not infer Transformer-only KD endpoint results and do not create `results_multiseed.csv` from smoke values.

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

## Required user decision

Either preserve the preregistration and terminate Phase 2E as gate-blocked, or explicitly authorize a prospective amendment before any test endpoint is observed. A defensible amendment would need to state why 100% finite-gradient clipping is acceptable despite the frozen `<0.95` condition, while retaining the same alpha, lambda, AMP, clipping value, seeds, and endpoint definitions. No formal run or alternate alpha has been launched.
