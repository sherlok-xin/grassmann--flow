# Phase 2E Teacher Branch Ablation

## Frozen scientific question

Phase 2E tests whether the jointly trained heterogeneous WikiText-2 teacher transfers more effectively than either component branch alone under the same warm-start KD protocol. It is a teacher-source ablation, not a test of Grassmann geometry as a unique causal mechanism.

The only new formal condition is Transformer-only teacher supervision. Existing Phase 2C WS+CE, fused-teacher, and Grassmann-only endpoints are reused without retraining.

## Frozen teacher and conditions

All conditions use `outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint/checkpoints/hybrid_best.pt`, required SHA256 `a5ad284f6895c0b4ed516c855c1b149e6636b945173aa2e842b8132299582694`.

The fusion convention is `alpha * Transformer logits + (1-alpha) * Grassmann logits`.

| Label | Effective alpha | Status |
|---|---:|---|
| T_F | 0.5 | reuse Phase 2C C1 |
| T_T | 1.0 | new Phase 2E arm |
| T_G | 0.0 | reuse Phase 2C C3 |

No teacher checkpoint is retrained or created.

## Reused controls and student initializations

| Seed | S0 run | Required checkpoint SHA256 |
|---:|---|---|
| 42 | `outputs/hybrid_experiments/20260515_004119_wt2_hybrid_lite_baseline_224x56_l6_e20` | `9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef` |
| 123 | `outputs/hybrid_experiments/20260922_071001_phase2c_wt2_s0_seed123` | `a44b64ba5a84d3bd2708a96aad4eae829ff7ab0f00476f0e31146fc314da1f13` |
| 456 | `outputs/hybrid_experiments/20260922_071449_phase2c_wt2_s0_seed456` | `3fbe985d6474c84d1c61fd7937ea906102cf460846829eb3af4c709593b01757` |

Phase 2C C0, C1/T_F, and C3/T_G are accepted only after their recorded S0 hashes, alpha values, lambda values, and run completeness are rechecked.

## Offline preflight

Teacher and student diagnostics use the same deterministic 512-chunk WikiText-2 validation subset selected with seed 20260920. The required selected-index SHA256 is `236633612db1e3144b8f3187a5bfb1795f558a50f26368252a8adf3f0b49413f`.

For T_F, T_T, and T_G, the preflight records teacher NLL, PPL, entropy, student-specific KL, utility, positive-utility fraction, hardest-quintile utility and positive fraction, rescue/harm rates, CE-KD cosine, and KD logit-gradient norm. It also aggregates branch-complementarity groups and student-loss quintiles without storing raw text or logits.

Gradient-scale calibration uses the first eight non-shuffled training batches of size two for each condition and S0. It records raw and weighted token-normalized KL, CE and KD logit-gradient norms, full-parameter KD-gradient norm, CE-KD cosine, and negative-cosine fraction. `KD_SCALE_MISMATCH` is triggered only when the T_T/T_F full-parameter ratio falls outside `[0.5, 2.0]`; the logit ratio is the fallback if parameter gradients are unavailable.

## Smoke and stability gate

Seed 42 first receives a one-epoch, 2,000-line T_T smoke with otherwise formal settings. It passes only if losses are finite, the teacher and S0 hashes are exact, effective alpha is 1.0, the checkpoint reloads, clipping is below 0.95, and AMP overflow/non-finite fractions are below 0.05.

If the reduced smoke reproduces the known Phase 2D short-window artifact, the preauthorized fallback is a matched full-data one-epoch T_F/T_T stability gate from the same seed-42 S0 and data order. Only the teacher alpha differs. Both arms must pass the same absolute clipping and AMP thresholds. The full-data gate is used only for optimization integrity; its validation or test values cannot select a condition or alter the endpoint definitions.

## Formal T_T configuration

Each new T_T arm uses effective alpha 1.0, temperature 2, token-normalized KL, lambda 5, the seed-specific S0, 10 epochs, batch size 32, learning rate `1e-4`, AdamW weight decay 0.01, warmup ratio 0.05, cosine decay, sequence length 256, AMP, gradient clipping at 1, full WikiText-2 data, the existing seeded data-loader policy, and validation-NLL checkpoint selection. Test data are used only for the final selected checkpoint.

For seed `s`, `Gain_X(s)=NLL_C0(s)-NLL_X(s)`. Paired contrasts are `C_FT=NLL_T-NLL_F`, `C_FG=NLL_G-NLL_F`, and `C_TG=NLL_G-NLL_T`. Positive values favor the first condition named in each definition. `C_FT` is primary.

Before endpoints are observed, a practically nontrivial paired effect is frozen as an absolute mean contrast of at least 0.02 test NLL with consistent seed signs. Smaller consistent effects are reported as partial; material sign changes across seeds are mixed evidence.

## Execution and stopping

Heavy computation runs only on server `10.42.0.197`. Datasets and historical checkpoints are not modified. Failed or retried runs are retained. Phase 2E stops after completing the required report and choosing one next experiment; that experiment is not launched automatically.

## Authorized protocol amendment after the stability gate

On 2026-09-29, after the full-data one-epoch gate and before any formal Transformer-only endpoint existed, the research lead explicitly authorized continuation despite Transformer-only `clip_fraction=1.0`. Authorization applies only because parameter gradients were finite, the three T/F parameter-gradient ratios remained inside `[0.5, 2.0]`, and overflow/non-finite fractions were both `0.013605`. No value of alpha, lambda, temperature, AMP, gradient clipping, training budget, data, S0, or evaluation protocol may change.

All endpoint reports must carry `CLIP_SATURATION_WARNING` and state that continuous gate-stage clipping can constrain Transformer-only optimization. NaN, training failure, materially increased overflow/non-finite fractions, or an anomalous endpoint stops execution without hyperparameter repair.
