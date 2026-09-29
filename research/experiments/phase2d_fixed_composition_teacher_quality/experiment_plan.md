# Phase 2D Fixed-Composition Teacher Quality Intervention

## Frozen scientific question

This phase tests whether teacher training state and resulting teacher quality modulate downstream knowledge distillation when teacher architecture, source branch identities, fusion semantics, and Transformer/Grassmann composition are fixed. It does not test a universal law that a lower-NLL teacher must always produce a better student.

The domain is WikiText-2 only. The student objective is unchanged: `CE + 5 * KL_token_mean`, with temperature 2. No new loss, routing rule, architecture, dataset, or manuscript claim is introduced.

## Frozen teacher conditions

Teacher J is `outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint/checkpoints/hybrid_best.pt`, expected SHA256 `a5ad284f6895c0b4ed516c855c1b149e6636b945173aa2e842b8132299582694`. Teacher A is `outputs/hybrid_experiments/20260328_084319_wt2_v1_hybrid_alpha_only/checkpoints/hybrid_best.pt`, expected SHA256 `43d288db1dba826d8ad1bb3fcbd09b9720bdde24c198e858e00ce4c4658df012`.

Both conditions use `--teacher-alpha-override 0.5`. Thus the deployed teacher logits are always `0.5 * Transformer logits + 0.5 * Grassmann logits`. The checkpoint-stored alpha is not the intervention.

Teacher J and Teacher A must have identical architecture schema, source branch run identities, source checkpoint hashes, compatible state-dict keys and shapes, and the same fusion implementation. The expected source hashes are `6916246499dc2e6066c21760421571c76d9f63f1006fab7de838534ef2f733f4` for Grassmann and `4c166d20245bd923dbe126b3f9be39dbb1947d568358fd6cc152c7b9faaafe6c` for Transformer. The expected checkpoint-state difference is 179 of 191 tensors. Any failed invariant stops the experiment without checkpoint substitution.

## Reused controls and student initializations

Phase 2C C0 and C1 endpoints are reused for seeds 42, 123, and 456. They are not retrained. The new D2 arm changes only the teacher run from Teacher J to Teacher A.

| Seed | S0 run | Required checkpoint SHA256 |
|---:|---|---|
| 42 | `outputs/hybrid_experiments/20260515_004119_wt2_hybrid_lite_baseline_224x56_l6_e20` | `9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef` |
| 123 | `outputs/hybrid_experiments/20260922_071001_phase2c_wt2_s0_seed123` | `a44b64ba5a84d3bd2708a96aad4eae829ff7ab0f00476f0e31146fc314da1f13` |
| 456 | `outputs/hybrid_experiments/20260922_071449_phase2c_wt2_s0_seed456` | `3fbe985d6474c84d1c61fd7937ea906102cf460846829eb3af4c709593b01757` |

## Offline preflight

Both teachers are evaluated at alpha 0.5 on the deterministic 512-chunk WikiText-2 validation subset selected with seed 20260920. The required selected-index SHA256 is `236633612db1e3144b8f3187a5bfb1795f558a50f26368252a8adf3f0b49413f`. The preflight reports fused and branch NLL, fused PPL, fusion gain over the better branch, branch JSD, branch top-1 agreement, fused entropy, branch gold probabilities, and seed-specific teacher-student utility, KL, hardest-quintile utility, positive-utility fraction, rescue rate, and harm rate. The conditions are frozen before these values are observed.

Gradient-scale calibration uses the first eight non-shuffled training batches of size two, for 4,080 prediction tokens per teacher/student pair. It reports unweighted token-mean KL, the contribution at lambda 5, per-token CE and KD logit-gradient norms, CE-KD cosine, negative-cosine fraction, and the full student-parameter gradient norm of the unweighted KD loss. No optimizer step is taken and no test data are used.

For each seed, `R_logit = G_KD_A / G_KD_J` and `R_param = G_param_A / G_param_J`. A ratio outside `[0.5, 2.0]` is flagged `KD_SCALE_MISMATCH`. Full parameter gradients are primary; logit gradients are the fallback only if parameter gradients are unavailable.

## Smoke gate

Before formal training, seed 42 receives a one-epoch, 2,000-line Teacher-A smoke with the otherwise frozen D2 settings. The formal matrix proceeds only if all losses are finite, serialization and reload complete, alpha equals 0.5, the S0 hash matches, AMP overflow and non-finite fractions are below 0.05, clipping fraction is below 0.95, and no persistent numerical instability occurs.

### Authorized protocol amendment (2026-09-29)

The reduced-data smoke failed the clipping and AMP-frequency thresholds. The inherited Teacher-J reduced-data smoke had the same clipping and overflow frequencies, whereas its full-data epoch was stable. The research lead therefore authorized one matched full-data, one-epoch stability comparison before any formal endpoint training. This amendment does not replace or erase the original gate outcome.

The amended gate runs Teacher J and Teacher A from the same seed-42 S0 checkpoint with identical formal settings, data order, effective alpha 0.5, lambda 5, and one complete WikiText-2 training epoch. Only the teacher checkpoint differs. Formal D2 training is authorized only if both runs have finite reported losses, reloadable checkpoints, exact alpha and S0 identity, AMP overflow and non-finite fractions below 0.05, and clipping fraction below 0.95. The gate is an optimization-integrity check; validation and test NLL from these one-epoch runs are not used to select the teacher, tune a hyperparameter, or change the frozen endpoint contrast.

For the scale-control rule, "close to clipping saturation" is frozen as clipping fraction at least 0.90. If `KD_SCALE_MISMATCH` occurs and the fixed-lambda smoke is close to saturation, a seed-42-only D2-scale diagnostic is created with `lambda_A_scale = 5 * mean(G_param_J) / mean(G_param_A)`, fixed from calibration before endpoint training. It is not selected using validation or test performance. Otherwise no scale-control arm is created.

## Formal D2 configuration

Each new D2 arm uses Teacher A at effective alpha 0.5, the seed-specific S0 checkpoint above, 10 epochs, batch size 32, learning rate 1e-4, AdamW weight decay 0.01, warmup ratio 0.05, cosine decay, sequence length 256, token-normalized KL, lambda 5, temperature 2, AMP, gradient clipping at 1, the same WikiText-2 preprocessing, the existing seeded training-loader policy, and validation-NLL checkpoint selection. Test data are used only for the final selected endpoint.

For seed `s`, `Gain_J(s) = NLL_C0(s) - NLL_J(s)`, `Gain_A(s) = NLL_C0(s) - NLL_A(s)`, and the frozen primary contrast is `Q(s) = Gain_J(s) - Gain_A(s) = NLL_A(s) - NLL_J(s)`. No endpoint is redefined after observing results.

## Execution and stopping

Heavy computation runs only on server `10.42.0.197`. Failed and retried runs remain preserved with explicit status. Datasets and historical checkpoints are not modified. After the report, compact artifacts, plots, metadata, tests, and source changes may be committed, while datasets, checkpoints, raw logits, and full high-volume logs remain outside GitHub. Phase 2D ends after choosing one next experiment; that experiment is not launched automatically.
