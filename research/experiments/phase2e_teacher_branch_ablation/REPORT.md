# Phase 2E Teacher Branch Ablation

Status: complete. Phase 2E ran only the three missing Transformer-only formal arms and reused the matched C0, fused, and Grassmann-only endpoints from Phase 2C. No additional alpha, teacher, loss, dataset, or student condition was trained.

## Scientific question and claim boundary

Phase 2E tests whether KD from a fixed jointly trained heterogeneous Transformer--Grassmann teacher transfers more effectively than KD from either component branch alone. All three teacher conditions come from checkpoint SHA256 `a5ad284f6895c0b4ed516c855c1b149e6636b945173aa2e842b8132299582694`; only the effective logit mixture changes. This experiment can measure an incremental downstream benefit from adding the Grassmann branch to Transformer supervision, but it cannot distinguish Grassmann geometry from generic ensemble diversity.

## Frozen protocol and authorized amendment

The new Transformer-only arms use alpha 1.0, token-mean KD, lambda 5, temperature 2, 10 epochs, batch size 32, learning rate `1e-4`, weight decay 0.01, warmup ratio 0.05, cosine scheduling, sequence length 256, AMP, gradient clipping at 1.0, full WikiText-2 data, seed-matched S0 checkpoints, and validation-NLL checkpoint selection. The Phase 2C C0, fused alpha-0.5, and Grassmann-only alpha-0.0 arms were not rerun.

The reduced smoke failed with clipping 1.00 and overflow/non-finite fractions 0.16. In the matched full-data gate, fused passed with clipping 0.2347, while Transformer-only retained clipping 1.00 but had finite gradients, T/F parameter-gradient ratios of 1.3164/1.2788/1.2324, and overflow/non-finite fractions of 0.013605. Before any formal endpoint existed, the research lead explicitly authorized continuation without changing any training or evaluation setting.

`CLIP_SATURATION_WARNING`: Transformer-only gradients continuously triggered clipping during the full-data gate. Formal endpoint differences may therefore partly reflect an optimization constraint in addition to teacher-source differences. This warning is permanent.

## Teacher and utility preflight

The deterministic validation subset contains 512 chunks with selected-index SHA256 `236633612db1e3144b8f3187a5bfb1795f558a50f26368252a8adf3f0b49413f`.

| Condition | Alpha | Teacher NLL | Teacher PPL | Mean residual advantage over S0 |
|---|---:|---:|---:|---:|
| Fused | 0.5 | 4.300646 | 73.747 | +0.074515 |
| Transformer-only | 1.0 | 4.445647 | 85.255 | -0.070486 |
| Grassmann-only | 0.0 | 4.652064 | 104.801 | -0.276903 |

Transformer assigned higher gold-token probability than Grassmann on 51.56% of validation tokens, while Grassmann was higher on 48.44%. In the hardest student-loss quintile, mean utility was 0.6356 for Transformer-only, 0.2133 for Grassmann-only, and 0.7396 for fused. These are descriptive observations, not causal token subsets.

## Formal endpoints

All new runs completed 10 epochs without NaN or training failure. Validation selection chose epoch 10 for every seed. The maximum AMP overflow and non-finite fraction remained 0.013605 for all three runs.

| Seed | C0 test NLL | Fused test NLL | Transformer test NLL | Grassmann test NLL |
|---:|---:|---:|---:|---:|
| 42 | 4.270742 | 4.108241 | 4.127957 | 4.223972 |
| 123 | 4.277520 | 4.130183 | 4.152294 | 4.243922 |
| 456 | 4.280830 | 4.123673 | 4.144237 | 4.238226 |

The endpoint ordering is fused < Transformer-only < Grassmann-only < C0 for every seed. Equivalently, the KD gain ordering is fused > Transformer-only > Grassmann-only > 0 for all three independent S0 checkpoints.

## Paired gains and contrasts

| Metric | Seed 42 | Seed 123 | Seed 456 | Mean | Sample SD | Sign consistency |
|---|---:|---:|---:|---:|---:|---:|
| `Gain_F` | 0.162501 | 0.147337 | 0.157157 | 0.155665 | 0.007692 | 3/3 positive |
| `Gain_T` | 0.142785 | 0.125226 | 0.136593 | 0.134868 | 0.008906 | 3/3 positive |
| `Gain_G` | 0.046771 | 0.033598 | 0.042604 | 0.040991 | 0.006733 | 3/3 positive |
| `C_FT` | 0.019716 | 0.022110 | 0.020564 | 0.020797 | 0.001214 | 3/3 positive |
| `C_FG` | 0.115731 | 0.113739 | 0.114553 | 0.114674 | 0.001001 | 3/3 positive |
| `C_TG` | 0.096015 | 0.091629 | 0.093989 | 0.093877 | 0.002195 | 3/3 positive |

The primary `C_FT` contrast is positive for every seed and has mean 0.020797, narrowly exceeding the preregistered 0.02 practical threshold by 0.000797 NLL. The evidence therefore meets the frozen YES rule, but its magnitude is boundary-close and must not be described as a large effect. The much larger `C_FG` and `C_TG` contrasts establish that Grassmann-only supervision is less transferable than both fused and Transformer-only supervision under this protocol.

## Optimization integrity and clipping interpretation

Transformer-only clipping was strongest early and declined during training. The mean epoch-level clipping fractions were 0.4990, 0.4605, and 0.4398 for seeds 42, 123, and 456; each run ranged from 1.00 in epoch 1 to 0.0034 in later epochs. Overflow/non-finite fractions never exceeded 0.013605. Thus the formal runs are numerically valid rather than failed or NaN-confounded, but the gate-stage saturation and early optimization constraint remain a credible alternative contributor to the small fused-versus-Transformer gap.

This experiment supports the bounded statement that adding the Grassmann branch to the fixed jointly trained Transformer teacher yields a small, stable additional KD gain under the tested protocol. It does not show that Grassmann geometry uniquely causes the gain, nor that the same gain would persist under a homogeneous ensemble or a clipping-insensitive optimization control.

## Figures and artifacts

`figures/fig1_paired_kd_gain` shows paired seed gains for Grassmann-only, Transformer-only, and fused KD. `figures/fig2_teacher_nll_vs_kd_gain` shows the three teacher conditions without fitting a three-point regression. `figures/fig3_hard_token_utility` shows hardest-quintile utility with individual S0 seeds visible. Vector PDF and 300-dpi PNG versions are provided. Compact configs, summaries, epoch metrics, checkpoint hashes, preflight tables, smoke summaries, and the strict result collector are stored in this directory; datasets, checkpoints, outputs, and full logs remain excluded from GitHub.

## Decision 1 — Does fused-teacher KD outperform Transformer-only KD?

YES

## Decision 2 — Does fused-teacher KD outperform Grassmann-only KD?

YES

## Decision 3 — Does Transformer-only KD provide positive transfer despite its global teacher quality relative to the student?

YES

## Decision 4 — Does Grassmann-only KD provide positive transfer across all independent student seeds?

YES

## Decision 5 — Does adding the Grassmann branch provide measurable downstream transfer beyond Transformer-only teacher supervision?

YES

## Decision 6 — Can the fused-teacher advantage be attributed specifically to Grassmann geometry?

INCONCLUSIVE

## Decision 7 — What is the single highest-information next experiment?

B. homogeneous Transformer+Transformer ensemble teacher control versus Transformer+Grassmann teacher
