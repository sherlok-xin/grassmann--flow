# Phase 2B Controlled Teacher Intervention

## Executive judgment

Phase 2B provides single-seed controlled intervention evidence that teacher residual advantage changes the magnitude of KD transfer in the predicted direction. On WikiText-2, the teacher conditions span good, near, and bad validation residual advantage while every training arm starts from the same S0 checkpoint and uses a matched optimization protocol. The resulting KD gains are strictly ordered as `T_good > T_near > T_bad`.

The evidence is partial rather than strong. The clearly bad teacher still improves over matched WS+CE, so the intervention does not cross the positive-transfer/negative-transfer boundary. This pilot therefore supports a graded association between residual advantage and KD benefit under the tested protocol, but it does not establish a deterministic sign law or statistical generality.

## Preflight and design

The mandatory validation-only preflight found a complete triplet within the frozen WikiText-2 teacher branch pair. T_good uses alpha 0.5 with `Delta_teacher=+0.084900`, T_near uses alpha 0.3 with `Delta_teacher=+0.000330`, and T_bad uses alpha 0.0 with `Delta_teacher=-0.266510`. Alpha denotes the Transformer logit weight, so T_good and T_near are heterogeneous fusions while T_bad is the pure Grassmann branch. Teacher selection used no test outcomes.

All four arms use seed 42 and the exact S0 checkpoint with SHA256 `9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef`. The student architecture, data and ordering policy, optimizer, scheduler, batch size, sequence length, epoch count, AMP behavior, and validation-based checkpoint selection are matched. C0 uses WS+CE. C1, C2, and C3 use `CE + 5 * KL_token_mean` at temperature 2. The collector verified all recorded invariants before producing the consolidated table.

## Primary results

| Arm | Teacher | Delta_teacher | Best epoch | Val NLL | Test NLL | Test PPL | Delta_KD |
|---|---|---:|---:|---:|---:|---:|---:|
| C0 | WS+CE | +0.084900 | 1 | 4.383154 | 4.270742 | 71.5748 | 0.000000 |
| C1 | T_good, alpha 0.5 | +0.084900 | 9 | 4.216686 | 4.108241 | 60.8396 | +0.162501 |
| C2 | T_near, alpha 0.3 | +0.000330 | 8 | 4.250850 | 4.139916 | 62.7975 | +0.130826 |
| C3 | T_bad, alpha 0.0 | -0.266510 | 8 | 4.340190 | 4.223972 | 68.3042 | +0.046771 |

Here `Delta_KD = test_NLL_C0 - test_NLL_arm`, so larger positive values indicate stronger transfer. The predicted ordering is exact: C1 exceeds C2 by 0.031675 NLL, C2 exceeds C3 by 0.084056 NLL, and the good-to-bad contrast is 0.115731 NLL. Relative to C0, test PPL decreases by 10.7351 for C1, 8.7772 for C2, and 3.2705 for C3.

The outcome does not satisfy the ideal sign pattern. T_good and T_near improve the student, but T_bad also yields a smaller positive gain. The experiment therefore shows that residual advantage modulates the strength of KD under this fixed protocol, not that negative teacher residual advantage necessarily produces negative transfer.

## Loss-scale and optimization audit

The initial S0 KD logit-gradient norms are 0.115484, 0.115873, and 0.118931 for C1, C2, and C3, respectively. Their maximum difference is about 3.0% relative to C1. Initial CE logit-gradient norm is identical at 0.802336. The initial CE-KD cosine values are also similar at -0.074001, -0.071588, and -0.072195.

Epoch-1 weighted KD contributions are 2.544399, 2.731717, and 3.642114. C3 consequently receives a larger loss contribution, as expected from its greater teacher-student KL. However, the corresponding total parameter-gradient norms are 1.373554, 1.333842, and 1.417159, while clipping fractions are 0.547619, 0.394558, and 0.585034. All three KD arms have identical AMP overflow and non-finite fractions of 0.013605 in epoch 1. C3 is neither clip-saturated nor unstable, and its total gradient norm is only about 3.2% larger than that of C1. After epoch 1, clipping and overflow rapidly return to low levels.

The formal runs therefore do not meet the preregistered `LOSS_SCALE_CONFOUNDER` criterion. Loss scale remains a limitation because the teacher intervention necessarily changes KL magnitude, but it is unlikely to be the primary explanation for the ordered endpoint outcomes. No lambda was retuned.

## Interpretation and scope

The controlled result is consistent with the Phase 2A hypothesis: reducing teacher residual advantage weakens the benefit of fixed KD. It upgrades the earlier cross-domain observation to a within-dataset intervention, but only for one seed and one branch pair. It does not prove that residual advantage alone determines transfer, and it does not establish statistical significance.

The architectural interpretation is also limited. C1 and C2 are heterogeneous fusions, whereas C3 is pure Grassmann. The experiment changes teacher distribution quality through alpha but does not independently randomize teacher composition. It supports neither Grassmann-specific causality nor a claim that Grassmann geometry causes the KD ordering.

The minimum informative replication is C0/C1/C3 for seeds 123 and 456. C2 is not required for the next confirmation because the decisive contrast is the clear good-versus-bad separation, while C0 is required to preserve the transfer reference. No replication was launched in Phase 2B.

Primary machine-readable evidence is in `results.csv` and `raw/formal_summary.json`. Validation and scale plots are provided as PDF and 300-dpi PNG files under `figures/`. Run configs, summaries, metrics, and reports are copied under `raw/formal/C0` through `raw/formal/C3`.

## Decision 1 — Did manipulating teacher residual advantage change KD outcome in the predicted direction?

PARTIAL

KD gain decreases strictly from T_good to T_near to T_bad, but all three teachers still produce positive transfer.

## Decision 2 — Did any teacher cross the positive-transfer / negative-transfer boundary?

NO

T_bad has negative residual advantage but still improves test NLL over C0 by 0.046771.

## Decision 3 — Can the result be explained primarily by loss-scale or clipping differences?

UNLIKELY

C3 has the largest weighted KD loss, but initial KD logit-gradient norms are closely matched, no arm is clip-saturated, total gradient norms are similar, and instability rates are identical and low.

## Decision 4 — Is a three-seed replication justified?

YES

Replicate only C0, C1, and C3 for seeds 123 and 456. These are the minimum arms needed to test the 0.115731-NLL good-to-bad contrast against the matched CE reference.

## Decision 5 — What should be tested next?

A. multi-seed replication of the controlled teacher intervention

The preregistered direction holds with a practically clear good-to-bad separation and without a dominant optimization confound, but the current evidence is only seed 42. Replication is required before treating the intervention as a stable empirical result.
