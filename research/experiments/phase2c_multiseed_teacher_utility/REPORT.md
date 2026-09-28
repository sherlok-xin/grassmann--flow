# Phase 2C Multiseed Teacher Utility Audit

## Executive judgment

The alpha-0.5 versus alpha-0.0 KD difference is robust across the three independently trained student initializations. The paired contrast `D = NLL_C3 - NLL_C1` is positive for seeds 42, 123, and 456, with mean 0.114674 NLL and sample standard deviation 0.001001. T_a00 also remains a positive-transfer teacher for every seed even though its validation NLL is worse than the seed-matched C0 endpoint. Teacher residual advantage therefore predicts the magnitude of transfer in this intervention, but it does not determine the sign of transfer.

The offline audit gives partial evidence for a complementarity explanation. T_a00 has negative mean gold-token utility overall and on the full student-error partition, but positive mean utility in the hardest student-loss quintile for all three seeds. Its top-1 rescue rate is small, so this pattern is conditional gold-token information rather than a demonstrated correction mechanism.

## Replication integrity

Seeds 123 and 456 were trained independently from scratch with the canonical 20-epoch WikiText-2 Hybrid-lite CE recipe. Their selected S0 checkpoint SHA256 values are `a44b64ba5a84d3bd2708a96aad4eae829ff7ab0f00476f0e31146fc314da1f13` and `3fbe985d6474c84d1c61fd7937ea906102cf460846829eb3af4c709593b01757`; the seed-42 hash is `9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef`. The three hashes are distinct.

Every continuation arm uses the same frozen teacher checkpoint, temperature 2, token-normalized KD, and the Phase 2B schedule. Within each seed, C0, C1, and C3 start from the exact same S0 checkpoint. C0 is WS+CE, C1 uses T_a05 with alpha 0.5 and lambda 5, and C3 uses T_a00 with alpha 0.0 and lambda 5. Alpha 0.3 was not retrained. The collector verified the S0 hashes, effective teacher alpha, lambda, and matched configuration fields for all nine reused or new arms.

## Primary endpoints

| Seed | C0 test NLL | C1 T_a05 test NLL | C3 T_a00 test NLL | Delta_KD_a05 | Delta_KD_a00 | D |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 4.270742 | 4.108241 | 4.223972 | +0.162501 | +0.046771 | +0.115731 |
| 123 | 4.277520 | 4.130183 | 4.243922 | +0.147337 | +0.033598 | +0.113739 |
| 456 | 4.280830 | 4.123673 | 4.238226 | +0.157157 | +0.042604 | +0.114553 |

| Quantity | Mean | Sample SD | Minimum | Maximum | Sign consistency |
|---|---:|---:|---:|---:|---|
| Delta_KD_a05 | 0.155665 | 0.007692 | 0.147337 | 0.162501 | 3/3 positive |
| Delta_KD_a00 | 0.040991 | 0.006733 | 0.033598 | 0.046771 | 3/3 positive |
| D | 0.114674 | 0.001001 | 0.113739 | 0.115731 | 3/3 positive |

The result is unusually consistent at the observed scale: the range of D is only 0.001992 NLL. This is descriptive evidence from three seeds, not a significance claim.

## Seed-matched teacher residual advantage

| Seed | Delta_teacher_a05 | Classification | Delta_teacher_a00 | Classification | Ordered |
|---:|---:|---|---:|---|---|
| 42 | +0.084898 | good | -0.266520 | bad | yes |
| 123 | +0.097926 | good | -0.253492 | bad | yes |
| 456 | +0.100412 | good | -0.251006 | bad | yes |

Residual advantage remains ordered for every seed. The neutral names T_a05 and T_a00 are retained because the interventions were selected using seed-42 validation results, although their residual classifications happen to remain unchanged.

## Optimization integrity

The new C1/C3 arms have epoch-1 clipping fractions of 0.367/0.463 for seed 123 and 0.378/0.497 for seed 456. AMP overflow and non-finite fractions are 0.0136 for all four arms. Initial KD logit-gradient norms are 0.0999/0.1055 for seed 123 and 0.1108/0.1157 for seed 456, close to the seed-42 values 0.1155/0.1189. No arm is clip-saturated, no arm crosses the instability threshold, and no initial KD-gradient ratio falls outside the preregistered 0.5--2.0 interval. There is no qualitative optimization mismatch to average away.

## Offline teacher utility

The same 512 WikiText-2 validation chunks and 130,560 prediction tokens are used for every S0 student. The selected-index SHA256 is `236633612db1e3144b8f3187a5bfb1795f558a50f26368252a8adf3f0b49413f`.

| Seed | Alpha | Mean utility | Fraction positive | Rescue rate | Harm rate | KL at T=1 | CE-KD cosine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.5 | +0.065472 | 0.4848 | 0.0671 | 0.1195 | 0.5346 | -0.0740 |
| 42 | 0.3 | -0.019102 | 0.4659 | 0.0650 | 0.1303 | 0.5671 | -0.0716 |
| 42 | 0.0 | -0.285946 | 0.4010 | 0.0600 | 0.1676 | 0.7994 | -0.0722 |
| 123 | 0.5 | +0.078231 | 0.5079 | 0.0660 | 0.1106 | 0.4749 | -0.0526 |
| 123 | 0.3 | -0.006342 | 0.4847 | 0.0634 | 0.1204 | 0.5109 | -0.0503 |
| 123 | 0.0 | -0.273187 | 0.4126 | 0.0590 | 0.1593 | 0.7406 | -0.0550 |
| 456 | 0.5 | +0.079842 | 0.5019 | 0.0682 | 0.1168 | 0.5062 | -0.0593 |
| 456 | 0.3 | -0.004731 | 0.4816 | 0.0655 | 0.1265 | 0.5394 | -0.0573 |
| 456 | 0.0 | -0.271575 | 0.4139 | 0.0601 | 0.1632 | 0.7741 | -0.0606 |

Global utility follows teacher quality, but it does not explain why T_a00 produces a consistent positive endpoint gain. The conditional analysis narrows the candidate explanation. For T_a00, mean utility on all student-wrong tokens remains negative at -0.2306, -0.2485, and -0.2339. In the highest student-loss quintile, however, mean utility becomes +0.2736, +0.1660, and +0.2003, and the positive-utility fractions are 0.5746, 0.5460, and 0.5567. Top-1 rescue within this quintile is nearly zero. The teacher therefore supplies improved probability to the gold token on a concentrated hard-token region without usually changing the top-1 prediction.

This observation is compatible with useful soft-target structure on difficult tokens, but it is not causal evidence that those tokens produce the final KD gain. A masked or fixed-composition intervention would be needed for a mechanism claim.

## Fixed-composition audit

The repository contains two naturally existing teachers that preserve architecture, source branch identities, and evaluation alpha while differing in branch training. The alpha-only checkpoint has validation NLL 6.376223 at fixed alpha 0.5, while the jointly trained checkpoint has 4.300646 on the same subset. Their source checkpoints and fusion composition are matched. This is a valid minimum two-condition candidate for a future fixed-composition teacher-quality intervention, although the quality gap is coarse and also represents frozen versus jointly continued branch weights. No Phase 2D training was launched.

## Limitations

The three seeds vary student initialization only; teacher-seed variability is not measured. The utility audit is observational and uses a fixed validation subset. Alpha changes the teacher distribution and, in the alpha-0.0 condition, removes Transformer contribution, so the completed alpha intervention does not isolate quality from composition. The fixed-composition checkpoints identified by the audit can address that remaining confound, but they have not yet been distilled in a matched experiment.

## Decision 1 — Is the alpha-0.5 versus alpha-0.0 KD difference robust across independent student seeds?

YES

D is positive for all three seeds and varies by only 0.001992 NLL across them.

## Decision 2 — Does alpha 0.0 remain a positive-transfer teacher across all student seeds?

YES

Delta_KD_a00 is positive for all three seeds, with mean 0.040991 NLL.

## Decision 3 — Does teacher residual advantage remain ordered consistently across seeds?

YES

Delta_teacher_a05 is positive and greater than Delta_teacher_a00 for every seed; Delta_teacher_a00 remains negative.

## Decision 4 — Is global teacher NLL sufficient to explain teacher usefulness?

NO

Global teacher quality explains the relative ordering of KD gain but not the consistent positive transfer from T_a00 despite negative residual advantage. Conditional utility reveals information not represented by a single global NLL.

## Decision 5 — Is there evidence that the globally weaker teacher contains useful information concentrated on student-weak tokens?

PARTIAL

The hardest student-loss quintile has positive mean utility and a positive-utility majority for every seed. However, the full student-error partition has negative mean utility, top-1 rescue is small, and the audit does not causally link the hard-token utility to endpoint improvement.

## Decision 6 — Does the repository contain a valid fixed-composition teacher-quality intervention for Phase 2D?

YES

The minimum conditions are the alpha-only and jointly trained teacher checkpoints, both evaluated and distilled at fixed alpha 0.5 with the same architecture and source branch identities.

## Decision 7 — What is the single most informative next experiment?

A. fixed-composition teacher-quality intervention

This experiment directly separates teacher quality from the alpha/composition change that remains confounded in Phase 2B/2C. It has higher information value than adding more domains or developing a new routing method before the teacher-quality explanation is isolated.

STOP. Phase 2D was not launched.
