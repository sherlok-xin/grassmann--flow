# Phase 2D Fixed-Composition Teacher Quality Intervention

## Executive judgment

Phase 2D provides strong, seed-consistent evidence that teacher training state and resulting teacher quality affect KD under fixed architecture and fixed Transformer/Grassmann composition. Teacher J and Teacher A were both deployed as `0.5 * Transformer logits + 0.5 * Grassmann logits`; only the trained teacher parameter state changed. Teacher J improved every student, whereas Teacher A slightly harmed every student. The paired primary contrast `Q=NLL_A-NLL_J` was positive for all three seeds and had mean 0.183319 with sample standard deviation 0.001375.

This result does not show that teacher NLL alone causally determines KD effectiveness. Joint continuation changed the teacher representations in both branches. Moreover, Phase 2C already showed that another globally weak condition, the pure-Grassmann alpha-0.0 teacher, transferred positively for all three seeds. The defensible conclusion is that teacher training state and resulting quality strongly modulate transfer in this fixed-composition intervention.

## Teacher integrity and intervention

Teacher J checkpoint SHA256 is `a5ad284f6895c0b4ed516c855c1b149e6636b945173aa2e842b8132299582694`; Teacher A is `43d288db1dba826d8ad1bb3fcbd09b9720bdde24c198e858e00ce4c4658df012`. Both teachers have the same architecture schema, 191 state keys with compatible shapes, source branch run identities and hashes, late-k setting, and logit-fusion implementation. Exactly 179 of 191 state tensors differ. The effective alpha was forced to 0.5 in every evaluation and training run.

Phase 2C C0 and Teacher-J C1 endpoints were reused without retraining. The new Teacher-A D2 arms used the exact seed-specific S0 checkpoints and the frozen C1 protocol: 10 epochs, batch size 32, learning rate `1e-4`, AdamW weight decay 0.01, warmup ratio 0.05, cosine decay, sequence length 256, token-normalized KL, lambda 5, temperature 2, AMP, gradient clipping at 1, full WikiText-2 data, and validation-NLL checkpoint selection.

## Fixed-composition teacher preflight

Both teachers were evaluated on the same 512 WikiText-2 validation chunks and 130,560 prediction tokens. The selected-index SHA256 is `236633612db1e3144b8f3187a5bfb1795f558a50f26368252a8adf3f0b49413f`.

| Teacher | Fused NLL | Fused PPL | Transformer NLL | Grassmann NLL | Fusion gain | Branch JSD | Top-1 agreement | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| J | 4.300644 | 73.7473 | 4.445648 | 4.652064 | 0.145003 | 0.181288 | 0.508395 | 4.402921 |
| A | 6.376220 | 587.7021 | 6.540640 | 6.696787 | 0.164420 | 0.176641 | 0.543719 | 4.551656 |

Joint training improved Transformer branch NLL by 2.094992 and Grassmann branch NLL by 2.044724. The improvements are similar in magnitude, with a 0.050268-NLL larger change in the Transformer branch. Teacher J has slightly higher branch JSD and lower top-1 agreement, which suggests more branch disagreement, but its fusion gain over the better branch is 0.019416 smaller. Thus the clear change is broad branch-quality improvement; the complementarity indicators are mixed and fusion gain itself did not increase.

## Conditional teacher utility

Teacher J has positive mean gold-token utility for every S0 seed. Teacher A is globally much worse than every S0, but its utility remains positive in the highest student-loss quintile for all seeds.

| Teacher | Seed | Global utility | Positive fraction | Hardest-quintile utility | Hardest-quintile positive fraction | Rescue rate | Harm rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| J | 42 | +0.065473 | 0.4848 | +0.803154 | 0.7444 | 0.0671 | 0.1197 |
| J | 123 | +0.078232 | 0.5080 | +0.686087 | 0.7264 | 0.0659 | 0.1108 |
| J | 456 | +0.079843 | 0.5019 | +0.729579 | 0.7296 | 0.0682 | 0.1170 |
| A | 42 | -2.010102 | 0.4076 | +0.305649 | 0.6225 | 0.0543 | 0.5086 |
| A | 123 | -1.997342 | 0.4261 | +0.183726 | 0.5967 | 0.0524 | 0.5011 |
| A | 456 | -1.995731 | 0.4217 | +0.229669 | 0.6049 | 0.0553 | 0.5081 |

Joint training increased hardest-quintile utility by 0.497505, 0.502361, and 0.499910 for seeds 42, 123, and 456. Teacher A therefore retains localized useful information, but that observational signal is insufficient to produce positive full-objective transfer at fixed lambda 5.

## Optimization integrity

The same first eight non-shuffled training batches were used for each teacher within each seed. Full-parameter KD-gradient ratios `G_A/G_J` were 1.102792, 1.148522, and 1.093056; logit-gradient ratios were 1.136287, 1.206410, and 1.168444. None crossed the preregistered `[0.5, 2.0]` scale-mismatch interval, so no D2-scale arm was created.

The original 2,000-line Teacher-A smoke failed because clipping was 1.00 and AMP overflow/non-finite fractions were 0.16. This failure is retained. After explicit research-lead authorization, a matched full-data one-epoch J/A gate was added because the inherited reduced-data Teacher-J smoke had the same 25-step behavior despite stable full-data training. In the amended gate, Teacher J/A clipping fractions were 0.234694/0.275510, and both had overflow/non-finite fractions 0.013605. Both checkpoints reloaded and all identity checks passed.

Formal Teacher-A epoch-1 clipping fractions were 0.666667, 0.503401, and 0.513605 for seeds 42, 123, and 456. Overflow and non-finite fractions were 0.013605 for every seed. These values are below the frozen mismatch thresholds and do not materially confound the endpoint comparison.

## Formal endpoints

Positive gain means that KD improves on the matched WS+CE control. The primary contrast is `Q=Gain_J-Gain_A=NLL_A-NLL_J`.

| Seed | C0 test NLL | Teacher J test NLL | Teacher A test NLL | Gain J | Gain A | Q |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 4.270742 | 4.108241 | 4.293136 | +0.162501 | -0.022394 | +0.184895 |
| 123 | 4.277520 | 4.130183 | 4.312882 | +0.147337 | -0.035362 | +0.182699 |
| 456 | 4.280830 | 4.123673 | 4.306036 | +0.157157 | -0.025206 | +0.182364 |

| Quantity | Mean | Sample SD | Min | Max | Sign consistency |
|---|---:|---:|---:|---:|---|
| Gain J | +0.155665 | 0.007692 | +0.147337 | +0.162501 | 3/3 positive |
| Gain A | -0.027654 | 0.006822 | -0.035362 | -0.022394 | 3/3 negative |
| Q | +0.183319 | 0.001375 | +0.182364 | +0.184895 | 3/3 positive |

The result matches preregistered Outcome A. The effect is practically nontrivial and unusually consistent across the tested student initializations. Teacher A crosses the transfer boundary: although it contains positive utility on hard validation tokens, fixed-lambda KD from Teacher A degrades every endpoint relative to its matched CE continuation.

## Decision 1 — Does teacher quality/training state affect KD under fixed Transformer/Grassmann composition?

YES. `Q` is positive for all three seeds, with mean 0.183319 and a narrow range of 0.182364--0.184895. Under the fixed protocol, Teacher J consistently transfers better than Teacher A.

## Decision 2 — Does the alpha-only teacher still provide positive KD transfer?

NO. `Gain_A` is negative for all three seeds: -0.022394, -0.035362, and -0.025206 NLL.

## Decision 3 — Is the fixed-composition effect robust across independent student seeds?

YES. All three independently trained S0 checkpoints show positive `Q`, positive `Gain_J`, and negative `Gain_A`.

## Decision 4 — Is the result materially confounded by KD gradient scale?

NO. The preflight parameter-gradient ratios are 1.093--1.149, the matched full-data stability gate passed for both teachers, and every formal run stayed below the clipping and AMP mismatch thresholds.

## Decision 5 — Does joint teacher training improve branch quality, fusion complementarity, or both?

Joint training clearly improves both branch qualities by about 2.0 NLL, with a slightly larger Transformer improvement. Complementarity indicators are mixed: Teacher J increases JSD and reduces agreement, but its fusion gain over the better branch decreases from 0.164420 to 0.145003. The evidence therefore supports broad branch-quality improvement, not an increased fusion-gain margin.

## Decision 6 — Is global teacher NLL now sufficient to explain KD usefulness?

NO. Global NLL correctly orders J and A in this intervention, but Phase 2C showed positive transfer from a globally worse alpha-0.0 teacher. Conditional utility and composition still affect whether a globally weak teacher is useful.

## Decision 7 — What is the single highest-information next experiment?

A. Transformer-only vs Grassmann-only vs fused-teacher KD. This directly separates the contributions of the two teacher branches and fusion after Phase 2D established that trained teacher state matters. It has higher mechanism information than another domain replication or an immediate conditional intervention. This experiment is selected but not launched.
