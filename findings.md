# Findings

更新时间：2026-09-16

## Current understanding

The strongest completed result is a PTB quality--efficiency trade-off produced by warm-start KD from a heterogeneous late-fusion teacher. The 31.434M student reaches 51.84 PPL versus 50.11 for the 36.264M teacher, while the 23.565M student reaches 53.87 PPL with a larger latency gain. The best-student result and the trade-off result both have small multi-seed variation.

The cross-domain pattern is not yet confirmatory. Legacy KD helps the warm-start student on PTB, WikiText-2, and CodeParrot but hurts it on TinyStories. Most points are single-seed, and the KL implementation sums across sequence positions before dividing by batch size. As a result, the coefficient is coupled to sequence length and domain-dependent teacher entropy.

The new loss API now preserves the legacy objective and adds valid-token-normalized KL. In the PTB one-epoch smoke, token-normalized `lambda_kd=5` reproduced legacy `alpha=0.02` closely: validation PPL was 61.73 versus 61.69 and test PPL was 53.29 versus 53.26. Their finite gradient norms and AMP overflow rates also matched. This is expected because the exact coefficient conversion at 255 prediction positions is 5.204.

## Patterns and insights

Warm-start is empirically important in the completed PTB runs, but the historical random-init controls are not matched closely enough to establish causality. Small legacy coefficients work because the raw KL is one to two orders of magnitude larger than CE. The TinyStories failure is the most useful boundary condition for a new contribution: a heterogeneous teacher should not be trusted uniformly when its branches are uncertain or disagree.

The strongest one-epoch PTB point was `lambda_kd=10`, but it clipped or overflowed on 54.7% of steps. It was rejected under the preregistered stability rule. `lambda_kd=5` is the fixed coefficient for confirmatory runs; it must not be retuned independently on TinyStories.

## Lessons and constraints

Do not compare runs by experiment name alone. Check effective dataset path, token counts, initialization checkpoint, learning rate, epoch count, model shape and loss reduction. CodeParrot runs may carry incorrect `ptb` or `wikitext2` metadata. Do not report CUDA speedups until correctness and import tests pass. Do not describe the 31.434M student as strong compression relative to the 36.264M teacher.

## Open questions

The first open question is whether the domain pattern survives token-normalized KD and matched seeds. The second is whether teacher branch disagreement predicts token-level student harm. The third is whether branch-aware weighting or branch-aligned distillation improves the Pareto frontier without additional student parameters.
