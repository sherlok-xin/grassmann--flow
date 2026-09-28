# Phase 2B Preregistered Interpretation

This document is frozen before the controlled training runs are launched. Phase 2B uses Route A on WikiText-2 with a common seed-42 S0 parameter tensor. The only intended intervention across KD arms is the fixed alpha used to combine the same frozen Transformer and Grassmann teacher branches.

The primary outcome is `Delta_KD = test_NLL_C0 - test_NLL_arm`, where positive values indicate improvement over matched WS+CE. Checkpoints are selected by validation NLL only. Test results will not be used to choose teachers, epochs, or hyperparameters.

## Conditions

C0 is WS+CE with `lambda_kd=0`. C1 uses T_good at alpha 0.5, whose validation residual advantage is +0.084900 nat/token. C2 uses T_near at alpha 0.3, whose residual advantage is +0.000330. C3 uses T_bad at alpha 0.0, whose residual advantage is -0.266510. C3 is the pure Grassmann branch; C1 and C2 are heterogeneous fusions. No pure-Transformer arm is part of the primary experiment.

## Strong support

Strong support for the residual-advantage hypothesis requires KD gain to decrease monotonically as `C1 > C2 > C3`. Ideally C1 produces positive transfer, C2 little or no transfer, and C3 negative transfer.

## Partial support

Partial support means that teacher condition changes KD gain in the expected ordering but the values do not cross zero, or one adjacent pair is effectively tied while the good-to-bad contrast remains directionally clear.

## Contradiction

The result contradicts the hypothesis if C3 performs as well as or better than C1 under the matched protocol. A non-monotonic result in which the clearly worse teacher produces the largest gain also counts as contradiction unless a preregistered confound makes the outcome inconclusive.

## Inconclusive

The result is inconclusive if loss-scale, clipping, AMP overflow, non-finite gradients, checkpoint failure, or data/configuration mismatch dominates the comparison. In particular, an arm is flagged `LOSS_SCALE_CONFOUNDER` when its epoch-1 weighted KD contribution or KD logit-gradient norm is materially larger than the other KD arms and this coincides with substantially higher clipping or instability. Lambda will not be retuned in response.

Because the preflight S0 diagnostics already show similar mean KD logit-gradient norms across C1/C2/C3 (0.11548/0.11587/0.11893), large optimization differences are not expected at initialization. This expectation is descriptive and does not alter the decision rules.

## Replication rule

Seed 42 is a mechanism pilot. Three-seed replication is justified only if the good-to-bad intervention follows the preregistered direction with a practically clear separation and is not dominated by `LOSS_SCALE_CONFOUNDER`. If justified, only the minimum decisive teacher arms and matched CE control will be replicated. Mixed or very small effects will not trigger automatic replication.

## Interpretation boundary

The experiment intervenes on the distribution produced by a frozen heterogeneous teacher. It may support a teacher-quality mechanism, but it cannot establish that Grassmann geometry causes the transfer effect. Alpha 0.0 is pure Grassmann, while alpha 0.3 and 0.5 are heterogeneous fusions; architectural composition and teacher quality are therefore not independently randomized.
