# Phase 2D Preregistered Interpretation

This document is frozen before the Teacher-A formal endpoints are trained. The intervention changes the trained teacher representation state while holding architecture, source branch identities, fusion semantics, and effective alpha fixed. It does not isolate teacher NLL as an abstract scalar cause.

On 2026-09-29, before formal endpoint training, the research lead authorized the matched full-data one-epoch stability amendment documented in `experiment_plan.md`. This amendment changes only the adequacy of the optimization-integrity observation window. It does not change the objective, teachers, student initializations, seeds, formal duration, endpoint definitions, or interpretation categories below.

## Outcome A: strong support

If `Q(s) > 0` for all three seeds and the differences are practically nontrivial, the result supports that the better jointly trained teacher transfers more effectively than the much weaker alpha-only teacher under fixed Transformer/Grassmann composition. If Teacher A also gives zero or negative KD gain, the observed transfer boundary is recorded without claiming a universal teacher-quality law.

## Outcome B: graded support

If Teacher A transfers positively but substantially less than Teacher J, teacher training state and resulting quality modulate transfer magnitude even under fixed composition, while the globally weak teacher retains transferable information. This is compatible with, but does not prove, the Phase 2C conditional-utility explanation.

## Outcome C: weak or no quality effect

If `Gain_A` is close to `Gain_J`, the teacher-quality explanation is weakened. Composition, soft-target structure, regularization, or other compatibility factors may dominate global teacher NLL.

## Outcome D: contradiction

If Teacher A consistently transfers better than Teacher J despite its substantially worse standalone NLL, the current quality-modulation hypothesis is contradicted and the result must be reported unchanged.

## Outcome E: optimization-confounded

If Teacher A has severe gradient-scale mismatch, clipping saturation, numerical instability, checkpoint-integrity failure, or another qualitative optimization mismatch, endpoint differences are not interpreted as clean teacher-training-state evidence. The preregistered scale diagnostic is used only when its trigger fires.

## Statistical reporting

All seed-level values, mean, sample standard deviation, minimum, maximum, and sign consistency are reported for `Gain_J`, `Gain_A`, and `Q`. With three independent student seeds, no conclusion is based on a p-value. A paired visualization must retain every seed.

## Required final decisions

The final `REPORT.md` ends with the seven questions and answer vocabularies specified in the governing Phase 2D request. Decision 5 separates branch quality from fusion complementarity. Decision 6 must respect the Phase 2C evidence that a globally weak teacher can still transfer positively. Decision 7 selects exactly one next experiment on information gain, after which execution stops.
