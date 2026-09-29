# Phase 2E Preregistered Interpretation

This document is frozen before Transformer-only formal endpoints are trained. The intervention changes only the effective mixture of logits from one fixed jointly trained teacher checkpoint. It can establish a teacher-source contribution under the tested protocol, but it cannot isolate Grassmann geometry from generic ensemble diversity.

## Outcome A: fused dominates both components

If `Gain_F > Gain_T` and `Gain_F > Gain_G` for all three seeds, and both paired mean gaps are at least 0.02 NLL, the result supports that heterogeneous fused supervision transfers more effectively than either component branch alone. Positive transfer from a globally weaker component teacher is recorded separately.

## Outcome B: fused approximately equals Transformer-only

If the mean absolute `C_FT` is below 0.02 NLL or its seed signs are not stable, while both T_F and T_T substantially exceed T_G, most transferable information is interpreted as already present in the Transformer branch. This weakens the incremental Grassmann contribution to KD.

## Outcome C: Transformer-only dominates fused

If `C_FT < 0` for all seeds with mean magnitude at least 0.02 NLL, fused teacher likelihood improvement does not translate into improved distillability under the tested protocol. The negative result is retained without selecting a favorable seed.

## Outcome D: Grassmann-only dominates Transformer-only

If `C_TG < 0` for all seeds with mean magnitude at least 0.02 NLL, the surprising ordering is reported as descriptive teacher-source evidence, not geometric superiority.

## Outcome E: mixed seeds

If a primary paired contrast changes sign materially across seeds, teacher-source contribution is classified as unstable. Means do not override sign reversals.

## Statistical reporting

All seed values, means, sample standard deviations, minima, maxima, and sign counts are reported for `Gain_F`, `Gain_T`, `Gain_G`, `C_FT`, `C_FG`, and `C_TG`. No conclusion is based on an asymptotic p-value with three student seeds.

## Final decision rules

Decision 1 is YES only for consistent, practically nontrivial positive `C_FT`; PARTIAL covers smaller positive or mixed evidence; NO covers consistent nonpositive evidence. Decision 2 uses the same rule for `C_FG`. Decision 3 follows the sign consistency of `Gain_T`, and Decision 4 reuses the established sign consistency of `Gain_G`. Decision 5 is governed primarily by `C_FT`. Decision 6 remains NO or INCONCLUSIVE because no homogeneous-ensemble causal control is added. Decision 7 selects one follow-up by information gain and does not launch it.
