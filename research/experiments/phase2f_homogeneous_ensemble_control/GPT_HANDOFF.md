# Phase 2F GPT Handoff

## Read this first

Phase 2F is complete and stopped. Do not launch Phase 2G, modify the manuscript, or reinterpret this phase as evidence for a Grassmann-specific KD mechanism without new authorization.

## Primary result

The homogeneous Transformer+Transformer teacher outperforms the Transformer+Grassmann teacher for all three independent WikiText-2 students. With `H = Gain_TG - Gain_TT = NLL_TT - NLL_TG`, the values for seeds 42/123/456 are -0.013952/-0.011329/-0.012099. Mean H is -0.012460 and sample SD is 0.001348, giving 3/3 negative signs.

Mean KD gain relative to matched C0 is 0.155665 for TG and 0.168125 for TT. Both teachers transfer positively, but TT is consistently better. The Grassmann-specific KD mechanism claim must therefore be removed. The Phase 2E fused advantage is compatible with generic ensemble supervision and teacher quality rather than unique Grassmann signal.

## Teacher facts

T1 is the existing WT2 Transformer from `outputs/experiments/20260317_114514_wt2_baseline_both`. T2 was independently trained with seed 123 under the same 20-epoch protocol and selected epoch 12, with validation/test NLL 5.230674/5.288460. Its checkpoint SHA256 is `d56b566023118a1089ee908cde6822f5e9b0bcd961aa1d6d3da8a2f776aad442`.

The TT joint teacher is `outputs/hybrid_experiments/20260930_034520_phase2f_tt_teacher_joint_e10`. It selected epoch 10; its checkpoint SHA256 is `870d2667f0c5169fb42498a7195551fdc92f5bdf3fd41869d2167998c964e6f9`. Evaluation and KD force alpha to 0.5.

On the common full validation set, TT/TG NLL is 4.293589/4.303697. TT is better by 0.010108. TT has 35,340,801 parameters and TG has 37,749,633; the teachers are not parameter-matched. TG has greater branch JSD and fusion gain, but those metrics do not yield a better KD endpoint.

## Confounds and interpretation boundary

Teacher quality is a material direction-aligned confound because TT already has lower validation NLL. The user-authorized TT teacher used alpha learning rate `1e-2`, while the historical TG teacher used `5e-3`. Parameter count does not explain the TT win as a capacity advantage because TT is smaller, but exact parameter matching is absent.

TT has higher early clipping than TG, with formal epoch-1 clipping fractions 0.8435--0.9286, but it is not saturated; clipping falls to 0.0034 and maximum overflow/non-finite fractions remain 0.013605. This asymmetry should be disclosed but does not obviously create the TT advantage. The Phase 2E `CLIP_SATURATION_WARNING` belongs to the separate Transformer-only condition and must not be transferred to the Phase 2F fused teachers.

Do not write that Plucker geometry, architectural heterogeneity, or Grassmann complementarity improves KD. The current evidence supports only that both ensemble teachers transfer positively and that the matched homogeneous control is better under the tested protocol.

## Files

- `REPORT.md`: complete scientific interpretation
- `results_multiseed.csv`: exact per-seed endpoints and diagnostics
- `raw/results_summary.json`: machine-readable aggregate statistics
- `raw/teacher_diagnostics.json`: fixed-alpha teacher comparison
- `provenance.md`: source paths, hashes, smoke gates, and protocol deviations
- `figures/fig1_paired_kd_gain.pdf`: paired KD gain comparison
- `figures/fig2_teacher_comparison.pdf`: teacher and branch diagnostics

## Stop state

Phase 2G is not necessary for the Grassmann-specific claim, because the decisive control has already contradicted it. Any further experiment must answer a newly scoped question, be preregistered, and receive explicit authorization.
