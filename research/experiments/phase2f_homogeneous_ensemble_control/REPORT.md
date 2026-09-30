# Phase 2F Report: Homogeneous Ensemble Control

## Status and decision

Phase 2F is complete. Under the frozen WikiText-2 protocol, the homogeneous Transformer+Transformer (TT) teacher produces better downstream KD endpoints than the Transformer+Grassmann (TG) teacher for all three independently initialized students. The primary contrast `H = Gain_TG - Gain_TT = NLL_TT - NLL_TG` is negative for seeds 42, 123, and 456. The mean is -0.012460 NLL with sample SD 0.001348. Consequently, the Phase 2E fused-teacher advantage does not identify a Grassmann-specific transferable mechanism. The Grassmann-specific KD claim must be removed.

No Phase 2G experiment was launched.

## Frozen question and comparison

This phase asks only whether the TG teacher retains a downstream KD advantage over a homogeneous TT ensemble. T1 is the validation-selected checkpoint from `outputs/experiments/20260317_114514_wt2_baseline_both`. T2 was independently trained with seed 123 using the same architecture, tokenizer, WikiText-2 preprocessing, sequence length, optimizer, batch size, 20-epoch budget, and validation-NLL checkpoint selection as T1. T2 selected epoch 12 and obtained validation/test NLL 5.230674/5.288460.

The TT teacher was jointly trained for 10 epochs with one frozen-branch epoch, branch learning rate `1e-5`, alpha learning rate `1e-2`, initial alpha 0.5, `late_k=1`, AMP, and validation-NLL selection. It selected epoch 10. Teacher comparison and all KD runs forced effective alpha to exactly 0.5. Test data were not used for teacher selection.

The implementation adds explicit `teacher_type=tg|tt` loading while retaining `tg` as the default and preserving the historical TG checkpoint keys and behavior. Seven remote tests passed, including loading the existing TG checkpoint. Reduced teacher and KD smoke tests were followed by a matched full-data one-epoch KD gate before formal training.

## Teacher comparison

| Teacher | Parameters | Validation NLL | Validation PPL | Branch 1 NLL | Branch 2 NLL | Branch JSD | Top-1 agreement | Fusion gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TG | 37,749,633 | 4.303697 | 73.9728 | 4.451680 | 4.655020 | 0.182087 | 0.507034 | 0.147983 |
| TT | 35,340,801 | 4.293589 | 73.2288 | 4.406160 | 4.422151 | 0.099793 | 0.630170 | 0.112570 |

The teachers are not parameter-matched. TT has 2,408,832 fewer parameters, or 6.38% fewer than TG, yet its validation NLL is lower by 0.010108. The parameter-count difference therefore does not explain the TT advantage through greater TT capacity, although the lack of exact parameter matching remains a design limitation. TG has higher branch JSD, lower agreement, and a larger fusion gain than TT, but those diversity indicators do not translate into a better KD endpoint.

Teacher quality is a relevant confound because the validation-NLL difference is direction-aligned with the downstream result and is similar in magnitude to the mean endpoint difference. In addition, the authorized TT teacher used alpha learning rate `1e-2`, whereas the historical TG Teacher J used `5e-3`. This training mismatch was recorded before endpoint generation and prevents a causal attribution solely to branch architecture.

## Downstream KD results

All new TT arms use the same three Phase 2C S0 checkpoints and the same `token_mean` KD objective as the reused TG arms: `lambda=5`, `T=2`, 10 epochs, batch size 32, learning rate `1e-4`, weight decay 0.01, warmup ratio 0.05, cosine schedule, sequence length 256, AMP, gradient clip value 1.0, and validation-NLL checkpoint selection.

| Seed | C0 test NLL | TG test NLL | TT test NLL | Gain_TG | Gain_TT | H |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 4.270742 | 4.108241 | 4.094289 | 0.162501 | 0.176453 | -0.013952 |
| 123 | 4.277520 | 4.130183 | 4.118854 | 0.147337 | 0.158666 | -0.011329 |
| 456 | 4.280830 | 4.123673 | 4.111574 | 0.157157 | 0.169257 | -0.012099 |

| Quantity | Mean | Sample SD | Sign consistency |
|---|---:|---:|---:|
| Gain_TG | 0.155665 | 0.007692 | 3/3 positive |
| Gain_TT | 0.168125 | 0.008948 | 3/3 positive |
| H | -0.012460 | 0.001348 | 0/3 positive; 3/3 negative |

Both ensembles provide positive transfer relative to C0, but TT provides the larger gain for every student. With only three paired seeds, the result is reported through effect sizes and sign consistency rather than a significance claim.

## Optimization audit

All formal TT runs completed normally, selected epoch 10, and contained no NaN endpoint. Their epoch-1 clipping fractions were 0.928571, 0.843537, and 0.874150; mean fractions across training were 0.170068, 0.137755, and 0.127551. Maximum AMP overflow and non-finite fractions were 0.013605 for every seed. Thus TT experienced substantially more early clipping than the reused TG arms, but it was not clip-saturated and clipping declined to 0.003401.

This asymmetry is an optimization limitation, but it does not provide an evident explanation for the observed ordering: the more heavily clipped TT arms still won all three comparisons. The Phase 2E `CLIP_SATURATION_WARNING` applies to the separate Transformer-only alpha-1.0 condition, not to either fused TG or TT endpoint in this phase.

## Final interpretation

TG does not outperform TT. TT improves test NLL over TG by 0.011329--0.013952 across the three seeds, with mean 0.012460. The teacher-quality difference is a material, direction-aligned confound; the teachers are also not parameter-matched and their alpha learning rates differ. Parameter count and clipping do not offer an obvious alternative explanation in favor of TT because TT is smaller and more strongly clipped, but neither observation removes the teacher-quality confound.

The allowed conclusion is therefore limited: generic ensemble supervision remains beneficial, while the present experiment provides no downstream evidence that a Grassmann branch supplies uniquely transferable KD signal. The Grassmann-specific mechanism claim should be deleted rather than merely softened. Phase 2G is not required to preserve that claim. A further phase would only be justified under a newly defined objective, such as isolating teacher quality from generic ensemble diversity, and would require a new preregistration and explicit authorization.

## Artifacts

The exact per-seed endpoints are in `results_multiseed.csv`; machine-readable aggregates and teacher diagnostics are under `raw/`. `figures/fig1_paired_kd_gain.pdf` shows the paired endpoint ordering, and `figures/fig2_teacher_comparison.pdf` shows teacher quality, parameter counts, diversity, and fusion metrics. PDF and 300-dpi PNG versions are retained. Full run paths, checkpoint hashes, data hashes, smoke gates, and known protocol differences are recorded in `provenance.md`.
