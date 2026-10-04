# table_teacher_source_ablation

Frozen Phase-2 snapshot, copied without changing values. Source: research/final_evidence/table_teacher_source_ablation.csv. Teacher NLL scope: the archived 512-validation-chunk subset (130560 predicted targets), not the full-split Phase-2F evaluation.

| row_type | label | alpha | teacher_validation_nll | test_nll_mean | test_nll_sample_sd | kd_gain_mean | kd_gain_sample_sd | paired_contrast_mean | paired_contrast_sample_sd | sign_consistency | status | warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| condition | Fused | 0.5 | 4.300646 | 4.120699 | 0.011269 | 0.155665 | 0.007692 |  |  | 3/3_positive | PARTIAL | bounded_source_ablation |
| condition | Transformer_only | 1.0 | 4.445647 | 4.141496 | 0.012398 | 0.134868 | 0.008906 |  |  | 3/3_positive | PARTIAL | CLIP_SATURATION_WARNING |
| condition | Grassmann_only | 0.0 | 4.652064 | 4.235373 | 0.010277 | 0.040991 | 0.006733 |  |  | 3/3_positive | CONFIRMED | no_geometry_causality |
| contrast | Fused_minus_Transformer |  |  |  |  |  |  | 0.020797 | 0.001214 | 3/3_positive | PARTIAL | small_boundary_close_effect_with_transformer_clipping |
| contrast | Fused_minus_Grassmann |  |  |  |  |  |  | 0.114674 | 0.001001 | 3/3_positive | CONFIRMED | composition_and_quality_both_change |
| contrast | Transformer_minus_Grassmann |  |  |  |  |  |  | 0.093877 | 0.002195 | 3/3_positive | CONFIRMED | composition_and_quality_both_change |
