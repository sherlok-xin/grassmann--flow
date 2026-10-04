# table_tt_vs_tg

Frozen Phase-2 snapshot, copied without changing values. Source: research/final_evidence/table_tt_vs_tg.csv. Teacher NLL/JSD/agreement use the common full validation split. TT/TG are not strictly parameter/training matched; alpha learning rate differs.

| row_type | teacher_or_contrast | parameters | validation_nll | validation_ppl | branch_jsd | top1_agreement | fusion_gain | test_nll_mean | test_nll_sample_sd | kd_gain_mean | kd_gain_sample_sd | H_mean | H_sample_sd | sign_consistency | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| teacher | TG | 37749633 | 4.303697 | 73.9728 | 0.182087 | 0.507034 | 0.147983 | 4.120699 | 0.011269 | 0.155665 | 0.007692 |  |  | 3/3_positive | COMPLETE |
| teacher | TT | 35340801 | 4.293589 | 73.2288 | 0.099793 | 0.630170 | 0.112570 | 4.108239 | 0.012617 | 0.168125 | 0.008948 |  |  | 3/3_positive | COMPLETE |
| contrast | H_Gain_TG_minus_Gain_TT |  |  |  |  |  |  |  |  |  |  | -0.012460 | 0.001348 | 3/3_TT_better | CONTRADICTS_TG_ADVANTAGE |
