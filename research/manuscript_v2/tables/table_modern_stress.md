# table_modern_stress

Primary trained-checkpoint endpoints remain frozen; secondary step0 selection gives zero diagnostic gains. CE itself worsens all S0 test endpoints. Do not headline this as clean modern KD failure.

| dataset | strength | seeds | mean_gain | sample_sd | sign_consistency | validation_test_agreement | best_including_s0 | evidence_role | warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FineWeb-Edu | 1 | 3 | -0.013550460934897904 | 0.003542273400225629 | 3/3_negative | 3/3 | 9/9_arms_select_S0 | bounded_stress_test | frequent_clipping |
| FineWeb-Edu | 5 | 3 | -0.04392408220210579 | 0.004016675497401846 | 3/3_negative | 3/3 | 9/9_arms_select_S0 | bounded_stress_test | CLIP_SATURATION_WARNING |
