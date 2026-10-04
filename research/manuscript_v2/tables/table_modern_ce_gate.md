# table_modern_ce_gate

One seed42, three CE LR candidates on an 8M-target disjoint stream. Gate improvement >=0.01 failed. No KD or new test; missing quantities are NOT_RUN, not zeros. Small positive validation headroom is acknowledged.

| arm | lr | s0_validation_nll | best_validation_nll | CE_improvement_validation | selected_step | selected_sha256 | trained_selected | gate_threshold | passes_individual_gate | test_evaluated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ce_lr5e-6 | 5e-06 | 2.9671119410592275 | 2.9628974995455346 | 0.004214441513692879 | 977 | 127b31e39515af6f7f91d2577ff0fa947b88bdcf7914aaa8d0d0591818f61234 | True | 0.01 | False | False |
| ce_lr1e-5 | 1e-05 | 2.9671119410592275 | 2.9622626482505976 | 0.00484929280862989 | 977 | e1ea4d380f6feda66e6e9321a3fd9a42979fae8f4945b90b5942695a780cc234 | True | 0.01 | False | False |
| ce_lr2e-5 | 2e-05 | 2.9671119410592275 | 2.9631713048530237 | 0.003940636206203774 | 977 | 9e1f1218d8aa4b323849c03313dccc7e2b8797252b1352f50813162678b7dbeb | True | 0.01 | False | False |
