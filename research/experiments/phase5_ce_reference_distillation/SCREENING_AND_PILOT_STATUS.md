# Development screening and pilot status

Snapshot: 2026-11-26T11:52:04+08:00.

These are seed-42 development results. They are not test endpoints or independent-seed confirmations.
The frozen promotion rule chooses the lowest trained screen dev NLL; subsequent checkpoint selection includes S0.

| Domain | Method | Trained screen dev NLL | Selected dev NLL | Gain vs selected CE | Clip fraction |
|---|---|---:|---:|---:|---:|
| wt2 | ce | 4.348807 | 4.348015 | +0.000000 | 0.0034 |
| wt2 | kd | 4.266468 | 4.266468 | +0.081546 | 0.2483 |
| wt2 | reference | 4.283045 | 4.283045 | +0.064970 | 1.0000 |
| wt2 | atkd | 4.282010 | 4.282010 | +0.066004 | 1.0000 |
| wt2 | todi | 4.297731 | 4.297731 | +0.050284 | 0.0374 |
| wt2 | wclip | 4.270367 | 4.270367 | +0.077648 | 1.0000 |
| wt2 | ls | 4.384453 | 4.348015 | +0.000000 | 0.0034 |
| ts | ce | 1.370287 | 1.370287 | +0.000000 | 0.0000 |
| ts | kd | 1.371282 | 1.371282 | -0.000994 | 0.0000 |

All method comparisons remain exploratory. A saturated arm retains `CLIP_SATURATION_WARNING`; finite clipping does not establish numerical failure or explain an outcome causally.

In the completed WT2 short screen, candidate A is weaker than the selected ordinary KD configuration. The two completed WT2 CE trajectories also select S0. Therefore neither candidate superiority nor a continuously improving held-out CE reference is supported by these observations. TinyStories short CE improves over S0; its full-training direction remains to be established.

| Full run | Status | Observed / total updates | Selected dev endpoint |
|---|---|---:|---:|
| ts_s42_full_ce | IN_PROGRESS_OR_INCOMPLETE | 3850 / 81890 | pending |
| wt2_s42_full_ce | COMPLETE | 2940 / 2940 | 4.348015 |
| wt2_s42_full_kd | IN_PROGRESS_OR_INCOMPLETE | 1450 / 2940 | pending |
| wt2_s42_full_reference_ce | COMPLETE | 2940 / 2940 | 4.348015 |

The isolated retry of the full-state replay smoke completed; it checks optimizer/RNG replay and finite H=1/8 branches only. It is not a formal mechanism result. Priority ablations, full forks, online search and seed confirmations must be reported from their own completed artifacts.

No test scores are read or selected by this script. Incomplete runs never receive a completed endpoint value.
