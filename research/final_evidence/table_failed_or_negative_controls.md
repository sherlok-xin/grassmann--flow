# Failed and Negative Controls

| Control | Exact result | Decision | Manuscript consequence |
|---|---|---|---|
| TinyStories fixed KD | Mean gain `-0.107219±0.000415` NLL, 3/3 negative | Confirmed negative transfer | Central negative boundary |
| Fixed-composition Teacher A | Mean gain `-0.027654±0.006822`, 3/3 negative | Teacher state matters | Contrast with Teacher J |
| TG superiority | `H=-0.012460±0.001348`, TT better 3/3 | Contradicted | Delete Grassmann-specific KD advantage |
| Strong JSD predictor | AUROC deltas `-0.000323/+0.010910`; R2 deltas `+0.000533/+0.003061` | Contradicted as strong mechanism | Report as negative diagnostic |
| CRBD repair | Test PPL CE/fixed/CRBD `5.0195/5.3018/5.5067`; repair `-0.7220` | Failed | Do not present CRBD as a method |
| CRBD versus shuffle | `5.5067` versus `5.5010` test PPL | Failed | Routing signal does not beat shuffle |
| Plücker causality | No causal isolation; TT beats TG | Unsupported | Delete causal geometry narrative |
| Warm-start necessity | Random-init runs are unmatched | Unsupported | Exclude causal initialization claim |
| CodeParrot transfer | Legacy gain `+0.188309` NLL, one seed | Exploratory | Appendix only |

The Stage C CRBD results are additionally confounded by 100% clipping in every KD arm. They remain useful as a transparent negative result but cannot establish that every fixed-budget routing design would fail.
