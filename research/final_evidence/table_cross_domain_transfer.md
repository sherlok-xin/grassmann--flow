# Cross-Domain Warm-Start Transfer

Positive gain means that KD improves over the seed-matched WS+CE continuation. Values are mean ± sample SD over three independent student initializations.

| Dataset | WS+CE test NLL | WS+KD test NLL | KD gain | PPL change | Sign consistency | Status |
|---|---:|---:|---:|---:|---:|---|
| PTB | 4.082352 ± 0.005423 | 3.937415 ± 0.001649 | +0.144937 ± 0.003938 | -13.49% | 3/3 positive | CONFIRMED |
| WikiText-2 | 4.276364 ± 0.005142 | 4.120699 ± 0.011269 | +0.155665 ± 0.007692 | -14.41% | 3/3 positive | CONFIRMED |
| TinyStories | 1.582197 ± 0.001173 | 1.689416 ± 0.000768 | -0.107219 ± 0.000415 | +11.32% | 3/3 negative | CONFIRMED |

The table contains only matched token-mean results. Dataset, teacher, and teacher residual quality co-vary, so the table establishes cross-domain variation but not a causal effect of domain identity.

## Separately labeled legacy observation

| Dataset | Seeds | WS+CE NLL | WS+KD NLL | KD gain | Status | Reason excluded from confirmatory table |
|---|---:|---:|---:|---:|---|---|
| CodeParrot common-5k | 1 | 1.980364 | 1.792055 | +0.188309 | EXPLORATORY | Legacy batch-normalized loss, single seed, and winner selection |
