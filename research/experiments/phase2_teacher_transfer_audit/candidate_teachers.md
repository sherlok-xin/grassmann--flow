# Candidate Teachers from the Frozen Alpha Landscape

Candidates use validation data only and the same frozen Transformer and Grassmann branches. A condition is operationally treated as near the WS+CE student when the absolute NLL difference is at most 0.025 nat/token; good and bad require margins larger than 0.025. This threshold is a descriptive selection rule, not a significance test.

## ptb

| Condition | Alpha | Validation NLL | PPL | Delta teacher | KL to S0 (T=1) | CE–KD cosine S0 | Negative fraction S0 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T_good | 0.500000 | 4.053784 | 57.615040 | 0.183495 | 0.345104 | 0.039809 | 0.368280 | 4.019185 |
| T_near | 1.000000 | 4.241051 | 69.480820 | -0.003772 | 0.633507 | 0.013914 | 0.387675 | 4.149372 |
| T_bad | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

The frozen alpha grid does not provide T_bad under the stated margin rule; no synthetic condition was invented.

## tinystories

| Condition | Alpha | Validation NLL | PPL | Delta teacher | KL to S0 (T=1) | CE–KD cosine S0 | Negative fraction S0 | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T_good | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| T_near | 0.600000 | 1.600853 | 4.957259 | -0.020675 | 0.181978 | 0.043799 | 0.452865 | 1.503116 |
| T_bad | 0.000000 | 2.112059 | 8.265239 | -0.531881 | 0.697439 | -0.028402 | 0.514438 | 1.700211 |

The frozen alpha grid does not provide T_good under the stated margin rule; no synthetic condition was invented.
