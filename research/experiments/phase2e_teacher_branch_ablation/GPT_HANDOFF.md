# Phase 2E GPT Handoff

## Completion state

Phase 2E is complete. Stop here: no next experiment has been launched. Formal Transformer-only runs were launched from amended-gate Git state `a94132b8c2cea41d034f33016d5485cebc78a997`; the Phase 2E start state was `a9642cf9393defa2f335811ea77a069becb1747a`.

Only the three missing Transformer-only arms were trained. C0, fused alpha-0.5, and Grassmann-only alpha-0.0 endpoints were reused from Phase 2C. The frozen teacher checkpoint SHA256 is `a5ad284f6895c0b4ed516c855c1b149e6636b945173aa2e842b8132299582694`.

## Student initializations

| Seed | S0 checkpoint SHA256 |
|---:|---|
| 42 | `9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef` |
| 123 | `a44b64ba5a84d3bd2708a96aad4eae829ff7ab0f00476f0e31146fc314da1f13` |
| 456 | `3fbe985d6474c84d1c61fd7937ea906102cf460846829eb3af4c709593b01757` |

## Teacher preflight

The frozen 512-chunk validation selection SHA256 is `236633612db1e3144b8f3187a5bfb1795f558a50f26368252a8adf3f0b49413f`. Teacher NLL is 4.3006458619 for fused, 4.4456469292 for Transformer-only, and 4.6520635191 for Grassmann-only. Mean residual teacher advantage over the three S0 students is +0.0745149494, -0.0704861179, and -0.2769027078, respectively.

## Formal results

| Seed | `Gain_F` | `Gain_T` | `Gain_G` | `C_FT` | `C_FG` | `C_TG` |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.162501 | 0.142785 | 0.046771 | 0.019716 | 0.115731 | 0.096015 |
| 123 | 0.147337 | 0.125226 | 0.033598 | 0.022110 | 0.113739 | 0.091629 |
| 456 | 0.157157 | 0.136593 | 0.042604 | 0.020564 | 0.114553 | 0.093989 |
| Mean | 0.155665 | 0.134868 | 0.040991 | 0.020797 | 0.114674 | 0.093877 |
| Sample SD | 0.007692 | 0.008906 | 0.006733 | 0.001214 | 0.001001 | 0.002195 |

All six metrics have 3/3 positive signs. Fused > Transformer-only > Grassmann-only > C0 in endpoint quality for every seed. The primary `C_FT` mean is only 0.000797 above the preregistered 0.02 threshold, so the extra fused gain is stable but small and boundary-close.

Transformer-only test NLL is 4.1279572060, 4.1522935062, and 4.1442370310 for seeds 42, 123, and 456. The corresponding validation-selected checkpoints are SHA256 `234d650a6062de6b819ef89f2804406095cc3cb1df2422daba497ca55a9c4eb4`, `e8044966f966dc72907e21cf77341ebc7c5445fe3cc11f966d073fe4fda99be9`, and `054602d0e4e0cd125c85611a62bed2aaa7a06f6b3ec7baffb9d1ba8c819ccacf`.

## Optimization integrity

`CLIP_SATURATION_WARNING`: the Transformer-only full-data gate had clipping fraction 1.0. The explicitly authorized amendment was recorded before formal endpoints and changed no training or evaluation setting. In formal training, epoch-1 clipping was 1.0 for all seeds, then declined to 0.0034 in later epochs. Mean epoch-level clipping was 0.4990/0.4605/0.4398. Maximum AMP overflow and non-finite fractions remained 0.013605 for every seed. There was no NaN, failed run, elevated overflow, or anomalous endpoint, but early clipping may partly affect the small `C_FT` difference.

## Conditional utility

Transformer gold probability exceeds Grassmann on 51.56% of validation tokens; Grassmann exceeds Transformer on 48.44%. Transformer-correct/Grassmann-wrong accounts for 5.89%, and Grassmann-correct/Transformer-wrong for 5.20%. In the hardest student-loss quintile, mean utility across seeds is 0.6356 for Transformer-only, 0.2133 for Grassmann-only, and 0.7396 for fused. This is observational complementarity, not a causal geometric mechanism.

## Final decisions

1. Fused beats Transformer-only: YES, 3/3 signs and mean `C_FT=0.020797`, with clipping caution.
2. Fused beats Grassmann-only: YES, 3/3 signs and mean `C_FG=0.114674`.
3. Transformer-only positive transfer: YES, 3/3 signs and mean `Gain_T=0.134868`.
4. Grassmann-only positive transfer: YES, reused Phase 2C 3/3 evidence.
5. Adding Grassmann yields extra downstream gain over Transformer-only: YES under the tested protocol, but the gain is small and optimization-constrained.
6. Grassmann geometry specifically causes the fused advantage: INCONCLUSIVE.
7. Single highest-information next experiment: homogeneous Transformer+Transformer ensemble teacher versus Transformer+Grassmann teacher.

Do not automatically launch Decision 7. The complete report is `REPORT.md`; the strict table is `results_multiseed.csv`; compact endpoint records are under `raw/formal/`; figures are under `figures/` as PDF and 300-dpi PNG.
