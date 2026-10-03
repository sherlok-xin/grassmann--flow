# Teacher-Source Ablation

| Teacher source | Teacher val NLL | Test NLL, mean ± SD | KD gain, mean ± SD | Signs |
|---|---:|---:|---:|---:|
| Fused TG | 4.300646 | 4.120699 ± 0.011269 | +0.155665 ± 0.007692 | 3/3 positive |
| Transformer-only | 4.445647 | 4.141496 ± 0.012398 | +0.134868 ± 0.008906 | 3/3 positive |
| Grassmann-only | 4.652064 | 4.235373 ± 0.010277 | +0.040991 ± 0.006733 | 3/3 positive |

| Paired contrast | Mean | Sample SD | Signs |
|---|---:|---:|---:|
| Fused minus Transformer | +0.020797 | 0.001214 | 3/3 positive |
| Fused minus Grassmann | +0.114674 | 0.001001 | 3/3 positive |
| Transformer minus Grassmann | +0.093877 | 0.002195 | 3/3 positive |

`CLIP_SATURATION_WARNING`: the Transformer-only gate and formal epoch 1 clipped every step. The fused-versus-Transformer difference is small and boundary-close. This table supports a bounded source ablation, not a Grassmann-specific mechanism.
