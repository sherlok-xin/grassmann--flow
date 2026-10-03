# Homogeneous TT versus Heterogeneous TG

| Teacher | Parameters | Val NLL | Branch JSD | Fusion gain | KD gain, mean ± SD |
|---|---:|---:|---:|---:|---:|
| TG | 37,749,633 | 4.303697 | 0.182087 | 0.147983 | +0.155665 ± 0.007692 |
| TT | 35,340,801 | 4.293589 | 0.099793 | 0.112570 | +0.168125 ± 0.008948 |

The paired contrast `H=Gain_TG-Gain_TT=NLL_TT-NLL_TG` is `-0.012460±0.001348` NLL, with TT better for 3/3 students. TG has greater branch diversity and fusion gain but does not provide the better transferable signal. TT is 6.38% smaller and has slightly better teacher validation NLL. The teachers are not strictly parameter- or training-matched, and their alpha learning rates differ; therefore, the table contradicts TG superiority but does not prove intrinsic TT superiority.
