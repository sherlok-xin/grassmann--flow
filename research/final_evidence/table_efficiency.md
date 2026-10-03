# PTB Quality--Efficiency Operating Points

| Model | Params | PPL | B1 latency | B8 latency | B32 latency | B32 throughput |
|---|---:|---:|---:|---:|---:|---:|
| Teacher | 36.264M | 50.1125 | 15.482 ms | 19.145 ms | 55.984 ms | 139,645 tok/s |
| Quality student | 31.434M | 51.8339 ± 0.0554 | 16.089 ms | 19.332 ms | 48.460 ms | 161,327 tok/s |
| Efficiency student | 23.565M | 53.9139 ± 0.0692 | 11.502 ms | 16.559 ms | 34.959 ms | 223,626 tok/s |

The quality student has 13.32% fewer parameters and is 13.44% faster only at batch 32; it is slightly slower at batches 1 and 8. The efficiency student has 35.02% fewer parameters and reduces latency by 25.71%, 13.51%, and 37.55% at batches 1, 8, and 32. PPL variation uses three student seeds, whereas latency uses one frozen checkpoint per model on a single RTX 3090 at sequence length 256. These historical KD endpoints are deployment operating points, not members of the later confirmatory CE/KD matrix.
