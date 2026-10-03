# Fixed-Composition Teacher-Quality Intervention

Both teachers use the same architecture, source branch identities, fusion implementation, and forced alpha 0.5. Positive gain means improvement over matched C0.

| Condition | Teacher val NLL | Seed 42 gain | Seed 123 gain | Seed 456 gain | Mean ± sample SD | Signs |
|---|---:|---:|---:|---:|---:|---:|
| Teacher J, jointly trained | 4.300644 | +0.162501 | +0.147337 | +0.157157 | +0.155665 ± 0.007692 | 3/3 positive |
| Teacher A, alpha-only | 6.376220 | -0.022394 | -0.035362 | -0.025206 | -0.027654 ± 0.006822 | 3/3 negative |
| `Q=NLL_A-NLL_J` | -- | +0.184895 | +0.182699 | +0.182364 | +0.183319 ± 0.001375 | 3/3 positive |

The intervention confirms that teacher training state and resulting quality strongly modulate transfer under fixed composition. It does not establish teacher NLL as the sole causal variable because joint continuation changes the internal branch representations.
