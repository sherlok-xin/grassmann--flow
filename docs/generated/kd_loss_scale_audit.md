# Legacy KD Loss-Scale Audit

Generated: 2026-09-16T11:52:11

The current implementation applies PyTorch `batchmean` to `[batch, sequence, vocabulary]` logits. With fixed-length, unpadded chunks, the table approximates token-mean KL by dividing the logged KL by `max_seq_len - 1`. This is an accounting audit, not a replacement training result.

The summary includes positive-alpha runs only. Coefficients still differ across runs, so these medians describe the historical sweep rather than a controlled domain comparison.

| Dataset | Complete epoch-1 runs | Median alpha | Median CE | Median raw KL | Median approximate token KL | Median KD share of objective |
|---|---:|---:|---:|---:|---:|---:|
| code | 10 | 0.035 | 1.535 | 230.710 | 0.9047 | 79.1% |
| ptb | 46 | 0.050 | 3.680 | 70.408 | 0.2761 | 49.7% |
| tinystories | 6 | 0.035 | 1.703 | 146.349 | 0.5739 | 75.2% |
| wikitext2 | 7 | 0.040 | 3.943 | 126.243 | 0.4951 | 56.9% |

For sequence length 256, legacy alpha values 0.01, 0.02 and 0.05 correspond algebraically to approximate token-KL multipliers 2.58, 5.20 and 13.42 when the objective is rewritten as `CE + lambda * KL_token`. This conversion matches objective scale only; gradient behavior still requires a parity smoke test.
