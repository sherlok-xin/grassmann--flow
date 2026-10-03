# Phase 2G Report: TinyStories Negative-Transfer Confirmation

## Status and decision

Phase 2G is complete. Under the frozen TinyStories warm-start protocol, KD degrades the validation-selected student endpoint for all three independently initialized students. With `Delta_KD = NLL_WS_CE - NLL_WS_KD`, the values for seeds 42, 123, and 456 are -0.107479, -0.107439, and -0.106740. The mean is -0.107219 NLL with sample SD 0.000415, giving 3/3 negative signs. Under the preregistered rule, the TinyStories negative-transfer boundary is `CONFIRMED`.

No coefficient, teacher, dataset, model, training budget, or selection rule was changed after observing an endpoint. No Phase 2H experiment was launched.

## Frozen protocol and independence

The historical seed-42 matched pair is reused without rerunning. Seeds 123 and 456 each use an independently trained 20-epoch Hybrid-lite S0 with the same architecture, tokenizer, TinyStories split, sequence length, optimizer, batch size, and validation-NLL checkpoint selection as seed 42. The three S0 checkpoint SHA256 values are distinct.

Each S0 is continued for 10 epochs in exactly two arms. WS+CE uses `kd_lambda=0`; WS+KD uses token-mean KD with `kd_lambda=5` and temperature 2. Both arms use the same frozen late-fusion teacher, batch size 32, learning rate `1e-4`, weight decay 0.01, warmup ratio 0.05, cosine schedule, AMP, and unchanged clipping. Checkpoints are selected by validation NLL, and test data are used only for the final report.

## Endpoint results

| Seed | S0 test NLL | WS+CE val NLL | WS+CE test NLL | WS+KD val NLL | WS+KD test NLL | Teacher residual advantage | Delta_KD |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1.622437 | 1.577206 | 1.581269 | 1.685600 | 1.688747 | -0.021960 | -0.107479 |
| 123 | 1.623954 | 1.577798 | 1.581807 | 1.685880 | 1.689246 | -0.021422 | -0.107439 |
| 456 | 1.624901 | 1.579197 | 1.583515 | 1.687236 | 1.690255 | -0.019714 | -0.106740 |

| Quantity | Mean | Sample SD | Sign consistency |
|---|---:|---:|---:|
| WS+CE test NLL | 1.582197 | 0.001173 | -- |
| WS+KD test NLL | 1.689416 | 0.000768 | -- |
| Teacher residual advantage | -0.021032 | 0.001173 | 3/3 negative |
| Delta_KD | -0.107219 | 0.000415 | 3/3 negative |

All six continuations selected epoch 10. The KD harm also appears on validation NLL for every seed, so the conclusion is not produced by test-set checkpoint selection. The very small between-seed variation in `Delta_KD` shows that the seed-42 finding was not an isolated initialization outcome under this protocol.

The teacher test NLL is 1.603229. Its residual advantage relative to WS+CE is negative for every seed, meaning that the frozen teacher is itself worse than the CE endpoint by 0.019714--0.021960 NLL. This observation is consistent with the earlier teacher-quality boundary hypothesis, but it does not by itself establish a causal mechanism.

## Optimization audit

| Seed | KD epoch-1 KL | KD mean KL | CE mean grad norm | KD mean grad norm | CE mean clip fraction | KD mean clip fraction | Max KD overflow/non-finite |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.578727 | 0.380224 | 0.486891 | 0.973546 | 0.000391 | 0.236891 | 0.000611 |
| 123 | 0.579751 | 0.381402 | 0.488263 | 0.981438 | 0.000391 | 0.283344 | 0.000611 |
| 456 | 0.582364 | 0.383282 | 0.483211 | 0.973942 | 0.000391 | 0.240090 | 0.000611 |

All endpoints and recorded gradients are finite, and no run contains a NaN endpoint or training failure. Maximum CE overflow/non-finite fraction is 0.000488 and maximum KD overflow/non-finite fraction is 0.000611 for every seed. KD approximately doubles the mean recorded gradient norm and increases the mean clipping fraction to 0.237--0.283, with epoch-1 clipping of 0.583--0.643. This is a reproducible optimization difference and must be disclosed. It is not clip saturation, and the low, matched overflow/non-finite rates do not indicate numerical failure. The result should therefore be interpreted as negative transfer under the frozen training protocol, not as evidence that all possible KD weights or optimization settings must fail.

The first seed-456 KD attempt on GPU 3 was excluded before endpoint analysis because that device remained under a severe software power cap and the run did not complete epoch 2. After explicit authorization, the arm was restarted from scratch on GPU 1 with the identical scientific protocol. No partial optimizer state was resumed. The valid retry completed all 10 epochs and is the only seed-456 KD endpoint in the table.

## Interpretation boundary

The allowed conclusion is narrow but robust: for this frozen teacher, warm-start student, `lambda=5`, temperature 2, and TinyStories protocol, token-mean KD consistently harms the endpoint relative to matched CE continuation across three independent initializations. Phase 2G confirms a domain- and protocol-specific negative-transfer boundary. It does not prove that TinyStories KD is universally harmful, identify Grassmann geometry as the cause, or authorize post hoc coefficient tuning.

Together with the positive three-seed WikiText-2 results, this experiment supports a domain-dependent KD account. Phase 2F already removed the Grassmann-specific KD mechanism claim; Phase 2G does not restore it. Experiments are now frozen and the next permitted activity is evidence-grounded manuscript reconstruction.

## Artifacts

Exact endpoints and diagnostics are in `results_multiseed.csv`. Machine-readable aggregates and compact copies of configs, summaries, metric traces, reports, and checkpoint hashes are under `raw/`. Source identities, run paths, the GPU-3 exclusion, and collector compatibility handling are recorded in `provenance.md`.
