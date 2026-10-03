# Phase 2G GPT Handoff

## Read this first

Phase 2G is complete and the experiment program is frozen. The project state is `experiments_frozen_manuscript_rewrite_next`. Do not launch Phase 2H, tune `lambda`, modify the teacher, or develop a new method without a new explicit authorization and preregistration.

## Primary result

TinyStories warm-start KD harms all three independently initialized students under the frozen matched protocol. For `Delta_KD = NLL_WS_CE - NLL_WS_KD`, seeds 42/123/456 give -0.107479/-0.107439/-0.106740. Mean `Delta_KD` is -0.107219 with sample SD 0.000415 and 3/3 negative signs. The preregistered decision is `CONFIRMED`.

The corresponding CE/KD test NLL pairs are 1.581269/1.688747, 1.581807/1.689246, and 1.583515/1.690255. All six continuations select epoch 10 by validation NLL, and the same degradation is present on validation endpoints.

## Teacher and optimization facts

The frozen teacher test NLL is 1.603229. Teacher residual advantage relative to WS+CE is negative for all seeds: -0.021960/-0.021422/-0.019714. Thus the teacher is worse than the matched CE endpoint, which is consistent with, but does not causally prove, the teacher-quality boundary hypothesis.

The three S0 checkpoint hashes are distinct. All formal endpoints and gradient records are finite. Mean KD gradient norm is 0.974--0.981 versus 0.483--0.488 for CE. Mean KD clipping fraction is 0.237--0.283 versus 0.000391 for CE; KD epoch-1 clipping is 0.583--0.643. Maximum KD AMP overflow and non-finite fractions are both 0.000611. Report this optimization asymmetry, but do not call the result numerically failed or clip-saturated.

The first seed-456 KD process on GPU 3 was excluded as an incomplete infrastructure attempt because the GPU remained software-power-capped. The accepted endpoint is the from-scratch GPU-1 retry at `outputs/distill_experiments/20261001_161938_phase2g_ts_kd_seed456`. Scientific settings and checkpoint identities were unchanged.

## Interpretation boundary

Write only that token-mean KD at the frozen `lambda=5`, temperature 2 protocol causes stable negative transfer on TinyStories for these three independent warm starts. Do not generalize to every KD coefficient or teacher, do not claim a Grassmann-specific mechanism, and do not use Phase 2G to revive the claim removed by Phase 2F. The cross-domain story is that matched KD is positive on WikiText-2 but negative on TinyStories under the tested protocols.

## Files

- `REPORT.md`: complete result and interpretation
- `results_multiseed.csv`: exact endpoints, diagnostics, paths, and hashes
- `raw/results_summary.json`: machine-readable aggregate statistics
- `raw/formal/`: compact configs, summaries, metric traces, reports, and checkpoint hashes
- `provenance.md`: source identities, execution history, exclusion, and protocol record

## Stop state

Experiments are frozen. The next task is manuscript reconstruction from the validated evidence. No Phase 2H exists in the authorized plan.
