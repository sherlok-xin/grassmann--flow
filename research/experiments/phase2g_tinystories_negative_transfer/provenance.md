# Phase 2G Provenance

Status: formal S0 training in progress; no new formal KD endpoint generated.

## Repository and execution

- Preregistration base HEAD: `f1ea69561f03f83b154b9f303c110744366989ff`
- Local project path: `/home/xin/fuwuqi/grassmann-flows`
- Remote execution host: `10.42.0.197`
- Remote project path: `/workspace/grassmannflows/grassmann-flows`
- Heavy training is remote-only through `~/fuwuqi/agent-tools/exec_grassmann.sh`.

## Frozen source identities

- Teacher checkpoint SHA256: `db9ffda81d1f95ebc3f3d5967e3121b1c6dd859104c9c47cc80b0bab1fcaedfb`
- Seed-42 S0 checkpoint SHA256: `e4d0959603bb39f0ad7b7a98025024975e2b87d117d36037e9f9aa9da966f9c2`
- Seed-42 WS+CE checkpoint SHA256: `36f31a2995283edec06f65276ebe2834d1c6bb0351908bb158bb6f82e83be378`
- Seed-42 WS+KD checkpoint SHA256: `d466d8ff9d5697bca338c2b444d11ba21e3de94e25b1db35b5324495f4eafc5d`

The protected `datasets/` and `checkpoints/` directories are read-only for this phase. Historical runs and checkpoints are not overwritten.

## Frozen data statistics from seed 42

- Train: 300,000 non-empty lines, 67,081,279 pre-trim tokens, 262,036 chunks
- Validation: 42,391 non-empty lines, 9,440,302 pre-trim tokens, 36,876 chunks
- Test: 21,990 non-empty lines, 4,765,822 pre-trim tokens, 18,616 chunks
- Sequence length: 256
- TinyStories split seed: 42

## Endpoint discipline

Only seeds 123 and 456 are new. For each seed, S0 is independently trained and then reused byte-identically for its CE and KD arms. Validation NLL selects checkpoints. Test NLL is read only after selection. The seed-42 matched result is reused without rerunning. No coefficient or protocol field may be changed in response to an endpoint.

## Pipeline smoke

The reduced-data smoke completed before formal training:

- S0: `outputs/hybrid_experiments/20260930_084815_phase2g_smoke_ts_s0_seed123_e1`
- WS+CE: `outputs/distill_experiments/20260930_084920_phase2g_smoke_ts_ce_seed123_e1`
- WS+KD: `outputs/distill_experiments/20260930_084919_phase2g_smoke_ts_kd_seed123_e1`

The smoke validated dataset loading, teacher and warm-start checkpoint loading, both objective paths, checkpoint writing, and final evaluation. Because it begins with a one-epoch S0 trained on only 2,000 stories, its 55-step CE/KD continuations have clip fraction 1.0 and AMP overflow/non-finite fractions 0.0364/0.0909. These values are retained as a short-window/random-S0 smoke artifact and are not treated as a formal stability endpoint. The formal matrix uses the frozen 300,000-story, 20-epoch S0 protocol; no hyperparameter was changed after the smoke.

## GPU-3 infrastructure retry

The first formal seed-456 KD attempt was launched at `outputs/distill_experiments/20261001_044148_phase2g_ts_kd_seed456`. Its first epoch was finite, but GPU 3 was persistently restricted to approximately 210--255 MHz with `SW Power Cap` active, compared with approximately 1,470--1,695 MHz on GPUs 0--2. Throughput fell to approximately 4,200 tokens/s versus 14,600--16,500 tokens/s on the healthy cards. No competing GPU process was present; the four GPU jobs all belonged to Phase 2G.

On 2026-10-01, the research lead explicitly approved the infrastructure retry. The GPU-3 process was terminated during epoch 2 after exact PID and command validation. Its logs, epoch-1 metrics, and checkpoint remain preserved and are excluded from endpoint analysis. Seed-456 KD will restart from scratch on GPU 1 after the successful seed-123 KD arm releases that card. The retry keeps the identical teacher checkpoint, seed-456 S0 hash, seed, objective, lambda, temperature, data, batch, optimizer, schedule, 10-epoch budget, AMP, clipping, and validation-selection rule. Only execution GPU and metadata identifying the infrastructure retry differ; no partial optimizer state is resumed.
