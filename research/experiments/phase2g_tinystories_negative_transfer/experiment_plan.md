# Phase 2G: TinyStories Negative-Transfer Confirmation

## Frozen question

Does the warm-start KD degradation observed for TinyStories seed 42 reproduce for independently trained student initializations with seeds 123 and 456?

This is the final confirmatory experiment. No new method, alpha sweep, routing rule, teacher modification, Grassmann mechanism test, or Phase 2H is permitted.

## Existing seed-42 reference

The matched seed-42 endpoints are reused and not rerun:

- S0: `outputs/hybrid_experiments/20260515_031101_ts_hybrid_lite_baseline_224x56_l6_e20`
- WS+CE: `outputs/distill_experiments/20260916_073243_h0_ts_confirm_token_l0_seed42`
- WS+KD: `outputs/distill_experiments/20260916_073219_h0_ts_confirm_token_l5_seed42`
- Teacher: `outputs/hybrid_experiments/20260515_022710_ts_latefusion_last1_joint_e10`
- CE/KD test NLL: 1.5812685042939885 / 1.6887474090795840
- `Delta_KD = NLL_WS_CE - NLL_WS_KD = -0.1074789047855955`

## Independent S0 training

Train one new Hybrid-lite S0 for each of seeds 123 and 456. Match seed 42 exactly except for the training seed and experiment metadata:

- architecture: model dimension 224, six layers, reduced dimension 56, windows 1/2/4, dropout 0.1, `late_k=1`
- data: saved TinyStories, 300,000-story limit, text field `text`, sequence length 256, fixed data `split_seed=42`
- tokenizer: repository `gpt2_local`
- optimization: batch 32, 20 epochs, AdamW-compatible script defaults, learning rate `2e-4`, weight decay 0.01, AMP
- checkpoint selection: validation NLL only

The two S0 checkpoint hashes must be distinct from each other and from seed 42. No test result may be used to choose or restart an S0.

## Matched endpoint matrix

For each new S0, run exactly two continuations: WS+CE and WS+KD. Both arms use the same seed-specific S0 tensor and match the seed-42 protocol:

- frozen Teacher J above, with the checkpoint alpha unchanged
- `student_type=hybrid_lite`
- token-mean KD objective
- CE arm: `kd_lambda=0`
- KD arm: `kd_lambda=5`
- temperature 2
- 10 epochs, batch 32, learning rate `1e-4`, weight decay 0.01
- warmup ratio 0.05, cosine schedule, AMP, gradient clip value inherited unchanged
- sequence length 256 and the same dataset/tokenizer/preprocessing
- validation-NLL checkpoint selection; test evaluation only after selection

No lambda adjustment is allowed after observing stability or endpoints.

## Diagnostics and decision rule

For each seed define:

`Delta_KD = NLL_WS_CE - NLL_WS_KD`.

A negative value means that KD harms the student. Report each seed, mean, sample SD, and sign consistency. Also record the teacher residual advantage `NLL_WS_CE - NLL_teacher`, KD KL trajectory, parameter-gradient norm, clipping fraction, AMP overflow fraction, and non-finite fraction. A failed or non-finite run is reported as optimization-confounded rather than retuned.

The frozen interpretation rule is:

- 3/3 negative: negative-transfer boundary `CONFIRMED`
- 2/3 negative: `PARTIAL`
- mixed signs without two negatives: `NEGATIVE TRANSFER NOT ROBUST`

## Execution and stopping

All heavy training runs on server `10.42.0.197`. A reduced-data S0 and paired CE/KD smoke precedes formal S0 training. After the formal matrix, produce compact raw artifacts, `REPORT.md`, `GPT_HANDOFF.md`, `results_multiseed.csv`, update project state/handoffs, push to GitHub, set the project state to `experiments_frozen_manuscript_rewrite_next`, and stop. Do not start Phase 2H.
