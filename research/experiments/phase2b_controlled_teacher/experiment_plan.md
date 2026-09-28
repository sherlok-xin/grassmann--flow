# Phase 2B Controlled Teacher Experiment Plan

## Selected route

Route A on WikiText-2 is selected by `preflight.md`. The frozen teacher branches come from `outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint`. Every arm starts from `outputs/hybrid_experiments/20260515_004119_wt2_hybrid_lite_baseline_224x56_l6_e20/checkpoints/hybrid_best.pt`, SHA256 `9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef`.

## Matched configuration

All arms use seed 42, Hybrid-lite 224 dimensions, six layers, eight Transformer heads, Grassmann reduced dimension 56, windows 1/2/4, dropout 0.1, late-k 1, 256-token sequences, batch size 32, 10 epochs, AdamW with learning rate 1e-4 and weight decay 0.01, cosine schedule with 5% warmup, AMP enabled, gradient clipping at 1.0, and validation-NLL checkpoint selection. Data are the frozen WikiText-2 splits at `/workspace/grassmannflows/datasets/wikitext2_v1_saved`. The train loader uses the existing seeded shuffle policy; validation and test loaders do not shuffle.

The loss is `CE + lambda_kd * KL_token_mean` with temperature 2 and KD chunk size 1024. C0 uses lambda 0. C1/C2/C3 use lambda 5. No arm-specific tuning is allowed.

| Arm | Condition | Teacher alpha | Teacher composition | Lambda |
|---|---|---:|---|---:|
| C0 | WS+CE | 0.5 | heterogeneous fusion, inactive KD | 0 |
| C1 | T_good | 0.5 | heterogeneous fusion | 5 |
| C2 | T_near | 0.3 | heterogeneous fusion | 5 |
| C3 | T_bad | 0.0 | pure Grassmann | 5 |

## Execution gates

The teacher-alpha override must preserve the existing behavior when omitted, must load the original teacher checkpoint before applying the fixed scalar, and must record both checkpoint alpha and effective alpha. A one-epoch, 200-line smoke will run before the formal arms. The smoke must produce finite losses, the requested effective alpha, identical S0 hashes, and complete summaries. Formal arms start only after this gate passes.

The formal four arms may run in parallel on separate RTX 3090 GPUs because their configuration and data order are independent but matched. No seed 123/456 job will be launched in Phase 2B.

## Diagnostics and stopping

Epoch 1 records CE, unweighted token-mean KL, `lambda * KL`, total gradient norm, clipping fraction, non-finite fraction, and AMP overflow fraction from training. Initial S0 teacher/student KL, CE logit-gradient norm, KD logit-gradient norm, cosine, and negative-cosine fraction come from the identical fixed validation subset in Phase 2A and are copied into the final result with provenance.

An arm is marked failed if no complete summary/checkpoint exists or non-finite behavior prevents evaluation. An arm is marked `LOSS_SCALE_CONFOUNDER` if scale differences coincide with materially different clipping or instability. Such a flag leads to an inconclusive mechanism decision; lambda is not retuned.

## Outputs

Training runs are written under `outputs/distill_experiments/` with `phase2b_wt2_` names. Logs, copied raw summaries/configs/metrics, the consolidated `results.csv`, plots, and the final report are stored in `research/experiments/phase2b_controlled_teacher/`.
