# Phase 2B Provenance

Phase 2A `REPORT.md` is the governing decision document. Historical priorities in `docs/codex_handoff.md` are not used. The Phase 2B preflight was completed before training and selected Route A on WikiText-2 without test-set selection.

The exact S1 checkpoint was evaluated on the same 512 validation chunks used by the frozen alpha landscape. Selection seed is 20260920 and the selected-index SHA256 is stored in `raw/preflight/preflight.json`. Dataset identity and preprocessing follow `research/snapshots/20260920/dataset_manifest_wikitext2.json`.

The common S0 checkpoint is `outputs/hybrid_experiments/20260515_004119_wt2_hybrid_lite_baseline_224x56_l6_e20/checkpoints/hybrid_best.pt`, with file SHA256 `9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef`. Each formal run must record this same resolved checkpoint, and the collector must reject any mismatch.

Teacher branches and weights come from `outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint`. The source convention is `alpha * Transformer logits + (1-alpha) * Grassmann logits`. Phase 2B changes only this fixed scalar after loading the same checkpoint. It does not optimize teacher parameters.

The canonical execution environment is server 197 as recorded in `research/snapshots/20260920/remote_environment.txt`. Heavy training is not run locally. Exact commands, run directories, source hashes after the bounded instrumentation change, raw run artifacts, and final integrity checks will be appended after execution.

The formal-run source hashes are: `abd501c5f43a8ea79316dcbe56710bb62afd06f1b24e18d848cd3ad6cbeb2202` for `train_distill_hybrid_lite_from_latefusion_teacher_v2.py` and `4a7d002db4f068d299d69e97832ca98c0d03644912c4e1d898ceb4a81592374d` for `src/kd_losses.py`. The frozen preflight, preregistration, and experiment-plan hashes are respectively `638c44c55ddfd544e5e9331c42937a4ae21c1e2647b00caaa576d74bd72e573d`, `aff878c80bac5360d23a8bdca66a5cfc7ba4ffb917f0d82f8358c78b43578ec8`, and `aa9041124b911a593a9a7dc447e6d154f6ddb0b27c0109d5df2646eabf40ee71`.

## Smoke gate

The bounded code change passed seven remote tests covering the established KD loss and the new alpha/hash controls. A 200-line smoke verified artifact creation but contained only one optimizer step; the T_bad KD step overflowed at the initial AMP scale. A follow-up 2,000-line, one-epoch smoke therefore evaluated all four arms over 25 steps.

All extended-smoke summaries were complete, all metrics were finite, all effective alpha values were correct, and every run recorded the required S0 SHA256. C0 had clip/non-finite/overflow fractions 0/0/0. C1, C2, and C3 each had clip fraction 1.0 and non-finite/overflow fraction 0.16. Their unweighted KL values were 0.7881, 0.8395, and 1.0776; weighted contributions were 3.9403, 4.1973, and 5.3881. This is a preregistered loss-scale warning: all KD arms are clip-saturated in the smoke, and T_bad has the largest KD contribution. Lambda remains fixed at 5 as required. Formal results must be interpreted with this limitation rather than retuning the objective.

## Formal execution

The formal command template below was executed in the server-197 `grassmann_lab` container from `/workspace/grassmannflows/grassmann-flows`. Bracketed values were instantiated exactly as shown in the arm table.

```bash
CUDA_VISIBLE_DEVICES=[GPU] python train_distill_hybrid_lite_from_latefusion_teacher_v2.py \
  --teacher-run-dir outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint \
  --student-type hybrid_lite --tokenizer-dir ./gpt2_local \
  --student-init-run-dir outputs/hybrid_experiments/20260515_004119_wt2_hybrid_lite_baseline_224x56_l6_e20 \
  --expected-student-init-sha256 9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef \
  --output-dir outputs/distill_experiments --seed 42 --batch-size 32 --epochs 10 \
  --lr 0.0001 --weight-decay 0.01 --warmup-ratio 0.05 --num-workers 4 --amp --log-interval 50 \
  --model-dim 224 --num-layers 6 --num-heads 8 --reduced-dim 56 --window-sizes 1,2,4 \
  --dropout 0.1 --student-late-k 1 --temperature 2 --kd-loss-mode token_mean --kd-chunk-tokens 1024 \
  --distill-strategy fixed_fused --dataset-name wikitext2 \
  --dataset-path /workspace/grassmannflows/datasets/wikitext2_v1_saved --text-field text \
  --max-seq-len 256 --max-lines 0 --encode-chars-per-batch 200000 \
  --tinystories-val-frac 0.02 --split-seed 42 --offline --gpu-id [GPU] \
  --experiment-name [NAME] --teacher-alpha-override [ALPHA] --kd-lambda [LAMBDA] \
  --notes [NOTES] --tags [TAGS]
```

| Arm | GPU | NAME | ALPHA | LAMBDA | NOTES | TAGS | Complete run directory |
|---|---:|---|---:|---:|---|---|---|
| C0 | 0 | `phase2b_wt2_c0_ws_ce_a05` | 0.5 | 0 | `phase2b_route_a_wt2_control` | `phase2b,controlled_teacher,wt2,control` | `outputs/distill_experiments/20260921_134300_phase2b_wt2_c0_ws_ce_a05` |
| C1 | 1 | `phase2b_wt2_c1_good_a05` | 0.5 | 5 | `phase2b_route_a_wt2_good` | `phase2b,controlled_teacher,wt2,good` | `outputs/distill_experiments/20260921_134302_phase2b_wt2_c1_good_a05` |
| C2 | 2 | `phase2b_wt2_c2_near_a03` | 0.3 | 5 | `phase2b_route_a_wt2_near` | `phase2b,controlled_teacher,wt2,near` | `outputs/distill_experiments/20260921_134301_phase2b_wt2_c2_near_a03` |
| C3 | 0 | `phase2b_wt2_c3_bad_a00` | 0.0 | 5 | `phase2b_route_a_wt2_bad` | `phase2b,controlled_teacher,wt2,bad` | `outputs/distill_experiments/20260922_001717_phase2b_wt2_c3_bad_a00` |

The first formal C3 attempt on physical GPU 3 was externally terminated during epoch 1 after the card remained clock-limited near 210 MHz. Its incomplete directory and log are retained and excluded by the collector because they contain no summary. C3 was rerun unchanged on physical GPU 0. This was an infrastructure retry, not a hyperparameter change.

The collector verified the common S0 hash, effective alpha, lambda, and all matched configuration invariants before copying raw artifacts. Best-checkpoint SHA256 values are `c0d373bd928c079938e4410cbd58121b916c363a3ec6e7903b7bf429514fb8a3` for C0, `a1b50e04cd58bfe0492af8f33d98ec8429296468be444fc4bad994a9190cb309` for C1, `5d0084d7633a562b960869d9c7e31657d62e3325d533266489bccc1e7fe6a337` for C2, and `229a9abb4aacdb39722e70ef8087a33eabf724a876556711811f4f707305fd61` for C3.

Formal epoch-1 clipping fractions are 0.0034, 0.5476, 0.3946, and 0.5850 for C0 through C3. The three KD arms each have AMP overflow and non-finite fractions of 0.0136 in epoch 1; subsequent values return to low or zero levels. Unlike the reduced smoke, no formal KD arm is clip-saturated. No `LOSS_SCALE_CONFOUNDER` flag is assigned.

Post-run collection used `python3 research/experiments/phase2b_controlled_teacher/collect_results.py`. Figures were generated with `python3 research/experiments/phase2b_controlled_teacher/plot_results.py`. The bounded source and control tests were re-run in the remote environment and passed 7/7.

## Final artifact integrity

| Artifact | SHA256 |
|---|---|
| `results.csv` | `e8c09f4219fa6d6557ac19c6c7f35e00b645b91bd709ba95c770767b96f2d676` |
| `REPORT.md` | `90c583434e44fbd2a127f0d5e38e1b5e240259a3292d5c8a9e6f142ebb966712` |
| `raw/formal_summary.json` | `447364a1db17411c72100873cfecd4022c650d7c004e477f98a607de06365974` |
| `figures/teacher_advantage_vs_kd_gain.pdf` | `a9deff2b68c92a5a9665d99c17e4717a18e3f8a91b414b3aa594ad7e7e1b11c5` |
| `figures/teacher_advantage_vs_kd_gain.png` | `224925c21b38970423582f2acbe5a245c9a80436d4dc3fb4d1b68ced1082ee78` |
| `figures/controlled_teacher_outcomes_and_scale.pdf` | `49afde4c5666888ac03d98371a32689130ac2f5956f8a350795cdb462f4175c8` |
| `figures/controlled_teacher_outcomes_and_scale.png` | `007310e042d09735e6325842329085f0d8dbc98527d3b79276f0f6ddd82fd0fe` |
