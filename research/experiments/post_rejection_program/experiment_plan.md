# Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-16
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

# Post-Rejection Experiment and Innovation Plan

## Experiment overview

- **Title**: From fixed fused-logit KD to consensus-routed branch distillation
- **Objective**: Repair the evidence gaps in the rejected CAC paper, explain the positive-transfer/negative-transfer boundary, and test one method-level contribution without increasing student inference cost.
- **Primary hypothesis**: Disagreement between the Transformer and Grassmann teacher branches identifies tokens for which fixed fused-logit KD conflicts with the supervised objective. Routing high-disagreement supervision to architecture-matched student branches should reduce this conflict.
- **Type**: analysis followed by staged training
- **Status**: author decision required before execution

No training command in this plan has been executed. The existing H0 PTB smoke is not repeated.

## Evidence diagnosis

The rejected manuscript contains a useful empirical boundary but not yet a sufficiently isolated method claim. Warm-start KD improves the student on PTB, WikiText-2, and code, while all tested positive legacy coefficients degrade the TinyStories student. Most sweeps are single-seed, the random-initialization comparisons are unmatched, the fusion comparisons use unmatched source checkpoints, and the code transfer comparison changes the data budget. These are evidence defects that can be repaired. They are not proof that the architecture or KD premise is invalid.

The H0 smoke established that `token_mean` with `lambda_kd=5` reproduces the scale of legacy `alpha=0.02` on the fixed 256-token chunks. Because all four current corpora use the same unpadded chunk length, the two objectives differ only by a positive global scale when `lambda_kd=255*alpha/(1-alpha)`. Token normalization is therefore required for interpretable reporting and future variable-length data, but it cannot by itself explain the TinyStories reversal. It must be presented as an implementation correction, not as the new contribution.

The strongest remaining scientific question is why the fused teacher helps on PTB but hurts on TinyStories. The first test must be diagnostic rather than another coefficient sweep.

## Innovation boundary

The working method is **Consensus-Routed Branch Distillation (CRBD)**. Let `p_T`, `p_G`, and `p_F` denote the Transformer-branch, Grassmann-branch, and fused teacher distributions, and let the student expose the corresponding distributions `q_T`, `q_G`, and `q_F`. For each prediction token, define detached branch disagreement

```text
d = JSD(p_T, p_G) / log(2),
w = exp(-d / tau).
```

The proposed objective is

```text
L = L_CE
  + lambda_f * w * KL(p_F || q_F)
  + lambda_b * (1 - w) / 2 * [KL(p_T || q_T) + KL(p_G || q_G)].
```

All KL terms use valid-token normalization and temperature scaling. CE is never gated. The routing weights are detached, normalized to mean one for the active component, and computed only during training, so inference parameters and latency are unchanged. Low-disagreement tokens use the stable fused consensus; high-disagreement tokens preserve architecture-specific signals instead of forcing their log-opinion pool into one target.

This is a candidate contribution, not a validated novelty claim. Plain confidence weighting, entropy weighting, disagreement suppression, multi-teacher gradient weighting, and branch-aligned KD each overlap prior work. The defensible claim, if supported, is narrower: token-level routing between fused consensus and architecture-matched branch supervision for a heterogeneous two-branch autoregressive teacher/student pair, supported by a measured link between branch disagreement and CE--KD gradient conflict.

Relevant novelty constraints include:

- Du et al., *Agree to Disagree* (NeurIPS 2020), which already uses teacher disagreement and gradient-space dynamic weighting: https://proceedings.neurips.cc/paper_files/paper/2020/hash/91c77393975889bd08f301c9e13a44b7-Abstract.html
- Zhong et al., *Revisiting Knowledge Distillation for Autoregressive Language Models* (ACL 2024), which already adapts teaching modes across tokens: https://aclanthology.org/2024.acl-long.587/
- Vu et al., *DWA-KD* (Findings of EACL 2026), which already upweights tokens using teacher confidence and student uncertainty: https://aclanthology.org/2026.findings-eacl.181/
- Li et al., *Fuse Before Transfer* (ICCV 2025), which already studies knowledge fusion for heterogeneous distillation in vision: https://openaccess.thecvf.com/content/ICCV2025/html/Li_Fuse_Before_Transfer_Knowledge_Fusion_for_Heterogeneous_Distillation_ICCV_2025_paper.html

Accordingly, entropy gating or disagreement gating alone is a baseline, not the paper contribution.

## Frozen inputs

| Input | Path | Role |
|---|---|---|
| PTB teacher | `outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint` | Frozen late-fusion teacher |
| PTB seed-42 initialization | `outputs/hybrid_experiments/20260328_142654_ptb_hybrid_lite_baseline_mild_e20` | Warm-start student |
| PTB seed-123 initialization | `outputs/hybrid_experiments/20260402_140126_ptb_hybrid_lite_baseline_224x56_l6_seed123_e20` | Warm-start student |
| PTB seed-456 initialization | `outputs/hybrid_experiments/20260402_140326_ptb_hybrid_lite_baseline_224x56_l6_seed456_e20` | Warm-start student |
| TinyStories teacher | `outputs/hybrid_experiments/20260515_022710_ts_latefusion_last1_joint_e10` | Frozen late-fusion teacher |
| TinyStories seed-42 initialization | `outputs/hybrid_experiments/20260515_031101_ts_hybrid_lite_baseline_224x56_l6_e20` | Warm-start student |
| PTB data | `/workspace/grassmannflows/datasets/ptb_text_only_saved` | Unmodified remote dataset |
| TinyStories data | `/workspace/grassmannflows/datasets/tinystories_saved` | Unmodified remote dataset |

The project runs under Python 3.12.3 and PyTorch 2.6.0a0+cu126 in `grassmann_lab` on `10.42.0.197`. Heavy jobs must use GPUs 1--3 unless a fresh GPU check shows GPU 0 is free. No experiment may edit `datasets/`, `checkpoints/`, or historical output directories.

## Stage A: necessary evidence repair

### A1. PTB matched three-seed confirmation

Run CE-only and fixed token-normalized KD from the seed-matched initialization checkpoint for seeds 42, 123, and 456. Both arms use 10 epochs, batch size 32, learning rate `1e-4`, temperature 2, and the same dataset split. Legacy KD is not a formal arm because its equivalence at sequence length 256 has already been shown analytically and by smoke test.

The exact entry command for each matrix row is:

```bash
~/fuwuqi/agent-tools/exec_grassmann.sh 'cd /workspace/grassmannflows/grassmann-flows && python train_distill_hybrid_lite_from_latefusion_teacher_v2.py --gpu-id <GPU> --teacher-run-dir outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint --student-init-run-dir <INIT> --student-type hybrid_lite --tokenizer-dir ./gpt2_local --output-dir outputs/distill_experiments --experiment-name h0_ptb_confirm_token_l<LAMBDA>_seed<SEED> --notes "H0 matched PTB confirmation; frozen protocol 2026-09-16" --tags h0,confirmatory,ptb,matched --seed <SEED> --batch-size 32 --epochs 10 --lr 1e-4 --weight-decay 0.01 --warmup-ratio 0.05 --num-workers 4 --model-dim 224 --num-layers 6 --num-heads 8 --reduced-dim 56 --window-sizes 1,2,4 --dropout 0.1 --student-late-k 1 --temperature 2.0 --distill-alpha 0.0 --kd-loss-mode token_mean --kd-lambda <LAMBDA> --dataset-name ptb --dataset-path /workspace/grassmannflows/datasets/ptb_text_only_saved --text-field sentence --max-seq-len 256 --max-lines 0 --encode-chars-per-batch 200000 --split-seed 42 --amp --offline'
```

Frozen row substitutions are:

| Seed | INIT | LAMBDA arms |
|---:|---|---|
| 42 | `outputs/hybrid_experiments/20260328_142654_ptb_hybrid_lite_baseline_mild_e20` | 0, 5 |
| 123 | `outputs/hybrid_experiments/20260402_140126_ptb_hybrid_lite_baseline_224x56_l6_seed123_e20` | 0, 5 |
| 456 | `outputs/hybrid_experiments/20260402_140326_ptb_hybrid_lite_baseline_224x56_l6_seed456_e20` | 0, 5 |

Expected cost is about 1.2 GPU-hours in total and about 25 minutes wall time on three GPUs. The gate passes if all six runs are finite and the mean paired effect `PPL(KD)-PPL(CE)` remains negative. Every seed is reported even if one direction differs.

### A2. TinyStories seed-42 matched confirmation

Before spending on multiple seeds, run only seed 42 with CE-only and `lambda_kd=5` from the same existing initialization.

```bash
~/fuwuqi/agent-tools/exec_grassmann.sh 'cd /workspace/grassmannflows/grassmann-flows && python train_distill_hybrid_lite_from_latefusion_teacher_v2.py --gpu-id <GPU> --teacher-run-dir outputs/hybrid_experiments/20260515_022710_ts_latefusion_last1_joint_e10 --student-init-run-dir outputs/hybrid_experiments/20260515_031101_ts_hybrid_lite_baseline_224x56_l6_e20 --student-type hybrid_lite --tokenizer-dir ./gpt2_local --output-dir outputs/distill_experiments --experiment-name h0_ts_confirm_token_l<LAMBDA>_seed42 --notes "H0 matched TinyStories confirmation; frozen protocol 2026-09-16" --tags h0,confirmatory,tinystories,matched --seed 42 --batch-size 32 --epochs 10 --lr 1e-4 --weight-decay 0.01 --warmup-ratio 0.05 --num-workers 4 --model-dim 224 --num-layers 6 --num-heads 8 --reduced-dim 56 --window-sizes 1,2,4 --dropout 0.1 --student-late-k 1 --temperature 2.0 --distill-alpha 0.0 --kd-loss-mode token_mean --kd-lambda <LAMBDA> --dataset-name tinystories --dataset-path /workspace/grassmannflows/datasets/tinystories_saved --text-field text --max-seq-len 256 --max-lines 300000 --encode-chars-per-batch 200000 --tinystories-val-frac 0.02 --split-seed 42 --amp --offline'
```

Run `<LAMBDA>=0` and `<LAMBDA>=5` on separate free GPUs. Expected cost is 22--24 GPU-hours and 11--12 hours wall time. Stop expansion if the sign no longer shows negative transfer; in that case, investigate historical-run drift before developing CRBD.

## Stage B: mechanism diagnostic before method training

Add a read-only diagnostic mode that exposes the teacher and student branch logits without changing default forward behavior. On fixed, manifest-recorded validation chunks from PTB and TinyStories, record only compact per-token statistics:

- teacher branch JSD, fused entropy, fused NLL, and branch correctness;
- analytic logit-gradient cosine and dot product between CE and fixed fused KD;
- endpoint `delta_NLL = NLL(KD model) - NLL(CE model)` after A1/A2;
- chunk identifier, token position, and dataset, but not raw text or full logits.

The principal mechanism test is whether JSD adds out-of-chunk predictive value beyond fused entropy and teacher NLL. Use chunk-grouped five-fold cross-validation and a chunk bootstrap with 2,000 resamples. Report the change in held-out AUROC for predicting negative CE--KD gradient cosine and the change in held-out `R^2` for endpoint `delta_NLL`. Token-level p-values that treat tokens as independent are prohibited.

Proceed to CRBD only if the JSD coefficient has the same harmful direction in PTB and TinyStories and improves at least one held-out predictive metric over the entropy/NLL baseline in both domains. Otherwise reject H1 and test branch-aligned KD without disagreement routing as a separate, weaker method candidate.

Expected cost is below 3 GPU-hours. A smoke on 32 chunks per domain must first verify finite statistics, deterministic chunk selection, and no saved raw logits. The final analysis uses at most 2,048 chunks per domain unless the bootstrap confidence interval is visibly unstable.

## Stage C: bounded CRBD prototype

Only after Stage B passes, implement CRBD behind new explicit flags while preserving the existing default. Run smoke tests on PTB and then a reduced TinyStories subset (`max_lines=20000`, two epochs). The prototype matrix is intentionally diagnostic:

| Arm | Supervision | Purpose |
|---|---|---|
| C0 | CE only | Optimization baseline |
| C1 | Fixed fused KD | Main existing baseline |
| C2 | Entropy-gated fused KD | Adaptive-token baseline |
| C3 | Disagreement-suppressed fused KD | Tests whether suppression alone is sufficient |
| C4 | Branch-aligned KD only | Tests structured transfer without routing |
| C5 | CRBD fused/branch routing | Proposed method |
| C6 | CRBD with shuffled routing weights | Tests whether measured disagreement matters |
| C7 | CRBD with swapped branch pairing | Tests whether architecture alignment matters |

Freeze `lambda_f=5`. Select `lambda_b` from `{1, 2.5, 5}` and `tau` from `{0.1, 0.25, 0.5}` using validation PPL in the reduced prototype only. Do not inspect test PPL during selection. After selection, freeze one setting for every dataset and seed.

The prototype passes if C5 is no worse than C1 by more than 0.3 validation PPL on PTB, reduces at least 50% of the C1-versus-C0 TinyStories excess validation PPL, and outperforms both C6 and C7 in the predicted direction. Otherwise do not launch full CRBD runs.

Expected prototype cost is 3--6 GPU-hours after implementation, depending on the number of settings eliminated by early stopping.

## Stage D: confirmatory method evaluation

If Stage C passes, run PTB and TinyStories with three paired seeds for C0, C1, the strongest adaptive baseline among C2/C3, C4, and C5. Initialize every pair from the same seed-specific baseline. For TinyStories, first repeat seeds 123 and 456 using the existing seed-42 initialization to estimate KD-stage variance. Training distinct TinyStories initialization checkpoints is a second gate, not the default.

The KD-stage-only TinyStories matrix costs about 69 GPU-hours for C0/C1/C5 across three seeds, or about 115 GPU-hours for all five confirmatory methods. Creating distinct seed-123 and seed-456 20-epoch initializations adds roughly 37 GPU-hours. The recommended plan is to postpone those new initializations until C5 has passed the reduced prototype and seed-42 full-data gate.

Only after success on both boundary domains should WikiText-2 and code be added. The code comparison must use one common file budget and manifest; historical 5k-versus-40k results remain descriptive only. Fusion-position and random-init claims remain out of scope unless new checkpoint-matched experiments are explicitly budgeted.

## Statistical and reporting rules

Model selection uses validation PPL; the test split is evaluated once per frozen run. Seeds 42, 123, and 456 are paired across methods. Report each seed, mean and standard deviation, and paired PPL differences. With only three seeds, do not base claims on null-hypothesis significance tests. Use effect direction and magnitude, and state the limited seed count.

For token diagnostics, chunks are the resampling and cross-validation unit. Report JSD-bin sample counts, bootstrap confidence intervals, held-out predictive deltas, and calibration plots. Apply Holm correction if more than three pre-registered inferential comparisons are reported. Exploratory plots and failed arms remain in the appendix or experiment journal.

Primary method success requires all of the following:

1. no increase in student inference parameters or latency beyond measurement noise;
2. PTB mean test PPL no more than 0.3 worse than fixed KD;
3. at least 50% reduction of the TinyStories fixed-KD excess PPL relative to CE-only;
4. the same direction for the CRBD-versus-fixed-KD paired difference in at least two of three seeds;
5. mechanism controls C6 and C7 weaker than C5 in the predicted direction.

These thresholds are pre-registered recommendations and become frozen only after author approval.

## Monitoring and outputs

- **Remote working directory**: `/workspace/grassmannflows/grassmann-flows`
- **Execution wrapper**: `~/fuwuqi/agent-tools/exec_grassmann.sh`
- **Per-run timeout**: PTB 30 minutes; TinyStories 14 hours; diagnostic 3 hours
- **Monitor files**: `logs/<experiment-name>.log`, the new run `config.json`, `distill_metrics.jsonl`, and `summary.json`
- **Failure policy**: preserve the failed directory and log; diagnose and request a new decision; never silently retry or change hyperparameters
- **Expected result table**: `research/experiments/post_rejection_program/results.csv`
- **Mechanism table**: `research/experiments/h1_branch_disagreement/token_diagnostics.csv`
- **Analysis report**: `research/experiments/post_rejection_program/analysis.md`
- **Figures**: `research/experiments/post_rejection_program/figures/`

Every completed run must record the teacher, initialization checkpoint, dataset manifest, git diff hash or snapshot identifier, seed, loss mode, coefficients, temperature, selection metric, stop reason, elapsed time, and GPU. Existing checkpoints and historical results are read-only.

## Decision requested

The recommended authorization is staged: execute A1, then A2, then Stage B; return with observed results before implementing or training CRBD. This spends about 25 GPU-hours before the next decision and prevents a 100+ GPU-hour method sweep on an unsupported mechanism.

Alternative author decisions are: evidence-only (A1+A2, no new method), or exhaustive (authorize Stages A--D and the full TinyStories seed-specific initialization cost). The exhaustive option is not recommended before the Stage B/C gates.
