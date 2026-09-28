# Phase 2A Provenance

## Scope and state freeze

This audit is evaluation-only. It did not optimize any model, launch any KD continuation, modify a checkpoint, alter a dataset, expand CRBD, or edit the manuscript. The pre-analysis source state is frozen under `research/snapshots/20260920/`. The repository was on `main` at `67efbc158ad823f7196f0696415f6e32b5e2e2fa` with a substantially dirty working tree; the active source is not recoverable from that commit alone.

The canonical runtime was server `10.42.0.197`: Python 3.12.3, PyTorch 2.6.0a0+df5bbc09d1.nv24.11, CUDA 12.6 as reported by PyTorch, cuDNN 90501, and four NVIDIA GeForce RTX 3090 GPUs with 24,576 MiB each. Full package and GPU records are in `research/snapshots/20260920/remote_environment.txt`.

## Data and selection

Dataset records and lightweight hashes are in `research/snapshots/20260920/dataset_manifests.json`. All evaluations use the local GPT-2 tokenizer at `/workspace/grassmannflows/grassmann-flows/gpt2_local`, vocabulary size 50,257, no added special tokens, contiguous 256-token chunks, and causal evaluation on positions 1–255 of each chunk.

PTB uses all 350 validation chunks (89,250 evaluated next-token targets). WikiText-2, TinyStories, and CodeParrot common-5k each use 512 validation chunks selected without replacement by Python `random.Random(20260920)` and then sorted, for 130,560 evaluated targets per domain. The exact indices and their SHA256 digests are stored in each `raw/<domain>/results.json`. TinyStories retains the historical 98/2 train-derived validation split with split seed 42; the original TinyStories validation split is the test split. CodeParrot uses the first 5,000 non-empty records of each native split, matching the frozen teacher configuration despite its historical `dataset_name=ptb` label.

## Models and checkpoints

The canonical teacher run directories are:

- PTB: `outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint`
- WikiText-2: `outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint`
- TinyStories: `outputs/hybrid_experiments/20260515_022710_ts_latefusion_last1_joint_e10`
- CodeParrot common-5k: `outputs/hybrid_experiments/20260529_120718_code_teacher_last1_joint`

Warm-start S0 and WS+CE S1 paths are recorded in the raw manifests and `transfer_table.csv`. The evaluator resolves migrated historical paths with `train_distill_hybrid_lite_from_latefusion_teacher_v2.py`, loads all tensors with `weights_only=True`, and never writes model state.

## Fusion and metrics

The verified source convention in `train_distill_hybrid_lite_from_latefusion_teacher_v2.py` is `fused_logits = alpha * transformer_logits + (1 - alpha) * grassmann_logits`. All audited teachers use `late_k=1`, so scalar offline reconstruction is exact up to mixed-precision rounding. The maximum absolute reconstruction error is recorded per domain in `teacher_branch_metrics.csv`.

NLL is the mean next-token cross entropy. PPL is `exp(NLL)`. Entropy, JSD, KL, accuracy, and agreement are token means. `KL(teacher || student)` at T=1 is the primary distribution-distance value. Gradient compatibility follows the corrected Stage-B logit-gradient definition at temperature 2: the CE logit gradient is `p_student - one_hot(y)`, and the temperature-scaled KD logit gradient is `T * (p_student,T - p_teacher,T)`. These are logit-level descriptive diagnostics, not full parameter-gradient measurements and not causal estimates.

The alpha grid is 0.0 through 1.0 in increments of 0.1, plus each learned canonical alpha and the historical static alpha 0.5. Neighboring alpha points reuse the same branches, students, and validation tokens and must not be treated as independent experiments.

## Evidence tiers and limitations

The transfer outcome table uses exact test NLL from original summaries. PTB is the only matched three-seed token-normalized comparison. TinyStories is matched token-normalized seed 42 only. WikiText-2 and CodeParrot are legacy single-seed batch-normalized sweep winners. The cross-domain predictor table therefore mixes evidence tiers and is descriptive at n=4. It is not a statistical test of a general law.

Branch and alpha metrics are validation-only. PTB uses the full validation split; the remaining domains use deterministic subsets. Candidate teachers are selected only from PTB and TinyStories validation landscapes. No test result was used to select an alpha candidate.

## Reproduction scripts

`run_transfer_landscape.py` performs the checkpoint-only evaluation. `build_dataset_manifests.py` creates dataset manifests. `build_transfer_table.py` reconstructs the outcome table. `aggregate_results.py` creates the analysis CSV files and descriptive correlations. `build_candidate_teachers.py` selects candidate conditions under a declared NLL-margin rule. `plot_landscapes.py` exports the exploratory PDF and 300-dpi PNG figures.
