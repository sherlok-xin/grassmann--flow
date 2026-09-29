# Phase 2D Provenance

## Start snapshot

- Experiment start date: 2026-09-29.
- Git commit at start: `9fec0ac48d6011d8f40ba52aabea7734353dbaa2`.
- Git state at start: clean `main`, tracking `personal/main`.
- Execution host for heavy work: server `10.42.0.197`, project path `/workspace/grassmannflows/grassmann-flows`.
- WikiText-2 dataset manifest: `research/snapshots/20260920/dataset_manifest_wikitext2.json`.
- Dataset-manifest SHA256: `2534df9ffce9a64cd8dc49ad267a1cb549c42fd8fd219b836bf15499de7ab0a5`.
- Validation selection seed: 20260920.
- Required selected-index SHA256: `236633612db1e3144b8f3187a5bfb1795f558a50f26368252a8adf3f0b49413f`.

At the start gate, all four RTX 3090 devices were idle. The two teacher hashes and three S0 hashes were recomputed on server 197 and matched the frozen values in `experiment_plan.md`. Exact commands and generated hashes are retained in `raw/start_manifest.json` and the phase logs as execution proceeds.

Phase 2C C0/C1 endpoints are reused rather than rerun. Their compact configs, summaries, and metrics remain under `research/experiments/phase2c_multiseed_teacher_utility/raw/formal/`.

## Source hashes at execution

- Continuation trainer: `5217f3d4a8b5e35ee4852194346f67c03d00d0a4ae829f0ed11770c865359c41`.
- KD loss implementation: `4a7d002db4f068d299d69e97832ca98c0d03644912c4e1d898ceb4a81592374d`.
- Teacher-integrity audit: `0be07da0513fa526e0005aa56158793c6809247fb4c674eb600ac8c020050ac2`.
- Teacher preflight: `387d941ee9788752889854c6314ff3ea98d9e8648110db628ee05de460a4d765`.
- Gradient-scale preflight: `900bf80697ac35448de4c11686f905341300156a224e3ecdbc49daa90a8b1b4b`.
- Preflight launcher: `cac83b66585168d0456c0a2ef7d74d738338a0a20ec47268c057b16d2ab41002`.
- Smoke launcher: `7950ee7dc422aa2c974074f7bbc132954b6741ab939bfbb801acfed507bb038a`.

Exact evaluation commands are frozen in `run_preflight.sh`; the exact reduced-data training command is frozen in `run_smoke.sh`. Formal D2 training was initially withheld because that smoke gate failed.

## Gate outcomes

The teacher-integrity audit passed every invariant and reproduced 179 differing tensors out of 191. The offline teacher and gradient preflights completed for both teachers and all three S0 checkpoints. Full-parameter gradient ratios `R_param = G_A/G_J` were 1.102792, 1.148522, and 1.093056 for seeds 42, 123, and 456. No `KD_SCALE_MISMATCH` flag was triggered, so no D2-scale arm was created.

The preregistered one-epoch, 2,000-line Teacher-A smoke completed at `outputs/distill_experiments/20260929_043856_phase2d_smoke_wt2_d2_a_seed42`. All scalar losses were finite, the effective alpha and S0 hash were exact, and the selected checkpoint reloaded successfully. However, its clipping fraction was 1.00 and its AMP overflow and non-finite fractions were both 0.16. These violate the frozen thresholds of clipping below 0.95 and overflow/non-finite below 0.05. Formal D2 training was therefore not authorized and was not launched. The failed gate is a protocol outcome, not a Teacher-A endpoint result.

After reviewing the matched historical observation that the reduced-data Teacher-J smoke also had clipping 1.00 and AMP overflow/non-finite fractions 0.16, while its full-data epoch had clipping 0.5476 and overflow/non-finite fractions 0.0136, the research lead explicitly authorized a protocol amendment on 2026-09-29. The amendment requires a matched full-data one-epoch Teacher-J/Teacher-A stability gate with the original absolute thresholds and no endpoint-based selection. The commands are frozen in `run_matched_full_epoch_smoke.sh`, and the resulting gate is evaluated by `evaluate_matched_smoke_gate.py` before any formal D2 launch.

The matched full-data gate completed before formal training. Teacher J had clipping fraction 0.234694 and AMP overflow/non-finite fractions 0.013605; Teacher A had clipping fraction 0.275510 and AMP overflow/non-finite fractions 0.013605. Both arms used 2,391,645 valid training tokens, passed every identity and numerical check, and produced reloadable checkpoints. The amended gate therefore authorized the three frozen formal D2 endpoints. The one-epoch validation and test values were not used for selection.

## Formal execution and collection

The exact three-seed launcher is `run_formal.sh`. It changed only the teacher checkpoint relative to the reused Phase 2C C1 protocol and ran Teacher A on GPUs 0, 1, and 2 for seeds 42, 123, and 456. All three processes exited successfully. The completed run directories are:

- seed 42: `outputs/distill_experiments/20260929_075825_phase2d_wt2_d2_a_seed42`
- seed 123: `outputs/distill_experiments/20260929_075826_phase2d_wt2_d2_a_seed123`
- seed 456: `outputs/distill_experiments/20260929_075826_phase2d_wt2_d2_a_seed456`

`collect_results.py` rejected incomplete or duplicate runs and verified the Teacher-A checkpoint hash, seed-specific S0 hashes, effective alpha, objective, all frozen training invariants, full-data setting, ten epoch records, finite stored metrics, and reused Phase 2C control hashes. The validation-selected Teacher-A checkpoint SHA256 values are `cab608a725a409d8cae7272888e61821a150ddc86705a7045c4a7de77e2e4ba9`, `2589a0df2b67f5898f2178e4cc0188d7474c649f3f5916f5acaf3629156ca10d`, and `349d14ab675912532217fbc28b46993de019904b76c560c0f12ed5ba6699a115` for seeds 42, 123, and 456.

Teacher-A test NLL is 4.293136, 4.312882, and 4.306036. `Gain_A` is -0.022394, -0.035362, and -0.025206; `Q=NLL_A-NLL_J` is +0.184895, +0.182699, and +0.182364. Full statistics and every reused endpoint are retained in `results_multiseed.csv` and `raw/results_summary.json`. Compact configs, summaries, metric JSONL files, and reports are copied under `raw/formal/`; checkpoints and full logs remain excluded from GitHub.

Final execution-script SHA256 values are:

- matched full-epoch launcher: `758d3aa17385c51c316c59175224916a177fb4a0a51ca4f28b7c7d29524f0ecf`
- amended-gate evaluator: `b8eddcd1f819a11e6024c5b5f8afe9ed84ef7cf8f461a7b09907eca7d4dff3ff`
- formal launcher: `9a4e900c06b258028784560a0f6066483a10c8055b438a572d30a243c8f836fb`
- strict collector: `b033276ad4f9a58355e0813e8bd35f5faf5c9a61160b3f19b50dc49a5ba74f3d`

The original failed smoke, the authorized amendment, and the successful formal execution are all preserved. No dataset, historical checkpoint, manuscript, KD loss, lambda, teacher alpha, or endpoint definition was changed.
