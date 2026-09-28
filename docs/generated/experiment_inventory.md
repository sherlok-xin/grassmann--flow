# Experiment Artifact Inventory

Generated: 2026-09-16T11:52:11

This inventory is generated without moving or deleting any artifact. The CSV registry is the source for detailed filtering and matched-run audits.

## Coverage

- Run directories indexed: 194
- Complete runs: 166
- Config-only runs: 26
- Summary-only runs: 0
- Other incomplete runs: 2
- Exact replicate groups with at least two completed rows: 16

## Runs by family

- `distill_experiments`: 106
- `experiments`: 39
- `hybrid_experiments`: 46
- `transformer_experiments`: 3

## Rows by dataset label

- `code`: 24
- `ptb`: 94
- `tinystories`: 35
- `wikitext2`: 43

## Storage

- `outputs/benchmark_reports`: 7.51 KiB
- `outputs/distill_experiments`: 9.70 GiB
- `outputs/experiments`: 2.75 GiB
- `outputs/hybrid_experiments`: 4.63 GiB
- `outputs/subspace_analysis`: 441.52 MiB
- `outputs/transformer_experiments`: 171.12 MiB
- `outputs/wikitext2_reproduction`: 135.02 MiB
- `logs`: 895.47 MiB

## Checkpoint manifest

- Checkpoints indexed: 185
- `keep_referenced`: 168 files, 16.17 GiB
- `review_unreferenced`: 0 files, 0.00 GiB
- `archive_candidate_incomplete`: 17 files, 1.50 GiB

Most storage is held by model checkpoints. Cleanup should therefore be based on an explicit keep/archive manifest, never on directory age alone.

## Incomplete run directories

- `outputs/distill_experiments/20260413_055429_ptb_hybrid_lite_warmstart_192x48_l6_kd005`: config_only
- `outputs/distill_experiments/20260413_085354_ptb_transformer_l6_stage1_kd005_e1_seed789_smoke`: config_only
- `outputs/distill_experiments/20260515_014145_wt2_warmstart_kd010`: config_only
- `outputs/distill_experiments/20260515_015700_wt2_warmstart_kd002`: config_only
- `outputs/distill_experiments/20260515_015701_wt2_warmstart_kd020`: config_only
- `outputs/distill_experiments/20260515_020932_wt2_warmstart_kd002`: config_only
- `outputs/distill_experiments/20260515_020933_wt2_warmstart_kd020`: config_only
- `outputs/distill_experiments/20260515_021831_wt2_warmstart_kd002`: config_only
- `outputs/distill_experiments/20260515_021831_wt2_warmstart_kd020`: config_only
- `outputs/distill_experiments/20260515_022154_wt2_warmstart_kd002`: config_only
- `outputs/distill_experiments/20260515_022154_wt2_warmstart_kd020`: config_only
- `outputs/distill_experiments/20260515_022421_wt2_warmstart_kd020`: config_only
- `outputs/distill_experiments/20260515_022422_wt2_warmstart_kd002`: config_only
- `outputs/distill_experiments/20260515_112045_ptb_warmstart_kd10`: config_only
- `outputs/distill_experiments/20260530_060326_code_ws_kd30`: config_only
- `outputs/distill_experiments/20260916_034049_h0_ptb_smoke_legacy_a002_seed42`: config_only
- `outputs/hybrid_experiments/20260515_021200_ts_latefusion_last1_joint_e10`: config_only
- `outputs/hybrid_experiments/20260515_021326_ts_latefusion_last1_joint_e10`: config_only
- `outputs/hybrid_experiments/20260515_021348_ts_latefusion_last1_joint_e10`: config_only
- `outputs/hybrid_experiments/20260515_021401_ts_latefusion_last1_joint_e10`: config_only
- `outputs/hybrid_experiments/20260515_021533_ts_latefusion_last1_joint_e10`: config_only
- `outputs/hybrid_experiments/20260515_021554_ts_latefusion_last1_joint_e10`: incomplete
- `outputs/hybrid_experiments/20260515_021640_ts_latefusion_last1_joint_e10`: incomplete
- `outputs/hybrid_experiments/20260527_072902_code_latefusion_teacher_joint`: config_only
- `outputs/hybrid_experiments/20260527_074051_code_hybrid_lite_baseline_ce_scratch`: config_only
- `outputs/hybrid_experiments/20260527_074223_code_hybrid_lite_baseline_ce_scratch`: config_only
- `outputs/hybrid_experiments/20260527_074319_code_hybrid_lite_baseline_ce_scratch`: config_only
- `outputs/hybrid_experiments/20260527_074805_code_hybrid_lite_baseline_ce_scratch`: config_only
