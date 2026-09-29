# Phase 2E Provenance

## Start snapshot

- Experiment start date: 2026-09-29.
- Git commit at start: `a9642cf9393defa2f335811ea77a069becb1747a`.
- Git state at start: clean `main`, matching `personal/main`.
- Heavy execution host: `10.42.0.197`, container project path `/workspace/grassmannflows/grassmann-flows`.
- Teacher checkpoint expected SHA256: `a5ad284f6895c0b4ed516c855c1b149e6636b945173aa2e842b8132299582694`.
- Dataset manifest: `research/snapshots/20260920/dataset_manifest_wikitext2.json`.
- Dataset-manifest SHA256: `2534df9ffce9a64cd8dc49ad267a1cb549c42fd8fd219b836bf15499de7ab0a5`.
- Validation selection seed: 20260920.
- Required selected-index SHA256: `236633612db1e3144b8f3187a5bfb1795f558a50f26368252a8adf3f0b49413f`.

## Source state

- Continuation trainer SHA256: `5217f3d4a8b5e35ee4852194346f67c03d00d0a4ae829f0ed11770c865359c41`.
- KD loss implementation SHA256: `4a7d002db4f068d299d69e97832ca98c0d03644912c4e1d898ceb4a81592374d`.

Phase 2C C0/T_F/T_G endpoints are reused rather than rerun. Their compact configs, summaries, metrics, checkpoint hashes, and collector validation remain under `research/experiments/phase2c_multiseed_teacher_utility/`. Phase 2E adds only T_T formal training after all preflight and smoke gates pass.

## Preflight completion

The immutable-input audit, teacher utility evaluation, branch-complementarity analysis, and gradient calibration completed on 2026-09-29. All teacher and S0 checkpoint hashes matched. The validation selection hash was `236633612db1e3144b8f3187a5bfb1795f558a50f26368252a8adf3f0b49413f`. Teacher NLL was `4.300645861868859` for F, `4.445646929177943` for T, and `4.652063519081662` for G. The three T/F full-parameter KD-gradient ratios were `1.3164257254`, `1.2788220243`, and `1.2323507323`; therefore the frozen scale-mismatch rule did not fire.

## Smoke-gate history

The reduced T smoke is retained at `outputs/distill_experiments/20260929_093434_phase2e_smoke_wt2_t_seed42`. Its compact gate record is `raw/smoke/reduced_smoke_summary.json`. It failed the frozen thresholds with clip fraction `1.0` and overflow/non-finite fractions `0.16`.

The fallback full-data arms are retained at `outputs/distill_experiments/20260929_093526_phase2e_full_epoch_smoke_F_seed42` and `outputs/distill_experiments/20260929_093527_phase2e_full_epoch_smoke_T_seed42`. F passed all checks. T had finite losses, correct hashes and alpha, a reloadable checkpoint, overflow `0.0136054422`, and non-finite fraction `0.0136054422`, but clipping remained `1.0`; hence the matched gate failed. Its compact record is `raw/smoke/matched_full_epoch_smoke_summary.json`.

No formal T endpoint was launched, inspected, or inferred. The frozen launcher will refuse to run while both gate records have `formal_training_authorized=false`.

## Authorized clipping amendment

On 2026-09-29, before any formal Transformer-only endpoint existed, the research lead explicitly authorized continuation with the observed full-data gate `clip_fraction=1.0`. The authorization requires finite parameter gradients, T/F parameter-gradient ratios within `[0.5, 2.0]`, and overflow/non-finite fractions remaining approximately `0.0136`. It forbids changes to alpha, lambda, temperature, AMP, the gradient clip value, training budget, data, student S0, or evaluation protocol.

The gate is re-evaluated with `--authorize-clip-saturation`; the original `gate_passed=false` is preserved while `formal_training_authorized` may become true only if every amendment check passes. All subsequent artifacts carry `CLIP_SATURATION_WARNING`. Formal runs must stop without hyperparameter repair on NaN, failure, materially increased overflow/non-finite fractions, or anomalous endpoints.
