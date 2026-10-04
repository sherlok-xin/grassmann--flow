# Phase4C — fixed-step, disjoint modern continuation control

## Decision

`DECISION = STOP_MODERN_CONTINUATION_NOT_ESTABLISHED`

The validation-only CE gate failed. Best LR candidate1e-5 selected a trained
step977 checkpoint and improved validation NLL by0.004849292809, below the
preregistered0.01 threshold. All three CE runs completed normally with exactly
8M additional targets each. No KD run, initial KD-gradient probe or new test
evaluation was performed. The entire modern extension is stopped; no more LR,
budget, clipping, method or seed experiment was launched.

This is not evidence of clean modern negative transfer or beneficial KD.
CE shows a small positive validation improvement, so this must not be described
as zero headroom or proof that all further training is useless. It failed to
establish the specified non-degenerate regime at the required magnitude.
Test improvement and CE-versus-KD gains remain unmeasured here.

## Fixed S0 and frozen disjoint continuation

The exact seed42 preparation step256 checkpoint exists, with SHA256 matching
the original record. Archived summary and metrics agree on step256 and
2,097,152 targets. Recomputed full validation NLL is2.9671119410592275.
The historically validation-selected file happens to equal this time point.
Phase4C chose it by the requested fixed step, not a new validation search;
no S0 retraining occurred.

continuation_manifest.json was frozen and committed before outcomes. It
excludes8192 chunks consumed in preparing early S0, filters the full seed4242
train permutation and fixes31250 new chunks. Their8M target positions neither
overlap S0 targets nor repeat. Adjacent chunks may share one context token,
not a supervised target position. Only train.bin indices are used; original
validation/test splits and tokenizer/data revisions remain unchanged.
Manifest SHA256: b81200c6648a8dfbe2a3d9b432800377152162f490ac7290b3f91fb8492974d1.

The same adapted frozen360M teacher is retained, not retrained. Archived
validation NLL2.74748592931419 gives residual advantage over this S0 of
0.2196260117450377. This likelihood advantage alone does not establish
distillability; Phase4C never reached an authorized KD comparison.

## CE-only gate results

CE_improvement_val=NLL_val(S0)−NLL_val(selected_CE), including S0 at step0.
All candidates select trained step977, but none meets improvement>=.01.
These are three LR candidates for ONE student seed, not three replications;
no cross-LR mean/sample SD or independent-seed3/3 robustness claim is made.

| Peak LR | Selected validation NLL | CE_improvement_val | Selected step | Gate |
|---|---:|---:|---:|---|
| 5e-6 | 2.962897499546 | +0.004214441514 | 977 | FAIL |
| 1e-5 | 2.962262648251 | +0.004849292809 | 977 | FAIL |
| 2e-5 | 2.963171304853 | +0.003940636206 | 977 | FAIL |

Full validation schedule0/256/512/768/977; ties earliest including S0.
LR ties would select the smaller LR. No test result selected an LR. There
is no conditional KD protocol because the gate failed. Full per-checkpoint
trajectories remain in raw/*metrics.jsonl.gz.

## Optimization audit

Fresh AdamW wd.01, betas(.9,.95), eps1e-8; batch32/micro4, seq256, fixed
clip1, FP32 weights/BF16 autocast/FP32 losses and SDPA. Unchanged Phase4A
scheduler form:49 warmup updates plus cosine over977 updates. Last update
has18 rows/4608 targets. Formal budget24M targets; disposable CE-only smoke
used another16384 without saving a state. No GradScaler or clip/teacher change.

| Peak LR | Clip fraction | Mean preclip norm | Maximum preclip norm | Nonfinite / overflow fraction |
|---|---:|---:|---:|---:|
| 5e-6 | 0.060388946 | 0.940982208 | 2.301602364 | 0 / 0 |
| 1e-5 | 0.079836233 | 0.943699946 | 1.375428677 | 0 / 0 |
| 2e-5 | 0.082906858 | 0.959165336 | 14.602655411 | 0 / 0 |

No CE arm is clip-saturated under the preregistered>=.95 rule; this says
nothing about unrun KD clipping. Losses, preclip norms and post-update
parameters passed finite checks throughout. The2e-5 arm has a large finite
norm outlier14.602655411 at step656, clipped at the unchanged threshold.
Cause and trajectory impact are UNKNOWN; it is disclosed, not excluded,
retried or used to change the gate. The best LR belongs to the separate1e-5
arm. See gradient_event_audit.json and full compressed logs. All historical
Phase4B CLIP_SATURATION_WARNING flags remain in its unchanged report.

KD025/KD1 selected steps, clipping, KL, initial KD/CE gradient ratio, test
values and Gain_lambda are NOT_RUN/NOT_MEASURED, not zero. results_seed42.csv
leaves them blank with explicit gate-failure status. Historical test numbers
are not inserted as new Phase4C endpoints.

## Interpretation, provenance and STOP

The bounded conclusion: this fixed early S0, disjoint8M-target stream and
three authorized LRs did not establish>=.01 validation CE headroom. This
does not prove impossibility for all modern continuation protocols or remove
Phase4B degeneracy. No clean negative-transfer or beneficial KD claim can
be added. Pretraining overlap, different135M/360M pretraining budgets, fixed
subset, short context, one student seed/teacher and validation LR selection
limit inference. No Grassmann-specific claim is restored.

Protocol commit dabd3f6e31e1a371cdf4a78ac7ca3c6bd9cfd65f preceded formal CE
execution. Launcher608564 owned only CE PIDs608566/608567/608568 on GPU0/1/2;
all exited0. Five new protocol and four original numerical tests plus CE-only
smoke passed. completed_run_audit.json is PASS: budgets, selection, state/order
hashes, finite logs and immutable inputs verified. Large states/data remain
outputs-only. Old Phase4A/B, manuscript, research/final_evidence/, datasets/
and checkpoints/ are unchanged. Compact artifacts only, local Git.
Phase4C COMPLETE / STOPPED; no automatic seed123/456 or further tuning.
