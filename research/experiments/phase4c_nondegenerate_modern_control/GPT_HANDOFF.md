# Phase4C — COMPLETE / STOPPED

`DECISION = STOP_MODERN_CONTINUATION_NOT_ESTABLISHED`

Read REPORT.md, ce_lr_gate.csv, results_seed42.csv, optimization_audit.csv,
continuation_manifest.json and provenance.md. completed_run_audit.json is PASS.
Fixed seed42 preparation step256 was reused with verified hash/metric/order
chain, not retrained or reselected by validation. It happens to equal the old
historically selected state; disclose that coincidence. Frozen8M manifest
excludes its8192 consumed chunks and fixes31250 new target chunks with no
overlap/repetition or heldout training.

Three CE candidates each completed8M targets/977 updates, ONE student seed.
S0 validation NLL=2.9671119410592275. Best validation NLL / improvement:
5e-6:2.9628974995455346 / +0.004214441513692879;
1e-5:2.9622626482505976 / +0.00484929280862989;
2e-5:2.9631713048530237 / +0.003940636206203774.
All select trained step977 but fall below the preregistered.01 gate. Best LR
candidate1e-5; no conditional KD protocol frozen/activated. These LR trials
are NOT three independent seeds; do not report sample SD/3-seed robustness.

No KD025/KD1, initial gradient-ratio probe or new test forward ran. Fields are
blank/NOT_RUN, never old test values or zeros. Teacher validation residual
over S0 is +0.2196260117450377 from the archived frozen teacher. Phase4C test
CE_improvement/Gain/teacher residual remain unmeasured by the stopping rule.

CE clip fractions.060388946/.079836233/.082906858, no saturation; observed
nonfinite/overflow0. The2e-5 arm has a finite norm outlier14.602655411 at656,
clipped; cause/impact UNKNOWN. No retry, exclusion or gate/threshold change;
best LR is another arm. Keep old Phase4B clipping and continuation warnings.

Small positive validation headroom exists but the required non-degenerate
regime was NOT established. Do not call this zero CE improvement or universal
continuation failure. No clean modern negative/beneficial KD conclusion can
be made. Phase4B degeneracy is not repaired. No better-teacher harm, corpus/
clipping causality, universal modern failure or Grassmann mechanism claim.

Formal protocol commit dabd3f6e31e1a371cdf4a78ac7ca3c6bd9cfd65f preceded
training. Heavy compute remote197 only; local Git. Raw summaries/compressed
logs and exact source/input hashes preserve provenance. No old Phase4A/B,
manuscript/final_evidence, datasets/checkpoints or .gitignore edit staged.

STOP the entire modern extension. No extra LR, budget, clipping sweep,
lambda5, new method or automatic seed123/456 replication is authorized.
