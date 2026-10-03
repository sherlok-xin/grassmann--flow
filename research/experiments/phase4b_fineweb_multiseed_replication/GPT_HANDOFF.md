# Phase 4B — COMPLETE / STOPPED

`DECISION = FINEWEB_L1_L5_NEGATIVE_REPLICATED`

Both lambda=1 and lambda=5 are negative for 3/3 independently prepared
student seeds, with validation/test agreement in all six paired comparisons.
The additional flag FINEWEB_L5_NEGATIVE_REPLICATED is also satisfied.
No further training, clipping control, TinyStories replication, new method,
manuscript edit or final_evidence edit is authorized. Phase 3A remains terminal.

## Read first

REPORT.md contains the complete result and scientific limits.
results_multiseed.csv contains the three matched primary rows.
summary_statistics.csv contains paired means, sample SD and sign counts.
best_including_s0.csv contains the nine secondary diagnostic choices.
optimization_audit.csv contains all twelve reused/new student run audits.
selected_state_manifest.csv records exact state/order hashes.
completed_run_audit.json reports PASS; raw/ has summaries and compressed
step telemetry. experiment_plan.md and provenance.md define the locked protocol.

## Primary test gains

Gain = NLL_CE - NLL_KD. Negative means KD harms matched CE continuation.

| Seed | Gain_1 | Gain_5 |
|---|---:|---:|
| 42, reused | -0.015718598299 | -0.046403663516 |
| 123 | -0.015470071017 | -0.046078763438 |
| 456 | -0.009462713488 | -0.039289819652 |
| Mean | -0.013550460935 | -0.043924082202 |
| Sample SD | 0.003542273400 | 0.004016675497 |
| Negative signs | 3/3 | 3/3 |

All validation/test signs agree. Lambda=1 has a smaller but reproducible
negative effect under the sign rule, not the original pilot magnitude gate.
The result is bounded to the specified continuation protocol and fixed subset;
do not infer universal harm or teacher-quality causality.

## Step-zero diagnostic and continuation regime

All 9/9 best_including_S0 choices select the original adapted S0 using ONLY
validation NLL; diagnostic gains are zero because all arms return the same
state within each seed. This never changes the primary endpoint.
S0 is target-adapted, not the raw pretrained base. Further CE itself worsens
test NLL for 3/3 seeds: +0.005562117276 / +0.005268004845 / +0.012924411904,
mean +0.007918178008, sample SD 0.004338019006.
Therefore the observed KD-specific harm occurs in a broader regime where
continuation is unnecessary or harmful. Always state this limitation.

Teacher advantage over S0 is positive for every seed: validation mean
+0.219866198612, test mean +0.218703697246. A better held-out likelihood
does not ensure beneficial transfer in this setting. This is not evidence
that better teachers generally harm students.

## Optimization limits that must remain

CLIP_SATURATION_WARNING applies permanently to ALL three lambda=5 arms:
clip_fraction=1.0. Lambda=1 clipping is 0.850123 / 0.854218 / 0.798526.
It is not fully saturated, but still frequently clipped. All observed
overflow/nonfinite fractions are zero; BF16 has no GradScaler.
Do not claim clipping causes harm, tune coefficients or discard constrained arms.

CE456_GRADIENT_SPIKE_WARNING: CE seed456 has one finite norm outlier
86.482155 at step711, clipped. Its selected step256 state predates the event,
so the saved weights are not directly changed by it; an effect on later
trajectory/selection cannot be isolated. Cause UNKNOWN. See the raw event
audit. No post-hoc exclusion, gate change, retry or new control was introduced.

S0 selected preparation steps are 256 / 256 / 1221 for seeds42/123/456.
Primary CE selected steps are 1221 / 1221 / 256; all KD select1221.
Every formal run still consumed its full10M-target budget.

## Provenance and STOP

Eight new formal runs: two independent S0 preparations plus six continuations,
exactly80M targets. Seed42/teacher/data are reused, not rerun. Each new S0
starts from the same official base snapshot with independent target-adaptation
seed/order; all continuations retain matched permutation4242. One teacher,
one corpus subset and three adapted-student seeds, not three pretrained models.

Formal protocol commit 1a525c379d7b72e809847dfed6ef1bf71fd0fe50 preceded training.
Validation-only diagnostic choice SHA256:
943442b28bd90f2cbd31fb0a0337b7a31c169446a021e9007c015073558580f5.
Choice serialization precedes new test forwards in finish.py; the subsequent
Git commit archives it, not a separate claim about the in-flight test timestamp.

Original Phase4A train/core, data, teacher, manuscript and final_evidence remain
unchanged. FineWeb-Edu pretraining overlap, different model pretraining budgets,
fixed subset, short context, finite budget, selection and optimizer constraints
limit generalization. No Grassmann/Plucker/heterogeneity/method claim follows.
All heavy work was remote on10.42.0.197 using the unchanged isolated packages.
Large states/tokens are excluded from Git; plots use compact artifacts only.

Review REPORT.md and the CSVs. STOP; do not start another experiment or edit
the manuscript without a new explicit task.
