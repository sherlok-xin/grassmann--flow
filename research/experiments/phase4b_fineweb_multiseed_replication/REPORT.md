# Phase 4B — FineWeb-Edu modern warm-start KD replication

## Decision and scope

`DECISION = FINEWEB_L1_L5_NEGATIVE_REPLICATED`

Both lambda=1 and lambda=5 have negative CE-relative test gains for all three
independently prepared students, with validation/test directions agreeing in
all six comparisons. The lambda=5-only flag is also satisfied:
`FINEWEB_L5_NEGATIVE_REPLICATED`. This is not the mixed-sign strength-dependent
case or failed replication. These are protocol-specific results, not a claim
that KD is universally harmful or that better teachers generally harm students.

Under the frozen SmolLM2-360M → 135M warm-start continuation protocol, KD
produces reproducible negative transfer on the fixed FineWeb-Edu subset despite
the teacher having substantially better held-out likelihood. This conclusion
must retain both sustained clipping and the continuation-regime limitation:
even CE continuation is worse than S0 for all three seeds, and all nine
validation-selected `best_including_S0` diagnostic choices return S0.

Phase4B is COMPLETE and STOPPED. Eight new formal runs consumed exactly 80M
predicted training targets. Seed42, the adapted teacher and data were reused,
not retrained or retokenized. No TinyStories replication, new method, tuning,
manuscript edit or final_evidence edit occurred. No subsequent training,
clipping-threshold control or modern experiment is authorized.

## Primary matched effects

Gain_1=NLL_CE−NLL_KD1 and Gain_5=NLL_CE−NLL_KD5; negative means KD harms the
matched CE continuation. Sample SD uses ddof=1 over the three paired gains.
The primary endpoint still excludes continuation step 0. No additional magnitude
cutoff was imposed on the lead's three-seed replication sign rules.

| Student seed | Gain_1, test | Gain_5, test | Gain_1, validation | Gain_5, validation | Val/test agrees, L1/L5 |
|---|---:|---:|---:|---:|---|
| 42, reused | -0.015718598299 | -0.046403663516 | -0.015706368649 | -0.046727056123 | yes / yes |
| 123 | -0.015470071017 | -0.046078763438 | -0.015470378595 | -0.046465882601 | yes / yes |
| 456 | -0.009462713488 | -0.039289819652 | -0.009913631193 | -0.040036665275 | yes / yes |
| Mean | -0.013550460935 | -0.043924082202 | | | |
| Sample SD | 0.003542273400 | 0.004016675497 | | | |
| Negative signs | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 for each strength |

The lambda=1 effect is smaller than lambda=5 and below the original single-seed
pilot's 0.02 clear-effect magnitude, but replicates in sign across independent
student preparations. Lambda5 exceeds 0.02 in absolute value for each seed.
Lower-strength harm is not restricted to the fully saturated lambda=5 arms,
although lambda=1 also clips frequently and remains optimization-constrained.
Three signs do not establish population-wide significance or universality.

## Selected endpoints and preparation variation

| Seed | S0 test NLL | CE test NLL | KD1 test NLL | KD5 test NLL |
|---|---:|---:|---:|---:|
| 42 | 2.944338195891 | 2.949900313167 | 2.965618911467 | 2.996303976684 |
| 123 | 2.944865172453 | 2.950133177297 | 2.965603248314 | 2.996211940735 |
| 456 | 2.944958744365 | 2.957883156270 | 2.967345869758 | 2.997172975922 |

| Seed | S0 preparation selected step | CE selected step | KD1 selected step | KD5 selected step |
|---|---:|---:|---:|---:|
| 42 | 256 | 1221 | 1221 | 1221 |
| 123 | 256 | 1221 | 1221 | 1221 |
| 456 | 1221 | 256 | 1221 | 1221 |

Every run completed 10M targets regardless of the selected step. Step 256
corresponds to 2,097,152 targets; step 1221 to 10M. Do not substitute the final
unselected S0 or force a common best step. Seed456 CE selected an early trained
checkpoint, which contributes to variation in the paired effect size. The
selected states were verified by hash and full validation reevaluation; the
GPU3 CE checkpoint also reproduced its validation NLL on GPU0 within the
predefined1e-5 tolerance. PPL and unrounded endpoint values are in raw final
results. No test value selected a state or teacher.

## Teacher quality and S0 → CE change

The same adapted 360M teacher has validation/test NLL 2.747485929314 /
2.726017006990 and test PPL 15.271937699026. These metrics are reused from
Phase4A with the same teacher SHA256. Residual advantage is NLL_S0−NLL_teacher.
S0→CE change is NLL_CE−NLL_S0, so positive means further CE degrades likelihood.

| Seed | Teacher residual, validation | Teacher residual, test | S0→CE, validation | S0→CE, test |
|---|---:|---:|---:|---:|
| 42 | +0.219626011745 | +0.218321188901 | +0.006042959097 | +0.005562117276 |
| 123 | +0.219904570533 | +0.218848165462 | +0.005676122436 | +0.005268004845 |
| 456 | +0.220068013559 | +0.218941737375 | +0.012954244717 | +0.012924411904 |
| Mean | +0.219866198612 | +0.218703697246 | +0.008224442083 | +0.007918178008 |
| Sample SD | 0.000223485355 | 0.000334549550 | 0.004100233774 | 0.004338019006 |

The teacher is better for all three S0 states, while both KD strengths are
worse than matched CE. This is a counterexample to sufficient prediction of
beneficial KD from global teacher likelihood alone in this setting, not a
causal intervention on teacher quality. Teacher quality itself is unchanged
across student seeds. Further CE offers no held-out improvement in this
continuation regime, and KD adds deterioration beyond that CE baseline.

## Secondary best_including_S0 diagnostic

This is no-training analysis, not a replacement primary endpoint. The original
validation-selected adapted S0 is allowed as continuation step 0 alongside
each arm's already-selected trained checkpoint. Only validation NLL selects
the candidate; ties choose S0. The choice artifact was written before new test
forwards and is separately archived at commit c89cf5b4c8ed060fa5416c1c817448273c6c394a.
Its SHA256 is 943442b28bd90f2cbd31fb0a0337b7a31c169446a021e9007c015073558580f5.
The commit archives the choice; the executable ordering, not an assumed Git
timestamp relative to an in-flight evaluation, establishes pre-test selection.

| Seed | CE choice | KD1 choice | KD5 choice | Diagnostic Gain_1 | Diagnostic Gain_5 |
|---|---|---|---|---:|---:|
| 42 | S0, step 0 | S0, step 0 | S0, step 0 | 0 | 0 |
| 123 | S0, step 0 | S0, step 0 | S0, step 0 | 0 | 0 |
| 456 | S0, step 0 | S0, step 0 | S0, step 0 | 0 | 0 |

All 9/9 arm choices select S0. Hence both diagnostic paired gains are exactly
zero within each seed because all three arms return the identical S0 state.
That state is target-adapted, not the raw pretrained base. This does not erase
the frozen trained-step negative comparisons, but narrows their practical
interpretation: harm occurs when continuation is performed in a regime where
stopping at S0 already gives better validation and test likelihood. Do not
present the primary results as a failure of every warm-start KD or deployment
policy. best_including_s0.csv preserves all nine choices and candidate NLLs.

## Optimization audit and warnings

| Seed | Arm | Clip fraction | Mean preclip norm | Max preclip norm | Nonfinite fraction |
|---|---|---:|---:|---:|---:|
| 42 | CE | 0.058968 | 0.929848 | 1.394476 | 0 |
| 42 | KD1 | 0.850123 | 1.084600 | 2.502529 | 0 |
| 42 | KD5 | 1.000000 | 2.557157 | 11.615945 | 0 |
| 123 | CE | 0.056511 | 0.929663 | 1.203820 | 0 |
| 123 | KD1 | 0.854218 | 1.090724 | 3.144516 | 0 |
| 123 | KD5 | 1.000000 | 2.601404 | 19.092007 | 0 |
| 456 | CE | 0.045045 | 0.994147 | 86.482155 | 0 |
| 456 | KD1 | 0.798526 | 1.074831 | 2.708021 | 0 |
| 456 | KD5 | 1.000000 | 2.545350 | 12.600890 | 0 |

`CLIP_SATURATION_WARNING` is permanent for all three lambda=5 arms: every
optimizer step clips. Lambda1 is not fully saturated but clips on 79.85–85.42%
of steps. This strengthens the boundary beyond only the fully saturated
lambda=5 condition without removing the optimizer confound. No lambda or clip
value was changed. Clipping is not established as the cause of harm.

`CE456_GRADIENT_SPIKE_WARNING`: seed456 CE has one finite preclip-norm outlier
of 86.482155 at step 711, with CE loss 2.787737. It is the only CE456 norm above 2;
neighboring norms are approximately 0.914–1.025. The step was clipped and all
parameters remained finite. Its selected step 256 snapshot predates the event,
so the event does not directly change those saved weights. Its influence on
later trajectory and failure to select a later checkpoint cannot be isolated
here. Cause is UNKNOWN; do not attribute it to the data, GPU or clipping.
raw/ce456_gradient_event.json records the telemetry and matched-seed context.
No post-hoc failure threshold, exclusion, retry or new control was introduced.

All preparation/continuation losses, gradients and parameters remained finite,
with observed overflow/nonfinite fractions zero. BF16 has no GradScaler;
these are observed nonfinite-event fields, not FP16 scaler skip rates.
Norm/KL means use optimizer-step arithmetic averages; held-out NLL is exactly
token-weighted. optimization_audit.csv also includes the three S0 preparations,
initial full-parameter gradient norms/ratios, KL, selected tokens, memory and
wall time. Wall times are not an efficiency comparison because GPU placement
and shared-server conditions differ. No inference-speed claim follows.

## Protocol, provenance and limits

Protocol commit 1a525c379d7b72e809847dfed6ef1bf71fd0fe50 was pushed before
formal runs. Exact-source equivalence verifies that the Phase4B loop changes
only seeds, output namespaces, shared-module references and restricted CLI
relative to Phase4A. The original train/core files remain unchanged. Prepare
each new 135M S0 from the same official pretrained snapshot with independent
seed-specific target-adaptation order; all continuations retain permutation
seed 4242. Independence is three adapted S0 states, not three pretrained
models, teachers or corpus samples. All branches within a seed start from
the same selected state and reset the optimizer as in Phase4A.

Data remain the same 20M/1M/1M deterministic encoded-token splits, revision
87f09149ef4734204d70ed1d046ddc9ca3f2b8f9; validation/test each average 999,999
predicted targets. AdamW lr=5e-5, wd=0.01, betas=(0.9,0.95), eps=1e-8,5% warmup + cosine,
T=2, token_mean forwardKL, FP32 weights/BF16 autocast/FP32 losses,clip=1,
sequence length 256 / global batch 32 / microbatch 4,trained validation steps 256/512/768/1024/1221
are unchanged. All heavy computation ran on 10.42.0.197, using the existing
isolated Transformers 4.46.3/tokenizers 0.20.3 installation. No framework or
dependency replacement occurred.

completed_run_audit.json is PASS: eight new complete runs / 80M targets,
four reused seed42 student runs, three distinct S0 state/order hashes,
matched continuation order, frozen teacher/data, exact selection, finite
telemetry and unchanged manuscript/evidence. Compact raw summaries and
compressed step telemetry accompany the CSVs; model states/tokens remain
under ignored outputs and are not committed. Plots are reproducible from
compact files and exported as vector PDF and 300-dpi PNG.

The fixed subset, only three adaptation seeds, one frozen teacher, short
context, finite budget and selection constrain generalization. FineWeb-Edu
appeared in SmolLM2 pretraining, with unknown exact overlap; the model sizes
have different pretraining budgets. Sustained/frequent clipping, the CE456
outlier and the all-S0 early-stop diagnostic further limit mechanism claims.
Do not claim that FineWeb-Edu causes negative transfer, clipping causes it,
better teachers generally harm KD, all modern KD fails, or any Grassmann,
Plucker-geometry or architectural-heterogeneity advantage.

The final_evidence package and manuscript remain unchanged. STOP. Review of
these bounded results is the only next action; no automatic clipping control,
new strength, dataset, modern experiment or manuscript writing is authorized.
