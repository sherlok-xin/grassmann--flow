# V2 evidence integration: frozen endpoints, revised paper identity

Base commit: `64db3aebd158f56373385db18dcdf056aeea4c52`. Prepared 2026-10-04.
This package integrates archived evidence only. No new training, evaluation,
diagnostic run, hyperparameter search or method development is authorized.
`research/final_evidence/` remains the immutable Phase-2 snapshot; V2 does not
rewrite its history. The old CAC source is also preserved.

## Question and comparison unit

What determines whether KD transfers useful information to a warm-start LM
student? The operational endpoint is `Gain = test_NLL_CE - test_NLL_KD`, using
validation-selected checkpoints. A positive value means KD improves on the
matched CE continuation, not merely on the initial student or a legacy random
initialization. This definition is an evaluation convention, not theoretical
novelty. Each CE/KD pair starts from the exact same S0 within a seed, with the
recorded matched budget, data/order, selection and optimizer settings. Training
states and independent student seeds are the relevant units; tokens, repeated
calibration subsets and reused baseline arms are not new independent runs.

Three signs and sample SD describe observed replication, not population-level
significance. Report SD over paired gains, not the difference of separate arm
SDs, and do not relabel SD as a confidence interval or standard error. NLL
across different corpora/tokenizers is not commensurate for a pooled absolute
likelihood ranking. Teacher replications and corpus replications were not run.

## Main-text primary evidence

| ID | Supporting experiment / dataset | Independent student seeds | Exact effect, NLL mean ± sample SD | Status / allowed interpretation |
|---|---|---:|---|---|
| E1 | Matched PTB post-rejection confirmation | 3: 42/123/456 | Gain +0.144937 ± 0.003938; 3/3 positive | CONFIRMED: positive transfer under this protocol |
| E2 | Phase 2C WT2 fused teacher | 3: 42/123/456 | Gain +0.155665 ± 0.007692; 3/3 positive | CONFIRMED: positive transfer under this protocol |
| E3 | Phase 2G TinyStories | 3: 42/123/456 | Gain −0.107219 ± 0.000415; 3/3 negative | CONFIRMED: reproducible negative transfer boundary |
| E4 | Phase 2D WT2 fixed-composition J/A | same 3 WT2 students | J +0.155665 ± 0.007692; A −0.027654 ± 0.006822; Q=NLL_A−NLL_J +0.183319 ± 0.001375 | CONFIRMED sign reversal; teacher-state, not isolated scalar-quality causality |
| E5 | Phase 2C WT2 Grassmann-only source | same 3 WT2 students | Gain +0.040991 ± 0.006733; 3/3 positive despite negative teacher residual vs C0 | CONFIRMED counterexample to globally stronger-teacher necessity |
| E6 | Phase 2E WT2 fused/T-only/G-only | same 3 WT2 students | F−T +0.020797 ± 0.001214; F−G +0.114674 ± 0.001001; T−G +0.093877 ± 0.002195; each 3/3 positive | F−T PARTIAL because persistent T clipping; bounded source ranking |
| E7 | Phase 2F WT2 homogeneous TT control | same 3 WT2 students | Gain_TT +0.168125 ± 0.008948; H=Gain_TG−Gain_TT −0.012460 ± 0.001348; 3/3 TT better | CONTRADICTED TG-superiority hypothesis in this control |

E1–E3 use token-mean KD, lambda=5 and T=2, with validation selection and final
held-out reporting. PTB/WT2/TinyStories show positive/positive/negative signs,
but teacher quality, corpus, data scale and training states also differ. This
is cross-domain reproducibility of configuration-level outcomes, not evidence
that dataset identity causally determines the sign. TinyStories CE improves
over S0 in its matched three-seed runs; its main negative boundary must not be
confused with the degenerate modern FineWeb continuation regime.

In E4, architecture, branch source identities and forced alpha=0.5 are fixed.
J and A have validation-subset teacher NLL 4.300644 and 6.376220 respectively.
Here Q refers to the KD-student test NLL under A minus that under J, not the
difference in teacher validation NLL. Joint training changes branch parameter states/representations as well as
likelihood. Thus teacher training state and predictive quality strongly
modulate observed transfer, but the contrast does not isolate NLL as the sole
cause. The three-seed contrast reuses J, C0 and S0 from Phase 2C; do not count
them as independent new confirmations.

E5 defines residual as matched C0 CE validation-subset NLL minus teacher NLL,
not automatically as S0 minus teacher. The three residuals are
−0.266520/−0.253492/−0.251006, whereas test gains are
+0.046771/+0.033598/+0.042604. A globally weaker teacher can transfer positively
here. This does not establish that weak teachers are generally preferable,
that teacher quality is irrelevant, or that Grassmann geometry caused gain.

E6 source gains are fused +0.155665 ± 0.007692, Transformer-only
+0.134868 ± 0.008906, and Grassmann-only +0.040991 ± 0.006733. The ranking is
stable across these three students, while composition, quality and optimization
constraints change together. The Transformer-only arm retains the permanent
`CLIP_SATURATION_WARNING`. The small F−T contrast is therefore not an
unconfounded branch-specific mechanism result.

E7 compares full-validation teacher NLL 4.303697 (TG) and 4.293589 (TT), with
37,749,633 versus 35,340,801 parameters. TT has 0.010108 lower teacher NLL and
about 6.38% fewer parameters. Historical TG alpha learning rate is 0.005 versus
TT 0.01; branch learning rate/freeze/joint training structure were reused but
the controls are not strictly training- or parameter-matched. These differences
and differing clipping trajectories can favor TT; they prevent an intrinsic
architecture-superiority inference in either direction. TT nevertheless
provides a necessary counterexample to the original claimed TG advantage.

Teacher metric scope matters: E4/E5/E6 teacher NLL is the archived
512-validation-chunk subset (130,560 predicted targets). E7 teacher NLL/JSD/
agreement is the common full validation split. The two TG values are not
contradictory measurements of identical scopes. Do not silently mix them.

## Secondary falsification evidence

| ID | Supporting evidence / units | Exact result | Status / placement |
|---|---|---|---|
| E8 | PTB/TinyStories branch diagnostic; one endpoint pair per dataset, token-level associations | JSD incremental AUROC −0.000323 / +0.010910; incremental R² +0.000533 / +0.003061 | PARTIAL negative association result; strong transfer-mechanism claim unsupported |
| E9 | Phase 2F full-validation ensemble diversity | TG JSD 0.182087, agreement 0.507034; TT JSD 0.099793, agreement 0.630170 | Higher measured disagreement does not track better KD in this pair; no universal monotonicity inference |
| E10 | Phase 3A offline diagnostic; 7 families × 3 seed-level conditions × 3 calibration subsets | 21 conditions / 63 rows; first/second-order AUROC 0.644444, sign accuracy 0.571429, Spearman 0.249351; identical signs 63/63; stable subsets 11/21 | Negative diagnostic: no reliable universal endpoint predictor established |

E8 controls for simpler token-level quantities but is not a prospective
teacher-level prediction benchmark; token bootstrap does not create independent
model seeds. E9 compares only two trained ensemble teachers with different
quality and states. Together these observations do not support branch
disagreement as a strong transfer mechanism, but they do not show that
diversity is useless in all ensemble methods.

E10 is a post-freeze falsification check, not a successful proposed method.
First- and second-order signs are identical on all 63 calibration rows and all
21 seed means, missing positive WT2 J/G/TT families. Nominal virtual-step AUROC
is 0.700000 with sign accuracy 0.571429; this is not reliable prediction or a
replacement endpoint. Only 10/21 virtual-step signs are calibration-stable.
Median relative curvature correction is about 0.52%; it adds no sign
separation. Literal first scheduler LR is zero, so its virtual update is a
no-op. The reported candidate safe lambdas (56.44–6429.64 where finite and
positive) are not deployment recommendations or a learned tuning rule.

The 21 conditions are seven dependent families; WT2 arms share S0/C0 and
calibration data. They are not 21 independent students or 63 experiments.
Small-subset teacher residuals, including a TinyStories sign different from
the full held-out result, must not replace full-scope teacher-quality evidence.
No broad impossibility of gradient-aware or meta/validation-driven KD follows.

## External-validity stress test, not a main confirmation

| ID | Supporting experiment / dataset | Independent units | Exact result | Evidence status |
|---|---|---|---|---|
| E11 | Phase 4A/4B FineWeb-Edu, adapted SmolLM2-360M→135M | 3 adaptation seeds from one common pretrained base; one teacher/subset | Gain_1 −0.013550460935 ± 0.003542273400; Gain_5 −0.043924082202 ± 0.004016675497; both 3/3 negative, all val/test directions agree | CONFIRMED CE-relative trained-step comparison; PARTIAL external validity due to continuation/optimization confounds |
| E12 | Phase 4B best_including_S0, validation-only diagnostic | same 3 students / 9 arm choices | 9/9 select S0; diagnostic gains exactly 0; CE−S0 test NLL +0.007918178008 ± 0.004338019006 | CONFIRMED degenerate continuation regime; primary endpoints are not replaced |
| E13 | Phase 4C fixed step256 S0 / disjoint 8M targets / CE gate | one seed42, three LR candidates | improvements +0.004214441514 / +0.004849292809 / +0.003940636206; best <0.01 | Required regime NOT ESTABLISHED; no KD or new test evaluation |
| E14 | Phase 4A modern TinyStories pilot | one adaptation seed42 | Gain_1 +0.004948632001; Gain_5 −0.004797389422 | EXPLORATORY, unreplicated; appendix only, not old TinyStories confirmation |

E11 historical decision `FINEWEB_L1_L5_NEGATIVE_REPLICATED` remains correct
for its frozen trained-checkpoint comparator. Seed42 is reused from Phase 4A,
not counted twice. Selection excludes continuation S0 in the primary result.
The teacher is substantially better than S0: mean held-out residual advantage
is +0.219866198612 (validation) and +0.218703697246 (test). However, E12 shows
that continuing CE also worsens the matched S0 test endpoint for every seed,
and all step-0-eligible validation choices stop at S0. The practical boundary
therefore concerns a continuation regime with little/no further CE benefit,
not a clean confirmation in an improving modern continuation regime.

E13 used exact seed42 preparation step256 (2,097,152 targets), excluded its
8,192 consumed target chunks, froze 31,250 disjoint continuation chunks and
ran only CE peak LR 5e-6/1e-5/2e-5. All select trained step977 but none meets
the preregistered 0.01 validation improvement. Small positive validation
headroom is real and must be stated; failure does not prove all continuation
is harmful. Decision `STOP_MODERN_CONTINUATION_NOT_ESTABLISHED` is terminal.
Conditional KD0.25/KD1, test gains and initial KD/CE gradient ratios are
`NOT_RUN`/`NOT_MEASURED`, not zero. E13 does not repair E12 or provide a new
modern KD endpoint.

Modern student independence is adaptation-seed independence, not independent
pretraining, teacher training, dataset sampling or model-family replication.
SmolLM2 pretraining includes FineWeb-Edu; exact overlap is unknown and
pretraining budgets differ by size. Short context, a fixed subset, finite
budgets and checkpoint eligibility limit generalization. Phase 4 belongs in a
bounded stress-test/limitations section, never an Abstract headline.

## Permanent warning and protocol-amendment registry

| Scope | Required record | Interpretation boundary / source |
|---|---|---|
| Phase 2D | Original short Teacher-A smoke failed; explicitly authorized matched full-data one-epoch J/A gate then passed | Keep both gate records; formal AMP nonfinite/overflow maxima about 0.013605, not falsely zero. See Phase 2D REPORT and both smoke summaries. |
| Phase 2E T-only | `CLIP_SATURATION_WARNING`: gate and first formal epoch clipping fraction 1.0 | User-authorized unchanged protocol; no lambda/clip adjustment. F−T mechanism is optimization-constrained. |
| Phase 2F TT | Epoch1 clipping 0.928571/0.843537/0.874150; mean 0.170068/0.137755/0.127551; AMP maxima 0.013605 | Disclose transient high clipping/TT–TG trajectory differences; not strictly optimization-matched. |
| Failed CRBD | All KD arms clip-saturated in the one-seed 20k control | Do not present routing as effective; no repair inference. |
| Phase 4A/B lambda5 | Permanent `CLIP_SATURATION_WARNING`, every continuation step clips | Do not attribute harm causally to clipping; no post-hoc threshold sweep. |
| Phase 4B lambda1 | Clip fractions 0.850123/0.854218/0.798526 | Not fully saturated, still frequently constrained. BF16 observed nonfinite/overflow 0 is not an FP16 scaler skip rate. |
| Phase 4B CE456 | `CE456_GRADIENT_SPIKE_WARNING`, finite norm86.482155 at step711 | Selected step256 weights predate it; later trajectory/selection effect UNKNOWN, no exclusion/retry. |
| Phase 4C CE2e-5 | Finite norm14.602655 at step656; clipped; cause/impact UNKNOWN | Best LR is another arm; no post-hoc exclusion. No unrun KD clipping inference. |
| Phase 3A | Offline FP32 diagnostic, calibration dependence and literal-zero-LR no-op | Not online optimizer-equivalent endpoint prediction; not Safe-lambda success. |

The above preserves each report's recorded warnings and all material
amendments. Detailed logs and exact arm-wise telemetry remain authoritative;
the compact registry is not permission to drop other protocol caveats.

## Appendix-only / removed mechanisms

CRBD TinyStories20k (one seed, two continuation epochs): CE test PPL 5.0195,
fixed KD 5.3018, CRBD 5.5067, shuffle 5.5010; repair −0.7220. It failed its
bounded test and cannot be a successful-method contribution. Legacy CodeParrot
common5k gain +0.188309 is single-seed, legacy-loss/winner-selected evidence,
not confirmation. Old random-initialization comparisons have unmatched LR/
budget, so warm-start necessity cannot be claimed.

Legacy deployment numbers describe different archived teacher/student
configurations, not the current TT/TG teachers: 36.264305/31.434257/23.565121M
parameters and batch32 latency 55.984/48.460/34.959ms on RTX3090, seq256.
Batch1 quality-student latency16.089ms exceeds teacher15.482ms; blanket speedup
is false. Keep the frozen efficiency table in the appendix, including its
single-checkpoint timing and unmatched architecture limitations.

Grassmann-specific superiority is CONTRADICTED in the tested TT control;
Pluecker-geometry causality is UNSUPPORTED because no isolating intervention
exists. Architectural heterogeneity superiority, strong-JSD transfer
mechanisms, successful CRBD/Safe-lambda and universal weak-teacher-negative
rules are not allowed. Detailed Grassmann equations provide test-bed context
only, in the appendix.

## Source pointers and reproducibility of this package

Primary: frozen `../final_evidence/table_*.csv` plus Phase 2C/2D/2E/2F/2G
REPORTs and per-seed CSVs. Secondary:
`../experiments/phase3a_safe_lambda_diagnostic/REPORT.md`, `predictor_metrics.csv`
and `condition_summary.csv`. Stress test:
`../experiments/phase4a_modern_generalization_pilot/REPORT.md`, Phase 4B
REPORT/results/best_including_s0/optimization CSVs and Phase 4C REPORT/gate/
manifest/optimization CSVs. `tables/` holds writing copies only;
`table_source_manifest.json` hashes the formatted CSV inputs.

Protected Phase-2 aggregate SHA256 (sorted `sha256sum final_evidence/*` output,
then SHA256): `3d172e634db611edad0d566e74ea0f826ed8cdae69922120baa94ebf9e6e2b96`.
Old CAC source SHA256:
`8b8e3fcc3f04b596058cd4b63ddfe38b6eab91f338c7d4f12db4b602c9982ec8`.
The V2 claim set and story supersede obsolete writing/next-action statements
in historical handoffs, not their data. No full manuscript is written here.
