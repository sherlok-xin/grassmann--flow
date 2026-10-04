# Manuscript story V2: controlled evidence, not a Grassmann method paper

Recommended title: What Makes Knowledge Distillation Transfer? Controlled
Evidence from Warm-Start Language Modeling

Alternative conservative titles: Controlled Evidence on Distillability in
Warm-Start Language Modeling; Teacher Quality and Distillability in Warm-Start
Language Modeling. The old “Bridging the Gap: Warm-start Hybrid-lite
Distillation for Grassmann Flow Sequence Models” frames an unsupported
architecture-specific contribution and should not define V2.

Research question: What determines whether KD transfers useful information
to a warm-start LM student? Main contribution type: a bounded controlled
empirical study with matched CE-relative endpoints, replications and negative
controls. No new loss, route, architecture, theoretical distillability measure
or universal scalar predictor is claimed. No target venue is specified:
`criteria_binding_unavailable`; this is not submission-ready or venue-approved.

## Narrative spine and hypothesis tests

Open with a practical decision: once an LM student has already been trained,
adding teacher supervision need not improve on continuing CE. Define the
paired comparison before showing teacher rankings. C1's PTB/WT2/TinyStories
positive/positive/negative outcomes establish that the question is substantive,
not a single cherry-picked failure. C2 then changes teacher training state
without changing composition and reverses the sign, while C3 supplies the
counterexample that a globally weaker source can still transfer positively.
This combination rejects both “teacher quality does not matter” and “teacher
likelihood alone determines transfer.”

Test composition and diversity next. Fused supervision outperforms its
component sources, but TT outperforms TG despite smaller JSD; the earlier
Grassmann-specific story must be removed rather than cosmetically renamed.
Local validation gradients/curvature also do not reliably predict these
long-horizon endpoints in Phase3A. The result is a sequence of falsifications
of simple scalar explanations, not a successful predictor or new optimizer.
Close the empirical sections with modern-model stress testing: replicated
CE-relative harm survives seed variation but continuation degeneracy and
clipping remain, and the attempted headroom control fails its gate. This
explicitly bounds, rather than inflates, external validity.

`V2-H1`–`V2-H4` below are retrospective organizing hypotheses for writing.
They are not claims that this exact narrative was preregistered before all
experiments. Keep original phase plans, amendments and decisions separately.

| Writing hypothesis | Evidence / inference | What it does not establish |
|---|---|---|
| V2-H1: Better global teacher likelihood predicts beneficial KD | E4 shows strong teacher-state/quality modulation; E5 refutes a universal necessity rule | No isolated scalar-quality causality, no “quality irrelevant” |
| V2-H2: Disagreement/diversity explains distillability | E8 adds little token-level signal; E9 higher TG JSD accompanies worse KD than TT | No universal rejection of diversity or token uncertainty in adaptive KD |
| V2-H3: Heterogeneous TG is intrinsically more transferable | E7 TT better 3/3, with quality/parameter/LR confounds recorded | No intrinsic superiority claim for TT either |
| V2-H4: Local gradient/curvature predicts endpoint transfer | E10 misses positive WT2 families and is unstable across calibration subsets | No impossibility theorem or refutation of online meta/gradient-aware algorithms |

The concluding interpretation should use “the evidence supports” or “is
consistent with.” Distillability behaves as configuration-dependent in the
tested settings; a complete factor-by-factor causal account remains untested.
The joint-property sentence is a synthesis of controls and counterexamples,
not a formal theorem proved by these data.

## Section-by-section blueprint

| Section | Main argument / exact evidence | Required item | Must not claim |
|---|---|---|---|
| 1 Introduction | Warm-start deployment decision; C1–C4; contrast with existing quality/compatibility literature | F1 paired CE/KD setup; cite verified closest prior work | A new Grassmann KD method, first harmful LM KD, clean modern headline |
| 2 Related Work | Six overlap areas in RELATED_WORK_GAPS; closest ATKD; method versus empirical-audit distinction | Verified bibliography and overlap matrix | Full-text comparisons not yet checked; superiority over methods not run |
| 3 Problem Setting and Paired Distillability Evaluation | Exact S0 pairing; validation selection; Gain convention; independent units | F1; protocol eligibility notes | Definition as theory, unmatched RI as warm-start necessity |
| 4 Experimental Test Bed and Controlled Interventions | Phase2 protocols/teachers; fixed-alpha J/A; source and homogeneous controls | Protocol appendix; source identities / teacher NLL scopes | Strict parameter matching; isolated teacher-NLL causality |
| 5 Reproducible Positive and Negative Transfer | E1–E3, +/+/- three-seed outcomes | T1; F2 paired per-seed effects | Dataset causal effect; pooled cross-corpus NLL significance |
| 6 What Determines Distillability? | E4 teacher-state sign reversal; E5 weaker source counterexample | T2 and T3; F3 state/quality contrasts | Quality irrelevant; weak teachers generally best |
| 7 Teacher Source, Ensemble, and Diagnostic Controls | E6–E10 source ranking, TT rejection, weak JSD increment, failed local diagnostic | T4/T5/T6; F4 TT/TG and measured JSD | Geometry benefit, intrinsic heterogeneity benefit, universal predictor |
| 8 External-Validity Stress Test | E11/E12 CE-relative harm + 9/9 S0; E13 gate fails; E14 appendix-only | T7/T8; F5 selection/gate diagram | Clean modern confirmation, all FineWeb continuation harms, fabricated Phase4C KD/test |
| 9 Discussion and Limitations | Joint-configuration interpretation; shared seeds/teachers, protocol/amendment/optimization limits | Warning registry; appendix optimization tables | Fully identified causal decomposition, general impossibility of KD |
| 10 Conclusion | Conservative C1–C4 and configuration-dependent synthesis | No new figure or metric | New claims/method recommendations or automatic next experiments |

Appendix sections: Grassmann/test-bed details; full protocols and amendments;
per-seed endpoint tables; branch diagnostic details; Phase3A gradient/HVP/
virtual-step tables; Phase4A/B/C selection and optimization details; failed
CRBD and shuffle control; legacy CodeParrot/RI results; legacy PTB deployment.
Legacy results must never share confirmatory rows without explicit labeling.

## Abstract and prose boundaries for the next writing stage

An eventual abstract should summarize C1's reproducible signs, C2's fixed
composition sign reversal, C3's limited scalar-quality explanation and C4's
homogeneous control. It may mention simple diagnostics not resolving the
outcomes if space allows. It must not advertise a new method, Grassmann
superiority, universal transfer prediction or clean modern-scale confirmation.
Do not draft the abstract now: the source contains comments/TODOs only.

Final prose should use simple, precise academic English, no contractions,
unnecessary possessives or ornate vocabulary. Preserve technical abbreviations
and existing semantic LaTeX labels/citations. Keep narrative paragraphs rather
than converting prose into itemized claims; bullet notes are allowed only in
LaTeX comments at this skeleton stage. No decorative emphasis is necessary.
The old source remains unchanged; detailed old architecture mathematics is
contextual appendix material, not the centerpiece of the reconstruction.

## Readiness and stop

The outline workflow supplied evidence-to-claim mapping and a literature
overlap audit before any Related Work prose. The ML writing workflow supplied
publisher-verified citations and compiling-source checks. Neither substitutes
for venue selection, full-text novelty review, author approval or a future
complete manuscript review. At `manuscript_v2_ready_for_writing`, stop. Writing
the full paper requires a new instruction; new experiments remain forbidden.
