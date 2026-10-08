# Entry point for manuscript V2 writing

## Current authorized full-revision handoff — STOP (2026-10-08)

Input: `56ab6cee930da08637c3fa7c6b36d5cb22ccf09f`. The lead explicitly
authorized the full revision. The authoritative local Abstract/Introduction
are now included, with their prose byte-preserved; the remote skeleton was
not substituted. Section 2 and Sections 9–10/appendix are complete, Sections
3–8 revised into reader-facing prose, and five pure-vector figures integrated.
Section 8 source prose is approximately 46% shorter; modern telemetry and
unreplicated observations are appendix-only. No Figure 5 placeholder remains;
the architecture figure is numbered A1.

Read `FULL_MANUSCRIPT_REVISION_STATUS.md`, `full_revision_qa.json`,
`FULL_TEXT_REFERENCE_REVIEW.md`, `full_text_reference_verification.json`, and
the new top section of `FIGURE_VECTOR_QA.md`. Historical QA JSON files retain
their historical scope and are not current source hashes. Reproduce current
checks with `audit_body_draft.py --full-revision` and
`audit_vector_figures.py --readability-revision`. Build actual main with
`make -C 论文投稿/manuscript_v2` (23-page venue-neutral article).

Twenty-four references have primary metadata and relevant full-text reading
records. This is not a systematic review or adaptive-baseline evaluation.
No experimental artifact, frozen claim or endpoint changed. State and scalar
quality remain distinct; TT/TG teacher quality/count/history are confounded;
modern continuation is not clean confirmation. All permanent flags remain.

STOP after the compact revision commit. No training, models, diagnostics,
sweeps or methods. Further manuscript edits require a new instruction;
venue/template, page budget and authors remain unresolved.

## Historical scoped-body stage

## Current scoped body-draft handoff — STOP

From input commit `1c153e49efa19132481fb74bbc98032bf610544e`, the research
lead authorized first-draft prose for Sections 3–8 only. Those six sources
are now written and compile with the current local manuscript. Read
`BODY_DRAFT_STATUS.md` and `body_draft_qa.json` before continuing.
Scientific conclusions and archived experiments are unchanged; no model
training, evaluation, diagnostic run, tuning, or new method occurred.

Sections 2/9/10 and appendix remain unchanged skeletons. Eight numerical
tables retain their planning captions and five figures remain labeled boxes.
Foundational dataset/model citations and closest-prior-work connections
remain comment TODOs. This is not a complete or submission-ready paper.
The local Abstract/Introduction were user edits predating this stage and are
excluded from its commit. Only a duplicate Introduction heading/label and
its adjacent blank line were removed locally, preserving all prose. The
committed prior skeleton already has a single Introduction heading.

The scientific scope remains C1–C4: state/quality matter but scalar teacher
likelihood is insufficient; fused sources help without TG/heterogeneity
superiority; Phase3A is a failed retrospective diagnostic; Phase4 remains
continuation- and clipping-confounded. All warnings stay next to results.
STOP. Further writing requires a new instruction; experiments remain
permanently frozen.

## Historical skeleton-stage handoff

The following records the previous preparation stage, not current writing
authorization. Its scientific safeguards and evidence paths remain valid.

Status: `manuscript_v2_ready_for_writing`. Experiments are permanently frozen.
This stage has prepared evidence/story/verified literature and a compiling
venue-neutral source skeleton only. The full paper has NOT been drafted.
STOP until the research lead explicitly authorizes the next writing task.
There is no authorization for new experiments, methods, controls or tuning.

## Repository and protected history

Input HEAD: `64db3aebd158f56373385db18dcdf056aeea4c52`. The commit containing
this package is obtained with `git log -1 --format=%H -- research/manuscript_v2`
after delivery (no self-referential commit hash is embedded).
Edit local NFS checkout `/home/xin/fuwuqi/grassmann-flows`; Git runs locally.
Never alter datasets/checkpoints. Preserve `research/final_evidence/`, the
immutable Phase-2 snapshot, and `论文投稿/cac/conference_101719.tex`. All new
writing goes under `论文投稿/manuscript_v2/`; post-freeze evidence integration
goes under `research/manuscript_v2/`.

## Read in this order

Start with EVIDENCE_INTEGRATION_V2.md and its permanent warning registry,
then CLAIMS_V2.md, MANUSCRIPT_STORY_V2.md, TABLE_FIGURE_PLAN_V2.md and
RELATED_WORK_GAPS.md. `citation_verification.json` records fourteen
publisher-exported, metadata-verified citations; full-text protocol comparisons
and dataset/model references are still TODOs. `table_source_manifest.json`
hashes CSV sources; `tables/` contains CSV/Markdown writing copies. If any
number seems inconsistent, read the original report/scope, not a chat summary.

Primary endpoints: frozen Phase-2 tables and original Phase2C–2G reports/CSV.
Secondary falsification: Phase3A REPORT/predictor_metrics/condition_summary.
Stress test: Phase4A REPORT, Phase4B REPORT/results/best_including_s0/
optimization and Phase4C REPORT/ce_lr_gate/manifest/optimization. All these
are read-only history; V2 integrates them without changing them.

## Paper identity and approved central evidence

Recommended title: What Makes Knowledge Distillation Transfer? Controlled
Evidence from Warm-Start Language Modeling

Question: What determines whether KD transfers useful information to a
warm-start LM student? Endpoint convention: Gain=test_NLL_CE−test_NLL_KD,
validation-selected states, exact within-seed S0. This is not theoretical novelty.

The paper has four central claims: C1 reproducible PTB/WT2/TinyStories
positive/positive/negative transfer (+0.144937/+0.155665/−0.107219 NLL);
C2 fixed-composition J/A sign reversal (Q+0.183319); C3 globally weaker G
still transfers positively (+0.040991, weakness relative to C0 on the same
validation subset); C4 fused-source benefits without TG/heterogeneity
superiority (TT better for3/3, H−0.012460). Read exact SDs/scopes/limits in
CLAIMS_V2.md rather than quoting these compressed values alone.

Organize the argument as tests of global likelihood, disagreement/diversity,
heterogeneity and local gradient/curvature explanations. V2-H1–H4 are writing
labels, not newly claimed preregistration. The synthesis is “evidence is
consistent with” joint teacher/student/optimization dependence, not a theorem
or a fully identified causal decomposition.

Phase3A is a negative diagnostic, not a proposed failed-method contribution:
first/second-order signs coincide on63/63 rows; AUROC0.644444 and sign
accuracy0.571429 over21 dependent conditions fail the reliability objective.
Do not promote its candidate lambdas into a safe training rule.

Phase4 is a bounded stress/limitation, never an Abstract headline. FineWeb
lambda1/5 gains replicate negatively, but9/9 step0-eligible choices prefer
S0 and CE worsens S0; Phase4C's best validation improvement0.004849292809
fails the0.01 gate. No Phase4C KD or new test exists. Acknowledge small positive
CE validation headroom; do not claim zero improvement or clean modern transfer.

## Red lines and warnings

Grassmann/Pluecker causal superiority, homogeneous-ensemble inferiority,
strong JSD transfer mechanism, successful CRBD/Safe-lambda, reliable universal
local predictor, irrelevant teacher quality, generally preferable weak teachers,
dataset-caused sign and universal FineWeb/modern KD failure are forbidden.
Do not mix legacy CodeParrot/RI/deployment with confirmation.

Permanent `CLIP_SATURATION_WARNING` applies to Phase2E T-only and all modern
lambda5 arms. Preserve Phase2D failed smoke/amended gate, nonzero FP16 AMP event
rates, TT transient clipping, modern frequent lambda1 clipping and
`CE456_GRADIENT_SPIKE_WARNING`; Phase4C finite CE2e-5 outlier cause/impact is
UNKNOWN. TT/TG parameter counts, teacher quality and alpha-LR differences are
not parameter-matched. Teacher subset versus full-validation scope must be
explicit. Shared WT2 seeds/baselines and three adaptation versus pretraining
seeds must not be inflated into independent experimental units.

## Source layout and local compile

`论文投稿/manuscript_v2/main.tex` uses standard article, not a selected conference
template. It includes ten named sections, title candidates in comments,
claim/citation TODOs, evidence-backed comment notes, eight numerical planning
tables and five figure boxes. Appendix source covers architecture, protocols,
per-seed results, diagnostics, modern stress, failed controls and efficiency.
There is no polished abstract or complete prose. References are official
metadata exports, not recycled unverified old citations.

From `论文投稿/manuscript_v2/`, run `make` (latexmk/pdflatex/BibTeX installed).
Build products go to `outputs/manuscript_v2_build/` and are not committed.
`skeleton_qa.json` records actual source/compile checks. Regenerating tables
is `python3 research/manuscript_v2/scripts/build_tables.py` from the repo root;
it performs only archived-CSV formatting/descriptive consistency checks.
The bibliography fetcher contacts public publisher metadata only; no private
project text or experimental data are uploaded to those services.

## Next task, only upon explicit authorization

Agree on target venue and writing scope; read full text of closest prior work
and verify foundational dataset/model citations; then draft only the requested
sections from approved C1–C4, retaining caveats next to results. Do not silently
fill unknown quantities. A later writing request is not permission for new
training. Neither a compiling skeleton nor the outline skill implies a
submission-ready manuscript or automatic conference suitability.
