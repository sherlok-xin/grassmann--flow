# Full manuscript revision status

Date: 2026-10-08. Input HEAD: `56ab6cee930da08637c3fa7c6b36d5cb22ccf09f`.
Status: complete authorized revision; STOP. Experiments permanently frozen.

## Delivered manuscript

The actual `论文投稿/manuscript_v2/main.tex` now builds the full venue-neutral empirical draft, not a temporary figure-inclusion copy. Authoritative local user Abstract and Introduction prose is byte-preserved and included in the revision commit. Preservation hashes are in `full_revision_preservation.json`. Only surrounding planning commands/comments and document formatting were adjusted; neither local prose was replaced with the earlier remote skeleton. The user's `.gitignore` remains untouched and unstaged.

| Area | Completed revision |
|---|---|
| Section 2 | Three narrative groups, relevant full-text protocol checks, no citation catalogue or novelty inflation |
| Sections 3–4 | CE plus distillation notation, temperature τ, paired operational endpoint, conceptual controls and held-out-scope distinction |
| Section 5 | Three-seed PTB/WT2/TinyStories +/+/- findings prominent; detailed telemetry/retry moved to appendix |
| Section 6 | Fixed-composition state reversal and weaker-source counterexample; no scalar-likelihood causality |
| Section 7 | Exactly three conceptual subsections; source/TT controls and failed local diagnostics, with confounds retained |
| Section 8 | Bounded modern stress test, approximately 46% shorter source prose; 9/9 S0 choices and failed headroom gate explicit |
| Sections 9–10 | Configuration-dependent synthesis, complete limitations, short C1–C4 conclusion only |
| Appendix | Correct causal architecture, original protocols/amendments/hashes, per-seed endpoints, branch/local diagnostics, modern telemetry, failed controls and separately labeled legacy deployment |

No new endpoints were computed. PTB endpoint tables format stored NLL fields from archived summaries, not logarithms of rounded legacy PPL; differences agree with the frozen inventory to reported precision. The original confirmatory gains and SDs are unchanged. Teacher A has a five-epoch fusion-weight-only history, compared with ten-epoch joint J/TT histories; its original alpha LR is 0.01 versus J 0.005. This is now explicit in the appendix, preserving the distinction between a trained-state intervention and a pure scalar-quality manipulation.

## Files changed

Manuscript: `main.tex`, `references.bib`, `README.md`, all Section 01–10 sources and `sections/appendix.tex` under `论文投稿/manuscript_v2/`. Section 01 includes the user's authoritative existing prose, not a new rewrite. Eight existing table caption/source-label files were polished. New appendix tables are `per_seed_gains.tex`, `ptb_endpoints.tex`, `wt2_endpoints.tex`, `modern_endpoints.tex`, `tinystories_optimization.tex`, `modern_optimization.tex`, `full_predictors.tex` and `teacher_training_protocol.tex`; the TinyStories endpoint table is inline. There are 17 tables in total, six in the main narrative and eleven in the appendix.

Figures: SVG/PDF/600-dpi PNG exports for `fig1_paired_protocol`, `fig2_cross_domain_gain`, `fig3_teacher_quality`, `fig4_ensemble_control`, `figA1_teacher_architecture`, plus figure README. Four existing masters received only authorized readability changes; A1 folds the same causal processing sequence into two rows. Figure 2 plots nine archived seed gains and mean±sample SD with zero. No seaborn, new model evaluation, CI, p-value or fitted trend. Figure 4 preserves exact gain numbers and moves the exact caution sentences to its caption. Main figures are 1–4; architecture is A1. No Figure 5 placeholder remains.

Compact metadata/tooling: this status, `full_revision_preservation.json`, `full_revision_qa.json`, `endpoint_summary_sources.json`, `FULL_TEXT_REFERENCE_REVIEW.md`, `full_text_reference_verification.json`, `figure_readability_qa.json`, `figure_readability_audit.json`, updated `FIGURE_VECTOR_QA.md`, `RELATED_WORK_GAPS.md`, `GPT_WRITING_HANDOFF.md`, and document-only scripts under `research/manuscript_v2/scripts/`. `AI_RESEARCH_HANDOFF.md` and `research-state.yaml` record completion/STOP. Project-local memory/journal were maintained locally, but excluded along with unrelated `.gitignore` edits from the compact manuscript commit.

## References and scientific boundaries

The bibliography has 24 unique primary-export-verified entries: the existing fourteen plus Hinton, Distillation Scaling Laws, GRACE, PTB, WikiText, TinyStories, Transformer, SmolLM2, FineWeb and Grassmann background. Relevant methods/background sections were read and the scope is recorded, rather than claiming an exhaustive review. A verified author-maintained Plücker background page is cited as a footnote; unverified book metadata were not invented. Scratch PDFs/text/raw exports are not committed.

Main-paper search finds no internal Phase 2/3/4 identifiers, visible writing TODOs or placeholders. Forbidden-claim hits were reviewed in context: geometry/heterogeneity superiority is explicitly unsupported; local diagnostics are bounded failures, not universal incompatibility theory; prior harmful/weak-teacher findings are acknowledged; adaptive KD/GRACE are not evaluated or refuted. TT/TG likelihood, count and training confounds remain, as do permanent T-only and modern clipping warnings. CE seed-456 gradient-event and failed modern-gate labels are permanently retained in the appendix. No unrun KD/test result is represented as zero or fabricated.

## Final QA

Actual-main command: `make -C 论文投稿/manuscript_v2`. Exit 0, 23 pages. Final pass: zero undefined references/citations, duplicate labels/destinations, overfull boxes and underfull notices. All five vector PDFs are actual recorded build inputs; the manuscript PDF contains zero raster images. Every page was rendered and viewed; figure pages 4/6/8/9/16 also received enlarged inspection. Reference-vs-revision comparison preserves scientific wording, gains, arrows and causal ordering. A detected Figure 2 legend collision and appendix vertical-table overflow were fixed before the final QA.

`audit_body_draft.py --full-revision`: PASS, all 62 original scientific numeric checks retained across authorized body-to-appendix relocations. Original CSV hashes, immutable final-evidence aggregate, CAC hash and protected-path diff checks pass. Seven generated appendix tables also match stored records. User prose hash checks pass; final index/commit verification confirms `main.tex` and `01_introduction.tex` are included. `audit_vector_figures.py --readability-revision`: PASS for pure-vector exports, editable text, embedded fonts, current hashes, wording, causal connectivity and build inclusion. Normal figure text is ≥7 pt and mathematical scripts ≥6 pt at 6.8 inches. This is not single-column-width certification. New/current Python scripts pass syntax compilation.

Build output: `outputs/manuscript_v2_build/main.pdf`; QA previews and literature downloads are under `outputs/`. These are local build/scratch products, not committed. Figures remain explicit compact publication assets. Scientific/compile hashes are in the current QA JSON, not the older historical skeleton/body/frozen-layout JSON snapshots.

## Remaining decisions and limitations

Venue/template, conference page budget, approved authors/affiliations and any submission-format shortening remain open. Source comments mark these decisions; no scientific endpoint or required citation remains a visible TODO. The current 23-page article is not an ACL/EMNLP template or page-limit certification. User-authored Introduction/Abstract remain intentionally unchanged, including their broader framing; Sections 6–9 state the confounds and bounded interpretations in detail.

Small primary models, three student seeds, fixed teachers/corpora, shared controls, lack of factorial/adaptive comparisons, TT/TG quality/history/count differences, clipping/AMP warnings, degenerate modern continuation and unknown exact pretraining overlap are unresolved scientific limitations, not repaired by rewriting. No new experiment, training, model forward, diagnostic, clipping control, sweep or method was launched. `research/experiments/`, `research/final_evidence/`, old CAC, datasets and checkpoints are unchanged. STOP after the compact commit; do not start further work automatically.
