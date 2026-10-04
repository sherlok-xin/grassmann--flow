# Exact V2 table and figure plan

This is a writing plan, not new experimental analysis or finished artwork.
Main table values are drawn from archived matched endpoints only. The formatting
script `scripts/build_tables.py` hashes inputs in `table_source_manifest.json`,
checks the modern paired gain identity, and emits CSV/Markdown/LaTeX tables.
Phase-2 writing copies preserve the precision of the frozen snapshot (six
decimals for NLL effects). New stress/gate CSV copies retain source precision.
Never invent precision beyond the underlying source or recompute endpoints
from rounded PPL when exact NLL is already archived.

## Main tables: exact locations and scientific role

| ID / LaTeX label | Section / file under new manuscript tables/ | Writing CSV under this package tables/ | Required fields and caption caveat |
|---|---|---|---|
| T1 / tab:cross_domain_transfer | S5 / cross_domain_transfer.tex | table_cross_domain_transfer.csv | Dataset, CE/KD mean NLL, paired gain mean/sample SD, seeds/signs; matched lambda5 T2, one teacher/domain, no dataset causality |
| T2 / tab:teacher_state | S6 / teacher_state.tex | table_teacher_quality_intervention.csv | J/A teacher validation-subset NLL, three gains each, Q mean/SD/signs; fixed alpha0.5, changed branch states, not isolated quality causality |
| T3 / tab:weaker_teacher | S6 / weaker_teacher.tex | table_weaker_teacher.csv | Seed, teacher residual vs selected C0 on same subset, test gain; residual comparator is C0 CE, not automatically S0 |
| T4 / tab:teacher_sources | S7 / teacher_sources.tex | table_teacher_source_ablation.csv | F/T/G alpha, subset teacher NLL, mean gain/SD, paired contrasts; T-only CLIP_SATURATION_WARNING |
| T5 / tab:tt_vs_tg | S7 / tt_vs_tg.tex | table_tt_vs_tg.csv; table_tt_vs_tg_per_seed.csv | Parameters, full-validation NLL, JSD/agreement, KD gain, H mean/SD/signs; quality/parameter/alpha-LR/optimization differences |
| T6 / tab:local_diagnostics | S7 / local_diagnostics.tex | table_local_diagnostics.csv | First-order, quadratic, virtual nominal AUROC/sign accuracy/Spearman; 21 dependent conditions, not 21 independent students |
| T7 / tab:modern_stress | S8 / modern_stress.tex | table_modern_stress.csv; table_modern_per_seed.csv | lambda1/5 paired mean/SD/signs; 9/9 S0, CE deterioration, full/frequent clipping; bounded stress, not clean modern confirmation |
| T8 / tab:modern_ce_gate | S8 / modern_ce_gate.tex | table_modern_ce_gate.csv | LR, selected validation NLL, improvement, selected step, failed>=0.01 gate; one seed, NO KD/test |

Teacher NLL scope is mandatory in T2–T5: T2/T3/T4 use the original 512-chunk
validation subset, whereas T5 uses the common full validation split. TG
4.300644/4.300646 versus4.303697 is a scope difference, not rounding noise.
T1 mean PPL is the average per-seed PPL from the frozen table, not necessarily
exp(mean NLL); never silently substitute one for the other.

Main-text tables are allowed to be combined during future venue-specific
writing (for example T2/T3 and T7/T8) but not to mix comparator eligibility,
scopes or units. A shorter main-text version can move T6 to the appendix while
retaining the bounded negative diagnostic in S7. No new table selection based
on more favorable endpoints is allowed.

## Figure placeholders and required data mapping

| ID / label | Section / planned final filename under figures/ | Exact source | Design and mandatory caveat |
|---|---|---|---|
| F1 / fig:paired_protocol | S3 / fig_paired_protocol.pdf | Phase2C/2G plans and hashes; Phase4B/4C selection records | Same S0 forks into matched CE and KD, independently selects validation checkpoint, test once for report. Show primary/step0 selection distinction as separate inset, no new method module. |
| F2 / fig:cross_domain_gain | S5 / fig_cross_domain_gain.pdf | table_endpoint_inventory.csv, frozen T1 and original Phase2 endpoint CSVs | Paired test gain dots for 42/123/456, per-domain mean/sample SD, zero line. No pooled absolute NLL axis or p-value inferred from three signs. |
| F3 / fig:teacher_state_quality | S6 / fig_teacher_state_quality.pdf | Phase2D results; table_teacher_quality_intervention.csv; table_weaker_teacher.csv | Panel A J/A paired gains, panel B weaker G residual vs C0 alongside positive gain. Distinguish state intervention from correlation; not one causal regression curve. |
| F4 / fig:ensemble_control | S7 / fig_ensemble_control.pdf | table_tt_vs_tg_per_seed.csv; full-validation teacher diagnostics in Phase2F | Paired TG/TT endpoint or gain dots and separate teacher-quality/diversity panel. Print H direction, parameter counts, alpha-LR mismatch. Do not draw a geometry-causes-gain arrow. |
| F5 / fig:modern_selection_gate | S8 / fig_modern_selection_gate.pdf | table_modern_per_seed.csv; table_modern_best_including_s0.csv; table_modern_ce_gate.csv | Selection diagram: negative trained-only comparison → 9/9 S0 → disjoint CE gate<0.01 → STOP. Do not visually imply Phase4C KD results exist. |

The LaTeX skeleton renders labeled white boxes for F1–F5. There are no invented
curves, scores, model icons or full final figures. Future plots should use
clean white background, vector PDF, English labels, muted colors, legible
fonts, signed NLL effects and a zero line where applicable. Distinguish sample
SD from uncertainty intervals and identify reused seed42 points. Any final
plot requires source checks and visual QA, not a new experiment.

## Appendix tables / figures

Per-seed endpoint inventory retains family/condition/S0 dependence, not a new
confirmatory pool. Branch JSD incremental AUROC/R² belongs beside its original
PTB/TinyStories token-unit analysis. Phase3 detailed HVP/candidate lambda/
virtual-step/calibration tables come from its existing condition/family CSVs;
the existing diagnostic figures may be reused after caption/scope checking.

Modern appendix uses `table_modern_per_seed`, `table_modern_best_including_s0`,
`table_modern_optimization`, `table_modern_gate_optimization` and original
Phase4A results. Include S0/selected steps, teacher residuals, validation/test
agreement, BF16 finite checks, KL/preclip norms, clip fractions and both finite
gradient-event records. No new Phase4C test column is populated.

Legacy appendix uses frozen `table_efficiency` and
`table_failed_or_negative_controls`; CRBD20k PPLs and shuffle, legacy
CodeParrot common5k and unmatched RI comparisons remain visibly nonconfirmatory.
Deployment timings are single-checkpoint RTX3090 observations, not multiseed
speed benchmarks or current TT/TG costs. Detailed Grassmann math/architectures
and full protocol amendments have appendix text placeholders, not new derivations.

## Warning placement and completion checks

T-only saturation must appear with T4 and in full optimization details. All
modern lambda5 saturation, frequent lambda1 clipping, CE456 spike and failed
Phase4C gate must appear next to T7/T8, not hidden exclusively in an appendix.
The failed short Teacher-A smoke/amended gate belongs in the experimental
setting appendix and limitations. Keep no-testing-after-failed-gate explicit.

Check LaTeX label uniqueness, zero undefined citations/references, caption
scope, source hashes and float fit. The compilation is skeleton QA only;
figure boxes/TODOs mean it is not a submission-ready paper. No target venue
has been bound. Do not run training to fill any placeholder.
