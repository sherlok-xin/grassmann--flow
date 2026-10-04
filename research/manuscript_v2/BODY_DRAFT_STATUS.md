# Manuscript V2 body draft status

Date: 2026-10-04. Input commit:
`1c153e49efa19132481fb74bbc98032bf610544e`.
Status: `manuscript_v2_body_sections_3_to_8_drafted`. STOP.

## Written scope

Only the following six sources under `论文投稿/manuscript_v2/sections/`
were drafted, using ordinary academic paragraphs and the approved C1–C4:

| Section | Source | Evidence and existing references |
|---|---|---|
| 3 | `03_problem_setting.tex` | Exact-S0 CE/KD comparison, token-mean forward KL with T-squared scaling, validation selection, paired gain, seed units; F1 |
| 4 | `04_experimental_setting.tex` | Archived main protocols, J/A and source construction, TT preparation and matching limits, subset/full-validation scopes |
| 5 | `05_cross_domain_transfer.tex` | PTB/WT2/TinyStories three-seed positive/positive/negative pattern; T1/F2; TinyStories optimization and accepted restart caveats |
| 6 | `06_distillability.tex` | Fixed-composition J/A sign reversal and globally weaker G versus C0 counterexample; T2/T3/F3; failed smoke and authorized full-data gate |
| 7 | `07_controls.tex` | Fused/source ablation, TT better than TG, disagreement limits, failed retrospective Phase3A diagnostic; T4–T6/F4 |
| 8 | `08_external_validity.tex` | Bounded SmolLM2 pilot/replication, 9/9 step-0 selections, failed Phase4C headroom gate; T7/T8/F5 |

Source authority remains the frozen Phase2 tables and original reports,
Phase3A REPORT/metrics, and Phase4A–4C REPORT/selection/optimization records.
The loss and fresh optimizer construction were checked against the actual
trainer. There are no new experimental results or changes to conclusions.
Grassmann is an experimental source, not a method contribution.

## Compile and consistency checks

Local `latexmk`/BibTeX compilation passes. From the manuscript directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../../outputs/manuscript_v2_build main.tex
```

The local PDF has 15 pages. Sections are numbered 1–10; eight table and five
figure labels resolve, as do body equation and appendix references. No
undefined references/citations, duplicate labels, errors, or overfull boxes
remain. Three pre-existing underfull bibliography spacing notices remain.
The independent read-only audit
`python3 research/manuscript_v2/scripts/audit_body_draft.py`
passes 62 core numerical checks and protection/style/reference checks;
its output is archived in `body_draft_qa.json`.
Ancillary optimization/protocol statements were reviewed against original
reports, not inferred from table means. The historical `skeleton_qa.json`
is preserved rather than overwritten.

The existing local user Abstract and Introduction prose are unchanged.
To restore correct numbering, only the second Introduction heading/label
and adjacent blank line were removed locally; the final newline was also
normalized. The audit reconstructs the original file and verifies its hash.
These pre-existing user changes in `main.tex` and Introduction, and the
unrelated `.gitignore` change, are excluded from this stage's commit.
The prior committed Introduction already has a single heading. Thus the
local compiled PDF includes user prose that is not part of the new commit.
An independently exported Git-index snapshot also compiles successfully
(13 pages, no unresolved references, duplicate labels, or overfull boxes).
This verifies the committed source independently of those local user edits.
Both compilation scopes are recorded separately in `body_draft_qa.json`;
PDFs and temporary build exports are not committed.

## Remaining work, not performed

Five existing figures are still labeled placeholder boxes, not finished
artwork. The eight existing numerical tables retain planning captions.
Replace F1–F5 with evidence-backed graphics and polish captions/placement
only under a later instruction; do not invent curves or mechanisms.

Body citation TODO comments concern verified foundational references for
PTB/WikiText-2/TinyStories/tokenizer and SmolLM2/FineWeb-Edu, plus the connection
to prior teacher-quality work in Section 6. No guessed citation keys were
inserted. The existing fourteen bibliography records have prior metadata
verification, not a completed full-text novelty/protocol comparison.
Related Work must complete those comparisons before making novelty claims.

Sections 2, 9, 10 and the appendix were not drafted or edited. Existing
appendix labels resolve, but their content is still a skeleton; architecture,
domain-specific preparation/hashes, per-seed tables, detailed diagnostics,
failed CRBD/legacy controls, and hardware-specific efficiency remain to be
written. Visible old TODOs in protected Abstract/Introduction are also left
for a separately authorized cleanup. Human review should check narrative
balance, technical definitions, citation placement and figure/table
integration, then choose the venue/template and page budget.

The ML writing skill guided evidence-to-prose mapping and compilation QA.
It did not initiate a full-paper workflow. No training, model forward, new
test evaluation, diagnostic experiment, tuning, or method development was
performed. Frozen evidence, experiments, datasets/checkpoints, old CAC source,
and scientific conclusions remain unchanged. This is a scoped first draft,
not a complete or submission-ready manuscript.

STOP. Further sections or revisions require a new instruction; experiments
remain permanently frozen.
