# COLING 2027 / ARR conversion status

Date: 2026-10-10. Input commit:
`63d7eb72c01d26a22e921a73246ecc9944df9a3a`.

Outcome: a separately compiling anonymous long-paper review source, with
seven main-content pages and fourteen total pages. Document QA is complete;
human submission and research-lead approval are not. No training, forward pass,
model evaluation, new diagnostic, sweep, or endpoint recomputation was run.

## Source and official template

Created `论文投稿/coling2027_arr/` with `main.tex`, `Makefile`, `README.md`, eleven
section files (ten main sections plus Limitations), four appendix files, twelve
table files, `references.bib`, and five SVG/PDF figure pairs. The directory is
self-contained and has no symlinks or dependencies on archived source paths.
A copy compiled successfully in a temporary directory using plain `make`.

Title: When Does Knowledge Distillation Transfer? Controlled Evidence from
Warm-Start Language Modeling.

Official template source: https://github.com/acl-org/acl-style-files

Exact template commit: `d5adc823ff0f80f98c80405ca0ab66c68e684409`.
The official example was compiled before conversion. Vendored files are
byte-identical to this checkout:

| File | SHA-256 |
| --- | --- |
| `acl.sty` | `19dfeddc2c0e448f3926a0bef048a9db3f3611b46265b760caabd7ada4f361de` |
| `acl_natbib.bst` | `6fbb306202290f4b68e74ac1460a8b27398500cb6dfeb4492e74c457eae7cd1e` |

The template already selects `acl_natbib` internally. A duplicate selection
in an initial trial was removed from the new main source; the official files
were not edited. The final document uses review mode, 11pt Times, A4, and two
columns throughout. No margin/font/spacing hacks were introduced.
Two inherited trailing-whitespace lines in `acl_natbib.bst` are deliberately
retained so its byte identity with the official vendor file is not broken.

## Structure and scientific preservation

The last main-content page is 7; all four main figures and Table 1 fall within
pages 2–6. Unnumbered `Limitations` starts immediately after Conclusion on
page 7, without a forced page break, and before References. References spans
pages 7–8. A standard page break occurs only before the double-column appendix.
Total: 14 pages. Abstract: 185 words, with the matched-CE comparator distinction
and no modern stress-test claim.

Retained main figures: paired protocol (Figure 1), cross-domain seed gains
(Figure 2), teacher quality (Figure 3), ensemble control (Figure 4). Retained main
table: cross-domain paired transfer, with exact means, sample SD, and signs.
Full numeric checks also remain in text or the curated appendix, rather than
being dropped to fit the page budget.

Appendix A covers architecture and causal pairs; B covers actual preparation,
continuation, teacher histories, authorized clipping amendments, and accepted
retry; C covers full per-seed endpoints and teacher/source/ensemble metrics;
D covers disagreement, failed local diagnostics, modern continuation
eligibility, clipping, and the failed CE headroom gate. Figure A1 uses generic
alpha, not a hard-coded 0.5. There are eleven appendix tables.

Removed only from the submission view: unmatched legacy latency/deployment and
CodeParrot observations, random-init/CRBD research history, excessive safe-lambda
or calibration trace detail, raw path/hash inventories, and legacy pilot
material not needed for the confirmatory narrative. They remain in the archive
and evidence records. No unsupported successful-CRBD, geometric KD advantage,
heterogeneity superiority, universal teacher-NLL rule, or new method appears.
No adaptive-KD baseline superiority is asserted. No modern gate KD/test result
is invented. The model/data/optimization confounds are preserved explicitly.

Original V2 preservation: all 54 inventoried files retained their pre-conversion
SHA-256 values, including its original prose, figures, and local build products.
Protected experiment/final-evidence/dataset/checkpoint/CAC paths remain unchanged.
The existing scientific state stays permanently frozen. Local project-memory
notes are maintained separately; pre-existing user changes are not staged.

Citation changes: none added or removed. All 24 previously primary-verified
entries are cited; the new bibliography is byte-identical to the archive. This
format conversion did not rerun scientific reference research or claim a new
external bibliographic validation.

## Compilation, figures, and manual visual QA

Main build: `outputs/coling2027_arr/main.pdf`. Rendered previews:
`outputs/coling2027_arr/pages/page-01.png` through `page-14.png`.
Build products and previews are not committed. Earlier trial previews/builds
were moved into an intermediate-output directory, not confused with the final.

Final pdfLaTeX/latexmk/BibTeX build succeeds: no errors, undefined references,
duplicate labels/destinations, or overfull boxes; BibTeX reports zero warnings.
There are 20 underfull hbox and 6 underfull vbox notices, principally dense
numeric prose/reference wrapping and float-heavy appendix whitespace. They
were inspected rather than hidden with font or spacing hacks. A benign imported
PDF page-group warning is retained. None produces clipping or collisions.

All 14 final pages were visually inspected at 110 dpi. Page-level checks:

| Pages | Manual finding |
| --- | --- |
| 1 | Title and abstract fit; anonymous header and review rulers present. |
| 2 | Figure 1 labels/arrows clear; Introduction/Related Work flow correctly. |
| 3 | KD/gain/fusion equations stay in-column; no orphan heading. |
| 4 | Table 1 fits; controlled contrasts and clipping warning remain visible. |
| 5 | Figures 2–3 retain scientific labels; captions and text do not overlap. |
| 6 | Figure 4 gains and caveats readable; Conclusion heading has following text. |
| 7 | Conclusion precedes exact Limitations heading; references follow naturally. |
| 8 | Bibliography wraps without overflow or identifying project links. |
| 9 | Appendix remains double-column; causal equations and footnote fit. |
| 10 | Figure A1 corrected architecture legible; protocol table and text fit. |
| 11 | Telemetry/formula fit; appendix has uneven but non-overlapping whitespace. |
| 12 | Five full-width numeric tables and captions fit at default text size. |
| 13 | Modern warnings/failed gate preserved; no overlong identifier overflow. |
| 14 | Diagnostic/modern endpoint and audit tables fit without clipped labels. |

All five included figure PDFs have embedded fonts, extractable vector text,
and zero embedded raster images; the complete manuscript also has zero embedded
raster images. Three SVG/PDF pairs are unchanged. In the copied Figure 1 and
A1 masters, only the math-script size 22→24 canvas units changed; every text
string, other attribute, module, and arrow is unchanged. Re-exported PDFs retain
aspect ratios and geometry. Independent text-bound/overlap checks pass.

At the actual official 16 cm full width, the minimum figure text/script sizes
are 6.213, 7.133, 6.520, 6.633, and 6.136 pt for Figures 1, 2, 3, 4, and A1.
Thus mathematical scripts satisfy 6 pt. Some normal explanatory figure labels
are approximately 6.5–6.7 pt rather than uniformly 7 pt: they were readable in
full-width inspection, but a research-lead print-scale check remains advisable.
No source font reduction was used to bypass the page limit.

## Scientific and anonymity audits

Original auditor: `audit_body_draft.py --full-revision`, PASS, 62/62.
New submission auditor checks the exact original 62-field inventory against
both new LaTeX and rendered PDF: PASS, 62/62. It also checks archive hashes,
style hashes, abstract, citations, warning labels, scientific state, build log,
figure text preservation, page locations, and anonymous PDF metadata/links.
This is read-only record validation, not endpoint recomputation.

Visible text, PDF author metadata, link destinations, known personal names,
local filesystem paths, personal repository URLs, TODO/FIXME/placeholders,
and main-text internal phase labels were checked. No identifying project link
or affiliation appears. Source-copy compilation confirms portability. These
checks cannot prove unsearchability of already-public research history.

## Official pubcheck: raw result, not a clean PASS

Source: https://github.com/acl-org/aclpubcheck

Checker commit: `f626497051df8689e990fac36ba9f42cd630e31c`.
Run in an isolated environment against the final, stable review PDF:

```sh
python -m aclpubcheck --paper_type long --disable_name_check \
  --output-dir outputs/coling2027_arr/pubcheck outputs/coling2027_arr/main.pdf
```

Raw result: **890 Error.MARGIN findings; 0 warnings; no other error category**.
The separate read-only classification matches all findings exactly to 876
official gray review line numbers and 14 official page numbers. Independent
body-word/vector-drawing bounds checks found no substantive overflow, allowing
only standard microtype punctuation protrusion. No checker, review PDF, ruler,
or page number was modified to obtain this classification. The tool is oriented
to camera-ready proceedings; this review-mode result must not be described as
an upstream zero-error PASS.

The optional name/reference check was disabled to avoid sending the anonymous
PDF to an external name-check service. This means pubcheck did not perform that
external check; the separate 24-entry frozen verification and anonymity audits
are not claimed as its result. Raw logs and annotated pages are local build
outputs; compact numeric/classification reports accompany this status.

## Remaining decisions and research-lead risks

Use `COLING2027_ARR_SUBMISSION_CHECKLIST.md` for all account-level actions.
The verified public schedule lists October 12 submission, October 14 registration,
and December 23 COLING commitment; exact console times and updated policy must
be confirmed by the authors. Codex performed no account actions.

The contribution remains controlled empirical evidence, not a new KD algorithm
or a clean modern-scale negative-transfer confirmation. The primary student is
small; three student preparations do not replicate teacher training; shared
endpoints make controls dependent; TT/TG differ in teacher quality, parameter
count, and histories; clipping constrains interpretation; adaptive baselines,
generation quality, and complete aggregate compute accounting are unavailable.
These are stated limitations, not defects repaired by this format conversion.
The lead should approve this novelty/evidence scope, figure fine-print sizing,
dataset/model licensing and release disclosure, prior ARR history, and public
repository anonymity risk before upload. No further experiment is authorized.

STOP after the compact source/QA commit. Do not launch a follow-up experiment.
