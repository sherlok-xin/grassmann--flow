# Frozen figure vector reconstruction QA

## Current authorized readability revision (2026-10-08)

This section supersedes the historical frozen-layout readability/temporary-compile status below. The lead authorized local typography/canvas fitting, minimal A1 cropping/folding, actual manuscript PDF inclusion, and a new numerical Figure 2. The original QA JSON remains a historical record; current hashes and geometry are in `figure_readability_qa.json`, and the current audit result is `figure_readability_audit.json`.

| Figure | SVG canvas | PDF width | Minimum normal / all text (pt) | 600-dpi PNG |
|---|---|---|---|---|
| 1 | 1752 × 941 | 6.8 in | 7.266 / 6.148 | 4080 × 2191 |
| 2 | 489.6 × 190.8 | 6.8 in | 8.000 / 7.700 | 4080 × 1590 |
| 3 | 1774 × 887 | 6.8 in | 7.038 / 7.038 | 4080 × 2040 |
| 4 | 1983 × 793 | 6.8 in | 7.160 / 7.160 | 4080 × 1632 |
| A1 | 1774 × 730 | 6.8 in | 7.728 / 6.072 | 4080 × 1679 |

Every figure has an editable `figures/<name>.svg` master, vector `figures/<name>.pdf`, and `figures/<name>_600dpi.png`, under `论文投稿/manuscript_v2/`. Names are `fig1_paired_protocol`, `fig2_cross_domain_gain`, `fig3_teacher_quality`, `fig4_ensemble_control`, and `figA1_teacher_architecture`. Liberation Sans is used, with Arial/Helvetica/sans-serif fallbacks for the semantic SVGs; Matplotlib Figure 2 uses the same sans-serif preference. PDF fonts are embedded and text is extractable. Every figure PDF has zero embedded raster images.

Figure 1 widens only the evidence region by 80 source units, enlarges fine print and wraps the weaker-teacher heading. Figure 3 preserves all approved wording, enlarging qualifier/student boxes as needed. Figure 4 wraps two headings, retains exact numerical gains and five “Positive transfer” labels, and moves the exact T-only/TT–TG cautions to its LaTeX caption. Figure A1 crops unused whitespace and folds the unchanged Grassmann module sequence into two rows. The return arrow runs from normalization to geometric projection through the inter-row gap, not backwards through causal pairs. Local direction remains past to current, with valid-Δ averaging and free-alpha late fusion.

The limited typography revision does not preserve identical canvas proportions for Figures 1/A1. These are explicitly authorized readability differences, not an unreported redesign. Other unavoidable differences are line wrapping, font metrics and antialiasing; colors, scientific ranking and module identities remain fixed. Normal labels meet 7 pt and mathematical scripts meet 6 pt at the native 6.8-inch width. A 3.3-inch single-column reduction is not approved.

Figure 2 plots only `research/manuscript_v2/tables/table_endpoint_inventory.csv` (SHA256 `28e2ebe006e4c6f1f1a3369d65efdc5f77db16bd20c9fcb8896f80760f8db2d4`): nine archived seed gains, three means and sample SD, plus zero. There are no confidence intervals, p-values, fitted trends, model calls or new endpoints. Visual QA found a legend/annotation collision in the first draft; the legend was moved outside the axes before final export.

All five exports were viewed, and the four original references were compared side-by-side with the final revisions. Bounds/owner-box and pairwise text-ink checks find zero violations in Figures 1/3/4/A1. Figure 2 passes PDF span bounds and manual inter-label review. Equations, subscripts, arrows, panel captions and final-page placement were reviewed. The actual `main.tex` uses a 6.8-inch text block, so figures are not silently reduced below the tested font sizes.

Actual-main build: `make -C 论文投稿/manuscript_v2`, exit 0, 23 pages. Final log has zero undefined citations/references, duplicate labels/destinations, overfull boxes and underfull notices. Figures appear on pages 4, 6, 8, 9 and 16, numbered 1/2/3/4/A1. All pages were rendered and reviewed; selected figure pages received enlarged inspection. A short appendix optimization table was changed from longtable to an ordinary float to remove a detected vertical overflow, without changing any data. The PDF figures are recorded as actual build inputs in `main.fls`. No Figure 5 placeholder remains. The main build PDF/page renders are local build products and not committed.

Current checks: `audit_vector_figures.py --readability-revision` and `audit_body_draft.py --full-revision`. Old default audit mode targets the old frozen-layout snapshot, not the newly authorized sources. The original 62 scientific checks remain intact. No conference-specific two-column template or page-limit certification is implied.

## Historical frozen-layout reconstruction (2026-10-04)

Date: 2026-10-04

Input repository HEAD: `11b3ed94344c7c8e41ced2d3e03311da1ad759c8`

Scope: reconstruction of four approved figures, not redesign, new science, manuscript drafting, or experiments.

## Outcome

Vector integrity, editable text, source aspect ratios, module/text bounds, mathematical notation, authorized wording, and temporary full-manuscript PDF inclusion checks passed. All four PDFs contain **zero embedded images** and embedded/subset fonts. Reference bitmaps were viewed only; they were not traced, OCR-processed, sampled to derive paths, or placed inside an export.

Print readability is **LIMITED**, not unconditionally certified. The frozen references contain dense fine print, especially Figure A1 and the Figure 4 cautions. At a 6.8-inch two-column span, several labels are below 7 pt and the smallest A1 mathematical subscript is approximately 2.78 pt. Vector reconstruction cannot eliminate that inherited limitation without changing the approved layout. No modules were enlarged, cropped, removed, or rearranged to obtain a misleading readability pass.

The publication assets are complete; production manuscript source integration was deliberately not performed because the requested commit is figure/QA-only. Actual PDF inclusion was verified in a temporary full copy of the current manuscript. No new training, forward pass, experiment, data access, or numerical-result generation occurred.

## Source-to-output inventory

Reference directory: `论文投稿/manuscript_v2/figures/reference_png/`.

Output directory: `论文投稿/manuscript_v2/figures/`.

| Figure | Source reference | Editable master | LaTeX PDF | 600-dpi QA/fallback |
|---|---|---|---|---|
| 1 | `fig1_paired_protocol_ref.png` | [fig1_paired_protocol.svg](../../论文投稿/manuscript_v2/figures/fig1_paired_protocol.svg) | [fig1_paired_protocol.pdf](../../论文投稿/manuscript_v2/figures/fig1_paired_protocol.pdf) | [fig1_paired_protocol_600dpi.png](../../论文投稿/manuscript_v2/figures/fig1_paired_protocol_600dpi.png) |
| 3 | `fig3_teacher_quality_ref.png` | [fig3_teacher_quality.svg](../../论文投稿/manuscript_v2/figures/fig3_teacher_quality.svg) | [fig3_teacher_quality.pdf](../../论文投稿/manuscript_v2/figures/fig3_teacher_quality.pdf) | [fig3_teacher_quality_600dpi.png](../../论文投稿/manuscript_v2/figures/fig3_teacher_quality_600dpi.png) |
| 4 | `fig4_ensemble_control_ref.png` | [fig4_ensemble_control.svg](../../论文投稿/manuscript_v2/figures/fig4_ensemble_control.svg) | [fig4_ensemble_control.pdf](../../论文投稿/manuscript_v2/figures/fig4_ensemble_control.pdf) | [fig4_ensemble_control_600dpi.png](../../论文投稿/manuscript_v2/figures/fig4_ensemble_control_600dpi.png) |
| A1 | `figA1_architecture_ref.png` | [figA1_teacher_architecture.svg](../../论文投稿/manuscript_v2/figures/figA1_teacher_architecture.svg) | [figA1_teacher_architecture.pdf](../../论文投稿/manuscript_v2/figures/figA1_teacher_architecture.pdf) | [figA1_teacher_architecture_600dpi.png](../../论文投稿/manuscript_v2/figures/figA1_teacher_architecture_600dpi.png) |

| Figure | Reference / SVG viewBox | Aspect ratio | PDF dimensions (pt) | PNG dimensions (px) |
|---|---:|---:|---:|---:|
| 1 | 1672 × 941 | 1.776833156 | 489.600 × 275.546 | 4080 × 2296 |
| 3 | 1774 × 887 | 2.000000000 | 489.600 × 244.800 | 4080 × 2040 |
| 4 | 1983 × 793 | 2.500630517 | 489.600 × 195.791 | 4080 × 1632 |
| A1 | 1774 × 887 | 2.000000000 | 489.600 × 244.800 | 4080 × 2040 |

All SVG/PDF canvases preserve the reference ratio exactly. The PNG pixel height is rounded to the nearest integer; this changes the ideal height by less than half a pixel. PNG metadata reports 599.9988 dpi, the normal integer-pixels-per-meter representation of 600 dpi.

## Reconstruction and typography

Each SVG was authored manually from rectangles, rounded modules, panel borders, paths, lines, arrow markers, circles, and editable text. Modules have semantic IDs, and text has invisible `data-box` metadata for checking its owning rectangle. No pixel-derived path construction was used. The diagram wording was manually transcribed from the visually reviewed references and the explicit instructions; no OCR was used.

The font stack is `"Liberation Sans", Arial, Helvetica, sans-serif`. Labels retain a common sans-serif family and bold/regular hierarchy. Mathematical variables use italic tspans, with true subscript/superscript positioning. The normalization fraction, its bar, and the hat over p are independently editable vector primitives. The primary font is embedded in all PDFs. The PDF renderer uses embedded Arimo for the calligraphic loss symbol in Figure 1 and embedded Nimbus Sans for a mathematical glyph in Figure A1; these are glyph-level vector fallbacks, not raster content.

Native librsvg 2.52 / pycairo exported PDF text and paths. The 600-dpi PNGs and reference-size QA previews were rendered directly from the SVG masters. No image-generation model was used. The academic-plotting export/visual-QA guidance and PDF inspection workflow informed the font, embedding, rendering, and compile checks; generative design steps were not used because the layout is frozen.

## Figure-specific semantic checks

### Figure 1

The same validation-selected warm-start S0 forks into matched CE and KD continuation. The two best-checkpoint modules and their separate paths to the gain box are retained. Gain is typeset as `Gain = NLL_CE − NLL_KD`, with true subscripts. Both beneficial/harmful sign interpretations and the four evidence-map boxes remain in their original positions.

The final evidence-map statement is exactly “TT > TG in the tested control; the tested local diagnostics are not reliable predictors”, with line breaks fitted inside the original box. No universal predictor or mechanistic claim was added.

### Figure 3

All 29 visible text labels were manually checked against the reference and locked in the read-only audit. Panel A preserves the teacher-state intervention with the same architecture, T/G composition and alpha=0.5, matched independent students, Teacher J/A labels and opposite signs. Panel B preserves the worse-held-out-likelihood comparison to the matched CE comparator and positive transfer from the Grassmann-only teacher. Neither panel is relabeled as proof that predictive quality alone causes transfer.

The title, both panel conclusions, and the complete bottom sentence remain unchanged, including “held-out teacher likelihood alone does not determine distillability.”

### Figure 4

The five source/control result boxes retain their original dimensions and positions. Exact displayed gains are G +0.0410 NLL, T +0.1349 NLL, TG +0.1557 NLL in both panels, and TT +0.1681 NLL. All five boxes say “Positive transfer”.

The decorative green one/two/three-arrow rank icons were removed. All verbal gain-size labels were removed, including the reference label “moderate gain”, so the numbers alone state the ranking. This is the instructed exception to layout/content fidelity; the resulting whitespace is retained instead of resizing the modules.

Both cautions are preserved verbatim: “T-only: early clipping saturation; interpret F–T gap cautiously.” and “TT and TG teachers are not strictly quality- or training-matched.” The TT-over-TG and component-source comparisons retain the reference interpretation; no geometry-specific advantage is introduced.

### Figure A1

The generic architecture retains the reference canvas, branches, nine Grassmann processing modules, logits boxes and fusion wiring. Two explicit label corrections supersede the visible raster labels: W_in → W_red and p → p_ij.

The manually reconstructed Grassmann path is:

`Token embedding → linear projection W_red → causal local pairs (z_t, z_(t−Δ)), Δ∈{1,2,4} → Plücker coordinates p_ij → p-hat = p / max(‖p‖_2, ε) → geometric projection → window mean over valid Δ → gated mixing → LM head → G logits`.

The causal direction label is `z_(t−Δ) → z_t`, not the reverse. The normalization epsilon floor and valid-Δ qualification are retained. Fusion is `z = α z^(T) + (1 − α) z^(G)`, with true superscripts; no fixed alpha=0.5 appears anywhere in the visible architecture.

## Visual and structural inspection

Every master was rendered at the reference pixel dimensions and inspected against its reference. All four were also rendered at 6.8-inch width / 180 dpi for a two-column-span inspection. PDF figures were then inspected inside the compiled manuscript, not only as SVG previews.

| Check | Result |
|---|---|
| Pure white background / flat pastel vector modules | PASS |
| Same aspect ratio, panel composition, module sizes, whitespace and arrow routing | PASS, manually reviewed; only authorized gain decorations removed |
| SVG image / foreignObject / filter / external bitmap dependency | None |
| Embedded PDF images | 0 / 0 / 0 / 0 |
| Embedded, subset fonts and extractable PDF text | PASS |
| Editable SVG text elements | 37 / 29 / 44 / 47 |
| Text ink beyond canvas or owning module | 0 / 0 / 0 / 0 |
| Pairwise text ink overlap | 0 / 0 / 0 / 0 |
| Arrow collisions, clipping, broken math glyphs | None observed in rendered review |
| Universal small-print readability at two-column span | NOT FULLY PASSED; frozen-layout limitation |

The bounds test uses librsvg ink geometry, converted from intrinsic CSS pixels to source viewBox units. Pairwise checks use a one-source-unit intersection tolerance. Manual inspection supplements these checks for paths, hats, fraction bars and superscripts; numerical bounds checks alone are not claimed to prove perceptual legibility.

| Figure | Base text size range at 6.8 in (pt) | Smallest text including scripts (pt) | Limitation |
|---|---:|---:|---|
| 1 | 5.534–10.249 | 4.919 | Evidence-map fine print is below 7 pt |
| 3 | 5.931–15.179 | 5.931 | Fine-print qualifications are below 7 pt |
| 4 | 4.815–9.629 | 4.815 | Caution labels remain small |
| A1 | 3.795–8.280 | 2.782 | Dense local formulas and normalization subscript are very small |

A single 3.3-inch column would roughly halve these sizes and is not approved by this QA. At the current manuscript width of 6.5 inches, sizes are 6.5/6.8 times the values above. No blanket publication-readiness claim is made for those annotations.

Unavoidable visual differences are known-font metrics and antialiasing, minor font-size fitting for long labels inside unchanged boxes, line breaking for the explicitly changed Figure 1 statement, and flat pastel fills replacing mild shading in the raster references. There is no pixel-identical claim. The large white margins in A1 were intentionally retained, not cropped to make the lettering look larger.

## Manuscript compile result

The current full manuscript, including the pre-existing user Abstract and Introduction, was copied to:

`outputs/manuscript_v2_vector_qa/manuscript-wUmJtk/`

Only this disposable copy received `graphicx`, actual PDF inclusion in place of the Figure 1/3/4 placeholder calls, and an A1 figure in the appendix. Captions identify the existing figure semantics; no scientific prose was rewritten. Figure A1 numbering and the corresponding hyperref destination were scoped to the appendix in the copy. Figures 2 and 5 remain placeholders and are outside this task.

Command: `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex`.

Result: exit 0, 17 pages. Final-pass log: zero overfull boxes; zero underfull figure-placement notices; zero unresolved references/citations or duplicate destinations. Three underfull bibliography-spacing notices remain from the existing bibliography and are unrelated to the figures.

The actual vector PDF files were confirmed as build inputs. Pages 4 (Figure 1), 8 (Figure 3), 10 (Figure 4) and 16 (Figure A1) were rendered with Poppler and visually inspected. No cropped figure labels, page-margin overflow, caption collision, or broken equation glyphs were observed. Small lettering remains as recorded above. Page 10 places Figure 4 below the existing control tables without collisions; A1 preserves its inherited white margins.

The current manuscript uses a venue-neutral single-column article class. This compile verifies its 6.5-inch full-width placement, comparable to a two-column-spanning figure, not placement inside a future venue-specific two-column template. No single-column-width legibility or final conference-template certification is claimed.

The temporary compiled manuscript and page previews remain local and are not committed. Their exact PDF/log hashes are recorded in `figure_vector_qa.json`. Production `main.tex`, all sections, existing labels/captions, and frozen experimental evidence were not changed by this task. Existing user edits are excluded from the figure commit.

## Reproducibility and commit boundary

Authoritative hashes, dimensions, checks and compile provenance are in [figure_vector_qa.json](figure_vector_qa.json).

To verify the frozen artifacts without modifying them:

`python3 research/manuscript_v2/scripts/audit_vector_figures.py`

To regenerate exports from the editable masters:

`python3 research/manuscript_v2/scripts/render_vector_figures.py`

The renderer uses only SVG masters, never PNG references. It requires system librsvg/gi, pycairo, Pillow and pypdf. The read-only audit requires Pillow, pypdf and Poppler `pdffonts`. Re-exporting PDFs may change creation metadata and therefore hashes; update the QA snapshot only after renewed inspection.

The commit contains only the twelve requested figure outputs, the figures README, this QA report, its JSON snapshot, and the two compact export/audit scripts. Local source references, large previews, temporary manuscript copies, build artifacts, unrelated user edits and project-memory/journal entries are excluded. Experiments and manuscript content remain frozen.

STOP after this reconstruction.
