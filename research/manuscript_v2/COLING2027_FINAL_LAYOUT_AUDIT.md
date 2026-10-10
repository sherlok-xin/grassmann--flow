# Final COLING 2027 / ARR layout audit

Date: 2026-10-11. Authoritative input commit:
`cd8b5504403a21cb2e98efb446e3ca680340032f`.

Outcome: final document-polish pass completed. Main content ends on page **7**;
the PDF has **14 pages**. There is no identified BLOCKING document-formatting
defect after this pass. This is not scientific acceptance certification or an
account-level submission. Experiments remain permanently frozen; no training,
model forward pass, new diagnostic, endpoint recomputation, or remote GPU work
was performed. The PDF skill workflow required baseline rendering, local crops,
and a final visual pass rather than relying on successful compilation alone.

## Artifacts and provenance

Source: `论文投稿/coling2027_arr/`. Final PDF:
`outputs/coling2027_arr/main.pdf`.

Final PDF SHA-256:
`0408342b232c6e3ba86deae8129cec552ed9e3e5a5c7b3ac4060a27a28792e20`.

The unedited source was force-recompiled first with `latexmk -g -pdf
-interaction=nonstopmode -halt-on-error`, before any manuscript edit. Its PDF,
log, auxiliary file, and all-page renders are retained locally in
`outputs/coling2027_arr/polish_baseline/`. Baseline PDF SHA-256:
`b5fc5b3bc387472520fc05e9f83111b231e0988d4d2aedfc610dc4501acf7f8f`.
Recompilation can change PDF metadata and therefore its byte hash.

Final previews are in `outputs/coling2027_arr/final_pages/`: all fourteen pages
at 150 dpi, all fourteen additionally at 96 dpi in `print_scale_96dpi/`, and
seventeen 300-dpi figure/table crops in `local_300dpi/`. The latter include all
five figures and all twelve tables with their captions. Baseline and final
pages were visually inspected, not only checked by text extraction. Final
pages 11–14 were inspected in the last layout trial; the retained audit verifies
that their final preview files are pixel-identical to those inspected images.

Compact machine evidence: `coling2027_final_layout_qa.json`. It retains the
complete 62-field audit inventory, source hashes, baseline/final warning
inventories with log locations, render inventory, and pubcheck classification.
Generated manuscript PDFs, build products, trial renders, and preview PNGs
are excluded from the commit. The already tracked Figure 2 PDF is retained as
a required compact LaTeX figure input, not as a generated manuscript build.

## Exact changes

Only two scientific-prose refinements were made.

In `main.tex`, the sole Abstract replacement is `whether KD helps depends on`
→ `whether KD helps varies with`. Every other Abstract word is unchanged;
the Abstract remains 185 words.

In `sections/06_distillability.tex`, the old text was:

> The comparator is the selected CE continuation, not an arbitrary initialization, and residual likelihood and transfer gain use different held-out scopes (Appendix~\ref{app:per-seed}).

It is now:

> The comparator is the selected CE continuation, not an arbitrary initialization. Teacher likelihood is compared against the matched CE endpoint on the same validation subset, whereas transfer gain is evaluated on the selected test endpoints (Appendix~\ref{app:per-seed}).

All numerical values, scope distinctions, caveats, citations, and conclusions
are preserved. No other prose was shortened to gain space.

The layout-only changes are:

- `appendix/results.tex`: move only `\input{tables/per_seed_gains}` to the
  natural boundary immediately before the opening paragraph of Appendix C.
  Table order and paragraph order remain unchanged. Table 4 can now appear
  on page 11 instead of waiting until page 12. All table source files are
  byte-identical to the input commit.
- Add `\allowbreak` after the slash in the three full-validation TG/TT NLL
  pairs and the modern teacher validation/test NLL pair. This permits ordinary
  line breaking without changing characters, numbers, or spacing parameters.
- In `appendix/diagnostics_modern.tex`, permit optional breaks at underscore
  boundaries in `STOP_MODERN_CONTINUATION_NOT_ESTABLISHED`, preserving the
  identifier and existing monospace style. This avoids isolating the preceding
  prose on an extremely sparse line, although two underfull notices remain.
- In `figures/fig2_cross_domain_gain.svg`, move only the text `-0.107219`
  from y=171.967564 to y=165.967564, also updating its matching rotation-center
  coordinate. This six-native-point upward move clears the lower axis. The
  plotted data, axes, uncertainty bars, all text, and all font sizes are
  unchanged. Re-export its pure-vector PDF. No PNG tracing or raster embedding.

No float specifier, caption wording, figure width, body font, page dimension,
paragraph spacing, penalty, template file, or citation was changed. The source
README documents the annotation repair. Two document auditors were updated:
one allows only that precise Figure 2 coordinate change; the other reports the
current pubcheck count dynamically. Two new helpers render QA previews and
verify final-polish scope. This report, its JSON companion, and links/current
locations in the existing conversion/checklist/page-budget reports complete
the committed QA changes. Local project-memory/journal notes are maintained
but not staged with pre-existing user edits.

## Page-by-page visual inspection

On every page the inspection covered margins, column flow, headings,
paragraph/line breaks, equations, floats, caption wrapping, bibliography or
appendix structure, gray review rulers, and the footer number. No text/arrow
collision, clipped scientific label, new isolated citation, invisible content,
or overflow was found. Table/figure-specific details follow below.

| Page | Final visual finding |
| --- | --- |
| 1 | Two-line title and anonymous header fit. Official title-block whitespace remains. Abstract edit wraps normally; Introduction has following text. Both columns fill normally. |
| 2 | Full-width Figure 1 retains its evidence map, independent selections, gain notation, arrows, and caption. Its internal bottom canvas whitespace remains. Introduction/Related Work flow without a stranded heading. |
| 3 | Two balanced text columns; equations (1)–(3) remain in-column. The first protocol line is somewhat widely justified but readable. No equation or citation is clipped. |
| 4 | Main Table 1 is legible at natural one-column width. The revised weaker-source paragraph continues into the right column clearly. Heading 6.2 has two following lines before the break. Equation (5) has its number on a separate line naturally; no collision. |
| 5 | Figures 2 and 3 remain full-width and stacked. TinyStories numeric annotation now clears the axis. Both captions fit. The two short text columns below the floats retain normal template heading separation. |
| 6 | Figure 4 gains and caption caveats are intact. Discussion follows the bounded stress test. Conclusion has two following text lines before continuing on page 7; it is not a bare heading. |
| 7 | Conclusion → exact unnumbered Limitations → References, without a forced break. Bibliography spacing is regular. A long cited author list crosses a column naturally, without losing content. |
| 8 | References continue cleanly. The right column ends earlier than the left; remaining space is the existing bibliography/appendix boundary, not omitted text. No malformed entry, broken line, or identifying project link found. |
| 9 | Appendix is two-column. Equations (6)–(11), normalization, causal direction, and footnote URL fit. The URL wraps at its existing legal break; the PDF link destination is intact. |
| 10 | Figure A1 and Table 2 remain full-width; causal arrows, normalization, free alpha, and caption are readable. Protocol text has mildly uneven paragraph spacing. Heading C has four following lines. |
| 11 | Tables 3 and 4 fit at the top. Moving Table 4 earlier reduces the previously conspicuous repeated paragraph gaps in C. Optional breaks fix the most stretched TG/TT NLL paragraph. D and D.1 have adequate following text. A moderate gap before the G paragraph remains. |
| 12 | Four full-width Tables 5–8 now replace the former five-table stack. The body below has materially more room. Diagnostic equation (12) fits. Heading D.2 has two following lines before page 13, not a single-line orphan. |
| 13 | Table 9 now precedes continued D.2/D.3 text. The former large blank right-column tail and oversized heading separation are reduced. Teacher NLL pairs wrap normally. The permanent decision/warning identifiers remain legible. Right column ends about three lines above the left, an acceptable final-text imbalance. |
| 14 | Final float-only appendix page contains Tables 10–12, with all captions and decimals readable. Standard float-page vertical separation remains; no columns were manually widened, tables shrunk, or rows removed. Footer number retained; no body lines require rulers here. |

## Whitespace ledger

Classification: A normal ACL two-column float behavior; B `figure*` placement;
C full-width tables; D section/page break; E appendix float imbalance;
F avoidable float ordering; G avoidable paragraph/heading issue; H other.
The ledger includes every visually unusual large region, including noteworthy
float/caption/heading gaps smaller than one-fifth of a column. Ordinary line,
equation, and caption separation is not treated as an error. Dimensions below
are approximate visual PDF-point extents, not an optimization target.

| Baseline/final page and region | Cause | Action and reason |
| --- | --- | --- |
| 1, title/anonymous-header/Abstract separation | H: official title block | Retained byte-identical preamble/template; not missing content and not adjustable with spacing hacks. |
| 2, bottom of Figure 1 canvas to caption, about 40 pt; caption to body | B/H, then A | Retained frozen figure aspect/composition and normal double-float separation. Cropping or one-column reduction would alter the approved layout or harm text size. |
| 5, stacked wide figures and short-column heading gaps | B/A | Retained. Two diagrams plus captions legitimately leave short text columns; no custom glue introduced. |
| 6, Figure 4 caption to body | B/A | Retained normal float/text separation; moving the figure later would not improve readability. |
| 8, below the final right-column reference, over 500 pt | D: bibliography ends before existing appendix page break | Retained. Appendix starts on the next page in two columns. Did not pull appendix content into References or change the official vertical spacing to cosmetically balance it. |
| 10, A1/Table 2 to body and protocol paragraph spacing | B/C/E | Retained readable full-width architecture and table. Some unequal paragraph separation is preferable to resizing figures or changing template glue. |
| Baseline 11, several C paragraph gaps roughly 40–55 pt | F/E | Improved: place Table 4 call one paragraph earlier, allowing it on page 11. Final C uses normal text flow below two tables; a smaller gap before the G paragraph remains (E). |
| Baseline 12, five-table stack leaving very short body columns | C/F/E | Improved: Table 4 moves to page 11, leaving Tables 5–8 and longer diagnostic text on page 12. No table order/content changes. Normal caption-to-body separation remains (C/A). |
| Baseline 13, large D.2 heading gap and blank right tail about 280 pt | F/E/G | Improved by upstream float timing and optional line breaks. Table 9 fits on page 13 and both columns carry the remaining diagnostics. Final right tail is about three lines, not a blank half-page. |
| Final 14, two large inter-table bands about 80 pt each, plus top/bottom float-page slack | C/E: ordinary final float page | Retained and explicitly acknowledged. Tables 10–12 remain at natural size with complete captions. Forcing top packing would require unnecessary spacing controls; the blank regions are less harmful than the baseline stretched body paragraphs. |

An intermediate trial moving all four Appendix C table calls earlier was
rejected because it produced a five-table stack with too little opening prose.
The final edit moves only Table 4. No additional experiment or scientific
reordering accompanied these document-only layout trials.

## Figure and table readability / print-size approximation

All five SVG/PDF figure pairs remain true vector graphics with extractable
text and embedded fonts. The complete manuscript contains zero embedded raster
images. Figures 1, 3, 4, and A1 are byte-identical to this pass's input commit;
only Figure 2 has the precise collision repair above. A1 retains generic alpha.

| Figure | PDF page | Minimum text/script at 16 cm width | Full-width decision / inspection |
| --- | ---: | ---: | --- |
| 1 | 2 | 6.213 pt | Keep wide: protocol and evidence map need simultaneous reading. Small evidence-map labels remain readable but require attention at the 96-dpi proxy. No safe content movement was warranted with frozen wording. |
| 2 | 5 | 7.133 pt | Keep wide: three seed groups, mean/SD symbols, legend, six-decimal means, and axis labels stay distinguishable. Negative mean no longer touches the axis. |
| 3 | 5 | 6.520 pt | Keep wide: both causal-boundary panels remain legible. Bottom summary and small panel annotations are the least comfortable normal-size labels, but neither clipped nor overlapping. No content or font reduction. |
| 4 | 6 | 6.633 pt | Keep wide: five source/control boxes, exact rounded gains, and bottom caveat fit. Caption retains clipping and quality/history caveats at normal caption size. |
| A1 | 10 | 6.136 pt | Keep wide: subscripts, causal local pair direction, normalization, projection/mean order, and free-alpha fusion remain distinguishable. Dense appendix notation is small but intact. |

Reducing these unchanged full-width compositions to one column would nearly
halve their type size (approximately 3–3.4 pt for the current minima), which
is not acceptable. No figure was mechanically forced to a 7-pt minimum.

The extra 96-dpi A4 rendering is approximately 794 × 1123 pixels, a common
100%-size desktop PDF proxy. Pages 2, 5, 6, and 10 were inspected at that
unmagnified rendering, together with table-heavy pages 11, 12, and 14. It is
not a calibrated monitor or a physical print test. The small explanatory
figure labels and A1 mathematical scripts remain a NON-BLOCKING comfort
limitation, not proof that every reader will find small type comfortable.
The 300-dpi local checks establish glyph integrity, not a substitute for scale.

All twelve tables were individually inspected in 300-dpi crops, including
caption notes. Tables 2–12 remain `table*` in a two-column appendix, not a
one-column-mode appendix. Table 1 remains a natural one-column table. Default
table/body sizes were not reduced; decimal columns align consistently, and
positive/negative directions are unambiguous. Existing math-minus glyphs in
Table 1/prose versus text-minus glyphs in some archived numeric tables remain
a minor cross-table typographic difference; no values or sign conventions
were changed to homogenize their appearance. No table source or caption changed.

## Compilation and warning-by-warning review

Final latexmk/pdfLaTeX/BibTeX build succeeds. Undefined references/citations,
duplicate labels/destinations, and overfull boxes: **0**. BibTeX warnings: **0**.
Baseline: **20 hbox / 6 vbox underfull**. Final: **14 hbox / 3 vbox underfull**.
An existing imported-PDF page-group warning for Figure 3 remains non-blocking;
its visual rendering is intact. Warning counts were not suppressed.

IDs below enumerate each kind in log order; the JSON records exact log lines
and badness for every baseline and final notice. Grouped rows name every
individual notice rather than omitting warnings with the same cause.

| Baseline ID → final ID | Page / location | Classification and disposition |
| --- | --- | --- |
| H1 → H1 (3138) | 3, first protocol line | Mild paragraph justification; retained, no clipping or reading ambiguity. Rewriting frozen prose solely for this line is unwarranted. |
| H2 → H2 (4647) | 4, weaker-source numeric line | Mild numeric-prose justification; retained. Approved scope clarification is complete. |
| H3 → H3 (1014) | 4, TT numeric comparison | Mild numeric-prose justification; retained, legible. |
| H4 → H4 (4832) | 7, Ben Allal author list | Harmless bibliography wrapping; no author/title/citation removed. |
| H5 → H5 (1043) | 8, Panigrahi reference | Harmless bibliography wrapping; no overflow. |
| H6 → H6 (10000) | 9, external background URL footnote | Harmless URL wrapping at an existing break; full link target verified. Not an identifying project URL. |
| H7 → H7 (2261) | 10, TT clip-fraction list | Numeric paragraph justification; retained without changing recorded values. |
| H8, H9, H10 → same IDs (1038, 1540, 2134) | 11, G residual/gain list | Numeric paragraph justification; readable despite slightly loose spacing. All signs and precision retained. |
| H11, H12, H13, H14 → removed | 11, TG/TT teacher and branch NLL pairs | Genuinely poor numeric paragraph justification; repaired using optional slash breaks. No prose or numbers changed. |
| H15 → H11 (3635) | 12, start of modern parameter paragraph | Numeric paragraph justification; retained. Heading has two following lines. |
| H16, H17, H18 → H12 (1163) | 13, modern teacher validation/test paragraph | Genuinely poor justification substantially improved by slash break; remaining mild line is readable. |
| H19, H20 → H13, H14 (10000 each) | 13, long frozen decision identifier | Optional underscore breaks improve preceding prose flow. Remaining monospace line justification retained; no overflow or identifier corruption. |
| V1 → V1 (3118) | 5, short body under Figures 2–3 | Normal short columns caused by full-width floats. Retained. |
| V2 → V2 (10000) | 10, appendix protocols below wide floats | Appendix float/page balance; retained normal template spacing. |
| V3 → V3 (10000 → 4531) | 11, appendix results | Genuinely poor float layout improved by Table 4 timing. Residual moderate paragraph gap retained. |
| V4, V5 → removed | Baseline 12, both short columns below five tables | Appendix float/page balance improved by moving only Table 4 earlier. |
| V6 → removed | Baseline 13, final diagnostic text imbalance | Appendix heading/float imbalance improved by upstream reflow and optional breaks. |

## Scientific preservation and anonymity

`audit_body_draft.py --full-revision` is rerun by the submission auditor:
**62/62 PASS**. The submission auditor separately tests the exact same
62-field inventory in LaTeX and the rendered PDF: **62/62 PASS**.
`audit_vector_figures.py --readability-revision` invokes the existing archival
readability auditor and passes without editing the archive. Geometry/bounds
and final-polish scope audits also pass. These are checks against frozen
records, not fresh model measurements.

| Preserved quantity | Exact value |
| --- | --- |
| PTB gain | +0.144937 ± 0.003938 |
| WikiText-2 / Teacher J gain | +0.155665 ± 0.007692 |
| TinyStories gain | −0.107219 ± 0.000415 |
| Teacher A gain | −0.027654 ± 0.006822 |
| Q | +0.183319 ± 0.001375 |
| G gain | +0.040991 ± 0.006733 |
| T gain | +0.134868 |
| TT gain | +0.168125 |
| H | −0.012460 ± 0.001348 |
| FineWeb lambda 1 / lambda 5 | −0.013550 ± 0.003542 / −0.043924 ± 0.004017 |
| Best CE headroom | +0.004849 < 0.01 |

The two approved prose substitutions are checked exactly against the input
commit. Normalizing only float-call positions and discretionary breaks proves
appendix wording unchanged. All other 36 existing submission files outside
the seven specifically edited files remain byte-identical, including all
twelve tables, all references, and both official styles. The earlier 54-file
archival preservation inventory also passes. Protected evidence, archived V2,
experiments, datasets, checkpoints, and CAC source have no changes.

Anonymity: source and rendered PDF scans, PDF author metadata, link targets,
and visual title/reference inspection pass. No personal affiliation, author
email, local filesystem path, personal GitHub/repository URL, acknowledgments,
or identifying self-reference was found. Legitimate cited author names remain
in References. PDF author metadata is empty; title metadata is anonymous.
No TODO/FIXME/PLACEHOLDER or hidden submission instruction appears. This does
not guarantee that already-public research cannot be identified by search.

## Official pubcheck and ARR formatting

Official checker commit: `f626497051df8689e990fac36ba9f42cd630e31c`.
Unmodified upstream command, run against the final stable PDF:

```sh
python -m aclpubcheck --paper_type long --disable_name_check \
  --output-dir outputs/coling2027_arr/pubcheck outputs/coling2027_arr/main.pdf
```

Raw result: **890 errors, 0 warnings**, all `Error.MARGIN`. This is **not** a
zero-error pubcheck PASS. The independent classification matches **451 left
review rulers + 425 right review rulers + 14 page numbers = 890**. There are
zero non-review-marker margin findings. Independent body-word and vector-path
bounds also pass, allowing only the standard microtype punctuation protrusion
already present in the unmodified template. All these review numbers are
**EXPECTED REVIEW-TEMPLATE ARTIFACT**, not an unwanted watermark.

Review mode, rulers, page numbers, margins, and official styles were not
disabled, shifted, covered, or altered. The optional external name/reference
service was disabled to avoid transmitting the anonymous PDF; pubcheck did
not perform that check. Existing 24-entry reference verification and local
anonymity checks are reported separately, not attributed to pubcheck.

The current [ARR author checklist](https://aclrollingreview.org/authorchecklist)
and [ACL formatting instructions](https://acl-org.github.io/ACLPUB/formatting.html)
were checked during this pass. The source retains official 11pt body type,
`[review]`, A4 and two columns, standard vertical spacing, seven main-content
pages within the eight-page allowance, and Conclusion → Limitations →
References → Appendix. No main content was moved into Limitations to evade
the limit. There are no font/margin/spacing hacks or acknowledgments. Official
style hashes remain those recorded in the conversion report and final JSON.

## Remaining concerns and stop

BLOCKING: no identified formatting, anonymity, compile, or frozen-science
regression blocker in the inspected artifact. This statement is limited to
this document pass, not account eligibility, author attestations, or acceptance.

NON-BLOCKING: small but intact figure labels/scripts; the digital print proxy
does not certify a physical print; fourteen hbox and three vbox underfull
notices; ordinary bibliography-end and appendix float-page whitespace; minor
text-minus/math-minus styling differences; a benign imported-PDF page-group
warning. None was hidden or repaired with forbidden layout adjustments.

EXPECTED REVIEW-TEMPLATE ARTIFACT: official gray line numbers and all page
numbers, including the 890 raw pubcheck margin findings. Keep them in the
anonymous review PDF. Human submission responsibilities remain as listed in
the ARR checklist. STOP: no additional scientific work or polishing stage is
automatically started.
