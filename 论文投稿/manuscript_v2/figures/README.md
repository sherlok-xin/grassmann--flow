# Publication vector masters — current readability revision

2026-10-08: all five figures below are integrated in the actual manuscript,
not a temporary copy. The lead authorized enlarged type and minimal A1
folding/cropping. Ordinary text is at least 7 pt and mathematical scripts at
least 6 pt at 6.8-inch width. The frozen-layout notes below are historical.
Current metadata: `research/manuscript_v2/figure_readability_qa.json`.

Figure 2 is new: [SVG](fig2_cross_domain_gain.svg), [PDF](fig2_cross_domain_gain.pdf),
[600-dpi QA PNG](fig2_cross_domain_gain_600dpi.png). It contains only archived
paired gains, per-seed dots, mean ± sample SD and zero; no CI or p-value.
The four other file names remain unchanged. A1 has appendix numbering;
no F5 placeholder is retained. Figure 4 cautions are in its LaTeX caption.

Current read-only audit:
`python3 research/manuscript_v2/scripts/audit_vector_figures.py --readability-revision`.
Renderer inspection without changing exports: `render_vector_figures.py --audit-only`
(system Python with librsvg bindings). Full QA is in `FIGURE_VECTOR_QA.md`.

## Historical first reconstruction

Four approved references have been semantically reconstructed as editable SVG masters, vector PDFs, and 600-dpi PNG QA/fallback exports. References are local visual inputs under `reference_png/`; they are not embedded in any output.

| Figure | Master | LaTeX PDF | QA PNG |
|---|---|---|---|
| 1 | [fig1_paired_protocol.svg](fig1_paired_protocol.svg) | [fig1_paired_protocol.pdf](fig1_paired_protocol.pdf) | [fig1_paired_protocol_600dpi.png](fig1_paired_protocol_600dpi.png) |
| 3 | [fig3_teacher_quality.svg](fig3_teacher_quality.svg) | [fig3_teacher_quality.pdf](fig3_teacher_quality.pdf) | [fig3_teacher_quality_600dpi.png](fig3_teacher_quality_600dpi.png) |
| 4 | [fig4_ensemble_control.svg](fig4_ensemble_control.svg) | [fig4_ensemble_control.pdf](fig4_ensemble_control.pdf) | [fig4_ensemble_control_600dpi.png](fig4_ensemble_control_600dpi.png) |
| A1 | [figA1_teacher_architecture.svg](figA1_teacher_architecture.svg) | [figA1_teacher_architecture.pdf](figA1_teacher_architecture.pdf) | [figA1_teacher_architecture_600dpi.png](figA1_teacher_architecture_600dpi.png) |

The canvas width is 6.8 inches; every reference aspect ratio is preserved. SVG text uses Liberation Sans with Arial/Helvetica fallbacks, and PDF fonts are embedded. PDF exports contain zero raster images.

See [FIGURE_VECTOR_QA.md](../../../research/manuscript_v2/FIGURE_VECTOR_QA.md) for provenance, scientific-label checks, exact sizes/hashes, the temporary full-manuscript compile, and inherited fine-print readability limitations. No redesign was authorized: A1 formulas and Figure 4 cautions are still small at paper width.

The production manuscript sources still use their existing placeholder calls. PDF inclusion was tested in a temporary manuscript copy only; this figure/QA-only commit does not change prose or permanently replace those calls. Figures 2 and 5 remain planned placeholders. Earlier planned F1/F3/F4 filenames are superseded by the actual filenames above.

Export: `python3 research/manuscript_v2/scripts/render_vector_figures.py`.
Read-only audit: `python3 research/manuscript_v2/scripts/audit_vector_figures.py`.

No experimental content was generated. Reconstruction is complete; STOP.
