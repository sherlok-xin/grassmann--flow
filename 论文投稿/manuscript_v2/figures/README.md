# Frozen vector figure masters

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
