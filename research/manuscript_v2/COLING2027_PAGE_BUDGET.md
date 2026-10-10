# Verified COLING / ARR page budget

Final build checked on 2026-10-11: main-content last page **7**, total PDF pages
**14**. This conservatively counts the page containing the last Conclusion
lines and includes every main figure and table. Limitations, References, and
Appendix are not counted against the eight-page main-content allowance.

Approximate allocations below describe occupied page-equivalents, not separate
page breaks. Two-column text, wide floats, captions, and title matter share
pages; estimates are not an alternative to checking the actual PDF.

| Content | Approximate page-equivalents | Actual location |
| --- | ---: | --- |
| Abstract + Introduction | 1.15 | 1–2 |
| Related Work | 0.85 | 2–3 |
| Method / protocol / controlled setting | 0.85 | 3 |
| Results, controls, and bounded modern stress test | 0.95 | 4–6 |
| Discussion | 0.45 | 6 |
| Conclusion | 0.12 | 6–7 |
| Four main figures, one table, and captions | 1.60 | 2, 4, 5, 6 |
| Official title/author block and structural whitespace | 0.25 | 1, shared |

The body occupies roughly 6.2 page-equivalents and ends early on page 7. The
remaining space on that page is used by Limitations and References in the
official sequential flow, not by a page-limit workaround.

## Concrete float and section locations

| Item | PDF page |
| --- | ---: |
| Figure 1: paired protocol | 2 |
| Table 1: cross-domain transfer | 4 |
| Figure 2: per-seed cross-domain gain | 5 |
| Figure 3: teacher-state/likelihood controls | 5 |
| Figure 4: source and ensemble controls | 6 |
| Main-content endpoint | 7 |
| Limitations starts, immediately after Conclusion | 7 |
| References | 7–8 |
| Appendix A: architecture | 9–10, Figure A1 on 10 |
| Appendix B: protocols and amendments | 9–10, Tables 2–3 on 10–11 |
| Appendix C: full per-seed results | 10–11, Table 4 on 11; Tables 5–8 on 12 |
| Appendix D: diagnostic/modern details | 11–13, Table 9 on 13; Tables 10–12 on 14 |

The appendix float locations intentionally differ from their source insertion
points. All eleven appendix tables are readable at template-default size.
Figure/table placement uses ordinary LaTeX float placement, not forced
negative spacing or modified style dimensions. Any later author edit requires
rebuilding and rechecking this budget; the current counts are not transferable
to an edited source.

The final typography pass is recorded in `COLING2027_FINAL_LAYOUT_AUDIT.md`;
it preserves seven main-content pages and does not attempt to fill page 8.
