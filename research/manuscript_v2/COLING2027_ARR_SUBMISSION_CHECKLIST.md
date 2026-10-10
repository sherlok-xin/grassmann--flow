# COLING 2027 / October 2026 ARR submission checklist

Verified on 2026-10-10. This records document preparation, not an account-level
submission or an acceptance prediction. Experiments remain permanently frozen.

Document QA was refreshed on 2026-10-11 in
`COLING2027_FINAL_LAYOUT_AUDIT.md`; it remains 7 main-content / 14 total pages.
The original conversion checklist below is historical; its human-action items
remain outstanding unless separately completed by the authors.

## A. COMPLETED BY CODEX

- [x] Started from `63d7eb72c01d26a22e921a73246ecc9944df9a3a`.
- [x] Created a separate, portable anonymous submission source under
  `论文投稿/coling2027_arr/`; the archived V2 is byte-preserved.
- [x] Vendored the unmodified official ACL style and bibliography style from
  `d5adc823ff0f80f98c80405ca0ab66c68e684409`.
- [x] Used 11pt Times, official review mode, A4, two columns, and no margin,
  font-size, line-spacing, or caption-spacing hacks.
- [x] Used the approved title and a 185-word abstract with the matched-CE
  comparator distinction. No modern stress-test result enters the abstract.
- [x] Main content and all main figures/tables end on page 7, within 8 pages.
  Total PDF: 14 pages.
- [x] Put unnumbered `Limitations` immediately after Conclusion on page 7,
  without a forced page break. References follows; appendix remains two-column.
- [x] Retained four main vector figures, one main table, and appendix Figure A1.
  All are actual PDF compilation inputs, not raster placeholders.
- [x] Preserved all 62 original numerical checks, scientific signs, contrasts,
  confounds, and permanent optimization warnings. No new endpoint was computed.
- [x] Compiled successfully, checked reference/label resolution, inspected all
  14 rendered pages, and tested a stand-alone source-copy build.
- [x] Checked PDF metadata, visible text, links, source dependencies, and known
  identifying strings. No personal repository URL or local path is in the PDF.
- [x] Ran official pubcheck and retained its raw outcome, with a separate
  review-marker classification. This is not a clean upstream PASS; see status.
- [x] Preserved 24 previously verified references without additions/removals.

## B. REQUIRES HUMAN ACTION

- [ ] Confirm the complete author list, consent, ordering, and OpenReview IDs.
  Complete every author profile, ORCID, accurate affiliation histories, career
  information, and verified email addresses. Add DBLP and ACL Anthology links
  where applicable. Review conflicts of interest. Do this immediately; the
  published advance-profile guidance is already close to/past its ideal window.
- [ ] Confirm eligibility under the October 2026 sustainable reviewing policy.
  Designate a qualified service contributor and complete registration within
  the stated 48-hour window. Check contributor submission limits and follow
  every registration request in the actual author console. The dates page has
  a generic all-author registration instruction; do not assume this removes
  any requirement in the cycle-specific CFP or the submission interface.
- [ ] Select `Long Paper`; select COLING 2027 in the intended/preferred venue
  field if offered. ARR submission is not COLING commitment.
- [ ] Select the area using the verified labels below and appropriate keywords.
- [ ] Supply a concise TL;DR. Suggested text for author approval:
  “Matched warm-start language-model continuations reveal reproducible positive
  and negative KD transfer; controlled teacher and ensemble comparisons limit
  explanations based on likelihood, heterogeneity, and local diagnostics.”
- [ ] Copy the exact title and abstract from `论文投稿/coling2027_arr/main.tex`,
  not from an older submission or a paraphrased chat response.
- [ ] Complete the Responsible NLP Research Checklist honestly: disclose AI
  writing/coding assistance, actual dataset/model licenses and access, selection
  protocol, resource reporting limits, and release plans. Do not mark an
  unperformed evaluation or unavailable aggregate compute accounting as done.
- [ ] Choose preprint status intentionally. The existing public project history
  can make the work searchable despite the anonymous PDF. Decide whether/how
  to provide anonymized code or supplementary material, and inspect it for
  identifying filenames, commit history, authors, and URLs before upload.
- [ ] Confirm prior ARR submission history. A rejected CAC paper alone is not
  a prior ARR submission. If there was an ARR review, follow its resubmission
  and change-summary requirements rather than treating this as a new paper.
- [ ] Upload the final `outputs/coling2027_arr/main.pdf` before the October 12,
  2026 deadline. Confirm the exact timezone and closing time in OpenReview.
  Do not upload a stale intermediate PDF or replace the review style to satisfy
  a camera-ready check.
- [ ] Download and visually check the actual uploaded PDF, submission fields,
  author IDs, checklist, supplementary files, and service registration receipt.
- [ ] If later committing to COLING 2027, complete that separate action by the
  published December 23, 2026 commitment date; confirm any subsequent updates.
- [ ] Research lead approves the empirical novelty, bounded modern section,
  teacher-control confounds, and figure readability before submission.

No OpenReview login, profile modification, service registration, upload, or
conference commitment was performed by Codex.

## Area recommendation

First choice: `Machine Learning for NLP`. The main contribution concerns
controlled transfer and optimization under matched KD/CE continuations, not a
new geometric architecture or a deployment-speed method.

Second choice: `Interpretability and Analysis of Models for NLP`. This fits the
teacher/ensemble falsification and retrospective diagnostic analysis. Select it
if the author framing emphasizes understanding model transfer rather than KD
training methodology. These are exact current labels, not guessed dropdown
names. `Language Modeling` is also current but is not the recommended top two.

## Official references

Rules were checked against [ARR CFP](https://aclrollingreview.org/cfp),
[dates and venues](https://aclrollingreview.org/dates),
[current areas](https://aclrollingreview.org/areas),
[author checklist](https://aclrollingreview.org/authorchecklist), and
[ACL formatting instructions](https://acl-org.github.io/ACLPUB/formatting.html).
Always follow the actual October-cycle console if requirements are updated.
