# Venue-neutral V2 manuscript skeleton

Input commit64db3aebd158f56373385db18dcdf056aeea4c52; experiments frozen.
Ten sections plus eight appendix placeholders. Comments contain evidence and
claim/citation TODOs; tables contain archived planning numbers; figures are
labeled boxes. This is not a full manuscript or a submission-ready draft.

Start at research/manuscript_v2/GPT_WRITING_HANDOFF.md. No target venue has been
chosen. Do not overwrite the old CAC paper or frozen Phase-2 evidence.

Run `make` here. Build output is ../../outputs/manuscript_v2_build/, excluded
from commits. Requires local latexmk/pdflatex/BibTeX and ordinary article,
geometry, lmodern, natbib, booktabs and hyperref packages. No GPU/model access.

`references.bib` contains14 metadata-verified publisher exports; full-text
comparisons and foundational dataset/model citations remain TODOs. The
compile and evidence-protection QA record is research/manuscript_v2/skeleton_qa.json.
