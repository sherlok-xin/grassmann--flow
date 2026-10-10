# Anonymous ACL Rolling Review source

This is a self-contained anonymous long-paper source prepared for the October
2026 ARR cycle, with COLING 2027 as the intended venue. The scientific evidence
is frozen. This directory contains no training code, datasets, checkpoints,
author information, or link to a non-anonymous project repository.

## Build

Run `make` from this directory. This requires pdfLaTeX, BibTeX, latexmk, and
standard TeX Live packages. The generated PDF is `build/main.pdf`.

When building from the surrounding checkout, use:

```sh
make -C 论文投稿/coling2027_arr BUILD_DIR=../../outputs/coling2027_arr
```

Only `main.tex`, `sections/`, `appendix/`, `tables/`, `references.bib`, the
official style files, and `figures/*.pdf` are needed to compile. The editable
SVG masters are included for figure maintenance; they are not compilation
inputs. No symlinks or external data files are required.

## Official template

Source: https://github.com/acl-org/acl-style-files

Template commit: `d5adc823ff0f80f98c80405ca0ab66c68e684409`.
The vendored `acl.sty` and `acl_natbib.bst` are unmodified. The official style
selects `acl_natbib` internally; the document does not select it a second time.
The review mode intentionally prints anonymous authors, gray line numbers,
and page numbers. Do not remove them to satisfy a camera-ready checker.

## Submission structure

The ten main sections, four main vector figures, and one main table end on
page 7 of the verified build. Unnumbered Limitations follows Conclusion without
a page break, then References. Four double-column appendix sections contain
architecture, protocols and amendments, full per-seed results, and diagnostic
and modern stress-test details. The appendix architecture is Figure A1.

Scientific wording, uncertainty, and permanent optimization warnings must not
be changed through formatting. Two copied SVG/PDF figures only increase
mathematical script sizes for readability; their text, layout, and scientific
content are otherwise unchanged. Figure 2 additionally moves its TinyStories
numeric annotation upward to clear the horizontal axis, without changing data
or wording. Figure 3 and Figure 4 are unchanged.

The PDF is a review document, not a camera-ready paper or proof of acceptance.
Account registration, the Responsible NLP Research Checklist, conflicts,
preprint status, and actual upload remain author responsibilities. Generated
PDFs, rendered page previews, and build products are not part of the source
commit.
