#!/usr/bin/env python3
"""Read-only fidelity checks; --snapshot records the initial protected file hashes.

This script does not run models, evaluate endpoints, or rebuild the English PDF.
Translation completeness and wording still require a human paragraph review.
Layout changes are checked against the frozen Chinese translation and the
official English float architecture; this script does not certify visual QA.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ZH = Path(__file__).resolve().parents[1]
ROOT = ZH.parents[1]
EN = ROOT / "论文投稿/coling2027_arr"
COMMIT = "5955954eed279686be9f4b1ba6634967d7676325"
ZH_BASELINE_COMMIT = "d07a95e37a0c7f37ddb60ed7efa2dc2cbc0d5e08"
EN_PDF = ROOT / "outputs/coling2027_arr/main.pdf"
EN_PDF_SHA = "700490be24483fef4bfd69a367ade958981bbe7e05e4804c3c14efa40cde7255"
SNAPSHOT = ZH / "scripts/protected_source_hashes.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def uncomment(text):
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def compact(text):
    return re.sub(r"\s+", "", text)


def numbers(text):
    text = uncomment(text)
    result = []
    for match in re.finditer(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text):
        value = match.group(0)
        # In seed-42 / step-0 / GPT-2 the hyphen is lexical, not a minus sign.
        # Scientific notation such as 5e-06 is already consumed as one token.
        if value[0] in "+-" and match.start() and re.match(r"[A-Za-z_]", text[match.start() - 1]):
            value = value[1:]
        result.append(value)
    return result


def mathematics(text):
    text = uncomment(text)
    pattern = r"\$[^$]*\$|\\\[.*?\\\]|\\begin\{(equation\*?|align\*?|gather\*?)\}.*?\\end\{\1\}"
    return [compact(m.group(0)) for m in re.finditer(pattern, text, re.S)]


def structural_commands(text):
    pattern = r"\\(?:label|ref|eqref|citep|citet|cite|input|includegraphics)(?:\[[^\]]*\])*\{[^}]*\}"
    return [compact(m.group(0)) for m in re.finditer(pattern, uncomment(text))]


def layout_neutral_content(text):
    """Preserve all prose/notation while ignoring only permitted float/header changes.

    Short stacked table headers may wrap the original text, but must not change
    it. Other content or formatting changes remain visible to this comparison.
    """
    text = uncomment(text)
    text = re.sub(r"\\(begin|end)\{(figure|table)\*?\}", r"\\\1{\2}", text)
    pattern = re.compile(r"\\(?:shortstack|makecell)\*?(?:\[[^\]]*\])?\{")
    while match := pattern.search(text):
        opening, depth, closing = match.end() - 1, 1, match.end()
        while closing < len(text) and depth:
            # Braces escaped with an odd number of backslashes are literal.
            slash = closing - 1
            while slash >= 0 and text[slash] == "\\":
                slash -= 1
            escaped = (closing - slash - 1) % 2
            if not escaped and text[closing] == "{":
                depth += 1
            elif not escaped and text[closing] == "}":
                depth -= 1
            closing += 1
        if depth:
            raise ValueError("Unclosed stacked table-header argument")
        content = text[opening + 1:closing - 1]
        content = re.sub(r"\\\\(?:\[[^\]]*\])?", "", content)
        text = text[:match.start()] + content + text[closing:]
    return compact(text)


def scientific_content(text, relative_path):
    """Exclude only the authorized, float-local padding of the main transfer table."""
    if str(relative_path) == "tables/cross_domain_transfer.tex":
        return re.sub(r"\\setlength\s*\{\s*\\tabcolsep\s*\}\s*\{\s*5pt\s*\}", "", text)
    return text


def float_structure(text):
    pattern = r"\\(?:begin|end)\{(?:figure|table)\*?\}(?:\[[^\]]*\])?"
    return [compact(m.group(0)) for m in re.finditer(pattern, uncomment(text))]


def tabular_columns(text):
    pattern = r"\\begin\{tabular\}(?:\[[^\]]*\])?\{[^}]*\}"
    return [compact(m.group(0)) for m in re.finditer(pattern, uncomment(text))]


def braced_command(text, name):
    """Read the first simple braced command, allowing nested author/math braces."""
    match = re.search(r"\\" + re.escape(name) + r"\{", uncomment(text))
    if not match:
        return None
    clean = uncomment(text)
    start, depth, index = match.end(), 1, match.end()
    while index < len(clean) and depth:
        if clean[index] == "{" and (index == 0 or clean[index - 1] != "\\"):
            depth += 1
        elif clean[index] == "}" and (index == 0 or clean[index - 1] != "\\"):
            depth -= 1
        index += 1
    return clean[start:index - 1] if not depth else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()
    if args.snapshot:
        if SNAPSHOT.exists():
            raise SystemExit("Refusing to replace the original protection snapshot.")
        hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(EN.rglob("*")) if p.is_file()}
        hashes[str(EN_PDF.relative_to(ROOT))] = digest(EN_PDF)
        SNAPSHOT.write_text(json.dumps({"source_commit": COMMIT, "sha256": hashes}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Protection snapshot: {len(hashes)} existing English files; English PDF {digest(EN_PDF)}")
        return

    failures = []

    def check(ok, label):
        if not ok:
            failures.append(label)

    tracked = [p for p in git("ls-tree", "-r", "--name-only", "-z", COMMIT, "--", str(EN.relative_to(ROOT))).decode().split("\0") if p]
    check(len(tracked) == 43, "Expected original 43 protected English tracked files")
    for rel in tracked:
        check((ROOT / rel).read_bytes() == git("show", f"{COMMIT}:{rel}"), f"English differs from authoritative commit: {rel}")
    check(digest(EN_PDF) == EN_PDF_SHA, "Canonical English PDF hash changed")
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    check(len(snapshot["sha256"]) == 51, "Expected original 51 protected English snapshot hashes")
    for rel, sha in snapshot["sha256"].items():
        check((ROOT / rel).is_file() and digest(ROOT / rel) == sha, f"Protected English file changed: {rel}")
    check(not git("diff", "--name-only", COMMIT, "--", "research/final_evidence", "research/experiments", "experiments", "checkpoints", "datasets").strip(), "Protected evidence/training paths differ")
    check(not any(p.is_symlink() for p in ZH.rglob("*")), "Chinese archive contains a symlink")
    for name in ("references.bib", "acl_natbib.bst", "acl.sty"):
        check((ZH / name).is_file() and (ZH / name).read_bytes() == (EN / name).read_bytes(), f"Changed or missing copy: {name}")
    figures = sorted((EN / "figures").glob("*.pdf"))
    check(len(figures) == 5, "Figure count differs from five")
    for p in figures:
        check((ZH / "figures" / p.name).read_bytes() == p.read_bytes(), f"Changed figure: {p.name}")

    rows = []
    for dirname in ("sections", "appendix", "tables"):
        for source in sorted((EN / dirname).glob("*.tex")):
            rel = source.relative_to(EN)
            target = ZH / rel
            if not target.is_file():
                check(False, f"Missing translation: {rel}")
                continue
            en, zh = source.read_text(encoding="utf-8"), target.read_text(encoding="utf-8")
            baseline = git("show", f"{ZH_BASELINE_COMMIT}:{target.relative_to(ROOT)}").decode("utf-8")
            zh_science = scientific_content(zh, rel)
            if str(rel) == "tables/cross_domain_transfer.tex":
                paddings = list(re.finditer(r"\\setlength\s*\{\s*\\tabcolsep\s*\}\s*\{\s*5pt\s*\}", zh))
                check(len(paddings) <= 1 and all(zh.find(r"\begin{table}") < m.start() < zh.find(r"\end{table}") for m in paddings), "Main-table padding must occur once inside its local table group")
            check(layout_neutral_content(baseline) == layout_neutral_content(zh_science), f"Frozen Chinese prose/content changed: {rel}")
            check(Counter(numbers(baseline)) == Counter(numbers(zh_science)), f"Chinese baseline numeric tokens differ: {rel}")
            check(mathematics(baseline) == mathematics(zh_science), f"Chinese baseline mathematics/order differs: {rel}")
            check(float_structure(en) == float_structure(zh), f"English float architecture/options differ: {rel}")
            check(tabular_columns(en) == tabular_columns(zh), f"English table columns/widths differ: {rel}")
            check(Counter(numbers(en)) == Counter(numbers(zh_science)), f"Numeric tokens differ: {rel}")
            if dirname == "tables":
                check(numbers(en) == numbers(zh_science), f"Table numeric/seed order differs: {rel}")
            check(mathematics(en) == mathematics(zh), f"Mathematical content/order differs: {rel}")
            check(structural_commands(en) == structural_commands(zh), f"Citations/refs/labels/inputs differ: {rel}")
            identifiers = lambda t: re.findall(r"\\(?:texttt|url)\{(?:\\.|[^}])*\}", uncomment(t))
            check(Counter(identifiers(en)) == Counter(identifiers(zh)), f"Technical identifiers/URLs differ: {rel}")
            counts = {}
            for cmd in ("section", "subsection", "subsubsection", "caption", "footnote"):
                pat = rf"\\{cmd}(?:\*)?(?:\[[^\]]*\])?\{{"
                a, b = len(re.findall(pat, uncomment(en))), len(re.findall(pat, uncomment(zh)))
                check(a == b, f"{cmd} coverage differs: {rel}")
                counts[cmd] = b
            check(bool(re.search(r"[\u4e00-\u9fff]", zh)), f"No Chinese translation: {rel}")
            rows.append({"path": str(rel), "numeric_tokens": len(numbers(en)), "math_spans": len(mathematics(en)), "coverage": counts,
                         "chinese_baseline_content_unchanged": layout_neutral_content(baseline) == layout_neutral_content(zh_science),
                         "float_structure": float_structure(zh), "tabular_columns": tabular_columns(zh)})

    en_main = (EN / "main.tex").read_text(encoding="utf-8")
    zh_main = (ZH / "main.tex").read_text(encoding="utf-8")
    abstract = lambda t: re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", t, re.S).group(1)
    baseline_main = git("show", f"{ZH_BASELINE_COMMIT}:{ZH.relative_to(ROOT)}/main.tex").decode("utf-8")
    check(abstract(baseline_main) == abstract(zh_main), "Frozen Chinese abstract text changed")
    check(numbers(abstract(en_main)) == numbers(abstract(zh_main)), "Abstract numeric tokens differ")
    check(mathematics(abstract(en_main)) == mathematics(abstract(zh_main)), "Abstract mathematics differs")
    input_pattern = r"\\input\{[^}]+\}"
    check(re.findall(input_pattern, en_main) == re.findall(input_pattern, zh_main), "Main input order differs")
    tail = lambda t: compact(uncomment(t.split(r"\end{abstract}", 1)[1]))
    check(tail(en_main) == tail(zh_main), "English bibliography/clearpage/appendix/input flow differs")
    clean_main = uncomment(zh_main)
    check(bool(re.search(r"\\documentclass\[11pt\]\{article\}", clean_main)), "Chinese source is not the English 11pt article class")
    check(bool(re.search(r"\\usepackage\{acl\}", clean_main)), "Chinese source does not use non-review ACL")
    check(not re.search(r"\\usepackage\[[^\]]*review[^\]]*\]\{acl\}", clean_main), "Review mode must not be enabled")
    check(bool(re.search(r"\\usepackage(?:\[[^\]]*\])?\{fontspec\}", clean_main)) and
          bool(re.search(r"\\usepackage(?:\[[^\]]*\])?\{xeCJK\}", clean_main)), "Minimal XeLaTeX CJK support absent")
    forbidden_layout = r"ctexart|\\usepackage(?:\[[^\]]*\])?\{(?:geometry|caption|placeins)\}|\\(?:onecolumn|FloatBarrier|ctexset|captionsetup)\b|\\(?:setlength|addtolength)\b|\\(?:fontsize|vspace|hspace)\*?\{"
    check(not re.search(forbidden_layout, clean_main), "Report-style or custom layout override in Chinese main")
    # Retain only the original Chinese baseline safeguards against breaking a
    # mean +/- SD expression at a math operator. They change neither geometry
    # nor mathematical content, and arbitrary penalty changes are not allowed.
    penalties = lambda t: [compact(m.group(0)) for m in re.finditer(r"\\(?:binoppenalty|relpenalty)\b[^\n]*", uncomment(t))]
    expected_penalties = [r"\binoppenalty=10000", r"\relpenalty=10000"]
    check(penalties(baseline_main) == expected_penalties and penalties(zh_main) == expected_penalties,
          "Inline-math break penalties differ from the frozen Chinese baseline")
    check(compact(braced_command(baseline_main, "author") or "") == compact(braced_command(zh_main, "author") or ""), "Author 1 / 1,2 placeholder block changed")
    check("Anonymous" not in clean_main, "Anonymous author/header in non-review Chinese source")
    makefile = (ZH / "Makefile").read_text(encoding="utf-8")
    check("xelatex" in makefile and not re.search(r"(?m)^\s*(?:pdf|lua)latex\b", makefile), "Chinese build does not use XeLaTeX")
    check(len(rows) == 27, "Expected 11 sections, four appendices, twelve tables")
    bibliography = (ZH / "references.bib").read_text(encoding="utf-8")
    ref_count = len(re.findall(r"^@\w+\s*\{", bibliography, re.M))
    check(ref_count == 24, "Expected 24 bibliography records")

    log = ROOT / "outputs/coling2027_arr_zh/main.log"
    if log.is_file():
        log_text = log.read_text(encoding="utf-8", errors="replace")
        check("This is XeTeX" in log_text, "Build log is not from XeLaTeX")
        for pattern in (r"Missing character", r"LaTeX Error", r"undefined references", r"undefined citations", r"Citation .* undefined", r"Reference .* undefined", r"Overfull \\[hv]box", r"Font Warning"):
            check(not re.search(pattern, log_text), f"Build warning/error matching {pattern}")
    else:
        check(False, "Chinese build log missing")

    import fitz
    zh_pdf = ROOT / "outputs/coling2027_arr_zh/main.pdf"
    document = fitz.open(zh_pdf)
    pdf_text = "\n".join(page.get_text() for page in document)
    normalized = re.sub(r"\s+", "", pdf_text).replace("−", "-")
    required_pairs = {
        "PTB": ("+0.144937", "0.003938"),
        "WikiText-2": ("+0.155665", "0.007692"),
        "TinyStories": ("-0.107219", "0.000415"),
        "J": ("+0.155665", "0.007692"),
        "A": ("-0.027654", "0.006822"),
        "Q": ("+0.183319", "0.001375"),
        "G": ("+0.040991", "0.006733"),
        "T": ("+0.134868", "0.008906"),
        "TT": ("+0.168125", "0.008948"),
        "H": ("-0.012460", "0.001348"),
        "FineWeb_lambda1": ("-0.013550", "0.003542"),
        "FineWeb_lambda5": ("-0.043924", "0.004017"),
    }
    for name, (mean, sd) in required_pairs.items():
        # English prose sometimes prints a positive mean without an explicit +.
        # Accept that original display convention, but never accept a minus.
        if mean.startswith("+"):
            found = re.search(r"(?<![\d.+-])\+?" + re.escape(mean[1:] + "±" + sd), normalized)
        else:
            found = mean + "±" + sd in normalized
        check(bool(found), f"Required signed mean/SD absent from rendered PDF: {name}")
    check("+0.004849" in normalized and "0.01" in normalized, "CE gate improvement/threshold absent")
    for warning in ("CLIP_SATURATION_WARNING", "STOP_MODERN_CONTINUATION_NOT_ESTABLISHED", "CE456_GRADIENT_SPIKE_WARNING"):
        check(warning in normalized, f"Warning/decision identifier missing from PDF: {warning}")
    forbidden_note = r"归档说明|内部留档说明|报备说明|中文版说明|以英文版为准|本中文版仅用于|正式投稿版本为英文版|如中英文表述存在差异|以最终英文投稿稿为准|translation\s*note|internal[- ]use\s*disclaimer"
    check(not re.search(forbidden_note, normalized, re.I), "Archival/internal-use note remains in Chinese PDF")
    check(not re.search(forbidden_note, clean_main, re.I), "Archival/internal-use note remains in Chinese source")
    check("Anonymous ACL submission" not in pdf_text, "Anonymous review header remains in Chinese PDF")
    english_document = fitz.open(EN_PDF)
    canvas = english_document[0].rect
    check(all(abs(page.rect.width - canvas.width) < 0.02 and abs(page.rect.height - canvas.height) < 0.02 for page in document), "Chinese page geometry differs from English ACL PDF")
    # At the normal 11pt body size, Chinese prose must stay in a single column.
    # Captions and the full-width title/author block use different font sizes.
    # Pages consisting solely of English references or floats have no CJK body.
    body_column_pages, wide_body_lines = [], []
    # Unchanged acl.sty specifies symmetric columns with a 0.6cm separation.
    # xeCJK punctuation can protrude into the gutter, so line-bbox width is
    # not an overflow test. A true cross-column body line occupies both column
    # interiors. Actual overfull boxes remain separately forbidden above.
    column_gap = 6 / 25.4 * 72
    left_column_right = (canvas.width - column_gap) / 2
    right_column_left = (canvas.width + column_gap) / 2
    for page_index, page in enumerate(document):
        columns = set()
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(span["text"] for span in line["spans"])
                sizes = [span["size"] for span in line["spans"] if re.search(r"[\u4e00-\u9fff]", span["text"])]
                if len(re.findall(r"[\u4e00-\u9fff]", text)) < 12 or not sizes or not all(10.5 <= size <= 11.5 for size in sizes):
                    continue
                x0, _, x1, _ = line["bbox"]
                if x0 < left_column_right and x1 > right_column_left:
                    wide_body_lines.append({"page": page_index + 1, "text": text, "bbox": list(line["bbox"])})
                else:
                    columns.add("left" if (x0 + x1) / 2 < canvas.width / 2 else "right")
        if columns:
            body_column_pages.append({"page": page_index + 1, "columns": sorted(columns)})
    check(not wide_body_lines, "Normal-size Chinese body prose spans both columns")
    check(any(row["columns"] == ["left", "right"] for row in body_column_pages), "Rendered Chinese PDF has no two-column body page")
    check(all(not page.get_images(full=True) for page in document), "Unexpected raster image in Chinese PDF")
    check(all(page.get_text().strip() for page in document), "Empty page in Chinese PDF")
    aux = (ROOT / "outputs/coling2027_arr_zh/main.aux").read_text(encoding="utf-8")
    check(len(re.findall(r"\\bibcite\{", aux)) == 24, "Rendered bibliography does not have 24 entries")
    check(len(re.findall(r"\\newlabel\{fig:", aux)) == 5, "Rendered figure labels not five")
    check(len(re.findall(r"\\newlabel\{tab:", aux)) == 12, "Rendered table labels not twelve")
    check(bool(re.search(r"\\newlabel\{fig:teacher_architecture\}\{\{A1\}", aux)), "Appendix architecture figure is not numbered A1")

    result = {"source_commit": COMMIT, "chinese_translation_baseline_commit": ZH_BASELINE_COMMIT,
              "english_tracked_files_unchanged": len(tracked),
              "english_snapshot_files_unchanged": len(snapshot["sha256"]),
              "english_pdf_sha256": digest(EN_PDF), "figures_exact_copies": len(figures),
              "bibliography_records": ref_count, "translation_files": rows,
              "numeric_tokens_checked": sum(row["numeric_tokens"] for row in rows),
              "math_spans_checked": sum(row["math_spans"] for row in rows),
              "chinese_pdf_sha256": digest(zh_pdf), "chinese_pdf_pages": len(document),
              "required_signed_mean_sd_pairs": required_pairs,
              "ce_gate_improvement": "+0.004849", "ce_gate_threshold": "0.01",
              "layout": "ACL article, non-review, XeLaTeX, two-column",
              "inline_math_break_penalties": penalties(zh_main),
              "english_page_canvas_pt": [canvas.width, canvas.height],
              "chinese_body_column_pages": body_column_pages, "wide_body_lines": wide_body_lines,
              "archival_note_removed": not bool(re.search(forbidden_note, normalized, re.I)),
              "underfull_hbox_count": len(re.findall(r"Underfull \\hbox", log_text)) if log.is_file() else None,
              "underfull_vbox_count": len(re.findall(r"Underfull \\vbox", log_text)) if log.is_file() else None,
              "raster_images": sum(len(page.get_images(full=True)) for page in document),
              "failures": failures, "result": "PASS" if not failures else "FAIL"}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
