#!/usr/bin/env python3
"""Read-only fidelity checks; --snapshot records the initial protected file hashes.

This script does not run models, evaluate endpoints, or rebuild the English PDF.
Translation completeness and wording still require a human paragraph review.
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
    for rel in tracked:
        check((ROOT / rel).read_bytes() == git("show", f"{COMMIT}:{rel}"), f"English differs from authoritative commit: {rel}")
    check(digest(EN_PDF) == EN_PDF_SHA, "Canonical English PDF hash changed")
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    for rel, sha in snapshot["sha256"].items():
        check((ROOT / rel).is_file() and digest(ROOT / rel) == sha, f"Protected English file changed: {rel}")
    check(not git("diff", "--name-only", COMMIT, "--", "research/final_evidence", "research/experiments", "experiments", "checkpoints", "datasets").strip(), "Protected evidence/training paths differ")
    check(not any(p.is_symlink() for p in ZH.rglob("*")), "Chinese archive contains a symlink")
    for name in ("references.bib", "acl_natbib.bst"):
        check((ZH / name).read_bytes() == (EN / name).read_bytes(), f"Changed copy: {name}")
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
            check(Counter(numbers(en)) == Counter(numbers(zh)), f"Numeric tokens differ: {rel}")
            if dirname == "tables":
                check(numbers(en) == numbers(zh), f"Table numeric/seed order differs: {rel}")
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
            rows.append({"path": str(rel), "numeric_tokens": len(numbers(en)), "math_spans": len(mathematics(en)), "coverage": counts})

    en_main = (EN / "main.tex").read_text(encoding="utf-8")
    zh_main = (ZH / "main.tex").read_text(encoding="utf-8")
    abstract = lambda t: re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", t, re.S).group(1)
    check(numbers(abstract(en_main)) == numbers(abstract(zh_main)), "Abstract numeric tokens differ")
    check(mathematics(abstract(en_main)) == mathematics(abstract(zh_main)), "Abstract mathematics differs")
    input_pattern = r"\\input\{[^}]+\}"
    check(re.findall(input_pattern, en_main) == re.findall(input_pattern, zh_main), "Main input order differs")
    check(len(rows) == 27, "Expected 11 sections, four appendices, twelve tables")
    bibliography = (ZH / "references.bib").read_text(encoding="utf-8")
    ref_count = len(re.findall(r"^@\w+\s*\{", bibliography, re.M))
    check(ref_count == 24, "Expected 24 bibliography records")

    log = ROOT / "outputs/coling2027_arr_zh/main.log"
    if log.is_file():
        log_text = log.read_text(encoding="utf-8", errors="replace")
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
    note = "说明：本中文版仅用于课题组内部留档与报备。正式投稿版本为英文版 COLING 2027 / ACL Rolling Review 稿件。如中英文表述存在差异，以最终英文投稿稿为准。"
    check(re.sub(r"\s+", "", note) in re.sub(r"\s+", "", document[-1].get_text()), "Final internal-use note absent from final page")
    check(all(not page.get_images(full=True) for page in document), "Unexpected raster image in Chinese PDF")
    check(all(page.get_text().strip() for page in document), "Empty page in Chinese PDF")
    aux = (ROOT / "outputs/coling2027_arr_zh/main.aux").read_text(encoding="utf-8")
    check(len(re.findall(r"\\bibcite\{", aux)) == 24, "Rendered bibliography does not have 24 entries")
    check(len(re.findall(r"\\newlabel\{fig:", aux)) == 5, "Rendered figure labels not five")
    check(len(re.findall(r"\\newlabel\{tab:", aux)) == 12, "Rendered table labels not twelve")

    result = {"source_commit": COMMIT, "english_tracked_files_unchanged": len(tracked),
              "english_snapshot_files_unchanged": len(snapshot["sha256"]),
              "english_pdf_sha256": digest(EN_PDF), "figures_exact_copies": len(figures),
              "bibliography_records": ref_count, "translation_files": rows,
              "numeric_tokens_checked": sum(row["numeric_tokens"] for row in rows),
              "math_spans_checked": sum(row["math_spans"] for row in rows),
              "chinese_pdf_sha256": digest(zh_pdf), "chinese_pdf_pages": len(document),
              "required_signed_mean_sd_pairs": required_pairs,
              "ce_gate_improvement": "+0.004849", "ce_gate_threshold": "0.01",
              "raster_images": sum(len(page.get_images(full=True)) for page in document),
              "failures": failures, "result": "PASS" if not failures else "FAIL"}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
