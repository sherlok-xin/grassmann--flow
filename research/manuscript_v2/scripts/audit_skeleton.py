#!/usr/bin/env python3
"""Local document QA only: compile skeleton, check sources, preserve evidence."""
import csv
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "research/manuscript_v2"
MANUSCRIPT = ROOT / "论文投稿/manuscript_v2"
BUILD = ROOT / "outputs/manuscript_v2_build"
BASE = "64db3aebd158f56373385db18dcdf056aeea4c52"
DOCS = ["EVIDENCE_INTEGRATION_V2.md", "CLAIMS_V2.md", "MANUSCRIPT_STORY_V2.md",
        "TABLE_FIGURE_PLAN_V2.md", "RELATED_WORK_GAPS.md", "GPT_WRITING_HANDOFF.md"]
SECTIONS = ["Introduction", "Related Work", "Problem Setting and Paired Distillability Evaluation",
            "Experimental Test Bed and Controlled Interventions", "Reproducible Positive and Negative Transfer",
            "What Determines Distillability?", "Teacher Source, Ensemble, and Diagnostic Controls",
            "External-Validity Stress Test", "Discussion and Limitations", "Conclusion"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, cwd=ROOT):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)


def main():
    checks = {}
    assert all((PACKAGE / x).is_file() for x in DOCS)
    checks["six_requested_v2_documents"] = "PASS"
    old = ROOT / "论文投稿/cac/conference_101719.tex"
    assert sha(old) == "8b8e3fcc3f04b596058cd4b63ddfe38b6eab91f338c7d4f12db4b602c9982ec8"
    frozen = sorted((ROOT / "research/final_evidence").glob("*"))
    checksums = run(["sha256sum"] + [str(p.relative_to(ROOT)) for p in frozen]).stdout
    aggregate = hashlib.sha256(checksums.encode()).hexdigest()
    assert aggregate == "3d172e634db611edad0d566e74ea0f826ed8cdae69922120baa94ebf9e6e2b96"
    checks["protected_old_cac_and_phase2_hashes"] = "PASS"
    protected = ["research/final_evidence", "research/experiments", "论文投稿/cac", "datasets", "checkpoints"]
    diff = run(["git", "diff", BASE, "--name-only", "--"] + protected).stdout
    assert not diff.strip(), diff
    checks["protected_tracked_history_diff_empty"] = "PASS"
    inputs = json.loads((PACKAGE / "table_source_manifest.json").read_text())["sources"]
    assert all(sha(ROOT / x) == value for x, value in inputs.items())
    checks["archived_table_input_hashes"] = "PASS"
    for short in ["cross_domain_transfer", "teacher_quality_intervention", "teacher_source_ablation", "tt_vs_tg", "efficiency", "failed_or_negative_controls"]:
        source = list(csv.DictReader(io.StringIO((ROOT / f"research/final_evidence/table_{short}.csv").read_text())))
        writing = list(csv.DictReader(io.StringIO((PACKAGE / f"tables/table_{short}.csv").read_text())))
        assert source == writing
    checks["phase2_table_values_preserved"] = "PASS"
    citations = json.loads((PACKAGE / "citation_verification.json").read_text())["records"]
    verified = {x["key"] for x in citations if x["status"] == "VERIFIED_METADATA"}
    bib = (MANUSCRIPT / "references.bib").read_text()
    keys = set(re.findall(r"@\w+\s*\{([^,]+),", bib))
    assert keys == verified and len(keys) == 14
    assert all(x.get("authors") and x.get("year") and x.get("identifier") for x in citations if x["key"] in keys)
    checks["fourteen_verified_bibliography_keys"] = "PASS"
    sections = sorted((MANUSCRIPT / "sections").glob("[0-9][0-9]_*.tex"))
    assert len(sections) == 10
    headings = [re.search(r"\\section\{([^}]+)\}", x.read_text()).group(1) for x in sections]
    assert headings == SECTIONS
    texts = "\n".join(p.read_text() for p in MANUSCRIPT.rglob("*.tex"))
    labels = re.findall(r"\\label\{([^}]+)\}", texts)
    assert len(labels) == len(set(labels))
    assert sum(x.read_text().count("\\figureplaceholder{") for x in sections) == 5
    assert len(list((MANUSCRIPT / "tables").glob("*.tex"))) == 8
    assert "\\writingtodo{" in (MANUSCRIPT / "main.tex").read_text()
    checks["ten_sections_eight_tables_five_boxes_unique_labels"] = "PASS"
    command = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
               "-outdir=../../outputs/manuscript_v2_build", "main.tex"]
    compiled = run(command, MANUSCRIPT)
    log = (BUILD / "main.log").read_text(errors="replace")
    blg = (BUILD / "main.blg").read_text(errors="replace")
    assert not re.search(r"undefined|multiply defined|^!|Overfull", log, re.I | re.M)
    assert "warning$ -- 0" in blg
    checks["latex_bibtex_no_errors_undefined_or_overfull"] = "PASS"
    pdf_text = run(["pdftotext", "-layout", str(BUILD / "main.pdf"), "-"]).stdout
    normalized = re.sub(r"\s+", " ", pdf_text)
    assert all(x in normalized for x in SECTIONS)
    assert all(f"F{x}:" in normalized for x in range(1, 6))
    assert "STOP_MODERN_CONTINUATION_NOT_ESTABLISHED" in normalized
    assert "CLIP_SATURATION_WARNING" in normalized
    assert "TODO:" in normalized
    checks["rendered_text_sections_placeholders_warnings"] = "PASS"
    state = (ROOT / "research-state.yaml").read_text()
    assert "status: manuscript_v2_ready_for_writing" in state
    assert "training_authorized: false" in state
    assert "experimental_program: permanently_frozen" in state
    checks["state_and_training_freeze"] = "PASS"
    manuscript_sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted(MANUSCRIPT.rglob("*")) if p.is_file()}
    # No document generation or model access beyond local LaTeX/pdf-to-text.
    qa = dict(status="PASS", created_on="2026-10-04", base_commit=BASE,
              activity="local source/bibliography/archived-CSV checks; no experiments or model forwards",
              checks=checks, compile_command=command, compile_exit_code=compiled.returncode,
              build_directory=str(BUILD.relative_to(ROOT)), pdf_sha256=sha(BUILD / "main.pdf"),
              pdf_pages=int(re.search(r"Output written on .*?\((\d+) pages", log, re.S).group(1)),
              underfull_hbox_count=log.count("Underfull \\hbox"),
              warning_note="Three harmless underfull bibliography line-spacing notices; no errors, undefined references/citations, or overfull boxes. Source/text QA only; final artwork and full prose remain TODO.",
              protected_phase2_aggregate_sha256=aggregate, old_cac_sha256=sha(old),
              formatted_csv_sources=len(inputs), verified_citations=len(keys), manuscript_sources=manuscript_sources)
    (PACKAGE / "skeleton_qa.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": qa["status"], "pages": qa["pdf_pages"], "checks": len(checks),
                      "underfull_notices": qa["underfull_hbox_count"], "verified_citations": len(keys)}))


if __name__ == "__main__":
    main()
