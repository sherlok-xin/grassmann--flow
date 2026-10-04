#!/usr/bin/env python3
"""Read-only document audit: no model access, training, or file writes."""
import csv
import hashlib
import json
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MS = ROOT / "论文投稿/manuscript_v2"
PACKAGE = ROOT / "research/manuscript_v2"
BUILD = ROOT / "outputs/manuscript_v2_build"
BASE = "1c153e49efa19132481fb74bbc98032bf610544e"
BODY = ["03_problem_setting.tex", "04_experimental_setting.tex", "05_cross_domain_transfer.tex",
        "06_distillability.tex", "07_controls.tex", "08_external_validity.tex"]
PROTECTED = {
    "main.tex": "5d2dd6c64e804c5dd360e9a9837e1d3b9e38524ca61c53711d699fecc72559e0",
    "sections/02_related_work.tex": "27a71d8d0b616eb8bb864f4943abc2aa8b8aad9251e5e976ad3c3fee1c885471",
    "sections/09_discussion.tex": "ce7724aa176ac4fda4999dfa8a9622907ca930f5bb193a644ff9c7f0a956bcb2",
    "sections/10_conclusion.tex": "70237fa280923c40a6ea8f7bbb2c793fcb66cb5b46c9802e5c9741d0629d7a02",
    "sections/appendix.tex": "40b6b7619530655ec87fff41abd9b85bde4bf80a0173fb2bb820d8f59e8dc3d9",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    return subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout


def rows(name):
    with (PACKAGE / "tables" / f"table_{name}.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    checks = {}
    assert all(sha(MS / path) == expected for path, expected in PROTECTED.items())
    checks["main_and_undrafted_sections_unchanged"] = "PASS"
    intro = (MS / "sections/01_introduction.tex").read_text()
    before = intro.replace("Knowledge distillation (KD) transfers",
                           "\\section{Introduction}\n\\label{sec:introduction}\n\nKnowledge distillation (KD) transfers", 1)
    assert hashlib.sha256(before.rstrip("\n").encode()).hexdigest() == "396c4813ed62ba899ef77770dc09f3a70a96672b679ccb2d3100a2acd80896e2"
    checks["introduction_only_duplicate_structure_removed_final_newline_normalized"] = "PASS"
    protected = ["research/final_evidence", "research/experiments", "论文投稿/cac", "datasets", "checkpoints"]
    assert not run(["git", "diff", BASE, "--name-only", "--"] + protected).strip()
    checks["frozen_experiments_data_checkpoints_old_manuscript_unchanged"] = "PASS"
    paths = sorted((ROOT / "research/final_evidence").glob("*"))
    aggregate = hashlib.sha256(run(["sha256sum"] + [str(p.relative_to(ROOT)) for p in paths]).encode()).hexdigest()
    assert aggregate == "3d172e634db611edad0d566e74ea0f826ed8cdae69922120baa94ebf9e6e2b96"
    assert sha(ROOT / "论文投稿/cac/conference_101719.tex") == "8b8e3fcc3f04b596058cd4b63ddfe38b6eab91f338c7d4f12db4b602c9982ec8"
    checks["frozen_evidence_and_cac_hashes"] = "PASS"
    sources = json.loads((PACKAGE / "table_source_manifest.json").read_text())["sources"]
    assert all(sha(ROOT / path) == expected for path, expected in sources.items())
    checks["archived_numeric_source_hashes"] = "PASS"
    texts = {name: (MS / "sections" / name).read_text() for name in BODY}
    joined = "\n".join(texts.values())
    assert not re.search(r"\\(?:writingtodo|textbf|emph|item)\b|\\begin\{(?:itemize|enumerate)\}", joined)
    assert not re.search(r"\b(?:it's|doesn't|don't|can't|we're|isn't|aren't|won't)\b", joined, re.I)
    assert all(line.startswith("% TODO:") for line in joined.splitlines() if "TODO" in line)
    checks["six_sections_paragraph_style_and_scoped_todos"] = "PASS"
    numerical_checks = []

    def contains(file, value):
        token = f"{float(value):.6f}"
        assert token in texts[file], (file, token)
        numerical_checks.append([file, token])

    mappings = [
        ("cross_domain_transfer", BODY[2], ["ws_ce_nll_mean", "ws_kd_nll_mean", "kd_gain_nll_mean", "kd_gain_nll_sample_sd"]),
        ("teacher_quality_intervention", BODY[3], ["teacher_validation_nll", "mean_gain_or_contrast", "sample_sd"]),
        ("weaker_teacher", BODY[3], ["teacher_validation_subset_nll", "teacher_residual_vs_c0", "test_gain"]),
        ("teacher_source_ablation", BODY[4], ["kd_gain_mean", "kd_gain_sample_sd", "paired_contrast_mean", "paired_contrast_sample_sd"]),
        ("tt_vs_tg", BODY[4], ["validation_nll", "branch_jsd", "top1_agreement", "fusion_gain", "kd_gain_mean", "kd_gain_sample_sd", "H_mean", "H_sample_sd"]),
        ("modern_stress", BODY[5], ["mean_gain", "sample_sd"]),
        ("modern_ce_gate", BODY[5], ["CE_improvement_validation"]),
    ]
    for table, file, keys in mappings:
        for row in rows(table):
            if row.get("label") == "Transformer_minus_Grassmann":
                continue
            for key in keys:
                if row[key]:
                    contains(file, row[key])
            if table == "tt_vs_tg" and row["parameters"]:
                assert f"{int(row['parameters']):,}" in texts[file]
    gains = [float(row["test_gain"]) for row in rows("weaker_teacher")]
    contains(BODY[3], statistics.mean(gains))
    contains(BODY[3], statistics.stdev(gains))
    assert all(row["test_evaluated"] == "False" and row["passes_individual_gate"] == "False" for row in rows("modern_ce_gate"))
    checks["core_effects_scoped_to_archived_tables"] = f"PASS ({len(numerical_checks)} numeric checks)"
    assert "CLIP\\_SATURATION\\_WARNING" in texts[BODY[4]] and "CLIP\\_SATURATION\\_WARNING" in texts[BODY[5]]
    assert "CE456\\_GRADIENT\\_SPIKE\\_WARNING" in texts[BODY[5]]
    assert "STOP\\_MODERN\\_CONTINUATION\\_NOT\\_ESTABLISHED" in texts[BODY[5]]
    checks["permanent_warning_and_failed_gate_labels"] = "PASS"
    log = (BUILD / "main.log").read_text()
    assert not re.search(r"undefined|multiply defined|^!|Overfull|Label\(s\) may have changed", log, re.I | re.M)
    assert "warning$ -- 0" in (BUILD / "main.blg").read_text()
    checks["latex_bibtex_no_errors_unresolved_references_or_overfull"] = "PASS"
    aux = (BUILD / "main.aux").read_text()
    labels = re.findall(r"\\newlabel\{([^}]+)\}", aux)
    assert len(labels) == len(set(labels))
    sections = ["introduction", "related-work", "problem-setting", "experimental-setting",
                "cross-domain-transfer", "distillability", "controls", "external-validity", "discussion", "conclusion"]
    for number, name in enumerate(sections, 1):
        assert f"\\newlabel{{sec:{name}}}{{{{{number}}}" in aux
    assert all(key in labels for key in re.findall(r"\\ref\{([^}]+)\}", joined))
    assert len([x for x in labels if x.startswith("fig:")]) == 5
    assert len([x for x in labels if x.startswith("tab:")]) == 8
    checks["sections_1_to_10_unique_labels_and_body_references"] = "PASS"
    state = (ROOT / "research-state.yaml").read_text()
    assert "training_authorized: false" in state and "experimental_program: permanently_frozen" in state
    checks["training_permanently_frozen"] = "PASS"
    print(json.dumps({
        "status": "PASS", "date": "2026-10-04", "input_commit": BASE,
        "activity": "read-only document, archived-table, hash, compiled-LaTeX checks; no model access",
        "checks": checks, "numeric_checks": len(numerical_checks),
        "body_source_sha256": {str((MS / "sections" / name).relative_to(ROOT)): sha(MS / "sections" / name) for name in BODY},
        "compiled_workspace_main_sha256": sha(MS / "main.tex"),
        "compiled_workspace_introduction_sha256": sha(MS / "sections/01_introduction.tex"),
        "build_directory": str(BUILD.relative_to(ROOT)),
        "pdf_pages": int(re.search(r"Output written on .*?\((\d+) pages", log, re.S).group(1)),
        "pdf_sha256": sha(BUILD / "main.pdf"),
        "underfull_bibliography_notices": log.count("Underfull \\hbox"),
        "scope_note": "Local PDF includes pre-existing user Abstract/Introduction edits, excluded from this commit. Five figures and eight planning-table captions remain placeholders. Sections 2/9/10 and appendix were not drafted.",
        "limits": "Core table values automatically checked; ancillary protocol, optimization and interpretation claims reviewed against original reports. Not submission-readiness certification.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
