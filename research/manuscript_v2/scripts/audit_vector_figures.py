#!/usr/bin/env python3
"""Read-only frozen vector-artifact audit; never renders or edits a figure."""
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
NS = {"s": "http://www.w3.org/2000/svg"}
META = ROOT / "research/manuscript_v2/figure_vector_qa.json"
APPROVED_FIG3 = [
    "Teacher quality matters, but is not sufficient", "A",
    "Controlled teacher-state intervention", "B",
    "Global teacher quality is insufficient", "Same teacher architecture,",
    "same T/G composition, same α = 0.5", "Matched warm-start", "students",
    "3 independent", "student seeds", "Teacher J", "(jointly trained)",
    "Teacher A", "(alternative teacher state)", "Positive transfer",
    "Gain > 0 across 3/3 seeds", "Negative transfer", "Gain < 0 across 3/3 seeds",
    "Changing the trained teacher state reverses the transfer sign.",
    "worse held-out likelihood", "than matched CE comparator", "Grassmann-only",
    "teacher", "Still yields positive", "transfer", "Gain > 0 across 3/3 seeds",
    "A globally weaker teacher can still improve the student.",
    "Teacher state and predictive quality strongly modulate transfer, but held-out teacher likelihood alone does not determine distillability.",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value):
    return " ".join(value.split())


def main():
    meta = json.loads(META.read_text())
    texts_by_name, docs, reports = {}, {}, {}
    for name, frozen in meta["figures"].items():
        paths = {k: ROOT/v for k, v in frozen["paths"].items()}
        for key, suffix in [("svg", ".svg"), ("pdf", ".pdf"), ("png", "_600dpi.png")]:
            assert sha(paths[key]) == frozen["sha256"][suffix], (name, key, "hash drift")
        doc = ET.parse(paths["svg"]).getroot()
        docs[name] = doc
        assert list(map(float, doc.attrib["viewBox"].split())) == [0, 0, *frozen["canvas"]]
        assert doc.attrib["width"] == "6.8in"
        for tag in ("image", "foreignObject", "filter"):
            assert not doc.findall(".//s:"+tag, NS), (name, tag)
        for node in doc.iter():
            for key, value in node.attrib.items():
                assert "data:image" not in value
                if key.endswith("href"):
                    assert value.startswith("#"), (name, "external dependency")
        bg = doc.find("s:rect[@id='white-background']", NS)
        assert bg is not None and bg.attrib["fill"] == "#ffffff"
        texts = [normalize("".join(e.itertext())) for e in doc.findall(".//s:text", NS)]
        texts_by_name[name] = texts
        assert len(texts) == frozen["svg_text_nodes"]
        assert not frozen["bounds_violations"] and not frozen["text_overlaps"]
        pdf = PdfReader(paths["pdf"])
        assert len(pdf.pages) == 1 and not list(pdf.pages[0].images)
        assert len(pdf.pages[0].extract_text().strip()) > 50
        box = pdf.pages[0].mediabox
        assert abs(float(box.width)-frozen["pdf_points"][0]) < 0.002
        assert abs(float(box.height)-frozen["pdf_points"][1]) < 0.002
        for resource in pdf.pages[0]["/Resources"]["/Font"].get_object().values():
            font = resource.get_object()
            if "/DescendantFonts" in font:
                font = font["/DescendantFonts"][0].get_object()
            desc = font["/FontDescriptor"].get_object()
            assert any(key in desc for key in ["/FontFile", "/FontFile2", "/FontFile3"])
        font_output = subprocess.check_output(["pdffonts", str(paths["pdf"])], text=True)
        assert all(re.search(r"\byes\s+yes\s+yes\b", line)
                   for line in font_output.splitlines()[2:] if line.strip())
        with Image.open(paths["png"]) as png:
            assert list(png.size) == frozen["png_pixels"]
            assert all(abs(d-600) < .01 for d in png.info["dpi"])
        ref = ROOT/frozen["reference_path"]
        ref_status = "NOT_AVAILABLE_IN_THIS_CHECKOUT"
        if ref.exists():
            assert sha(ref) == frozen["reference_sha256"], (name, "reference drift")
            with Image.open(ref) as image:
                assert list(image.size) == frozen["canvas"]
            ref_status = "HASH_AND_CANVAS_MATCH"
        reports[name] = {"hashes": "PASS", "vector_only": "PASS",
                         "fonts_embedded": "PASS", "reference": ref_status,
                         "readability": frozen["readability_at_two_column_width"]}

    one = normalize(" ".join(texts_by_name["fig1_paired_protocol"]))
    assert "the tested local diagnostics are not reliable predictors" in one
    assert "Gain = NLLCE − NLLKD" in one
    assert "best CE checkpoint" in one and "best KD checkpoint" in one

    assert texts_by_name["fig3_teacher_quality"] == APPROVED_FIG3

    four = texts_by_name["fig4_ensemble_control"]
    assert [t for t in four if t.startswith("+")] == [
        "+0.0410 NLL", "+0.1349 NLL", "+0.1557 NLL", "+0.1557 NLL", "+0.1681 NLL"]
    assert four.count("Positive transfer") == 5
    four_joined = normalize(" ".join(four))
    assert "T-only: early clipping saturation; interpret F–T gap cautiously." in four_joined
    assert "TT and TG teachers are not strictly quality- or training-matched." in four_joined
    for banned in ("small gain", "larger gain", "largest gain", "moderate gain"):
        assert banned not in four_joined.lower()
    assert not any(p.attrib.get("fill") in ("#287c31", "#38863c", "#2f8438")
                   for p in docs["fig4_ensemble_control"].findall("s:path", NS))

    arch = docs["figA1_teacher_architecture"]
    labels = {el.attrib["id"]: normalize("".join(el.itertext()))
              for el in arch.findall(".//s:text", NS)}
    assert labels["label-21"] == "Wred"
    assert labels["label-24"] == "(zt, zt−Δ)"
    assert labels["label-25"] == "(zt−Δ → zt)"
    assert labels["label-26"] == "Δ ∈ {1, 2, 4}"
    assert labels["label-29"] == "pij"
    assert labels["label-32"] == "max(‖p‖2, ε)"
    assert labels["label-37"] == "valid Δ"
    assert labels["label-44"] == "z = αz(T) + (1 − α)z(G)"
    assert "0.5" not in " ".join(texts_by_name["figA1_teacher_architecture"])
    ordered = ["token-embedding-g", "linear-projection", "local-pairs",
               "plucker-coordinates", "normalization", "geometric-projection",
               "window-mean", "gated-mixing", "lm-head-g", "g-logits"]
    x_positions = [float(arch.find("s:rect[@id='"+i+"']", NS).attrib["x"])
                   for i in ordered]
    assert x_positions == sorted(x_positions)
    arrows = {p.attrib.get("d") for p in arch.findall("s:path", NS)
              if "marker-end" in p.attrib}
    assert {"M 416 546 L 439 546", "M 526 546 L 544 546",
            "M 653 546 L 661 546", "M 739 546 L 746 546",
            "M 878 546 L 887 546", "M 961 546 L 972 546",
            "M 1042 546 L 1051 546", "M 1112 546 L 1120 546",
            "M 1205 546 L 1229 546"}.issubset(arrows)

    for name, expected in meta["protected_user_source_sha256"].items():
        assert sha(ROOT/name) == expected, ("pre-existing prose changed", name)
    body = json.loads((ROOT/"research/manuscript_v2/body_draft_qa.json").read_text())
    for name, expected in body["body_source_sha256"].items():
        assert sha(ROOT/name) == expected, ("body prose changed", name)
    print(json.dumps({"status": meta["status"], "figures": reports,
                      "wording_and_scientific_notation": "PASS",
                      "pre_existing_prose_preserved": "PASS",
                      "compile_record": meta["compile"],
                      "scope": "Read-only artifact audit; no experiments or rendering."},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--readability-revision',action='store_true',help='Audit explicitly authorized typography revision and new numerical figure')
    if parser.parse_args().readability_revision:
        from audit_readability_revision import main as audit_revision
        audit_revision()
    else:
        main()
