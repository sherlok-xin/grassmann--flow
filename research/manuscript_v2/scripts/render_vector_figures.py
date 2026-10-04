#!/usr/bin/env python3
"""Render semantic SVG masters; reference bitmaps are never rendering inputs."""
import argparse
import hashlib
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cairo
import gi
from PIL import Image
from pypdf import PdfReader

gi.require_version("Rsvg", "2.0")
from gi.repository import Rsvg

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "论文投稿/manuscript_v2/figures"
QA = ROOT / "outputs/manuscript_v2_vector_qa"
NAMES = ["fig1_paired_protocol", "fig3_teacher_quality",
         "fig4_ensemble_control", "figA1_teacher_architecture"]
NS = {"svg": "http://www.w3.org/2000/svg"}


def viewport(width, height):
    r = Rsvg.Rectangle()
    r.x = r.y = 0
    r.width, r.height = width, height
    return r


def render(handle, path, width, height, pdf=False, dpi=None):
    surface = (cairo.PDFSurface(str(path), width, height) if pdf else
               cairo.ImageSurface(cairo.FORMAT_ARGB32, int(width), int(height)))
    ctx = cairo.Context(surface)
    assert handle.render_document(ctx, viewport(width, height))
    if pdf:
        ctx.show_page()
        surface.finish()
    else:
        data = io.BytesIO()
        surface.write_to_png(data)
        image = Image.open(data).convert("RGB")
        image.save(path, optimize=True, **({"dpi": (dpi, dpi)} if dpi else {}))
        surface.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=NAMES)
    args = parser.parse_args()
    QA.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in ([args.only] if args.only else NAMES):
        svg = FIG / f"{name}.svg"
        document = ET.parse(svg).getroot()
        assert not document.findall(".//svg:image", NS)
        assert not document.findall(".//svg:foreignObject", NS)
        assert not document.findall(".//svg:filter", NS)
        assert "data:image" not in svg.read_text()
        _, _, w, h = map(float, document.attrib["viewBox"].split())
        handle = Rsvg.Handle.new_from_file(str(svg))
        handle.set_dpi(96)
        ok, intrinsic_w, intrinsic_h = handle.get_intrinsic_size_in_pixels()
        assert ok
        pdfw = 6.8 * 72
        render(handle, FIG / f"{name}.pdf", pdfw, pdfw*h/w, pdf=True)
        pngw, pngh = 4080, round(4080*h/w)
        render(handle, FIG / f"{name}_600dpi.png", pngw, pngh, dpi=600)
        render(handle, QA / f"{name}_reference_size.png", int(w), int(h))
        render(handle, QA / f"{name}_two_column_180dpi.png", 1224, round(1224*h/w))
        pdf = PdfReader(FIG / f"{name}.pdf")
        assert len(pdf.pages) == 1
        assert not list(pdf.pages[0].images)
        assert pdf.pages[0].extract_text().strip()
        labels, violations = [], []
        for element in document.findall(".//svg:text", NS):
            ok, ink, logical = handle.get_geometry_for_layer("#"+element.attrib["id"], viewport(w, h))
            assert ok
            # librsvg 2.52 returns intrinsic CSS-pixel geometry even when a
            # viewport is supplied; convert explicitly to SVG viewBox units.
            rect = [ink.x*w/intrinsic_w, ink.y*h/intrinsic_h,
                    ink.width*w/intrinsic_w, ink.height*h/intrinsic_h]
            entry = {"id": element.attrib["id"], "text": "".join(element.itertext()),
                     "ink_box": [round(x, 3) for x in rect],
                     "font_size_canvas_units": float(element.attrib["font-size"])}
            labels.append(entry)
            x, y, tw, th = rect
            if x < -0.5 or y < -0.5 or x+tw > w+.5 or y+th > h+.5:
                violations.append({"kind": "canvas_clip", **entry})
            if "data-box" in element.attrib:
                bx, by, bw, bh = map(float, element.attrib["data-box"].split(","))
                if x < bx-.5 or y < by-.5 or x+tw > bx+bw+.5 or y+th > by+bh+.5:
                    violations.append({"kind": "owner_box_overflow", **entry})
        overlaps = []
        for i, a in enumerate(labels):
            ax, ay, aw, ah = a["ink_box"]
            for b in labels[i+1:]:
                bx, by, bw, bh = b["ink_box"]
                dx = min(ax+aw, bx+bw)-max(ax, bx)
                dy = min(ay+ah, by+bh)-max(ay, by)
                if dx > 1 and dy > 1:
                    overlaps.append({"a": a["id"], "b": b["id"],
                                     "text_a": a["text"], "text_b": b["text"],
                                     "overlap": [round(dx, 3), round(dy, 3)]})
        sizes = [x["font_size_canvas_units"]*pdfw/w for x in labels]
        results[name] = {
            "canvas": [int(w), int(h)], "aspect_ratio": w/h,
            "pdf_points": [pdfw, pdfw*h/w], "png_pixels": [pngw, pngh],
            "png_dpi": list(Image.open(FIG/f"{name}_600dpi.png").info["dpi"]),
            "svg_text_nodes": len(labels), "pdf_embedded_images": 0,
            "font_family": "Liberation Sans, Arial, Helvetica, sans-serif",
            "base_text_size_pt_at_6_8in": [round(min(sizes), 3), round(max(sizes), 3)],
            "bounds_violations": violations, "text_overlaps": overlaps,
            "sha256": {suffix: hashlib.sha256((FIG/f"{name}{suffix}").read_bytes()).hexdigest()
                       for suffix in [".svg", ".pdf", "_600dpi.png"]},
        }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
