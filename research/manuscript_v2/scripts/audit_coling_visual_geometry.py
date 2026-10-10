#!/usr/bin/env python3
"""Read-only review-marker classification and semantic figure geometry QA."""
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import fitz
import pdfplumber

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / 'outputs/coling2027_arr'
FIG = ROOT / '论文投稿/coling2027_arr/figures'


def main():
    raw = json.loads((BUILD / 'pubcheck/errors-main.json').read_text())
    counts = {k: len(v) for k, v in raw.items()}
    assert set(k for k, v in counts.items() if v) == {'Error.MARGIN'}
    markers, nonmarker, protrusions = Counter(), [], []
    with pdfplumber.open(BUILD / 'main.pdf') as doc:
        for pn, p in enumerate(doc.pages, 1):
            for word in p.extract_words(extra_attrs=['non_stroking_color', 'stroking_color']):
                # Mirror the official checker's color filter for raw classification.
                # Independently inspect every body word below: the upstream filter
                # skips some black text, so classification alone is not sufficient QA.
                if word['non_stroking_color'] in ((0, 0, 0), [0]) or (
                        word['non_stroking_color'] is None and word['stroking_color'] is None):
                    continue
                side = ('right' if 595 - word['x1'] < 66.5 else
                        'left' if word['x0'] < 69 else
                        'top' if word['top'] < 56 else None)
                if side:
                    # Mirror the official crop's <=1-pixel skip as well.
                    if side == 'left' and min(int(word['x1']), 66.5) - max(0, int(word['x0'])) <= 1:
                        continue
                    if side in ('left', 'right') and word['text'].isdigit() and (
                            word['x1'] < 60 or word['x0'] > 540):
                        markers[side] += 1
                    else:
                        nonmarker.append({'page': pn, 'side': side, 'word': word})
            bottom = [w for w in p.extract_words() if w['top'] >= 780]
            assert len(bottom) == 1 and bottom[0]['text'] == str(pn), (pn, bottom)
            markers['bottom_page_number'] += 1
            for word in p.extract_words():
                if word['text'].isdigit() and (word['x1'] < 60 or word['x0'] > 540 or word['top'] >= 780):
                    continue
                # The required, unmodified microtype package protrudes punctuation.
                # Allow at most 2.4 PDF pt; do not alter the document to suppress it.
                assert word['x0'] >= 68.466 and word['x1'] <= 526.81 and word['top'] >= 55.9 and word['bottom'] < 780, (pn, word)
                if word['x0'] < 69:
                    protrusions.append({'page': pn, 'text': word['text'], 'x0': word['x0'], 'explanation': 'Standard microtype leading plus-sign protrusion, not an upstream reported finding.'})
    assert not nonmarker, nonmarker
    assert sum(markers.values()) == counts['Error.MARGIN'], (markers, counts)
    pdf = fitz.open(BUILD / 'main.pdf')
    for page in pdf:
        for path in page.get_drawings():
            r = path['rect']
            # Publication graphics, table rules, and footnote separators only.
            assert r.x0 >= 68.9 and r.x1 <= 528.6 and r.y0 >= 55.9 and r.y1 <= 780

    # Import renderer helpers, never its main(): no archive export is written.
    sys.path.append('/usr/lib/python3/dist-packages')
    spec = importlib.util.spec_from_file_location('semantic_renderer',
        ROOT / 'research/manuscript_v2/scripts/render_vector_figures.py')
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    ns = {'s': 'http://www.w3.org/2000/svg'}
    geometry = {}
    for name in ['fig1_paired_protocol', 'figA1_teacher_architecture']:
        path = FIG / (name + '.svg')
        doc = ET.parse(path).getroot()
        _, _, w, h = map(float, doc.get('viewBox').split())
        handle = renderer.Rsvg.Handle.new_from_file(str(path))
        handle.set_dpi(96)
        ok, iw, ih = handle.get_intrinsic_size_in_pixels()
        assert ok
        boxes, violations, overlaps = [], [], []
        for e in doc.findall('.//s:text', ns):
            ok, ink, _ = handle.get_geometry_for_layer('#' + e.get('id'), renderer.viewport(w, h))
            assert ok
            x, y, tw, th = ink.x*w/iw, ink.y*h/ih, ink.width*w/iw, ink.height*h/ih
            boxes.append((e.get('id'), x, y, tw, th))
            if x < -.5 or y < -.5 or x+tw > w+.5 or y+th > h+.5:
                violations.append((e.get('id'), 'canvas'))
            if e.get('data-box'):
                bx, by, bw, bh = map(float, e.get('data-box').split(','))
                if x < bx-.5 or y < by-.5 or x+tw > bx+bw+.5 or y+th > by+bh+.5:
                    violations.append((e.get('id'), 'owner_box'))
        for i, (aid, ax, ay, aw, ah) in enumerate(boxes):
            for bid, bx, by, bw, bh in boxes[i+1:]:
                if min(ax+aw, bx+bw)-max(ax, bx) > 1 and min(ay+ah, by+bh)-max(ay, by) > 1:
                    overlaps.append((aid, bid))
        assert not violations and not overlaps, (name, violations, overlaps)
        geometry[name] = {'text_nodes': len(boxes), 'bounds_violations': [], 'text_overlaps': []}
    print(json.dumps({
        'status': 'PASS', 'date': '2026-10-10',
        'official_pubcheck_raw_counts': counts,
        'review_marker_classification': dict(markers),
        'non_review_margin_findings': nonmarker,
        'independent_all_body_word_bounds': 'PASS (2.4 pt standard microtype punctuation allowance)',
        'standard_microtype_protrusions': protrusions,
        'vector_drawing_bounds': 'PASS', 'adjusted_figure_geometry': geometry,
        'interpretation': 'Official raw check is NOT a clean pass: all 890 margin findings correspond to official review rulers or page numbers. No submission PDF or official checker was altered.'
    }, indent=2))


if __name__ == '__main__':
    main()
