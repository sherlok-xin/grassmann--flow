#!/usr/bin/env python3
"""Render manuscript pages and captioned float regions for human layout QA."""
import argparse
import json
import re
from pathlib import Path

import fitz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdf', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for child in ['print_scale_96dpi', 'local_300dpi']:
        (args.output / child).mkdir(exist_ok=True)
    doc = fitz.open(args.pdf)
    index = []
    for pn, page in enumerate(doc, 1):
        for dpi, folder in [(150, args.output), (96, args.output/'print_scale_96dpi')]:
            page.get_pixmap(dpi=dpi, alpha=False).save(folder / f'page-{pn:02d}.png')
        captions = []
        for block in page.get_text('dict')['blocks']:
            if 'lines' not in block:
                continue
            first = ''.join(s['text'] for s in block['lines'][0]['spans'])
            match = re.match(r'(Figure|Table) (A?\d+):', first)
            if match:
                captions.append((block, match.group(1), match.group(2)))
        top = 65.0
        for block, kind, number in sorted(captions, key=lambda x: x[0]['bbox'][1]):
            x0, y0, x1, y1 = block['bbox']
            rect = fitz.Rect(68, top, 526, min(780, y1+4))
            if kind == 'Table' and number == '1':
                rect.x1 = 295
            name = f'page-{pn:02d}-{kind.lower()}-{number}.png'
            page.get_pixmap(dpi=300, clip=rect, alpha=False).save(args.output/'local_300dpi'/name)
            index.append({'page': pn, 'kind': kind, 'number': number,
                          'crop_pdf_points': list(rect), 'dpi': 300, 'file': name})
            top = y1+5
    print(json.dumps({'pdf': str(args.pdf), 'pages': len(doc), 'whole_page_dpi': 150,
                     'print_scale_dpi': 96, 'local_inspections': index}, indent=2))


if __name__ == '__main__':
    main()
