#!/usr/bin/env python3
"""Read-only final vector/wording/font audit for the authorized full revision."""
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
from pypdf import PdfReader
from audit_vector_figures import APPROVED_FIG3

ROOT = Path(__file__).resolve().parents[3]
META = ROOT / 'research/manuscript_v2/figure_readability_qa.json'
NS = {'s': 'http://www.w3.org/2000/svg'}


def wording(node):
    children = list(node)
    if children and all('y' in x.attrib for x in children):
        return ' '.join(''.join(x.itertext()) for x in children)
    return ''.join(node.itertext())


def main():
    meta = json.loads(META.read_text())
    texts, docs, reports = {}, {}, {}
    for name, record in meta['figures'].items():
        paths = {k: ROOT/v for k,v in record['paths'].items()}
        for key, suffix in [('svg','.svg'),('pdf','.pdf'),('png','_600dpi.png')]:
            assert hashlib.sha256(paths[key].read_bytes()).hexdigest() == record['sha256'][suffix]
        doc = ET.parse(paths['svg']).getroot(); docs[name] = doc
        assert not any(doc.findall('.//s:'+tag,NS) for tag in ['image','foreignObject','filter'])
        assert 'data:image' not in paths['svg'].read_text()
        texts[name] = [' '.join(wording(e).split()) for e in doc.findall('.//s:text',NS)]
        reader = PdfReader(paths['pdf'])
        assert len(reader.pages)==1 and not list(reader.pages[0].images)
        assert reader.pages[0].extract_text().strip()
        page = reader.pages[0]
        assert abs(float(page.mediabox.width)-6.8*72)<0.002
        for resource in page['/Resources']['/Font'].get_object().values():
            font = resource.get_object()
            if '/DescendantFonts' in font: font = font['/DescendantFonts'][0].get_object()
            descriptor = font['/FontDescriptor'].get_object()
            assert any(k in descriptor for k in ['/FontFile','/FontFile2','/FontFile3'])
        with Image.open(paths['png']) as im:
            assert list(im.size)==record['png_pixels']
            assert all(abs(v-600)<.01 for v in im.info['dpi'])
        assert record['normal_text_min_pt']>=7 and record['all_text_min_pt']>=6
        assert not record['bounds_violations'] and not record['text_overlaps']
        if record.get('reference_path'):
            ref=ROOT/record['reference_path']
            assert hashlib.sha256(ref.read_bytes()).hexdigest()==record['reference_sha256']
        reports[name]={'pure_vector':'PASS','editable_svg_text':'PASS','embedded_pdf_fonts':'PASS','readability_thresholds':'PASS','export_hashes':'PASS'}
    one=' '.join(texts['fig1_paired_protocol'])
    assert 'the tested local diagnostics are not reliable predictors' in one
    assert 'Gain = NLLCE − NLLKD' in one
    assert 'best CE checkpoint' in one and 'best KD checkpoint' in one
    assert texts['fig3_teacher_quality']==APPROVED_FIG3
    four=texts['fig4_ensemble_control']
    assert [t for t in four if t.startswith('+')]==['+0.0410 NLL','+0.1349 NLL','+0.1557 NLL','+0.1557 NLL','+0.1681 NLL']
    assert four.count('Positive transfer')==5
    assert not any(t in ' '.join(four).lower() for t in ['small gain','larger gain','largest gain'])
    caption=(ROOT/'论文投稿/manuscript_v2/sections/07_controls.tex').read_text().replace('F--T','F–T')
    assert 'T-only: early clipping saturation; interpret F–T gap cautiously.' in caption
    assert 'TT and TG teachers are not strictly quality- or training-matched.' in caption
    arch=docs['figA1_teacher_architecture']; joined=' '.join(texts['figA1_teacher_architecture'])
    for token in ['zt−Δ → zt','Δ ∈ {1, 2, 4}','pij','max(‖p‖2, ε)','valid Δ','z = αz(T) + (1 − α)z(G)']:
        assert token in joined, token
    assert '0.5' not in joined
    for identifier in ['token-embedding-g','linear-projection','local-pairs','plucker-coordinates','normalization','geometric-projection','window-mean','gated-mixing','lm-head-g','g-logits']:
        assert arch.find("s:rect[@id='"+identifier+"']",NS) is not None
    arrows={p.get('d') for p in arch.findall('s:path',NS) if 'marker-end' in p.attrib}
    assert {'M446 431H473','M626 431H653','M856 431H883','M1056 431H1083','M1220 507V545H385V578','M476 631H508','M681 631H713','M886 631H918','M1091 631H1123'}.issubset(arrows), arrows
    assert 'sample SD' in ' '.join(texts['fig2_cross_domain_gain'])
    fls=(ROOT/'outputs/manuscript_v2_build/main.fls').read_text()
    for name in reports: assert 'figures/'+name+'.pdf' in fls
    print(json.dumps({'status':'PASS','date':'2026-10-08','figures':reports,'scientific_wording_and_causal_connectivity':'PASS','pdf_figures_are_actual_manuscript_build_inputs':'PASS','limits':'Geometry checks supplemented by manual visual comparison; not certification for a future single-column conference layout.'},indent=2))


if __name__=='__main__':main()
