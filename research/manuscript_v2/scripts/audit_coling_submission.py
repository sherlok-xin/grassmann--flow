#!/usr/bin/env python3
"""Read-only submission QA against frozen records; never accesses a model."""
import ast
import hashlib
import importlib.util
import json
import re
import statistics
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[3]
MS = ROOT / '论文投稿/coling2027_arr'
QA = ROOT / 'research/manuscript_v2'
BUILD = ROOT / 'outputs/coling2027_arr'
ARCHIVE = ROOT / '论文投稿/manuscript_v2'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    original_path = QA / 'scripts/audit_body_draft.py'
    original_result = json.loads(subprocess.check_output(
        ['python3', str(original_path), '--full-revision'], cwd=ROOT, text=True))
    assert original_result['status'] == 'PASS'
    assert original_result['numeric_checks'] == 62
    locked = json.loads((QA / 'coling2027_preservation.json').read_text())
    assert all(sha(ROOT / p) == h for p, h in locked['archive_files_sha256'].items())
    assert not subprocess.check_output(
        ['git', 'diff', locked['input_commit'], '--name-only', '--',
         'research/final_evidence', 'research/experiments', 'datasets',
         'checkpoints', '论文投稿/cac', '论文投稿/manuscript_v2'], cwd=ROOT, text=True).strip()

    spec = importlib.util.spec_from_file_location('frozen_audit', original_path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    tree = ast.parse(original_path.read_text())
    node = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'mappings' for t in n.targets))
    # The exact original field inventory, not a reduced set for the conversion.
    mappings = eval(compile(ast.Expression(node), '<frozen-inventory>', 'eval'),
                    {'BODY': original.BODY, '__builtins__': {}})
    source_paths = [MS / 'main.tex'] + sorted((MS / 'sections').glob('*.tex'))
    source_paths += sorted((MS / 'appendix').glob('*.tex')) + sorted((MS / 'tables').glob('*.tex'))
    source = '\n'.join(p.read_text() for p in source_paths)
    pdf = fitz.open(BUILD / 'main.pdf')
    # Exclude outer review rulers and page footers from text extraction only.
    # The submitted PDF retains every official review mark unchanged.
    content_lines = []
    for p in pdf:
        for b in p.get_text('dict')['blocks']:
            for line in b.get('lines', []):
                spans = [s for s in line['spans'] if s['bbox'][0] >= 60
                         and s['bbox'][2] <= 540 and s['bbox'][1] < 795]
                if spans:
                    content_lines.append(''.join(s['text'] for s in spans))
    pdf_text = '\n'.join(content_lines).replace('−', '-')
    pdf_text = re.sub(r'-(\s+)(?=\d)', '-', pdf_text)
    numeric_checks = []

    def contains(table, field, value):
        token = f'{float(value):.6f}'
        assert token in source, (table, field, token, 'source')
        assert token in pdf_text, (table, field, token, 'rendered PDF')
        numeric_checks.append({'table': table, 'field': field, 'value': token})

    for table, _, fields in mappings:
        for row in original.rows(table):
            if row.get('label') == 'Transformer_minus_Grassmann':
                continue
            for field in fields:
                if row[field]:
                    contains(table, field, row[field])
            if table == 'tt_vs_tg' and row['parameters']:
                assert f"{int(row['parameters']):,}" in source
    gains = [float(r['test_gain']) for r in original.rows('weaker_teacher')]
    contains('weaker_teacher', 'frozen_mean', statistics.mean(gains))
    contains('weaker_teacher', 'frozen_sample_sd', statistics.stdev(gains))
    assert len(numeric_checks) == 62

    expected_style = {
        'acl.sty': '19dfeddc2c0e448f3926a0bef048a9db3f3611b46265b760caabd7ada4f361de',
        'acl_natbib.bst': '6fbb306202290f4b68e74ac1460a8b27398500cb6dfeb4492e74c457eae7cd1e'}
    assert all(sha(MS / p) == h for p, h in expected_style.items())
    preamble = (MS / 'main.tex').read_text().split('\\begin{document}')[0]
    assert '\\usepackage[review]{acl}' in preamble
    assert not re.search(r'\\(?:usepackage\{(?:geometry|fullpage)|setlength|renewcommand|vspace|fontsize)', preamble)
    assert '\\bibliographystyle{acl_natbib}' in (MS / 'acl.sty').read_text()
    assert (MS / 'references.bib').read_bytes() == (ARCHIVE / 'references.bib').read_bytes()
    keys = set(re.findall(r'@\w+\{([^,]+),', (MS / 'references.bib').read_text()))
    cited = set()
    for group in re.findall(r'\\cite\w*\{([^}]+)\}', source):
        cited.update(group.split(','))
    assert cited == keys and len(keys) == 24
    assert all(r['metadata_status'] == 'VERIFIED_PRIMARY_EXPORT' for r in
               json.loads((QA / 'full_text_reference_verification.json').read_text())['records'])
    abstract = source.split('\\begin{abstract}')[1].split('\\end{abstract}')[0]
    assert len(abstract.split()) <= 200
    assert 'matched CE comparator' in abstract and 'FineWeb' not in abstract
    prose = '\n'.join(p.read_text() for p in (MS / 'sections').glob('*.tex'))
    assert not re.search(r'\bPhase\s*[234][A-Z]?\b|TODO|FIXME|PLACEHOLDER|CRBD|Safe-KD', prose)
    assert not re.search(r'\\(?:textbf|emph|item)\b|\\begin\{(?:itemize|enumerate)\}', prose)
    for warning in ['CLIP_SATURATION_WARNING', 'CE456_GRADIENT_SPIKE_WARNING',
                    'STOP_MODERN_CONTINUATION_NOT_ESTABLISHED']:
        # Whitespace in extraction may split the long diagnostic decision label.
        assert warning in re.sub(r'\s+', '', pdf_text)
    assert 'No adaptive-KD baseline' in source

    log = (BUILD / 'main.log').read_text(errors='replace')
    assert not re.search(r'undefined|multiply defined|^!|Overfull|Label\(s\) may have changed', log, re.I | re.M)
    assert 'warning$ -- 0' in (BUILD / 'main.blg').read_text()
    aux = (BUILD / 'main.aux').read_text()
    labels = re.findall(r'\\newlabel\{([^}]+)\}', aux)
    assert len(labels) == len(set(labels))
    assert all(k in labels for k in re.findall(r'\\ref\{([^}]+)\}', source))

    def page(label):
        return int(re.search(r'\\newlabel\{' + re.escape(label) + r'\}\{\{[^}]*\}\{(\d+)\}', aux)[1])

    figure_labels = ['fig:paired_protocol', 'fig:cross_domain_gain', 'fig:teacher_quality', 'fig:ensemble_control']
    main_end = max(page('end:main-text'), *(page(k) for k in figure_labels), page('tab:cross_domain_transfer'))
    assert main_end <= 8
    assert all(page(k) <= page('sec:limitations') for k in figure_labels)
    assert len([k for k in labels if k.startswith('fig:')]) == 5
    assert '\\newlabel{fig:teacher_architecture}{{A1}' in aux
    assert '\\section*{Limitations}' in (MS / 'sections/limitations.tex').read_text()
    assert 'Conclusion' in pdf_text and 'Limitations' in pdf_text and 'References' in pdf_text
    assert pdf_text.index('Conclusion') < pdf_text.index('Limitations') < pdf_text.index('References')
    assert not re.search(r'\\(?:onecolumn|twocolumn|clearpage|newpage)', (MS / 'sections/limitations.tex').read_text())

    forbidden = r'sherlok[-_]xin|/home/xin|grassmann--flow\.git|Ningxia|Anonymous authors.*@|xin@'
    assert not re.search(forbidden, pdf_text, re.I)
    assert not pdf.metadata.get('author')
    assert not re.search(forbidden, json.dumps(pdf.metadata), re.I)
    urls = [a['uri'] for p in pdf for a in p.get_links() if 'uri' in a]
    assert not re.search(forbidden, '\n'.join(urls), re.I)
    assert not list(MS.rglob('*.png')) and not any(p.is_symlink() for p in MS.rglob('*'))
    assert all(abs(p.rect.width - 595.276) < .1 and abs(p.rect.height - 841.89) < .1 for p in pdf)
    assert all(not p.get_images() for p in pdf)

    font_sizes = {}
    for path in sorted((MS / 'figures').glob('*.pdf')):
        fig = fitz.open(path)
        assert not fig[0].get_images() and fig[0].get_text().strip()
        assert all(len(fig.extract_font(f[0])[3]) > 0 for f in fig[0].get_fonts())
        spans = [s for b in fig[0].get_text('dict')['blocks'] if 'lines' in b
                 for l in b['lines'] for s in l['spans']]
        # Official full width is 16 cm; conversion from native 6.8 inches.
        factor = (16 / 2.54 * 72) / fig[0].rect.width
        font_sizes[path.stem] = round(min(s['size'] for s in spans) * factor, 3)
        assert font_sizes[path.stem] >= 6
        svg = ET.parse(path.with_suffix('.svg')).getroot()
        assert not any(n.tag.endswith('}image') for n in svg.iter())
        old = ET.parse(ARCHIVE / 'figures' / path.with_suffix('.svg').name).getroot()
        assert [n.text for n in svg.iter()] == [n.text for n in old.iter()]
        assert [n.tail for n in svg.iter()] == [n.tail for n in old.iter()]
        assert len(list(svg.iter())) == len(list(old.iter()))
        for a, b in zip(svg.iter(), old.iter()):
            allowed = dict(b.attrib)
            if path.stem in ['fig1_paired_protocol', 'figA1_teacher_architecture'] and allowed.get('font-size') == '22':
                allowed['font-size'] = '24'
            assert a.attrib == allowed

    state = (ROOT / 'research-state.yaml').read_text()
    assert 'training_authorized: false' in state and 'experimental_program: permanently_frozen' in state
    print(json.dumps({
        'status': 'PASS', 'input_commit': locked['input_commit'],
        'activity': 'Read-only document, archived record, PDF, and preservation checks. No model access or endpoint evaluation.',
        'original_audit': 'PASS (62/62)', 'submission_numeric_checks': len(numeric_checks),
        'numeric_check_inventory': numeric_checks,
        'archival_files_byte_preserved': len(locked['archive_files_sha256']),
        'official_template_commit': 'd5adc823ff0f80f98c80405ca0ab66c68e684409',
        'style_hashes': expected_style, 'abstract_words': len(abstract.split()),
        'main_content_last_page': main_end, 'pdf_pages': len(pdf),
        'limitations_page': page('sec:limitations'),
        'figure_pages': {k: page(k) for k in figure_labels + ['fig:teacher_architecture']},
        'verified_reference_count': len(keys), 'bib_identical_to_archive': True,
        'anonymity_identity_checks': 'PASS', 'embedded_raster_images': 0,
        'minimum_figure_text_or_script_pt_at_actual_acl_width': font_sizes,
        'underfull_hbox_notices': log.count('Underfull \\hbox'),
        'underfull_vbox_notices': log.count('Underfull \\vbox'),
        'pdf_sha256': sha(BUILD / 'main.pdf'),
        'limits': 'Automated checks are not a scientific acceptance guarantee; visual page inspection and human submission actions remain distinct.'
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
