#!/usr/bin/env python3
"""Read-only final-polish scope audit and compact QA collection; no model access."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MS = Path('论文投稿/coling2027_arr')
BUILD = ROOT / 'outputs/coling2027_arr'
BASE = 'cd8b5504403a21cb2e98efb446e3ca680340032f'


def old(relative):
    return subprocess.check_output(['git', 'show', f'{BASE}:{MS / relative}'], cwd=ROOT)


def current(relative):
    return (ROOT / MS / relative).read_bytes()


def normalize_layout(text):
    text = re.sub(r'\\input\{[^}]+\}', '', text)
    text = text.replace('\\allowbreak', '')
    text = re.sub(r'\\texttt\{([^}]+)\}', r'\1', text)
    return re.sub(r'\s+', ' ', text).strip()


def warning_inventory(path):
    lines = path.read_text(errors='replace').splitlines()
    counts = {'h': 0, 'v': 0}
    rows = []
    for lineno, line in enumerate(lines, 1):
        m = re.search(r'Underfull \\([hv])box \(badness (\d+)\)(.*)', line)
        if m:
            kind = m[1]
            counts[kind] += 1
            rows.append({'id': kind.upper() + str(counts[kind]),
                         'log_line': lineno, 'badness': int(m[2]),
                         'context': m[3].strip()})
    return {'hbox': counts['h'], 'vbox': counts['v'],
            'overfull': sum('Overfull \\' in x for x in lines), 'inventory': rows}


def main():
    assert current('main.tex').decode() == old('main.tex').decode().replace(
        'whether KD helps depends on', 'whether KD helps varies with')
    before = ('The comparator is the selected CE continuation, not an arbitrary '
              'initialization, and residual likelihood and transfer gain use '
              'different held-out scopes')
    after = ('The comparator is the selected CE continuation, not an arbitrary '
             'initialization. Teacher likelihood is compared against the matched '
             'CE endpoint on the same validation subset, whereas transfer gain '
             'is evaluated on the selected test endpoints')
    assert current('sections/06_distillability.tex').decode() == old(
        'sections/06_distillability.tex').decode().replace(before, after)
    for rel in ['appendix/results.tex', 'appendix/diagnostics_modern.tex']:
        a, b = old(rel).decode(), current(rel).decode()
        assert normalize_layout(a) == normalize_layout(b), rel
        assert re.findall(r'\\input\{[^}]+\}', a) == re.findall(r'\\input\{[^}]+\}', b)
    paths = subprocess.check_output(
        ['git', 'ls-tree', '-r', '-z', '--name-only', BASE, '--', str(MS)],
        cwd=ROOT, text=True).strip('\0').split('\0')
    exceptions = {'main.tex', 'sections/06_distillability.tex', 'appendix/results.tex',
                  'appendix/diagnostics_modern.tex', 'README.md',
                  'figures/fig2_cross_domain_gain.svg', 'figures/fig2_cross_domain_gain.pdf'}
    preserved = []
    for path in paths:
        rel = str(Path(path).relative_to(MS))
        if rel not in exceptions:
            assert current(rel) == old(rel), path
            preserved.append(path)
    assert not subprocess.check_output(['git', 'diff', BASE, '--name-only', '--',
        'research/final_evidence', 'research/experiments', '论文投稿/manuscript_v2',
        '论文投稿/cac', 'datasets', 'checkpoints'], cwd=ROOT, text=True).strip()
    for p in (ROOT / MS).rglob('*.tex'):
        text = p.read_text()
        assert not re.search(r'sherlok[-_]xin|/home/xin|xin@|Ningxia|Acknowledg|TODO|FIXME|PLACEHOLDER', text, re.I), p
    assert not (ROOT / MS / 'main.tex').read_text().split('\\begin{document}')[0].replace(
        old('main.tex').decode().split('\\begin{document}')[0], '')

    submission = json.loads((BUILD / 'final_submission_audit.json').read_text())
    geometry = json.loads((BUILD / 'final_visual_geometry_audit.json').read_text())
    archive = json.loads((BUILD / 'final_archive_vector_audit.json').read_text())
    assert all(x['status'] == 'PASS' for x in [submission, geometry, archive])
    sha = hashlib.sha256((BUILD / 'main.pdf').read_bytes()).hexdigest()
    assert sha == submission['pdf_sha256']
    index = json.loads((BUILD / 'final_render_index.json').read_text())
    assert index['pages'] == 14 and len(index['local_inspections']) == 17
    for p in range(1, 15):
        assert (BUILD / f'final_pages/page-{p:02d}.png').exists()
        assert (BUILD / f'final_pages/print_scale_96dpi/page-{p:02d}.png').exists()
    # Pages 11-14 were also inspected in the last layout trial, from this same
    # final PDF. Verify pixel identity rather than treating an earlier trial as QA.
    for p in range(11, 15):
        assert (BUILD / f'polish_trial3_pages/page-{p:02d}.png').read_bytes() == (
            BUILD / f'final_pages/page-{p:02d}.png').read_bytes()
    source_hashes = {str(MS / rel): hashlib.sha256(current(rel)).hexdigest()
                     for rel in sorted(exceptions)}
    result = {
        'status': 'PASS', 'date': '2026-10-11', 'input_commit': BASE,
        'scope': 'Document-only final polish; no experiments, endpoint recomputation, or model access.',
        'exact_two_micro_edits': 'PASS', 'appendix_visible_wording_and_input_order': 'UNCHANGED',
        'other_submission_files_byte_preserved': preserved,
        'protected_paths': 'UNCHANGED', 'official_preamble': 'BYTE_IDENTICAL',
        'source_identity_scan': 'PASS', 'modified_submission_files_sha256': source_hashes,
        'submission_audit': submission, 'visual_geometry_audit': geometry,
        'archive_vector_audit': archive,
        'baseline_warnings': warning_inventory(BUILD / 'polish_baseline/main.log'),
        'final_warnings': warning_inventory(BUILD / 'main.log'),
        'render_inventory': index,
        'manual_visual_record': 'COLING2027_FINAL_LAYOUT_AUDIT.md',
        'pdf_sha256': sha,
        'limits': 'Digital A4-size approximation, not a calibrated physical print certification. Official pubcheck is not zero-error.'}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
