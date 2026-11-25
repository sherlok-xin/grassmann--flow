"""Reverse-audit the explicitly incomplete V3 draft without running models."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / 'research/experiments/phase5_ce_reference_distillation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    tex = ROOT / '论文投稿/manuscript_v3/main.tex'
    text = tex.read_text()
    checks = []

    def present(value, source, field, rendered=None):
        expected = str(value) if rendered is None else rendered
        if expected not in text:
            raise AssertionError(f'{field}: expected manuscript rendering {expected}')
        checks.append(dict(value=value, rendered=expected, source=source, field=field))

    ce_path = 'outputs/phase5/runs/wt2_s42_screen_ce0/summary.json'
    kd_path = 'outputs/phase5/runs/wt2_s42_screen_kd0/summary.json'
    ce = json.loads((ROOT / ce_path).read_text())
    kd = json.loads((ROOT / kd_path).read_text())
    present(ce['s0_dev_nll'], ce_path, 's0_dev_nll', f"{ce['s0_dev_nll']:.6f}")
    present(kd['final_dev_nll'], kd_path, 'final_dev_nll', f"{kd['final_dev_nll']:.6f}")
    present(kd['steps'], kd_path, 'steps')
    modern_path = 'research/experiments/phase5_ce_reference_distillation/modern_data.json'
    modern = json.loads((ROOT / modern_path).read_text())
    present(modern['retained_rows'], modern_path, 'retained_rows', f"{modern['retained_rows']:,}")
    for split in ['preparation', 'continuation']:
        value = modern['files'][split]['unique_response_targets']
        present(value, modern_path, split + '.unique_response_targets', f'{value:,}')
    old_path = 'research/arr_response_20261125/verification.json'
    old = json.loads((ROOT / old_path).read_text())['table9']
    for key in ['TP', 'FP', 'TN', 'FN']:
        present(old[key], old_path, key, f'{key}={old[key]}')
    for key in ['accuracy', 'auroc']:
        present(old[key], old_path, key, f'{old[key]:.6f}')
    assert old['balanced_accuracy'] == .5 and old['mcc'] == 0
    checks.append(dict(source=old_path, balanced_accuracy=.5, MCC=0,
                       convention='zero MCC when constant predictions make its denominator zero'))
    protected = json.loads((PHASE / 'protected_files.json').read_text())['sha256']
    changed = [path for path, digest in protected.items() if sha(ROOT / path) != digest]
    if changed:
        raise AssertionError(f'protected files changed: {changed}')
    log = (ROOT / 'outputs/manuscript_v3/main.log').read_text()
    if 'Overfull' in log or 'undefined' in log.lower():
        raise AssertionError('LaTeX overfull or unresolved reference warning')
    pages = int(re.search(r'\((\d+) pages,', log).group(1))
    sources = json.loads((ROOT / '论文投稿/manuscript_v3/figures/partial_wt2_screen_sources.json').read_text())
    for name in sources['runs']:
        result = json.loads((ROOT / 'outputs/phase5/runs' / name / 'summary.json').read_text())
        assert result['status'] == 'COMPLETE' and result['steps'] == 294 and result['config']['seed'] == 42
    result = dict(status='PASS_FOR_INCOMPLETE_DRAFT_ONLY', manuscript_sha256=sha(tex),
                  pdf_sha256=sha(ROOT / 'outputs/manuscript_v3/main.pdf'), pages=pages,
                  numeric_checks=checks, figure_completed_source_runs=len(sources['runs']),
                  protected_files_verified=len(protected), protected_mismatches=changed,
                  underfull_warnings=log.count('Underfull'), overfull_warnings=0,
                  pending='full experiments, endpoint tables, seed variability and final scientific manuscript')
    (PHASE / 'draft_qa.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'numeric_checks'}))


if __name__ == '__main__':
    main()
