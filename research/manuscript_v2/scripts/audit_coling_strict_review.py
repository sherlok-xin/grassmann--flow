#!/usr/bin/env python3
"""Read-only final editorial whitelist, provenance, and existing PDF audits.

Never imports a model, reads checkpoints, or recomputes experiment endpoints.
Earlier exact-diff polish audits remain historical and are not relaxed here.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
QA = ROOT / 'research/manuscript_v2'
BUILD = ROOT / 'outputs/coling2027_arr'
PREFIX = '论文投稿/coling2027_arr/'
BASE = 'ab149eb20da37bdbe0d5862e67c9d6e1a8777bd6'

TEACHERS = r'''PTB and TinyStories teachers each combine a six-layer Transformer of width 256 with a six-layer Grassmann branch of width 240, reduced width 64, and lags $\{1,2,4\}$; both use dropout 0.1. Source preparation uses seed 42, 20 epochs, batch 32, sequence length 256, and learning rate $3\times10^{-4}$; all four sources select epoch 20 by validation. Fused teachers use ten joint epochs, freezing branches in epoch 1, branch learning rate $10^{-5}$, weight decay 0.01, AMP, initial $\alpha=0.5$, and late-$k=1$. Fusion-weight learning rates are $5\times10^{-3}$ and $10^{-2}$, respectively. Validation selects fused epochs 8 and 10, retaining learned $\alpha=0.470293$ and $0.599406$ for primary continuations; forced $\alpha=0.5$ applies to the WikiText-2 controls.'''

EDITS = [
    ('main.tex', 'A/C: avoid implying failure of every possible scalar predictor',
     'Together, these results support configuration-dependent distillability: whether KD helps varies with the teacher, warm-start student, and training configuration, rather than a single teacher-quality or compatibility statistic.',
     'Together, these results support configuration-dependent evaluation of warm-start KD against matched CE continuation, rather than inferring transfer from teacher likelihood ordering or the tested local diagnostics.'),
    ('sections/01_introduction.tex', 'A: state the exact weaker-source counterexample',
     'Global teacher likelihood alone therefore cannot determine distillability.',
     'A teacher likelihood advantage over the matched CE comparator is therefore not necessary for positive transfer.'),
    ('sections/04_experimental_setting.tex', 'A/C: expose primary hybrid architecture scope',
     'The 31.43M-parameter student uses a local GPT-2 tokenizer',
     'The 31.43M-parameter hybrid Transformer--Grassmann student uses a local GPT-2 tokenizer'),
    ('sections/02_related_work.tex', 'C: acknowledge closest existing negative-diagnostic overlap',
     'unlike our fixed-logit validation-gradient probes. We do not evaluate these adaptive algorithms.',
     'unlike our fixed-logit validation-gradient probes. It also reports weak validation-influence baselines for generative teacher selection; our probes target paired CE-relative corpus-likelihood endpoints. We do not evaluate these adaptive algorithms.'),
    ('appendix/architecture.tex', 'A/C: consistent primary hybrid identity',
     'The compact primary student uses six layers',
     'The compact primary hybrid student uses six layers'),
    ('appendix/architecture.tex', 'A: direct readers to authenticated teacher metadata',
     'Teacher dimensions and states follow their source configurations, not the student dimensions.',
     r'Primary teacher configurations are specified in Appendix~\ref{app:protocols}.'),
    ('appendix/protocols.tex', 'A: make PTB/TinyStories frozen teachers paper-level reproducible',
     r'\subsection{Teacher preparation and eligibility}' + '\n',
     r'\subsection{Teacher preparation and eligibility}' + '\n' + TEACHERS + '\n\n'),
    ('appendix/diagnostics_modern.tex', 'B: identify the pinned FineWeb revision and shard',
     r'FineWeb-Edu uses the first shard of the pinned \texttt{sample-10BT} revision.',
     r'FineWeb-Edu uses shard \texttt{sample/10BT/000\_00000.parquet} of \texttt{sample-10BT} at revision \texttt{87f09149ef4734204d70e}\allowbreak\texttt{d1d046ddc9ca3f2b8f9}.'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def main():
    tracked = git('ls-tree', '-r', '--name-only', '-z', BASE, '--', PREFIX).decode().split('\0')
    by_path, records = {}, []
    for name, reason, before, after in EDITS:
        path = PREFIX + name
        by_path.setdefault(path, []).append((before, after))
        records.append({'path': path, 'classification': 'SHOULD FIX',
                        'reviewer_concern': reason, 'before': before, 'after': after})
    preserved = []
    for path in filter(None, tracked):
        expected = git('show', f'{BASE}:{path}')
        if path in by_path:
            content = expected.decode()
            for before, after in by_path[path]:
                assert content.count(before) == 1, (path, before)
                content = content.replace(before, after, 1)
            assert (ROOT / path).read_bytes() == content.encode(), path
        else:
            assert (ROOT / path).read_bytes() == expected, path
            preserved.append(path)
    assert not git('diff', BASE, '--name-only', '--', 'research/final_evidence',
                   'research/experiments', 'datasets', 'checkpoints',
                   '论文投稿/manuscript_v2', '论文投稿/cac', 'research-state.yaml').strip()

    provenance_paths = []

    def read(path):
        p = ROOT / path
        provenance_paths.append({'path': path, 'sha256': sha(p)})
        return json.loads(p.read_text())

    sources = [
        ('20260318_094720_ptb_baseline_both', 'transformer', 256),
        ('20260318_102137_ptb_ws124_rd64_md240', 'grassmann', 240),
        ('20260322_095111_tinystories_transformer_300k_e20_split3', 'transformer', 256),
        ('20260322_094948_tinystories_ws124_rd64_md240_300k_e20_split3', 'grassmann', 240)]
    for run, branch, width in sources:
        path = f'outputs/experiments/{run}/'
        config = read(path + 'config.json')['config']
        for key, value in {'model_dim': width, 'num_layers': 6, 'dropout': .1,
                           'seed': 42, 'epochs': 20, 'batch_size': 32,
                           'max_seq_len': 256, 'lr': .0003}.items():
            assert config[key] == value, (run, key)
        if branch == 'grassmann':
            assert config['reduced_dim'] == 64 and config['window_sizes'] == '1,2,4'
        assert read(path + 'summary.json')[branch]['best_epoch'] == 20
    fused = [
        ('20260328_095917_ptb_latefusion_last1_joint', .005, 8, .47029295563697815),
        ('20260515_022710_ts_latefusion_last1_joint_e10', .01, 10, .5994059443473816)]
    for run, rate, epoch, alpha in fused:
        path = f'outputs/hybrid_experiments/{run}/'
        config = read(path + 'config.json')['config']
        for key, value in {'train_mode': 'joint', 'epochs': 10, 'freeze_branch_epochs': 1,
                           'lr': .00001, 'weight_decay': .01, 'amp': True,
                           'init_alpha': .5, 'late_k': 1, 'alpha_lr': rate}.items():
            assert config[key] == value, (run, key)
        summary = read(path + 'summary.json')['hybrid']
        assert summary['best_epoch'] == epoch and summary['best_alpha'] == alpha
    modern = read('research/experiments/phase4a_modern_generalization_pilot/data_manifest_fineweb.json')
    assert modern['source']['revision'] == '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
    assert modern['source']['source_shards'] == ['sample/10BT/000_00000.parquet']
    ptb = read('research/experiments/phase2_teacher_transfer_audit/raw/ptb/results.json')
    # Nested location is provenance rather than a newly recomputed teacher metric.
    assert '0.47029295563697815' in json.dumps(ptb)
    ts = read('research/experiments/phase2g_tinystories_negative_transfer/raw/formal/seed123/KD/config.json')
    assert ts['source_teacher']['alpha_override'] is None
    assert ts['source_teacher']['effective_alpha'] == [.5994059443473816]

    def audit(name):
        return json.loads(subprocess.check_output(['python3', str(QA / 'scripts' / name)],
                                                 cwd=ROOT, text=True))

    submission = audit('audit_coling_submission.py')
    geometry = audit('audit_coling_visual_geometry.py')
    assert submission['status'] == geometry['status'] == 'PASS'
    log = (BUILD / 'main.log').read_text()
    abstract = (ROOT / PREFIX / 'main.tex').read_text().split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
    print(json.dumps({
        'status': 'PASS', 'date': '2026-10-11', 'input_commit': BASE,
        'input_pdf_sha256': '0408342b232c6e3ba86deae8129cec552ed9e3e5a5c7b3ac4060a27a28792e20',
        'pdf_sha256': sha(BUILD / 'main.pdf'),
        'activity': 'Read-only editorial/provenance/PDF QA; no training, models, checkpoints, inference, endpoints, or experimental diagnostics.',
        'reviewer_scores_refer_to': 'Input manuscript; not upgraded after editorial corrections',
        'exact_editorial_whitelist': records,
        'unchanged_submission_files': preserved,
        'changed_submission_files': sorted(by_path),
        'figures_tables_bibliography_template_unchanged': True,
        'frozen_paths_unchanged': True,
        'provenance_artifacts': provenance_paths,
        'abstract_word_count': len(abstract.split()),
        'latex_notices': {'underfull_hbox': len(re.findall(r'Underfull \\hbox', log)),
                          'underfull_vbox': len(re.findall(r'Underfull \\vbox', log)),
                          'overfull': len(re.findall(r'Overfull', log))},
        'submission_audit': submission, 'geometry_audit': geometry,
        'historical_audits': 'Earlier final-polish exact-diff records are historical. No frozen assertions were relaxed.',
    }, indent=2))


if __name__ == '__main__':
    main()
