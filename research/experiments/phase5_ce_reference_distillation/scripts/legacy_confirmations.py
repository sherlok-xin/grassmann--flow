"""Finite WT2 confirmations after first-batch pilots and modern priority work.

No new tuning. All configuration decisions use seed-42 development endpoints.
The seed-42 estimates remain exploratory, not an independent confirmation.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PHASE = HERE.parent
RUNS = ROOT / 'outputs/phase5/runs'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    from resources import wait_idle
    prerequisites = [PHASE / 'wt2_followups_complete.json',
                     RUNS / 'ts_s42_full_reference/summary.json']
    # This low-priority work cannot consume the modern confirmation budget first.
    while True:
        decision_path = PHASE / 'modern_decision.json'
        decision = json.loads(decision_path.read_text()) if decision_path.exists() else {}
        modern_finished = decision.get('queue_complete') or decision.get('decision') == 'CE_HEADROOM_NOT_ESTABLISHED_AFTER_ONE_FALLBACK'
        if modern_finished and all(p.exists() for p in prerequisites):
            break
        time.sleep(30)
    selected = json.loads((PHASE / 'wt2_dev_selections.json').read_text())['winners']
    full = {mode: json.loads((RUNS / f'wt2_s42_full_{mode}/summary.json').read_text())
            for mode in ['ce', 'kd', 'reference', 'atkd', 'todi', 'wclip']}
    literature = min(['atkd', 'todi', 'wclip'], key=lambda m: (full[m]['selected_dev_nll'], m))
    improvement = min(full[m]['selected_dev_nll'] for m in ['ce', 'kd', literature]) - full['reference']['selected_dev_nll']
    confirm_a = improvement >= .005
    ledger = json.loads((ROOT / 'outputs/phase5/accounting.json').read_text())
    remaining = ledger['ceiling'] - ledger['smoke_reserve_gpu_hours'] - sum(
        x.get('gpu_hours', x['reservation']) for x in ledger['runs'].values())
    # Conservative per-run limits, plus evaluation reserve. No speculative credit
    # from projected speed is used to authorize a partial confirmation matrix.
    methods = ['ce', 'kd', literature] + (['reference'] if confirm_a else [])
    extra_reference = confirm_a and selected['ce']['lr'] != selected['reference']['lr']
    required = 2 * (len(methods) + int(extra_reference)) * 2
    lock = dict(seeds=[123, 456], seed42_role='exploratory configuration selection',
                methods=methods, source='full seed42 development NLL only; no test',
                A_improvement=improvement, confirm_A=confirm_a, selected=selected,
                maximum_additional_gpu_hours=required, evaluation_reserve_gpu_hours=8,
                remaining_gpu_hours_at_lock=remaining, test_access=False)
    if remaining < required + 8:
        lock['status'] = 'BUDGET_DEFERRED_NO_CONFIRMATION_RUN_STARTED'
        (PHASE / 'wt2_confirmation_lock.json').write_text(json.dumps(lock, indent=2) + '\n')
        return
    lock['status'] = 'LOCKED_BEFORE_CONFIRMATION'
    (PHASE / 'wt2_confirmation_lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    for seed in [123, 456]:
        ce_name = f'wt2_s{seed}_full_ce'
        reference = ce_name if not extra_reference else f'wt2_s{seed}_full_reference_ce'
        specifications = [('ce', ce_name, selected['ce'])]
        if extra_reference:
            specifications.append(('ce', reference, dict(selected['reference'], lam=0.)))
        specifications += [(m, f'wt2_s{seed}_full_{m}', selected[m]) for m in methods if m != 'ce']
        for mode, name, config in specifications:
            path = RUNS / name / 'summary.json'
            if path.exists():
                previous = json.loads(path.read_text())
                expected = dict(seed=seed, mode=mode, lr=config['lr'], lam=config['lam'],
                                temperature=config['temperature'], epochs=10)
                if previous['status'] != 'COMPLETE' or any(previous['config'][k] != v for k, v in expected.items()):
                    raise ValueError('existing confirmation is not protocol-identical')
                continue
            wait_idle(args.gpu)
            cmd = [sys.executable, '-u', str(HERE / 'train_legacy.py'), '--gpu', str(args.gpu),
                   '--domain', 'wt2', '--seed', str(seed), '--name', name, '--mode', mode,
                   '--lr', str(config['lr']), '--lam', str(config['lam']),
                   '--temperature', str(config['temperature']), '--epochs', '10', '--max-hours', '2']
            if confirm_a and name == reference:
                cmd.append('--cache-gold')
            if mode == 'reference':
                cmd += ['--reference-run', reference]
            with (ROOT / 'outputs/phase5/launcher_logs' / (name + '.log')).open('a') as stream:
                subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    lock['status'] = 'COMPLETE_DEV_ENDPOINTS_NO_TEST'
    (PHASE / 'wt2_confirmation_status.json').write_text(json.dumps(lock, indent=2) + '\n')


if __name__ == '__main__':
    main()
