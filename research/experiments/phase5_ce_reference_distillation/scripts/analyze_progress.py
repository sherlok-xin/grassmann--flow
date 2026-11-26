"""Summarize completed dev screens and live full-run status without model access."""
import csv
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / 'research/experiments/phase5_ce_reference_distillation'
RUNS = ROOT / 'outputs/phase5/runs'


def write_csv(name, rows):
    if not rows:
        return
    with (PHASE / name).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    screen_rows = []
    for domain in ['wt2', 'ts']:
        selection_path = PHASE / f'{domain}_dev_selections.json'
        if not selection_path.exists():
            continue
        winners = json.loads(selection_path.read_text())['winners']
        ce = json.loads((RUNS / winners['ce']['name'] / 'summary.json').read_text())
        for method, winner in winners.items():
            if not winner['name'].startswith(domain + '_'):
                continue  # Imported WT2 settings are not TinyStories endpoints.
            summary = json.loads((RUNS / winner['name'] / 'summary.json').read_text())
            assert summary['status'] == 'COMPLETE'
            screen_rows.append(dict(domain=domain, seed=42, method=method, run=winner['name'],
                lr=winner['lr'], lam=winner['lam'], temperature=winner['temperature'],
                smoothing=winner.get('smoothing') if method == 'ls' else '',
                steps=summary['steps'], trained_dev_nll=summary['final_dev_nll'],
                selected_dev_nll=summary['selected_dev_nll'], selected_step=summary['selected_step'],
                gain_vs_selected_CE=ce['selected_dev_nll'] - summary['selected_dev_nll'],
                clip_fraction=summary['clip_fraction'], nonfinite_fraction=summary['nonfinite_fraction'],
                warning=summary.get('warning', ''), scope='single_seed_short_screen_dev_only'))
    write_csv('screen_summary.csv', screen_rows)
    full_rows = []
    for folder in sorted(RUNS.glob('*_s42_full_*')):
        config_path = folder / 'config.json'
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        summary_path = folder / 'summary.json'
        completed = summary_path.exists()
        summary = json.loads(summary_path.read_text()) if completed else {}
        selections = json.loads((folder / 'selection.json').read_text()) if (folder / 'selection.json').exists() else []
        last_step = summary.get('steps', 0)
        trace = folder / 'metrics.jsonl'
        if not completed and trace.exists():
            with trace.open('rb') as stream:
                stream.seek(max(0, trace.stat().st_size - 4096))
                for line in stream.read().splitlines():
                    try:
                        last_step = max(last_step, json.loads(line)['step'])
                    except (ValueError, KeyError):
                        continue
        full_rows.append(dict(run=folder.name, domain=config['domain'], method=config['mode'],
            status='COMPLETE' if completed else 'IN_PROGRESS_OR_INCOMPLETE',
            observed_step=last_step, total_steps=config['steps'],
            selected_dev_nll=summary.get('selected_dev_nll', ''),
            final_dev_nll=summary.get('final_dev_nll', ''),
            selected_step=summary.get('selected_step', ''),
            last_observed_dev_nll=selections[-1]['dev_nll'] if selections else '',
            endpoint_available=completed, snapshot_at=stamp))
    write_csv('full_pilot_status.csv', full_rows)
    lines = ['# Development screening and pilot status', '', f'Snapshot: {stamp}.', '',
             'These are seed-42 development results. They are not test endpoints or independent-seed confirmations.',
             'The frozen promotion rule chooses the lowest trained screen dev NLL; subsequent checkpoint selection includes S0.', '',
             '| Domain | Method | Trained screen dev NLL | Selected dev NLL | Gain vs selected CE | Clip fraction |',
             '|---|---|---:|---:|---:|---:|']
    for row in screen_rows:
        lines.append(f"| {row['domain']} | {row['method']} | {row['trained_dev_nll']:.6f} | "
                     f"{row['selected_dev_nll']:.6f} | {row['gain_vs_selected_CE']:+.6f} | {row['clip_fraction']:.4f} |")
    lines += ['', 'All method comparisons remain exploratory. A saturated arm retains `CLIP_SATURATION_WARNING`; '
              'finite clipping does not establish numerical failure or explain an outcome causally.', '',
              'In the completed WT2 short screen, candidate A is weaker than the selected ordinary KD configuration. '
              'The two completed WT2 CE trajectories also select S0. Therefore neither candidate superiority nor '
              'a continuously improving held-out CE reference is supported by these observations. '
              'TinyStories short CE improves over S0; its full-training direction remains to be established.', '',
              '| Full run | Status | Observed / total updates | Selected dev endpoint |',
              '|---|---|---:|---:|']
    for row in full_rows:
        value = f"{row['selected_dev_nll']:.6f}" if row['endpoint_available'] else 'pending'
        lines.append(f"| {row['run']} | {row['status']} | {row['observed_step']} / {row['total_steps']} | {value} |")
    lines += ['', 'The isolated retry of the full-state replay smoke completed; it checks optimizer/RNG replay and '
              'finite H=1/8 branches only. It is not a formal mechanism result. Priority ablations, full forks, '
              'online search and seed confirmations must be reported from their own completed artifacts.', '',
              'No test scores are read or selected by this script. Incomplete runs never receive a completed endpoint value.']
    (PHASE / 'SCREENING_AND_PILOT_STATUS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(snapshot=stamp, screen_winners=len(screen_rows), full_runs=len(full_rows),
                          full_completed=sum(row['endpoint_available'] for row in full_rows))))


if __name__ == '__main__':
    main()
