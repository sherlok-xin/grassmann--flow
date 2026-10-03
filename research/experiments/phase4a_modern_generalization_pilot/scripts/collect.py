"""Export only complete paired endpoints plus compact raw reproducibility records."""
import csv
import gzip
import json
from pathlib import Path
import shutil
from train import ART, OUT


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    raw = ART / 'raw'
    raw.mkdir(exist_ok=True)
    results, opt, selected = [], [], []
    for dataset in ['tinystories', 'fineweb']:
        root = OUT / 'runs' / dataset
        result = json.loads((root / 'final_result.json').read_text())
        initial = json.loads((root / 'initialization_audit.json').read_text())
        shutil.copyfile(root / 'final_result.json', raw / (dataset + '_final_result.json'))
        shutil.copyfile(root / 'initialization_audit.json', raw / (dataset + '_initialization_audit.json'))
        shutil.copyfile(OUT / 'data' / dataset / 'document_order.jsonl.gz', raw / (dataset + '_document_order.jsonl.gz'))
        for arm in ['teacher', 's0', 'ce', 'kd1', 'kd5']:
            summary = json.loads((root / arm / 'summary.json').read_text())
            assert summary['consumed_target_tokens'] == 10_000_000
            shutil.copyfile(root / arm / 'summary.json', raw / (dataset + '_' + arm + '_summary.json'))
            with (root / arm / 'metrics.jsonl').open('rb') as source, gzip.open(raw / (dataset + '_' + arm + '_metrics.jsonl.gz'), 'wb') as target:
                shutil.copyfileobj(source, target)
            endpoint = result['arms'][arm]
            results.append({'dataset': dataset, 'seed': 42, 'arm': arm, 'lambda': summary['lambda'],
                            'train_target_tokens': summary['consumed_target_tokens'],
                            'validation_nll': endpoint['validation_nll'], 'test_nll': endpoint['test_nll'],
                            'test_ppl': endpoint['test_ppl'], 'gain_test_nll': endpoint.get('gain_test_nll', ''),
                            'gain_validation_nll': endpoint.get('gain_validation_nll', ''),
                            'clear_effect': endpoint.get('clear_effect', ''),
                            'best_validation_step': summary['best_validation_step'],
                            'best_validation_tokens': summary['best_validation_tokens'],
                            'teacher_residual_advantage_over_s0': initial['teacher_residual_advantage_over_s0'],
                            'numerically_valid': result['numerically_valid'], 'dataset_decision': result['decision']})
            opt.append({'dataset': dataset, 'seed': 42, 'arm': arm, 'lambda': summary['lambda'],
                        'clip_fraction': summary['clip_fraction'], 'nonfinite_fraction': summary['nonfinite_fraction'],
                        'overflow_fraction': summary['overflow_fraction'],
                        'gradient_norm_mean': summary['gradient_norm_mean'], 'gradient_norm_max': summary['gradient_norm_max'],
                        'gradient_norm_ce_initial': initial['gradient_norm_ce'] if arm in ['ce', 'kd1', 'kd5'] else '',
                        'gradient_norm_kd_initial_unweighted': initial['gradient_norm_kd_unweighted'] if arm in ['kd1', 'kd5'] else '',
                        'gradient_ratio_weighted_kd_over_ce_initial': summary['lambda'] * initial['gradient_ratio_kd_over_ce'] if arm in ['kd1', 'kd5'] else '',
                        'ce_mean': summary['ce_mean'], 'scaled_kl_mean_T2': summary['scaled_kl_mean'],
                        'peak_allocated_gib': summary['peak_allocated_gib'], 'wall_seconds': summary['wall_seconds'],
                        'clip_saturation_warning': summary['clip_fraction'] >= 0.95,
                        'overflow_note': summary['overflow_interpretation']})
            selected.append({'dataset': dataset, 'arm': arm, 'selected_sha256': summary['selected_sha256'],
                             's0_sha256': summary['s0_sha256'], 'teacher_sha256': summary['teacher_sha256'],
                             'training_order_sha256': summary['order_sha256'],
                             'source_state_path': str((root / arm / 'selected.pt').relative_to(OUT))})
    write_csv(ART / 'results_seed42.csv', results)
    write_csv(ART / 'optimization_audit.csv', opt)
    write_csv(ART / 'selected_state_manifest.csv', selected)
    decision = 'MODERN_REPLICATION_WORTHWHILE' if any(r['clear_effect'] is True for r in results) else 'STOP_MODERN_REPLICATION'
    (ART / 'decision.json').write_text(json.dumps({'decision': decision, 'replication_authorized': False, 'independent_training_seeds': [42]}, indent=2) + '\n')
    print(decision, flush=True)


if __name__ == '__main__':
    main()
