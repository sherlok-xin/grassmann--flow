"""Collect only complete paired results; separate step-zero diagnostic."""
import csv
import gzip
import json
from pathlib import Path
import shutil
import statistics
from selection import decisions

ART = Path(__file__).resolve().parents[1]
ROOT = ART.parents[2]
P4A = ROOT / 'research/experiments/phase4a_modern_generalization_pilot'
OLD = ROOT / 'outputs/phase4a_modern/runs/fineweb'
NEW = ROOT / 'outputs/phase4b_fineweb/runs'


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    raw = ART/'raw'
    raw.mkdir(exist_ok=True)
    teacher = json.loads((OLD/'final_result.json').read_text())['arms']['teacher']
    choices = json.loads((ART/'selection_with_s0.json').read_text())['choices']
    results, opt, diagnostic, selected = [], [], [], []
    for seed in [42,123,456]:
        root = OLD if seed==42 else NEW/str(seed)
        final = json.loads((root/'final_result.json').read_text())
        assert final['test_evaluation_completed'] and final['numerically_valid']
        initial = json.loads((root/'initialization_audit.json').read_text())
        summaries = {a:json.loads((root/a/'summary.json').read_text()) for a in ['s0','ce','kd1','kd5']}
        ep = final['arms']
        row = {'dataset':'FineWeb-Edu','seed':seed,'reused_phase4a':seed==42,
               's0_validation_nll':ep['s0']['validation_nll'],'s0_test_nll':ep['s0']['test_nll'],
               'ce_validation_nll':ep['ce']['validation_nll'],'ce_test_nll':ep['ce']['test_nll'],
               'kd1_validation_nll':ep['kd1']['validation_nll'],'kd1_test_nll':ep['kd1']['test_nll'],
               'kd5_validation_nll':ep['kd5']['validation_nll'],'kd5_test_nll':ep['kd5']['test_nll'],
               'Gain_1':ep['ce']['test_nll']-ep['kd1']['test_nll'],
               'Gain_5':ep['ce']['test_nll']-ep['kd5']['test_nll'],
               'gain1_validation':ep['ce']['validation_nll']-ep['kd1']['validation_nll'],
               'gain5_validation':ep['ce']['validation_nll']-ep['kd5']['validation_nll'],
               'teacher_residual_validation':ep['s0']['validation_nll']-teacher['validation_nll'],
               'teacher_residual_test':ep['s0']['test_nll']-teacher['test_nll'],
               's0_to_ce_validation_nll_change':ep['ce']['validation_nll']-ep['s0']['validation_nll'],
               's0_to_ce_test_nll_change':ep['ce']['test_nll']-ep['s0']['test_nll'],
               's0_selected_preparation_step':summaries['s0']['best_validation_step'],
               'ce_selected_step':summaries['ce']['best_validation_step'],
               'kd1_selected_step':summaries['kd1']['best_validation_step'],
               'kd5_selected_step':summaries['kd5']['best_validation_step']}
        row['gain1_validation_test_agrees']=row['Gain_1']*row['gain1_validation']>0
        row['gain5_validation_test_agrees']=row['Gain_5']*row['gain5_validation']>0
        results.append(row)
        diag_choices = {r['arm']:r for r in choices if r['seed']==seed}
        ce_diag = ep[diag_choices['ce']['selected_source']]['test_nll']
        for arm in ['ce','kd1','kd5']:
            choice = diag_choices[arm]
            value = ep[choice['selected_source']]
            diagnostic.append({'seed':seed,'arm':arm,'diagnostic_only':True,
                               'selected_source':choice['selected_source'],'selected_step':choice['selected_step'],
                               'selected_sha256':choice['selected_sha256'],
                               's0_preparation_selected_step':choice['s0_preparation_selected_step'],
                               's0_candidate_validation_nll':summaries['s0']['best_validation_nll'],
                               'trained_candidate_validation_nll':summaries[arm]['best_validation_nll'],
                               'selection_validation_nll':choice['selection_validation_nll'],
                               'test_nll':value['test_nll'],'test_ppl':value['test_ppl'],
                               'diagnostic_ce_minus_arm_test_nll':ce_diag-value['test_nll'],
                               'primary_endpoint_changed':False})
        for arm,s in summaries.items():
            assert s['consumed_target_tokens']==10_000_000
            opt.append({'seed':seed,'arm':arm,'reused_phase4a':seed==42,'lambda':s['lambda'],
                        'clip_fraction':s['clip_fraction'],
                        'warning':'CLIP_SATURATION_WARNING' if s['clip_fraction']>=.95 else '',
                        'gradient_norm_preclip_mean':s['gradient_norm_mean'],
                        'gradient_norm_preclip_max':s['gradient_norm_max'],
                        'nonfinite_fraction':s['nonfinite_fraction'],'overflow_fraction':s['overflow_fraction'],
                        'ce_mean':s['ce_mean'],'scaled_kl_T2_mean':s['scaled_kl_mean'],
                        'initial_ce_gradient_norm':initial['gradient_norm_ce'] if arm!='s0' else '',
                        'initial_unweighted_kd_gradient_norm':initial['gradient_norm_kd_unweighted'] if arm!='s0' else '',
                        'initial_weighted_kd_over_ce_ratio':s['lambda']*initial['gradient_ratio_kd_over_ce'] if arm in ['kd1','kd5'] else '',
                        'best_validation_step':s['best_validation_step'],'best_validation_tokens':s['best_validation_tokens'],
                        'target_tokens':s['consumed_target_tokens'],'peak_allocated_gib':s['peak_allocated_gib'],
                        'wall_seconds':s['wall_seconds'],'overflow_note':s['overflow_interpretation']})
            selected.append({'seed':seed,'arm':arm,'selected_sha256':s['selected_sha256'],
                             's0_sha256':s['s0_sha256'],'teacher_sha256':s['teacher_sha256'],
                             'order_seed':s['order_seed'],'order_sha256':s['order_sha256']})
            shutil.copyfile(root/arm/'summary.json',raw/f'seed{seed}_{arm}_summary.json')
            with (root/arm/'metrics.jsonl').open('rb') as source, gzip.open(raw/f'seed{seed}_{arm}_metrics.jsonl.gz','wb') as dest:
                shutil.copyfileobj(source,dest)
        shutil.copyfile(root/'final_result.json',raw/f'seed{seed}_final_result.json')
        shutil.copyfile(root/'initialization_audit.json',raw/f'seed{seed}_initialization_audit.json')
    stats=[]
    for metric in ['Gain_1','Gain_5','teacher_residual_validation','teacher_residual_test',
                   's0_to_ce_validation_nll_change','s0_to_ce_test_nll_change']:
        values=[r[metric] for r in results]
        stats.append({'metric':metric,'n':3,'mean':statistics.mean(values),'sample_sd':statistics.stdev(values),
                      'negative_signs':sum(v<0 for v in values),'positive_signs':sum(v>0 for v in values),
                      'zero_signs':sum(v==0 for v in values)})
    flags=decisions([r['Gain_1'] for r in results],[r['Gain_5'] for r in results],
                    [r['gain5_validation_test_agrees'] for r in results])
    decision={'decision':flags[-1],'flags':flags,'student_seeds':[42,123,456],
              'new_training_target_tokens':80_000_000,'new_formal_runs':8,
              'teacher_reused':True,'primary_step0_eligible':False,
              'best_including_s0_count':sum(r['selected_step']==0 for r in diagnostic),
              'best_including_s0_total_arms':9,'further_training_authorized':False}
    for name,rows in [('results_multiseed.csv',results),('optimization_audit.csv',opt),
                      ('best_including_s0.csv',diagnostic),('summary_statistics.csv',stats),
                      ('selected_state_manifest.csv',selected)]:
        write_csv(ART/name,rows)
    (ART/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(json.dumps({'decision':decision,'statistics':stats},indent=2),flush=True)


if __name__=='__main__':
    main()
