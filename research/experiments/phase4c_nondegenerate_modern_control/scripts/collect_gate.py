"""Validation-only CE gate audit and compact telemetry; no model/test forward."""
import csv
import gzip
import json
import math
from protocol import *

def csv_write(name, rows):
    with (ART / name).open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

verify_inputs()
completion = json.loads((OUT/'ce_launcher_complete.json').read_text())
assert completion['status']=='PASS'
summaries, rows, optimization = [], [], []
raw = ART/'raw'
raw.mkdir(exist_ok=True)
for arm in CE_ARMS:
    run = OUT/'runs'/arm
    s = json.loads((run/'summary.json').read_text())
    metrics = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    assert s['status']=='TRAIN_COMPLETE_VALIDATION_SELECTED' and s['consumed_target_tokens']==BUDGET
    assert len(metrics)==STEPS+1 and metrics[-1]['tokens']==BUDGET
    assert not s['test_evaluated'] and s['nonfinite_fraction']==0
    vals = [r for r in metrics if 'validation_nll' in r]
    assert [r['step'] for r in vals]==CHECK_STEPS
    best = select(vals)
    assert best['step']==s['best_validation_step'] and best['validation_nll']==s['best_validation_nll']
    assert sha(ROOT/s['selected_path'])==s['selected_sha256']
    assert all(math.isfinite(r['gradient_norm_preclip']) and math.isfinite(r['ce']) for r in metrics[1:])
    assert sum(r['clipped'] for r in metrics[1:])/STEPS==s['clip_fraction']
    summaries.append(s)
    rows.append({'arm':arm,'lr':s['optimizer']['lr'],'s0_validation_nll':s['initial_validation_nll'],
                 'best_validation_nll':s['best_validation_nll'],'CE_improvement_validation':s['initial_validation_nll']-s['best_validation_nll'],
                 'selected_step':s['best_validation_step'],'selected_sha256':s['selected_sha256'],
                 'trained_selected':s['best_validation_step']>0,'gate_threshold':.01,
                 'passes_individual_gate':s['best_validation_step']>0 and s['initial_validation_nll']-s['best_validation_nll']>=.01,
                 'test_evaluated':False})
    optimization.append({'arm':arm,'lr':s['optimizer']['lr'],'lambda':0.,'steps':STEPS,'targets':BUDGET,
                         'selected_step':s['best_validation_step'],'clip_fraction':s['clip_fraction'],
                         'preclip_gradient_norm_mean':s['gradient_norm_mean'],'preclip_gradient_norm_max':s['gradient_norm_max'],
                         'nonfinite_fraction':s['nonfinite_fraction'],'overflow_fraction':s['overflow_fraction'],
                         'warning':s['warning']})
    (raw/(arm+'_summary.json')).write_text(json.dumps(s,indent=2)+'\n')
    with gzip.open(raw/(arm+'_metrics.jsonl.gz'),'wb') as f:
        f.write((run/'metrics.jsonl').read_bytes())
best, improvement, passed = gate(summaries)
decision = {'status':'CE_GATE_PASSED' if passed else 'STOP_MODERN_CONTINUATION_NOT_ESTABLISHED',
            'ce_gate_passed':passed,'best_lr_by_validation':best['optimizer']['lr'],
            'best_arm_by_validation':best['arm'],'s0_validation_nll':best['initial_validation_nll'],
            'best_ce_validation_nll':best['best_validation_nll'],'ce_validation_improvement':improvement,
            'selected_step':best['best_validation_step'],'test_evaluated':False,
            'teacher_validation_nll':2.74748592931419,
            'teacher_residual_advantage_validation':best['initial_validation_nll']-2.74748592931419,
            'initial_kd_ce_gradient_ratio':'NOT_MEASURED_BEFORE_CE_GATE',
            'manifest_sha256':manifest()[1]['continuation_manifest_sha256'],'s0_sha256':S0_SHA,
            'audit':'PASS'}
csv_write('ce_lr_gate.csv',rows)
csv_write('optimization_audit.csv',optimization)
(ART/'ce_gate_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
print(json.dumps(decision,indent=2),flush=True)
