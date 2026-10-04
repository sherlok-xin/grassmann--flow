"""Independent compact-artifact QA, no torch/model/test execution."""
import csv
import gzip
import json
import math
from protocol import *

m, inputs = verify_inputs()
for name in ['REPORT.md','GPT_HANDOFF.md','ce_lr_gate.csv','results_seed42.csv',
             'optimization_audit.csv','continuation_manifest.json','provenance.md']:
    assert (ART/name).is_file()
assert not (ART/'kd_protocol_frozen.json').exists()
seen = set(m['excluded_chunk_ids'])
order = m['ordered_chunk_ids']
assert len(seen)==8192 and len(set(order))==31250 and not seen.intersection(order)
assert all(0<=i<m['available_full_train_chunks'] for i in order)
assert len(order)*256==BUDGET
summaries = []
for arm in CE_ARMS:
    s = json.loads((ART/'raw'/(arm+'_summary.json')).read_text())
    original = json.loads((OUT/'runs'/arm/'summary.json').read_text())
    assert s==original
    with gzip.open(ART/'raw'/(arm+'_metrics.jsonl.gz'),'rb') as f:
        data = f.read()
    assert data==(OUT/'runs'/arm/'metrics.jsonl').read_bytes()
    metrics = [json.loads(line) for line in data.splitlines()]
    trained = metrics[1:]
    assert [r['step'] for r in metrics]==list(range(STEPS+1))
    assert metrics[0]['eligible_for_selection']
    assert all(r['tokens']==min(r['step']*8192,BUDGET) for r in metrics)
    assert [r['step'] for r in metrics if 'validation_nll' in r]==CHECK_STEPS
    best = select([r for r in metrics if 'validation_nll' in r])
    assert best['step']==s['best_validation_step']==977
    assert best['validation_nll']==s['best_validation_nll']
    assert sha(ROOT/s['selected_path'])==s['selected_sha256']
    assert s['manifest_sha256']==inputs['continuation_manifest_sha256']
    assert s['s0_sha256']==S0_SHA and s['order_sha256']==m['order_sha256_int64_le']
    assert sum(r['clipped'] for r in trained)/STEPS==s['clip_fraction']
    assert math.isclose(sum(r['gradient_norm_preclip'] for r in trained)/STEPS,s['gradient_norm_mean'],rel_tol=1e-12)
    assert all(math.isfinite(r['ce']) and math.isfinite(r['gradient_norm_preclip']) and not r['nonfinite'] for r in trained)
    assert not s['test_evaluated'] and s['lambda']==0.
    summaries.append(s)
best, improvement, passed = gate(summaries)
assert not passed and improvement<.01
d = json.loads((ART/'ce_gate_decision.json').read_text())
assert d['status']=='STOP_MODERN_CONTINUATION_NOT_ESTABLISHED'
assert d['ce_validation_improvement']==improvement and d['best_arm_by_validation']==best['arm']
with (ART/'results_seed42.csv').open() as f:
    rows = list(csv.DictReader(f))
assert len(rows)==4 and rows[2]['status']==rows[3]['status']=='NOT_RUN_CE_GATE_FAILED'
assert all(r['test_nll']==r['Gain_lambda_validation']==r['Gain_lambda_test']==r['initial_kd_ce_gradient_ratio']=='' for r in rows)
assert float(rows[1]['CE_improvement_validation'])==improvement
event = json.loads((ART/'gradient_event_audit.json').read_text())
with gzip.open(ART/'raw/ce_lr2e-5_metrics.jsonl.gz','rt') as f:
    raw_event = next(json.loads(line) for line in f if json.loads(line)['step']==event['step'])
assert all(event[k]==raw_event[k] for k in ['tokens','ce','gradient_norm_preclip','clipped','nonfinite'])
state = (ROOT/'research-state.yaml').read_text()
assert 'training_authorized: false' in state and 'status: experiments_frozen_phase4c_gate_failed' in state
result = {'status':'PASS','scope':'independent read-only compact artifact QA',
          'decision':d['status'],'best_lr':best['optimizer']['lr'],
          'best_validation_improvement':improvement,'formal_targets':3*BUDGET,
          'new_training':False,'new_test_forward':False,'source_event_verified':True,
          'raw_archive_equals_original':True,'required_files_present':True,'stop_state_verified':True}
(ART/'final_artifact_qa.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
