"""Archive a failed CE gate without KD or test; never relax the gate."""
import csv
import json
from protocol import *

verify_inputs()
d = json.loads((ART/'ce_gate_decision.json').read_text())
assert d['status']=='STOP_MODERN_CONTINUATION_NOT_ESTABLISHED' and not d['ce_gate_passed']
assert not any((OUT/'runs'/arm).exists() for arm in ['kd025','kd1'])
base_row = {'seed':42,'arm':'S0','status':'FIXED_PREPARATION_STEP256',
            'lr':'','lambda':'','selected_step':0,'selected_preparation_step':256,
            'validation_nll':d['s0_validation_nll'],'test_nll':'',
            'CE_improvement_validation':'','CE_improvement_test':'',
            'Gain_lambda_validation':'','Gain_lambda_test':'',
            'teacher_residual_advantage_validation':d['teacher_residual_advantage_validation'],
            'teacher_residual_advantage_test':'','initial_kd_ce_gradient_ratio':'',
            'test_status':'NOT_EVALUATED_CE_GATE_FAILED','decision':d['status']}
rows = [base_row]
best = json.loads((OUT/'runs'/d['best_arm_by_validation']/'summary.json').read_text())
rows.append({**base_row,'arm':'CE_best_gate_candidate','status':'TRAINED_GATE_CANDIDATE_NOT_CONFIRMATORY_ENDPOINT',
             'lr':best['optimizer']['lr'],'lambda':0.,'selected_step':best['best_validation_step'],
             'selected_preparation_step':'','validation_nll':best['best_validation_nll'],
             'CE_improvement_validation':d['ce_validation_improvement']})
for arm, strength in [('KD025',.25),('KD1',1.)]:
    rows.append({**base_row,'arm':arm,'status':'NOT_RUN_CE_GATE_FAILED','lambda':strength,
                 'selected_step':'','selected_preparation_step':'','validation_nll':''})
with (ART/'results_seed42.csv').open('w',newline='') as f:
    w = csv.DictWriter(f,fieldnames=list(base_row),lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
for name in ['ce_launcher_registry.json','ce_launcher_complete.json']:
    (ART/'raw'/name).write_bytes((OUT/name).read_bytes())
audit = {'status':'PASS','decision':d['status'],'ce_gate_passed':False,
         'formal_runs':3,'targets_per_run':BUDGET,'total_formal_targets':3*BUDGET,
         'independent_seeds':1,'kd_runs':0,'new_test_forwards':0,
         'fixed_s0_provenance_verified':True,'disjoint_training_manifest_verified':True,
         'all_budgets_and_977_steps_complete':True,'all_selections_include_step0':True,
         'validation_only_lr_gate_verified':True,'selected_checkpoint_hashes_verified':True,
         'observed_nonfinite_fraction':0.,'observed_overflow_fraction':0.,
         'initial_kd_ce_ratio':'NOT_MEASURED_GATE_FAILED_NO_KD_PROBE',
         'protected_input_and_source_hashes_verified':True,
         's0_sha256':S0_SHA,'teacher_sha256':TEACHER_SHA,
         'manifest_sha256':manifest()[1]['continuation_manifest_sha256'],
         'protocol_commit':'dabd3f6e31e1a371cdf4a78ac7ca3c6bd9cfd65f',
         'stop':True,'no_additional_lr_budget_clip_seed_or_method':True}
(ART/'completed_run_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2),flush=True)
