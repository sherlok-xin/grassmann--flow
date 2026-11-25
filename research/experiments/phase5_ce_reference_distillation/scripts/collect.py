"""Collect completed endpoints and cost/status only; never run models or choose tests."""
import csv
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[4]
PHASE=ROOT/'research/experiments/phase5_ce_reference_distillation'


def write(name,rows,fields):
    with (PHASE/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    rows=[];manifest=[]
    for family in ['runs','modern_runs']:
        for p in sorted((ROOT/'outputs/phase5'/family).glob('*/config.json')):
            c=json.loads(p.read_text());sp=p.parent/'summary.json';s=json.loads(sp.read_text()) if sp.exists() else {}
            modern=family=='modern_runs';identity=c.get('identity',{})
            base=dict(run=p.parent.name,status=s.get('status','INCOMPLETE'),domain='smol_summarize' if modern else c['domain'],
                role=c.get('role','continuation'),seed=c['seed'],method=c['mode'],lr=c['lr'],lam=c['lam'],temperature=c['temperature'],
                config_path=str(p.relative_to(ROOT)),summary_path=str(sp.relative_to(ROOT)) if sp.exists() else '',
                S0_sha256=identity.get('S0_sha256',identity.get('expected_student_sha256')),
                teacher_sha256=c.get('teacher_sha256',identity.get('teacher_sha256')),
                data_sha256=c['data_manifest_sha256'],code_sha256=c.get('runner_sha256',c.get('source_sha256')),
                git_commit=c['git_commit'],reference_run=c.get('reference_run'),
                scope='smoke' if 'smoke' in p.parent.name else 'screen' if 'screen' in p.parent.name else 'full')
            manifest.append(base)
            if s:
                rows.append({**base,**{k:s.get(k) for k in ['s0_dev_nll','selected_dev_nll','final_dev_nll','selected_step',
                    'teacher_dev_nll','teacher_residual','steps','targets','clip_fraction','preclip_norm_mean','update_norm_mean',
                    'nonfinite_fraction','mean_beta','gpu_hours','peak_memory_GiB','warning']},'test_evaluated':False})
    write('RUN_MANIFEST.csv',manifest,list(manifest[0]) if manifest else ['run','status'])
    write('RESULTS.csv',rows,list(rows[0]) if rows else ['run','status'])
    account=json.loads((ROOT/'outputs/phase5/accounting.json').read_text());cost=[]
    for name,r in account['runs'].items():
        cost.append(dict(run=name,**r,current_elapsed_gpu_hours=(time.time()-r['started_unix'])/3600 if r['status']=='RUNNING' else r.get('gpu_hours'),
            includes='allocated GPU wall time including loading evaluation cache snapshots and contention'))
    for p in sorted(PHASE.glob('smoke*.json')):
        s=json.loads(p.read_text())
        if 'total_wall_seconds' in s:
            cost.append(dict(run=p.stem,status=s['status'],gpu=s['gpu'],gpu_hours=s['total_wall_seconds']/3600,
                current_elapsed_gpu_hours=s['total_wall_seconds']/3600,includes='bootstrap smoke including model/data setup; reconciled reserve'))
    write('COSTS.csv',cost,['run','status','gpu','reservation','gpu_hours','current_elapsed_gpu_hours','started_unix','finished_unix','includes'])
    complete=sum(r.get('gpu_hours',0) for r in account['runs'].values())
    print(json.dumps(dict(completed_endpoints=len(rows),complete_or_failed_gpu_hours=complete,
                         additional_smoke_reserve=account['smoke_reserve_gpu_hours'],active=[r['run'] for r in cost if r['status']=='RUNNING'])))


if __name__=='__main__':main()
