"""Finite modern pilot and dev-selected confirmations; no test selection or Git writes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
RUNS=ROOT/'outputs/phase5/modern_runs'


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--pilot-only',action='store_true');a=p.parse_args()
    from resources import wait_idle
    from core import digest
    progress=[]
    def remaining():
        d=json.loads((ROOT/'outputs/phase5/accounting.json').read_text())
        return d['ceiling']-d['smoke_reserve_gpu_hours']-sum(r.get('gpu_hours',r['reservation']) for r in d['runs'].values())
    def wait_file(path):
        start=time.monotonic()
        while not path.exists():
            if time.monotonic()-start>24*3600:raise TimeoutError('prerequisite not completed in 24h; no GPU reserved')
            time.sleep(30)
    def run(name,role,mode,lr,targets,seed=42,lam=1.,temperature=1.,s0=None,teacher=None,ref=None,cache=False,smoke=False,lora=False):
        expected=dict(name=name,role=role,mode=mode,lr=lr,targets=targets,seed=seed,lam=lam,temperature=temperature,student_lora=lora)
        path=RUNS/name/'summary.json'
        if path.exists():
            result=json.loads(path.read_text())
            if any(result['config'][k]!=v for k,v in expected.items()):raise ValueError('existing modern run mismatch')
            return result
        wait_idle(a.gpu)
        cmd=[sys.executable,'-u',str(HERE/'train_modern.py'),'--gpu',str(a.gpu),'--name',name,'--role',role,
            '--mode',mode,'--lr',str(lr),'--targets',str(targets),'--seed',str(seed),'--lam',str(lam),
            '--temperature',str(temperature),'--max-hours','.5' if smoke else '12']
        for flag,val in [('s0-run',s0),('teacher-run',teacher),('reference-run',ref)]:
            if val:cmd+=['--'+flag,val]
        if cache:cmd+=['--cache-gold']
        if smoke:cmd+=['--smoke-steps','8']
        if lora:cmd+=['--student-lora']
        logs=ROOT/'outputs/phase5/launcher_logs';logs.mkdir(parents=True,exist_ok=True)
        with (logs/(name+'.log')).open('a') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        result=json.loads(path.read_text())
        progress.append(dict(name=name,role=role,mode=mode,selected_dev_nll=result['selected_dev_nll'],
            final_dev_nll=result['final_dev_nll'],selected_step=result['selected_step'],gpu_hours=result['gpu_hours']))
        (PHASE/'modern_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        print(json.dumps(progress[-1]),flush=True)
        return result
    wait_file(PHASE/'modern_data.json')
    while not json.loads((PHASE/'modern_assets.json').read_text()).get('download_complete'):time.sleep(30)
    # Actual full-model memory failures require inspection, never an automatic
    # fallback triggered by another job entering the device.
    lora=False
    ce_smoke=run('qwen_resource_ce8','continuation','ce',1e-5,500000,lam=0,cache=True,smoke=True)
    kd_smoke=run('qwen_resource_kd8','continuation','kd',1e-5,500000,smoke=True)
    a_smoke=run('qwen_resource_A8','continuation','reference',1e-5,500000,ref='qwen_resource_ce8',smoke=True)
    run('qwen_resource_teacher8','teacher','ce',2e-4,500000,lam=0,smoke=True)
    rate=min(r['steady_response_targets_per_second'] for r in [ce_smoke,kd_smoke,a_smoke])
    choices=[n for n in [8000000,4000000,2000000] if 1.25*n/rate/3600<=8]
    if not choices:raise RuntimeError('2M resource projection exceeds gate; inspect budget before further runs')
    targets=choices[0]
    lock=dict(continuation_targets=targets,rate=rate,estimated_slowest_run_gpu_hours=1.25*targets/rate/3600,
        student_lora=lora,selection='precommitted throughput-only rule; no adaptation endpoints observed',
        smoke_names=['qwen_resource_ce8','qwen_resource_kd8','qwen_resource_A8'],test_access=False)
    (PHASE/'modern_protocol_selected.json').write_text(json.dumps(lock,indent=2)+'\n')
    teacher='qwen_teacher_s42';run(teacher,'teacher','ce',2e-4,1150454,lam=0)
    prep_targets=500000;s0='qwen_s42_S0';run(s0,'preparation','ce',2e-5,prep_targets,lam=0,lora=lora)
    ce_rows=[]
    for i,lr in enumerate([1e-6,5e-6,1e-5,2e-5,3e-5,5e-5]):
        ce_rows.append(run(f'qwen_s42_screen_ce{i}','continuation','ce',lr,250000,lam=0,s0=s0,cache=True,lora=lora))
    choose=lambda rs:min(rs,key=lambda r:(r['final_dev_nll'],r['config']['lam'],r['config']['temperature'],r['config']['lr']))
    ce_short=choose(ce_rows);lr=ce_short['config']['lr']
    ce_name='qwen_s42_full_ce';ce=run(ce_name,'continuation','ce',lr,targets,lam=0,s0=s0,cache=True,lora=lora)
    def gate(result,name):
        root=RUNS/name;step=result['selected_step']
        x=json.loads((root/'dev_0.json').read_text());y=json.loads((root/f'dev_{step}.json').read_text())
        sums=np.array([u['nll_sum']-v['nll_sum'] for u,v in zip(x,y)]);counts=np.array([u['targets'] for u in x])
        rng=np.random.default_rng(20261125);boot=[]
        for _ in range(2000):
            ii=rng.integers(0,len(sums),len(sums));boot.append(sums[ii].sum()/counts[ii].sum())
        low,high=np.quantile(boot,[.025,.975]);gain=result['s0_dev_nll']-result['selected_dev_nll']
        return dict(pass_gate=bool(step>0 and gain/result['s0_dev_nll']>=.002 and low>0),
            improvement=gain,relative_improvement=gain/result['s0_dev_nll'],selected_step=step,
            bootstrap_interval=[float(low),float(high)],uncertainty='selected-dev descriptive only, not selection-adjusted',test_access=False)
    gates=[dict(S0=s0,CE=ce_name,**gate(ce,ce_name))]
    if not gates[-1]['pass_gate']:
        prep_targets=100000;s0='qwen_s42_S0_early';run(s0,'preparation','ce',2e-5,prep_targets,lam=0,lora=lora)
        ce_name='qwen_s42_full_ce_early';ce=run(ce_name,'continuation','ce',lr,targets,lam=0,s0=s0,cache=True,lora=lora)
        gates.append(dict(S0=s0,CE=ce_name,**gate(ce,ce_name)))
        if gates[-1]['pass_gate']:
            ce_short=run('qwen_s42_screen_ce_early','continuation','ce',lr,250000,lam=0,s0=s0,cache=True,lora=lora)
    (PHASE/'modern_headroom_gate.json').write_text(json.dumps(gates,indent=2)+'\n')
    if not gates[-1]['pass_gate']:
        (PHASE/'modern_decision.json').write_text(json.dumps({'decision':'CE_HEADROOM_NOT_ESTABLISHED_AFTER_ONE_FALLBACK','test_access':False},indent=2)+'\n')
        return
    kd=[]
    for lam in [.25,1,5]:
        for t in [1,2]:
            name=f'qwen_s42_screen_kd_l{lam}_t{t}'
            kd.append(run(name,'continuation','kd',lr,250000,lam=lam,temperature=t,s0=s0,teacher=teacher,lora=lora))
    candidates=[run(f'qwen_s42_screen_A_l{lam}','continuation','reference',lr,250000,lam=lam,s0=s0,
        teacher=teacher,ref=ce_short['config']['name'],lora=lora) for lam in [.25,1,5]]
    wait_file(PHASE/'wt2_dev_selections.json')
    wt=json.loads((PHASE/'wt2_dev_selections.json').read_text())['winners']
    lit=min(['atkd','todi','wclip'],key=lambda name:wt[name]['final_dev_nll'])
    literature=[run(f'qwen_s42_screen_{lit}_l{lam}','continuation',lit,lr,250000,lam=lam,s0=s0,teacher=teacher,lora=lora) for lam in [1,5]]
    winners={k:choose(v)['config'] for k,v in [('kd',kd),('reference',candidates),('literature',literature)]}
    selected=dict(S0=s0,preparation_targets=prep_targets,lr=lr,targets=targets,teacher=teacher,winners=winners,
        source='development selection only; no test',data_manifest_sha256=digest(json.loads((PHASE/'modern_data.json').read_text())))
    (PHASE/'modern_dev_selections.json').write_text(json.dumps(selected,indent=2)+'\n')
    full={}
    for key,w in winners.items():
        full[key]=run(f'qwen_s42_full_{key}','continuation',w['mode'],lr,targets,lam=w['lam'],temperature=w['temperature'],
            s0=s0,teacher=teacher,ref=ce_name if key=='reference' else None,lora=lora)
    improve=min([ce['selected_dev_nll'],full['kd']['selected_dev_nll'],full['literature']['selected_dev_nll']])-full['reference']['selected_dev_nll']
    confirm_A=improve/ce['selected_dev_nll']>=.002
    decision=dict(first_batch_complete=True,confirm_A=confirm_A,A_gain_over_strongest=improve,
                  test_access=False,independent_pretraining_seeds=False,remaining_gpu_hours=remaining())
    (PHASE/'modern_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    if a.pilot_only:return
    # Reserve legacy queue completion and evaluation; finite confirmations only.
    keys=['kd','literature']+(['reference'] if confirm_A else [])
    for seed in [123,456]:
        required=4+(len(keys)+1)*lock['estimated_slowest_run_gpu_hours']
        if remaining()<required+45:
            decision['confirmation_budget_stop_before_seed']=seed;break
        init=f'qwen_s{seed}_S0';run(init,'preparation','ce',2e-5,prep_targets,seed=seed,lam=0,lora=lora)
        cname=f'qwen_s{seed}_full_ce';run(cname,'continuation','ce',lr,targets,seed=seed,lam=0,s0=init,cache=confirm_A,lora=lora)
        for key in keys:
            w=winners[key];run(f'qwen_s{seed}_full_{key}','continuation',w['mode'],lr,targets,seed=seed,lam=w['lam'],temperature=w['temperature'],
                s0=init,teacher=teacher,ref=cname if key=='reference' else None,lora=lora)
    if remaining()>=2*lock['estimated_slowest_run_gpu_hours']+50:
        teacher2='qwen_teacher_s123';run(teacher2,'teacher','ce',2e-4,1150454,seed=123,lam=0)
        for key in ['kd','reference' if confirm_A else 'literature']:
            w=winners[key];run(f'qwen_s42_teacher123_{key}','continuation',w['mode'],lr,targets,lam=w['lam'],temperature=w['temperature'],
                s0=s0,teacher=teacher2,ref=ce_name if key=='reference' else None,lora=lora)
    else:decision['second_teacher_budget_deferred']=True
    decision['queue_complete']=True;decision['remaining_gpu_hours']=remaining()
    (PHASE/'modern_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print('MODERN_BOUNDED_TRAINING_QUEUE_FINISHED; test and manuscript analysis remain',flush=True)


if __name__=='__main__':main()
