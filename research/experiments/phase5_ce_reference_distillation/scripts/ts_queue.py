"""Bounded TinyStories screening, full CE/reference/KD/A pilot; dev-only promotion."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
PHASE=HERE.parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    from resources import wait_idle
    # Complete the missing disposable KD/A smoke before the formal screen.
    for mode,name in [('kd','ts_s42_smoke_kd8'),('reference','ts_s42_smoke_A8')]:
        if (ROOT/'outputs/phase5/runs'/name/'summary.json').exists():continue
        wait_idle(a.gpu)
        cmd=[sys.executable,'-u',str(HERE/'train_legacy.py'),'--gpu',str(a.gpu),'--domain','ts',
            '--seed','42','--name',name,'--mode',mode,'--lam','1','--lr','1e-4','--epochs','1',
            '--step-limit','8','--max-hours','.25']
        if mode=='reference':cmd+=['--reference-run','ts_s42_smoke_ce8']
        with (ROOT/'outputs/phase5/launcher_logs'/(name+'.log')).open('a') as f:
            subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    rows=[]
    def run(name,mode,lr,lam=1.,temperature=1.,short=True,cache=False,ref=None):
        path=ROOT/'outputs/phase5/runs'/name/'summary.json'
        spec=dict(name=name,mode=mode,lr=lr,lam=lam,temperature=temperature,epochs=1 if short else 10)
        if path.exists():
            result=json.loads(path.read_text())
            if any(result['config'][k]!=v for k,v in spec.items()):raise ValueError('existing run mismatch')
        else:
            wait_idle(a.gpu)
            cmd=[sys.executable,'-u',str(HERE/'train_legacy.py'),'--gpu',str(a.gpu),'--domain','ts',
                '--name',name,'--mode',mode,'--lr',str(lr),'--lam',str(lam),'--temperature',str(temperature),
                '--epochs',str(spec['epochs']),'--max-hours','1' if short else '18']
            if short:cmd+=['--step-limit','819']
            if cache:cmd+=['--cache-gold']
            if ref:cmd+=['--reference-run',ref]
            with (ROOT/'outputs/phase5/launcher_logs'/(name+'.log')).open('a') as f:
                subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
            result=json.loads(path.read_text())
        row={**spec,**{k:result[k] for k in ['selected_dev_nll','final_dev_nll','selected_step','gpu_hours','clip_fraction']}}
        rows.append(row);(PHASE/'ts_progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(row),flush=True);return row
    ce=[run(f'ts_s42_screen_ce{i}','ce',lr,lam=0) for i,lr in enumerate([1e-6,5e-6,1e-5,2e-5,5e-5,1e-4])]
    kd=[run(f'ts_s42_screen_kd{i}','kd',lr,lam,t) for i,(lr,lam,t) in enumerate(
        [(5e-5,.25,1),(1e-4,.25,1),(5e-5,1,1),(1e-4,1,1),(5e-5,5,2),(1e-4,5,2)])]
    best=lambda rs:min(rs,key=lambda r:(r['final_dev_nll'],r['lam'],r['temperature'],r['lr']))
    winners=dict(ce=best(ce),kd=best(kd))
    path=PHASE/'wt2_dev_selections.json'
    while not path.exists():time.sleep(30)
    winners['reference']=json.loads(path.read_text())['winners']['reference']
    (PHASE/'ts_dev_selections.json').write_text(json.dumps(dict(criterion='trained dev; A imported from WT2 dev-only screen',winners=winners),indent=2)+'\n')
    w=winners['ce'];run('ts_s42_full_ce','ce',w['lr'],lam=0,short=False,cache=w['lr']==1e-4)
    ref='ts_s42_full_ce'
    if w['lr']!=1e-4:
        ref='ts_s42_full_reference_ce';run(ref,'ce',1e-4,lam=0,short=False,cache=True)
    w=winners['kd'];run('ts_s42_full_kd','kd',w['lr'],w['lam'],w['temperature'],short=False)
    w=winners['reference'];run('ts_s42_full_reference','reference',1e-4,w['lam'],1.,short=False,ref=ref)
    print('TINYSTORIES_FIRST_BATCH_COMPLETE_NO_TEST_ACCESS',flush=True)


if __name__=='__main__':main()
