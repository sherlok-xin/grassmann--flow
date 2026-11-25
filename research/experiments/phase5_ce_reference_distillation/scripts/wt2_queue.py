"""Finite preregistered WT2 seed42 screen and dev-only full promotions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
PHASE=HERE.parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--screen-only',action='store_true');a=p.parse_args()
    from resources import wait_idle
    rows=[]
    def run(name,mode,lr=1e-4,lam=1.,temperature=1.,epochs=1,cache=False,ref=None,smoothing=.1):
        spec=dict(name=name,mode=mode,lr=lr,lam=lam,temperature=temperature,epochs=epochs,smoothing=smoothing)
        path=ROOT/'outputs/phase5/runs'/name/'summary.json'
        if path.exists():
            result=json.loads(path.read_text())
            if result['status']!='COMPLETE' or any(result['config'][k]!=v for k,v in spec.items()):
                raise ValueError('existing run mismatch')
        else:
            wait_idle(a.gpu)
            cmd=[sys.executable,str(HERE/'train_legacy.py'),'--gpu',str(a.gpu),'--name',name,'--mode',mode,
                 '--lr',str(lr),'--lam',str(lam),'--temperature',str(temperature),'--epochs',str(epochs),
                 '--smoothing',str(smoothing),'--max-hours','2']
            if cache:cmd+=['--cache-gold']
            if ref:cmd+=['--reference-run',ref]
            log=ROOT/'outputs/phase5/launcher_logs';log.mkdir(parents=True,exist_ok=True)
            with (log/(name+'.log')).open('a') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
            result=json.loads(path.read_text())
        row={**spec,**{k:result[k] for k in ['selected_dev_nll','final_dev_nll','selected_step','gpu_hours','clip_fraction']}}
        rows.append(row);(PHASE/'wt2_progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        return row
    ce=[]
    for i,lr in enumerate([1e-6,5e-6,1e-5,2e-5,5e-5,1e-4]):
        ce.append(run(f'wt2_s42_screen_ce{i}','ce',lr,lam=0,cache=lr==1e-4))
    kd=[]
    for i,(lr,lam,t) in enumerate([(5e-5,.25,1),(1e-4,.25,1),(5e-5,1,1),(1e-4,1,1),(5e-5,5,2),(1e-4,5,2)]):
        kd.append(run(f'wt2_s42_screen_kd{i}','kd',lr,lam,t))
    groups={'ce':ce,'kd':kd,'reference':[],'atkd':[],'todi':[],'wclip':[],'ls':[]}
    for i,lam in enumerate([.25,1,5]):groups['reference'].append(run(f'wt2_s42_screen_A{i}','reference',lam=lam,ref='wt2_s42_screen_ce5'))
    for mode in ['atkd','todi','wclip']:
        for i,lam in enumerate([1,5]):groups[mode].append(run(f'wt2_s42_screen_{mode}{i}',mode,lam=lam))
    for i,smooth in enumerate([.05,.1]):groups['ls'].append(run(f'wt2_s42_screen_ls{i}','ls',smoothing=smooth))
    winners={k:min(v,key=lambda x:(x['final_dev_nll'],x['lam'],x['temperature'],x['lr'])) for k,v in groups.items()}
    (PHASE/'wt2_dev_selections.json').write_text(json.dumps({'criterion':'trained screen dev only; no test', 'winners':winners},indent=2)+'\n')
    if a.screen_only:return
    def analysis(name,source,reference,smoke=False,controller=False):
        if (ROOT/'outputs/phase5/analyses'/name/'summary.json').exists():return
        wait_idle(a.gpu)
        cmd=[sys.executable,'-u',str(HERE/'fork_legacy.py'),'--gpu',str(a.gpu),
             '--source-run',source,'--reference-run',reference,'--name',name]
        if smoke:cmd.append('--smoke')
        if controller:cmd.append('--controller')
        with (ROOT/'outputs/phase5/launcher_logs'/(name+'.log')).open('a') as f:
            subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    analysis('wt2_state_replay_smoke_idle_retry','wt2_s42_screen_ce5','wt2_s42_screen_ce5',smoke=True)
    # Selection artifact is serialized before full runs. Initial protocol preauthorizes this deterministic promotion.
    ce_w=winners['ce']
    full_ce=run('wt2_s42_full_ce','ce',ce_w['lr'],lam=0,epochs=10,cache=ce_w['lr']==1e-4)
    refname='wt2_s42_full_ce'
    if ce_w['lr']!=1e-4:
        refname='wt2_s42_full_reference_ce'
        run(refname,'ce',lr=1e-4,lam=0,epochs=10,cache=True)
    for mode in ['kd','reference','atkd','todi','wclip','ls']:
        w=winners[mode]
        run('wt2_s42_full_'+mode,mode,w['lr'],w['lam'],w['temperature'],epochs=10,
            ref=refname if mode=='reference' else None,smoothing=w['smoothing'])
    w=winners['reference']
    for mode in ['current','mean','permuted']:
        run('wt2_s42_ablation_'+mode,mode,w['lr'],w['lam'],1,epochs=10,ref=refname if mode!='current' else None)
    analysis('wt2_s42_real_adamw_forks','wt2_s42_full_reference',refname)
    analysis('wt2_s42_online_search_B',refname,refname,controller=True)
    print('WT2_FULL_PILOT_AND_PRIORITY_ABLATIONS_COMPLETE; no test accessed',flush=True)

if __name__=='__main__':main()
