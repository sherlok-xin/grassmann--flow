"""Secondary mandatory source/weight controls, after the primary WT2 queue."""
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
    end=ROOT/'outputs/phase5/analyses/wt2_s42_online_search_B/summary.json'
    while not end.exists():time.sleep(30)
    winners=json.loads((PHASE/'wt2_dev_selections.json').read_text())['winners']
    w=winners['reference'];runs=ROOT/'outputs/phase5/runs'
    ref='wt2_s42_full_ce' if winners['ce']['lr']==1e-4 else 'wt2_s42_full_reference_ce'
    def run(name,mode,lam,epochs=10,reference=None,scale=1.,lr=1e-4):
        if (runs/name/'summary.json').exists():return
        wait_idle(a.gpu)
        cmd=[sys.executable,'-u',str(HERE/'train_legacy.py'),'--gpu',str(a.gpu),'--name',name,
            '--mode',mode,'--lr',str(lr),'--lam',str(lam),'--epochs',str(epochs),'--temperature','1',
            '--loss-scale',str(scale),'--max-hours','2']
        if reference:cmd+=['--reference-run',reference]
        with (ROOT/'outputs/phase5/launcher_logs'/(name+'.log')).open('a') as f:
            subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    # A completed one-epoch exact reference cache provides a full trajectory replay test.
    run('wt2_reference_only_replay_smoke','reference_only',w['lam'],epochs=1,reference='wt2_s42_screen_ce5')
    run('wt2_s42_ablation_fixed','fixed',w['lam'])
    run('wt2_s42_ablation_reference_only','reference_only',w['lam'],reference=ref)
    A=json.loads((runs/'wt2_s42_full_reference'/'summary.json').read_text())
    beta=A['mean_beta'];scale=1+w['lam']*beta;effective=w['lam']*(1-beta)/scale
    plan=dict(beta_from='completed A training trace, no held-out outcomes',mean_beta=beta,
        fixed_lambda=effective,matched_constant_CE_coefficient=scale,
        A_one_use_cost_gpu_hours=A['gpu_hours']+json.loads((runs/ref/'summary.json').read_text())['gpu_hours'])
    (PHASE/'wt2_coefficient_controls.json').write_text(json.dumps(plan,indent=2)+'\n')
    run('wt2_s42_fixed_effective_lambda','kd',effective)
    run('wt2_s42_fixed_effective_lambda_scaled','kd',effective,scale=scale)
    # Approximate total-GPUh matching is selected from measured cost, not endpoints.
    # Report actual achieved ratios; never claim exactly matched compute.
    for mode in ['ce','kd']:
        s=json.loads((runs/f'wt2_s42_full_{mode}'/'summary.json').read_text());cw=winners[mode]
        epochs=max(10,min(40,int(10*plan['A_one_use_cost_gpu_hours']/s['gpu_hours'])))
        if epochs==10:continue
        name=f'wt2_s42_cost_control_{mode}'
        if (runs/name/'summary.json').exists():continue
        wait_idle(a.gpu)
        cmd=[sys.executable,'-u',str(HERE/'train_legacy.py'),'--gpu',str(a.gpu),'--name',name,
            '--mode',mode,'--lr',str(cw['lr']),'--lam',str(cw['lam']),'--temperature',str(cw['temperature']),
            '--epochs',str(epochs),'--max-hours','2']
        plan[mode+'_cost_control_epochs']=epochs
        (PHASE/'wt2_coefficient_controls.json').write_text(json.dumps(plan,indent=2)+'\n')
        with (ROOT/'outputs/phase5/launcher_logs'/(name+'.log')).open('a') as f:
            subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    (PHASE/'wt2_followups_complete.json').write_text(json.dumps({'complete':True,'test_access':False})+'\n')


if __name__=='__main__':main()
