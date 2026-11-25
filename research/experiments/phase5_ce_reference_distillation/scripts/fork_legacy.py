"""Real AdamW forks and a deliberately simple, budget-matched online search.

No test data. Snapshots include optimizer moments, RNG, schedule and cursor.
The CE fork is local to the selected state, not an independent CE trajectory.
"""
import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--source-run', required=True)
    p.add_argument('--reference-run', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--controller', action='store_true')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    used = int(subprocess.check_output(['nvidia-smi', f'--id={a.gpu}',
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    if used > 256:
        raise RuntimeError('GPU occupied; no launch')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    import numpy as np
    import random
    import torch
    from legacy import ROOT, context, logits, flat, sha
    from core import GoldCache, token_loss
    from budget import Budget
    torch.set_num_threads(4)
    phase = ROOT / 'research/experiments/phase5_ce_reference_distillation'
    runs = ROOT / 'outputs/phase5/runs'
    source = runs / a.source_run
    config = json.loads((source/'config.json').read_text())
    output = ROOT/'outputs/phase5/analyses'/a.name
    output.mkdir(parents=True, exist_ok=False)
    rows, diagnostics = [], []
    with Budget(a.name, a.gpu, .5 if a.smoke else (3 if a.controller else 2)) as budget:
        model, teacher, _, _, _ = context(config['domain'], config['seed'])
        opt = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(.9,.999), eps=1e-8, weight_decay=.01)
        manifest = json.loads((phase/f"data_{config['domain']}.json").read_text())
        data = {s: np.load(ROOT / v['path'], mmap_mode='r') for s,v in manifest['files'].items()}
        cache = GoldCache(runs/a.reference_run/'gold_cache', config['cache_identity'])
        micro = config['micro']; spe = config['steps_per_epoch']; total = config['steps']
        perms = {}
        def batch(step):
            ep, bi = divmod(step, spe)
            if ep not in perms: perms[ep] = np.random.default_rng(20261125+config['seed']+ep).permutation(len(data['train']))
            ids = perms[ep][bi*32:(bi+1)*32].tolist()
            return ids, torch.from_numpy(np.array(data['train'][ids], dtype=np.int64)).cuda()
        def rng():
            return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())
        def restore_rng(r):
            random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state(r['cuda'])
        def restore(s):
            model.load_state_dict(s['model']);opt.load_state_dict(copy.deepcopy(s['optimizer']));restore_rng(s['rng'])
            assert s['accumulation_position'] == 0 and s['scaler'] is None
        def capture(completed):
            return dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                optimizer=copy.deepcopy(opt.state_dict()),rng=rng(),scaler=None,accumulation_position=0,
                scheduler=dict(completed_steps=completed,total_steps=total,warmup=config['warmup_steps'],peak_lr=config['lr']),
                cursor=dict(epoch=completed//spe,batch=completed%spe),config=config)
        @torch.no_grad()
        def evaluate(split='control', count=None):
            model.eval();s=0.;nt=0
            arr=data[split]; n=len(arr) if count is None else min(count,len(arr))
            for i in range(0,n,micro):
                x=torch.from_numpy(np.array(arr[i:min(n,i+micro)],dtype=np.int64)).cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16): z=logits(model,x)
                zz,y=flat(z,x);s+=float(torch.nn.functional.cross_entropy(zz.float(),y,reduction='sum'));nt+=y.numel()
            return s/nt
        def update(step, mode, lam):
            budget.check(); ids,x=batch(step);nt=x[:,1:].numel()
            w=config['warmup_steps'];lr=config['lr']*((step+1)/w if step<w else .5*(1+math.cos(math.pi*(step-w)/(total-w))))
            for g in opt.param_groups:g['lr']=lr
            rp=cache.get(step,ids,x[:,1:]).cuda() if mode=='reference' else None
            model.train();opt.zero_grad(set_to_none=True)
            before=[v.detach().clone() for v in model.parameters()]
            for i in range(0,len(x),micro):
                xx=x[i:i+micro]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    with torch.no_grad():zt=logits(teacher,xx) if mode!='ce' else None
                    zs=logits(model,xx)
                sf,y=flat(zs,xx);tf=flat(zt,xx)[0] if zt is not None else None
                loss,_=token_loss(sf,tf,y,mode=mode,lam=lam,temperature=1,
                    reference_logp=rp[i:i+micro].reshape(-1) if rp is not None else None)
                (loss.sum()/nt).backward()
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
            if not math.isfinite(norm):raise FloatingPointError('nonfinite fork gradient')
            opt.step()
            un=float(sum((p.detach()-b).float().square().sum() for p,b in zip(model.parameters(),before)).sqrt())
            return dict(lr=lr,preclip_norm=norm,clipped=norm>1,update_norm=un,targets=nt)
        def probe(s):
            # Gradient diagnostics use deterministic eval mode, not training dropout.
            # Curvature is a symmetric finite difference along the normalized KD gradient;
            # it is explicitly a proxy, not an exact Hessian or a gate for training.
            restore(s);model.eval();params=list(model.parameters());step=s['scheduler']['completed_steps']
            _,x=batch(step);xx=x[:2]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                with torch.no_grad():zt=logits(teacher,xx)
                zs=logits(model,xx)
            sf,y=flat(zs,xx);tf=flat(zt,xx)[0]
            loss,stats=token_loss(sf,tf,y,mode='kd',lam=1.)
            gd=torch.autograd.grad(stats['kd'].mean(),params)
            gd=[g.detach().clone() for g in gd];gn=float(sum(g.square().sum() for g in gd).sqrt())
            del loss,stats,zs,sf,zt,tf
            for n in ([2] if a.smoke else [2,16,64]):
                restore(s);model.eval();opt.zero_grad(set_to_none=True)
                for i in range(0,n,micro):
                    xx=torch.from_numpy(np.array(data['control'][i:min(n,i+micro)],dtype=np.int64)).cuda()
                    with torch.autocast('cuda',dtype=torch.bfloat16):zz=logits(model,xx)
                    zf,y=flat(zz,xx)
                    (torch.nn.functional.cross_entropy(zf.float(),y,reduction='sum')/(n*255)).backward()
                gv=[v.grad.detach().clone() if v.grad is not None else torch.zeros_like(v) for v in params]
                dot=float(sum(u.mul(v).sum() for u,v in zip(gv,gd)));vn=float(sum(g.square().sum() for g in gv).sqrt())
                f0=evaluate(count=n);eps=.01
                vals=[]
                for sign in [-1,1]:
                    model.load_state_dict(s['model'])
                    with torch.no_grad():
                        for v,g in zip(params,gd):v.add_(g,alpha=sign*eps/max(gn,1e-30))
                    vals.append(evaluate(count=n))
                diagnostics.append(dict(source=a.source_run,step=step,control_chunks=n,
                    train_probe_chunks=2,gradient_dot=dot,gradient_cosine=dot/max(gn*vn,1e-30),
                    kd_gradient_norm=gn,control_gradient_norm=vn,curvature_epsilon=eps,
                    normalized_direction_curvature=(sum(vals)-2*f0)/eps**2,
                    curvature_estimator='symmetric finite difference BF16 forward; descriptive only'))
            restore(s)
        if a.controller:
            # Begin at the first saved warmup state, not at an invented zero-moment state.
            # This is a warmup-prefix CE continuation and is labeled as such.
            state_path=source/f"state_{config['warmup_steps']}.pt"
            state=torch.load(state_path,map_location='cpu',weights_only=False)
            completed=state['scheduler']['completed_steps'];restore(state)
            H,K=8,256;decisions=[];selected={'step':completed,'dev_nll':evaluate('dev')}
            while completed<total:
                start_state=capture(completed); trials=[]
                h=min(H,total-completed)
                for mode,lam in [('ce',0.),('kd',.25),('kd',1.)]:
                    restore(start_state)
                    for j in range(h):update(completed+j,mode,lam)
                    trials.append((evaluate(),mode,lam,capture(completed+h)))
                winner=min(trials,key=lambda z:(z[0],z[2]));restore(winner[3]);completed+=h
                decisions.append(dict(step=completed,mode=winner[1],lam=winner[2],control_nll=winner[0],
                    all_trials=[dict(control_nll=t[0],mode=t[1],lam=t[2]) for t in trials],
                    committed='model_optimizer_rng_cursor_full_state',discarded_updates=2*h))
                for j in range(min(K-h,total-completed)):
                    update(completed,winner[1],winner[2]);completed+=1
                dev=evaluate('dev')
                if dev<selected['dev_nll']:
                    selected=dict(step=completed,dev_nll=dev);torch.save(model.state_dict(),output/'best.pt')
                (output/'decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
                if a.smoke:break
            result=dict(name=a.name,kind='B_identical_to_budget_matched_online_HPO',prefix_run=a.source_run,
                prefix_steps=config['warmup_steps'],selected=selected,completed_steps=completed,
                dev_schedule='each controller block; differs from main epoch schedule and not checkpoint-budget matched',
                independent_method_success_claim=False,test_access=False)
            (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        else:
            paths=sorted(source.glob('state_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
            if a.smoke:paths=paths[:1]
            for path in paths:
                s=torch.load(path,map_location='cpu',weights_only=False);start=s['scheduler']['completed_steps']
                probe(s)
                for mode,lam in [('ce',0.),('kd',1.),('reference',config['lam'] or 1.)]:
                    restore(s);norms=[];updates=[];clips=[]
                    hs=[1,8] if a.smoke else [1,8,32,128]
                    for h in range(1,min(max(hs),total-start)+1):
                        m=update(start+h-1,mode,lam);norms.append(m['preclip_norm']);updates.append(m['update_norm']);clips.append(m['clipped'])
                        if h in hs:
                            rows.append(dict(source=a.source_run,state_sha256=sha(path),step=start,mode=mode,lam=lam,H=h,
                                control_nll=evaluate(count=2 if a.smoke else None),lr=m['lr'],
                                preclip_norm=float(np.mean(norms)),update_norm=float(np.mean(updates)),clip_fraction=float(np.mean(clips))))
            for row in rows:
                ce=next(x for x in rows if x['step']==row['step'] and x['H']==row['H'] and x['mode']=='ce')
                row['A_H']=ce['control_nll']-row['control_nll']
            for filename,values in [('fork_results.csv',rows),('local_diagnostics.csv',diagnostics)]:
                with (output/filename).open('w') as f:
                    w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
            (output/'summary.json').write_text(json.dumps(dict(name=a.name,smoke=a.smoke,rows=len(rows),test_access=False,
                source_run=a.source_run,reference_run=a.reference_run,independence_unit='training family/seed, not probe rows'),indent=2)+'\n')
        print((output/'summary.json').read_text(),flush=True)


if __name__=='__main__':main()
