"""Disposable real-model updates and resource measurement; no endpoints/test."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time

def main():
    p=argparse.ArgumentParser(); p.add_argument('--gpu',type=int,required=True); p.add_argument('--steps',type=int,default=8)
    a=p.parse_args()
    if a.steps>32: raise ValueError('disposable smoke capped at 32 steps per arm')
    mem=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if mem>256: raise RuntimeError(f'GPU {a.gpu} has {mem} MiB in use; no launch')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    os.environ['OMP_NUM_THREADS']='4'
    import torch
    from legacy import context,logits,flat,batch,seed_all,ROOT
    from core import token_loss
    torch.set_num_threads(4)
    seed_all(42)
    start=time.monotonic()
    student,teacher,old,common,identity=context()
    data=old.TextDataset('train',**common)
    base={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
    rows=[]
    for mode in ('ce','kd','reference'):
        student.load_state_dict(base); student.train(); seed_all(42)
        optimizer=torch.optim.AdamW(student.parameters(),lr=1e-4,weight_decay=.01)
        reference=copy.deepcopy(student).eval() if mode=='reference' else None
        ropt=torch.optim.AdamW(reference.parameters(),lr=1e-4,weight_decay=.01) if reference is not None else None
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); tic=time.monotonic()
        norms=[]; updates=[]; betas=[]
        for step in range(a.steps):
            ids=list(range(step*32,(step+1)*32)); x=batch(data,ids)
            rng=torch.cuda.get_rng_state()
            before=[v.detach().clone() for v in student.parameters()]
            optimizer.zero_grad(set_to_none=True)
            for j in range(0,32,2):
                xx=x[j:j+2]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    with torch.no_grad():
                        zt=logits(teacher,xx) if mode!='ce' else None
                        zr=logits(reference,xx) if reference is not None else None
                    zs=logits(student,xx)
                sf,y=flat(zs,xx); tf=flat(zt,xx)[0] if zt is not None else None
                rp=flat(zr,xx)[0].float().log_softmax(-1).gather(1,y[:,None]).squeeze() if zr is not None else None
                loss,stats=token_loss(sf,tf,y,mode=mode,lam=1,reference_logp=rp)
                (loss.mean()/16).backward(); betas.append(float(stats['beta'].mean()))
                del zs,zt,zr,sf,tf,loss,stats,rp
            norm=float(torch.nn.utils.clip_grad_norm_(student.parameters(),1.))
            if not torch.isfinite(torch.tensor(norm)): raise FloatingPointError('nonfinite gradient')
            optimizer.step(); norms.append(norm)
            updates.append(float(sum((p.detach()-b).float().square().sum() for p,b in zip(student.parameters(),before)).sqrt()))
            del before
            if reference is not None:
                # Independent CE update; identical student dropout stream before this step.
                kd_rng=torch.cuda.get_rng_state(); torch.cuda.set_rng_state(rng)
                reference.train(); ropt.zero_grad(set_to_none=True)
                for j in range(0,32,2):
                    xx=x[j:j+2]
                    with torch.autocast('cuda',dtype=torch.bfloat16): zr=logits(reference,xx)
                    rf,y=flat(zr,xx)
                    (torch.nn.functional.cross_entropy(rf.float(),y)/16).backward()
                torch.nn.utils.clip_grad_norm_(reference.parameters(),1.); ropt.step(); reference.eval()
                torch.cuda.set_rng_state(kd_rng)
        torch.cuda.synchronize(); wall=time.monotonic()-tic
        row=dict(mode=mode,steps=a.steps,prediction_targets=a.steps*32*255,wall_seconds=wall,
                 target_tokens_per_second=a.steps*32*255/wall,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                 clip_fraction=sum(n>1 for n in norms)/len(norms),mean_preclip_norm=sum(norms)/len(norms),
                 mean_update_norm=sum(updates)/len(updates),mean_beta=sum(betas)/len(betas),nonfinite_fraction=0,
                 includes_online_CE_reference_cost=reference is not None)
        print(json.dumps(row),flush=True); rows.append(row)
        del optimizer,reference,ropt
    out=ROOT/'research/experiments/phase5_ce_reference_distillation/smoke_audit.json'
    out.write_text(json.dumps(dict(status='PASS',gpu=a.gpu,rows=rows,identity=identity,
                                  total_wall_seconds=time.monotonic()-start,uses_test=False,
                                  precision='FP32 weights, BF16 autocast, FP32 losses',checkpoints_written=False),indent=2)+'\n')
if __name__=='__main__': main()
