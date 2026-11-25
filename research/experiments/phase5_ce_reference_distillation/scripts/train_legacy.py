"""Phase5 paired continuation, scalar CE cache and complete AdamW state snapshots."""
import argparse
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--domain',choices=['wt2','ts'],default='wt2')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--name',required=True)
    p.add_argument('--mode',choices=['ce','kd','reference','current','fixed','mean','permuted','ls','atkd','todi','wclip'],required=True)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--lam',type=float,default=1.)
    p.add_argument('--temperature',type=float,default=1.);p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--micro',type=int,default=16);p.add_argument('--reference-run');p.add_argument('--cache-gold',action='store_true')
    p.add_argument('--max-hours',type=float,default=2.);p.add_argument('--smoothing',type=float,default=.1)
    p.add_argument('--step-limit',type=int,help='preregistered short-budget prefix only; never changes full defaults')
    a=p.parse_args()
    mem=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if mem>256:raise RuntimeError(f'GPU {a.gpu} occupied ({mem} MiB), deferring')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);os.environ['OMP_NUM_THREADS']='4'
    import copy
    import random
    import numpy as np
    import torch
    from legacy import context,logits,flat,seed_all,sha,ROOT
    from core import token_loss,project,GoldCache,digest
    torch.set_num_threads(4)
    if a.micro>32 or 32%a.micro:raise ValueError('microbatch must divide 32')
    if a.cache_gold and a.mode!='ce':raise ValueError('only independent CE may generate cache')
    if '/' in a.name or '..' in a.name:raise ValueError('invalid run name')
    root=ROOT/'outputs/phase5';out=root/'runs'/a.name
    out.mkdir(parents=True,exist_ok=False)
    # Advisory accounting lock is project-specific; no other job is touched.
    account=root/'accounting.json';lock=root/'accounting.lock'
    def change_account(update):
        with lock.open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            data=json.loads(account.read_text()) if account.exists() else {'ceiling':230,'smoke_reserve_gpu_hours':.25,'runs':{}}
            update(data)
            tmp=account.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(account)
    def reserve(data):
        committed=data['smoke_reserve_gpu_hours']+sum(v.get('gpu_hours',v['reservation']) for v in data['runs'].values())
        if committed+a.max_hours>data['ceiling']:raise RuntimeError('Phase5 total budget reservation would exceed 230 GPUh')
        data['runs'][a.name]={'reservation':a.max_hours,'status':'RUNNING','gpu':a.gpu,'started_unix':time.time(),'pid':os.getpid()}
    change_account(reserve);start=time.monotonic();status='FAILED'
    try:
        phase=ROOT/'research/experiments/phase5_ce_reference_distillation'
        manifest=json.loads((phase/f'data_{a.domain}.json').read_text())
        arrays={k:np.load(ROOT/v['path'],mmap_mode='r') for k,v in manifest['files'].items()}
        for v in manifest['files'].values():
            if sha(ROOT/v['path'])!=v['sha256']:raise ValueError('token data hash mismatch')
        student,teacher,old,common,identity=context(a.domain,a.seed,teacher_needed=a.mode not in {'ce','ls'})
        seed_all(a.seed)
        initial={k:v.detach().cpu().clone() for k,v in student.state_dict().items()}
        fixed=copy.deepcopy(student).eval().requires_grad_(False) if a.mode=='fixed' else None
        n=len(arrays['train']);spe=math.ceil(n/32);steps=spe*a.epochs
        if a.step_limit is not None:
            if a.step_limit<=0:raise ValueError('positive short-budget step limit required')
            steps=min(steps,a.step_limit)
        warmup=max(1,int(.05*steps))
        optimizer=torch.optim.AdamW(student.parameters(),lr=a.lr,weight_decay=.01,betas=(.9,.999),eps=1e-8)
        cache_identity=dict(seed=a.seed,student_hash=identity['expected_student_sha256'],
            data_hash=manifest['files']['train']['sha256'],steps=steps,lr=a.lr,weight_decay=.01,
            betas=[.9,.999],eps=1e-8,warmup_steps=warmup,batch=32,micro=a.micro,precision='BF16',
            order_seed=20261125+a.seed,dropout='train_for_update_eval_for_reference',reference_timing='preupdate')
        cache=None
        if a.cache_gold:cache=GoldCache(out/'gold_cache',cache_identity,write=True)
        elif a.mode in {'reference','mean','permuted'}:
            if not a.reference_run:raise ValueError('independent CE reference run required')
            cache=GoldCache(root/'runs'/a.reference_run/'gold_cache',cache_identity)
            summary=json.loads((root/'runs'/a.reference_run/'summary.json').read_text())
            if summary['status']!='COMPLETE' or summary['mode']!='ce':raise ValueError('reference is not completed pure CE')
        config=dict(vars(a),steps=steps,steps_per_epoch=spe,warmup_steps=warmup,identity=identity,
            cache_identity=cache_identity,data_manifest_sha256=sha(phase/f'data_{a.domain}.json'),
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            protocol_sha256=sha(phase/'PROTOCOL.yaml'),loss_source_sha256=sha(Path(__file__).parent/'core.py'),
            runner_sha256=sha(__file__),test_access=False,selection='dev NLL includes S0; earliest tie',
            optimizer='AdamW(.9,.999),eps1e-8,wd.01',clip=1.)
        (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        def xbatch(split,ix):return torch.from_numpy(np.array(arrays[split][ix],dtype=np.int64)).cuda()
        @torch.no_grad()
        def evaluate(model,split):
            was=model.training;model.eval();total=0.;count=0;docs=[]
            for j in range(0,len(arrays[split]),a.micro):
                x=xbatch(split,slice(j,j+a.micro))
                with torch.autocast('cuda',dtype=torch.bfloat16):z=logits(model,x)
                zz,y=flat(z,x);loss=torch.nn.functional.cross_entropy(zz.float(),y,reduction='none').reshape(len(x),-1)
                total+=float(loss.sum());count+=loss.numel();docs.extend(loss.mean(-1).cpu().tolist())
            model.train(was);return total/count,docs
        def get_rng():return {'python':random.getstate(),'numpy':np.random.get_state(),
                             'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}
        def save_state(path,completed):
            torch.save(dict(model=student.state_dict(),optimizer=optimizer.state_dict(),rng=get_rng(),
                scheduler={'completed_steps':completed,'total_steps':steps,'warmup':warmup,'peak_lr':a.lr},
                scaler=None,precision='BF16',cursor={'epoch':completed//spe,'batch':completed%spe},
                accumulation_position=0,config=config),path)
        initial_dev,_=evaluate(student,'dev');best=initial_dev;best_step=0;selected=[{'step':0,'dev_nll':best}]
        torch.save(initial,out/'best.pt')
        teacher_dev=evaluate(teacher,'dev')[0] if teacher is not None else None
        nonfinite=0;clipped=0;norm_sum=0.;update_sum=0.;total_targets=0;beta_sum=0.;beta_count=0
        milestones={max(1,warmup),steps//2,max(1,steps-128)}
        torch.cuda.reset_peak_memory_stats()
        with (out/'metrics.jsonl').open('w') as trace:
            for epoch in range(a.epochs):
                if epoch*spe>=steps:break
                perm=np.random.default_rng(20261125+a.seed+epoch).permutation(n)
                for bi in range(spe):
                    step=epoch*spe+bi
                    if step>=steps:break
                    if (time.monotonic()-start)/3600>a.max_hours:raise TimeoutError('run GPU-hour limit reached')
                    ids=perm[bi*32:(bi+1)*32].tolist();x=xbatch('train',ids);yall=x[:,1:];nt=yall.numel()
                    lr=a.lr*((step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/(steps-warmup))))
                    for g in optimizer.param_groups:g['lr']=lr
                    rp=None;batch_beta=None;hard=None;scale=None
                    # Cache uses eval-mode probabilities with no dropout RNG consumption.
                    if a.cache_gold or fixed is not None:
                        model=student if a.cache_gold else fixed;was=model.training;model.eval();vals=[]
                        with torch.no_grad():
                            for j in range(0,len(x),a.micro):
                                xx=x[j:j+a.micro]
                                with torch.autocast('cuda',dtype=torch.bfloat16):zz=logits(model,xx)
                                zf,y=flat(zz,xx);vals.append(zf.float().log_softmax(-1).gather(1,y[:,None]).reshape(len(xx),-1))
                        rp=torch.cat(vals);model.train(was)
                        if a.cache_gold:cache.put(step,ids,yall,rp)
                    elif cache is not None:rp=cache.get(step,ids,yall).cuda()
                    if a.mode in {'mean','atkd'}:
                        vals=[]
                        with torch.no_grad():
                            for j in range(0,len(x),a.micro):
                                xx=x[j:j+a.micro]
                                with torch.autocast('cuda',dtype=torch.bfloat16):zz=logits(teacher,xx)
                                zf,y=flat(zz,xx)
                                if a.mode=='mean':vals.append(project(zf,y,rp[j:j+a.micro].reshape(-1))[1])
                                else:vals.append(1-(zf.float()/a.temperature).softmax(-1).gather(1,y[:,None]).squeeze(-1))
                        values=torch.cat(vals)
                        if a.mode=='mean':batch_beta=values.mean()
                        else:
                            hard=values>=values.median();ne=(~hard).sum();nh=hard.sum()
                            # Separate group means, author alpha=.2. Empty easy group contributes zero.
                            scale=torch.where(hard,torch.tensor(.8*nt,device='cuda')/nh.clamp_min(1),torch.tensor(.2*nt,device='cuda')/ne.clamp_min(1))
                    student.train();optimizer.zero_grad(set_to_none=True)
                    before=[v.detach().clone() for v in student.parameters()]
                    sums={k:0. for k in ['loss','ce','kd','beta']}
                    for j in range(0,len(x),a.micro):
                        xx=x[j:j+a.micro]
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            with torch.no_grad():zt=logits(teacher,xx) if teacher is not None else None
                            zs=logits(student,xx)
                        sf,y=flat(zs,xx);tf=flat(zt,xx)[0] if zt is not None else None
                        sl=slice(j*255,(j+len(xx))*255)
                        loss,stats=token_loss(sf,tf,y,mode=a.mode,lam=a.lam,temperature=a.temperature,
                            reference_logp=rp[j:j+a.micro].reshape(-1) if rp is not None else None,
                            beta_mean=batch_beta,smoothing=a.smoothing,
                            atkd_weights=(hard[sl],scale[sl]) if hard is not None else None)
                        (loss.sum()/nt).backward();sums['loss']+=float(loss.detach().sum())
                        for k in stats:sums[k]+=float(stats[k].detach().sum())
                        del zs,zt,sf,tf,loss,stats
                    norm=float(torch.nn.utils.clip_grad_norm_(student.parameters(),1.))
                    if not math.isfinite(norm):nonfinite+=1;raise FloatingPointError('nonfinite gradient; no automatic retry')
                    optimizer.step()
                    update=float(sum((v.detach()-b).float().square().sum() for v,b in zip(student.parameters(),before)).sqrt())
                    del before
                    if not math.isfinite(update):raise FloatingPointError('nonfinite parameter update')
                    clipped+=norm>1;norm_sum+=norm;update_sum+=update;total_targets+=nt;beta_sum+=sums['beta'];beta_count+=nt
                    telemetry=dict(step=step+1,lr=lr,lam=a.lam,temperature=a.temperature,targets=nt,
                        preclip_norm=norm,clipped=norm>1,update_norm=update,nonfinite=0,
                        elapsed_seconds=time.monotonic()-start,**{k:v/nt for k,v in sums.items()})
                    trace.write(json.dumps(telemetry)+'\n')
                    if (step+1)%50==0:trace.flush();print(json.dumps({'run':a.name,**telemetry}),flush=True)
                    if step+1 in milestones:save_state(out/f'state_{step+1}.pt',step+1)
                completed=min((epoch+1)*spe,steps)
                val,_=evaluate(student,'dev');selected.append(dict(step=completed,dev_nll=val))
                print(json.dumps({'run':a.name,'epoch':epoch+1,'dev_nll':val,'best_including_s0':min(best,val)}),flush=True)
                if val<best:best=val;best_step=completed;torch.save(student.state_dict(),out/'best.pt')
                (out/'selection.json').write_text(json.dumps(selected,indent=2)+'\n')
        torch.save(student.state_dict(),out/'final.pt');save_state(out/'final_state.pt',steps)
        wall=time.monotonic()-start;status='COMPLETE'
        summary=dict(status=status,mode=a.mode,seed=a.seed,domain=a.domain,config=config,
            s0_dev_nll=initial_dev,selected_dev_nll=best,selected_step=best_step,final_dev_nll=selected[-1]['dev_nll'],
            teacher_dev_nll=teacher_dev,teacher_residual=initial_dev-teacher_dev if teacher_dev is not None else None,
            steps=steps,targets=total_targets,clip_fraction=clipped/steps,preclip_norm_mean=norm_sum/steps,
            update_norm_mean=update_sum/steps,nonfinite_fraction=nonfinite/steps,mean_beta=beta_sum/beta_count,
            wall_seconds=wall,gpu_hours=wall/3600,target_tokens_per_second=total_targets/wall,
            peak_memory_GiB=torch.cuda.max_memory_allocated()/2**30,
            warning='CLIP_SATURATION_WARNING' if clipped/steps>=.95 else '',test_evaluated=False,
            selected_sha256=sha(out/'best.pt'),reference_cost_included_in_this_run=a.cache_gold)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps({k:v for k,v in summary.items() if k!='config'}),flush=True)
    finally:
        elapsed=(time.monotonic()-start)/3600
        def finish(d):d['runs'][a.name].update(status=status,gpu_hours=elapsed,finished_unix=time.time())
        change_account(finish)

if __name__=='__main__':main()
