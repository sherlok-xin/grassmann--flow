"""Paired, response-masked SFT with pinned Qwen3 models and exact vocabulary KL.

Only Phase 5 output directories are writable. Test is never loaded here.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[4]
PHASE=ROOT/'research/experiments/phase5_ce_reference_distillation'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--name',required=True)
    p.add_argument('--role',choices=['teacher','preparation','continuation'],required=True)
    p.add_argument('--mode',choices=['ce','kd','reference','current','mean','permuted','ls','todi','wclip','atkd'],default='ce')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--targets',type=int,required=True)
    p.add_argument('--lr',type=float,required=True);p.add_argument('--lam',type=float,default=1.)
    p.add_argument('--temperature',type=float,default=1.);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--micro',type=int,default=1);p.add_argument('--s0-run');p.add_argument('--teacher-run')
    p.add_argument('--reference-run');p.add_argument('--cache-gold',action='store_true')
    p.add_argument('--eval-targets',type=int,default=500000);p.add_argument('--smoke-steps',type=int)
    p.add_argument('--max-hours',type=float,default=16.);p.add_argument('--student-lora',action='store_true')
    p.add_argument('--save-states',action='store_true')
    a=p.parse_args()
    used=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if used>256:raise RuntimeError('GPU is occupied')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    import random
    import numpy as np
    import torch
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig,get_peft_model,get_peft_model_state_dict
    from modern_checks import load_adapter_exact
    from core import GoldCache,token_loss,project,digest
    from budget import Budget
    from prepare_modern import sha
    torch.set_num_threads(4)
    if a.micro<1 or a.batch%a.micro:raise ValueError('micro must divide batch')
    if a.cache_gold and (a.mode!='ce' or a.role!='continuation'):raise ValueError('cache only the independent CE continuation')
    if a.role!='continuation' and a.mode!='ce':raise ValueError('preparation/teacher must use CE only')
    if a.targets<=0:raise ValueError('positive target budget required')
    out=ROOT/'outputs/phase5/modern_runs'/a.name;out.mkdir(parents=True,exist_ok=False)
    with Budget(a.name,a.gpu,a.max_hours) as budget:
        start=time.monotonic();assets=json.loads((PHASE/'modern_assets.json').read_text())
        manifest=json.loads((PHASE/'modern_data.json').read_text())
        def load_data(split):
            ent=manifest['files'][split];path=ROOT/ent['path']
            if sha(path)!=ent['sha256']:raise ValueError('SFT data hash mismatch')
            with np.load(path) as d:return {k:d[k] for k in d.files}
        train=load_data('continuation' if a.role=='continuation' else 'preparation')
        dev=load_data('dev')
        # Teacher uses only preparation documents; no hidden training on control/dev/test.
        nt=(train['labels'][:,1:]!=-100).sum(1);n=len(nt);plan=[];used_targets=0;epoch=0
        while used_targets<a.targets:
            perm=np.random.default_rng(20261125+a.seed+epoch).permutation(n)
            for i in range(0,n,a.batch):
                ids=perm[i:i+a.batch].tolist();keep=min(int(nt[ids].sum()),a.targets-used_targets)
                plan.append(dict(ids=ids,targets=keep,epoch=epoch));used_targets+=keep
                if used_targets>=a.targets:break
            epoch+=1
        if a.smoke_steps:plan=plan[:a.smoke_steps]
        total=len(plan);warmup=max(1,int(.05*total))
        def seed(seed):
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        seed(a.seed)
        lcfg=dict(r=16,lora_alpha=32,lora_dropout=0.,bias='none',task_type='CAUSAL_LM',
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
        is_lora=a.role=='teacher' or a.student_lora
        repo='Qwen/Qwen3-1.7B-Base' if a.role=='teacher' else 'Qwen/Qwen3-0.6B-Base'
        model=AutoModelForCausalLM.from_pretrained(ROOT/assets['models'][repo]['local_dir'],
            local_files_only=True,torch_dtype=torch.bfloat16 if is_lora else torch.float32,attn_implementation='sdpa').cuda()
        model.config.use_cache=False
        if is_lora:model=get_peft_model(model,LoraConfig(**lcfg))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        if is_lora:model.enable_input_require_grads()
        runs=ROOT/'outputs/phase5/modern_runs'
        s0_hash=assets['models'][repo]['revision']
        def read_weights(path):return torch.load(path,map_location='cpu',weights_only=True)
        def set_weights(weights):
            if is_lora:load_adapter_exact(model,weights)
            else:model.load_state_dict(weights,strict=True)
        def save_weights(path):
            weights=get_peft_model_state_dict(model) if is_lora else model.state_dict()
            torch.save(weights,path)
        if a.role=='continuation':
            if not a.s0_run:
                if not a.smoke_steps:raise ValueError('exact prepared S0 required')
            else:
                previous=json.loads((runs/a.s0_run/'summary.json').read_text())
                if previous['config']['student_lora']!=a.student_lora:raise ValueError('student trainable parameter set changed')
                init=runs/a.s0_run/'s0.pt';s0_hash=sha(init);set_weights(read_weights(init))
        teacher=None;teacher_hash=None
        if a.mode not in ['ce','ls']:
            if not a.teacher_run and not a.smoke_steps:raise ValueError('frozen adapted teacher required')
            teacher=AutoModelForCausalLM.from_pretrained(ROOT/assets['models']['Qwen/Qwen3-1.7B-Base']['local_dir'],
                local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
            teacher=get_peft_model(teacher,LoraConfig(**lcfg))
            if a.teacher_run:
                path=runs/a.teacher_run/'best.pt';load_adapter_exact(teacher,read_weights(path));teacher_hash=sha(path)
            else:teacher_hash='unadapted_resource_smoke:'+assets['models']['Qwen/Qwen3-1.7B-Base']['revision']
            teacher.eval().requires_grad_(False);teacher.config.use_cache=False
        seed(a.seed)
        params=[v for v in model.parameters() if v.requires_grad]
        opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=.01,betas=(.9,.999),eps=1e-8)
        identity=dict(seed=a.seed,S0_sha256=s0_hash,data_sha256=manifest['files']['continuation' if a.role=='continuation' else 'preparation']['sha256'],
            plan_sha256=digest(plan),total_steps=total,targets=sum(x['targets'] for x in plan),lr=a.lr,
            warmup=warmup,batch=a.batch,micro=a.micro,optimizer='AdamW .9 .999 eps1e-8 wd.01',
            precision='FP32 trainable student parameters BF16 autocast' if not is_lora else 'BF16 frozen base FP32 LoRA',
            clip=1.,reference_timing='preupdate_eval',trainable_lora=is_lora)
        config=dict(vars(a),identity=identity,teacher_sha256=teacher_hash,models=assets['models'],
            data_manifest_sha256=sha(PHASE/'modern_data.json'),lora=lcfg if is_lora else None,
            trainable_parameters=sum(v.numel() for v in params),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_sha256=sha(__file__),core_sha256=sha(Path(__file__).parent/'core.py'),
            adapter_loader_sha256=sha(Path(__file__).parent/'modern_checks.py'),
            steps=total,warmup_steps=warmup,test_access=False,selected_including_S0=True,
            response_mask='assistant response plus im_end only; final batch shortened to exact target budget',
            evaluation_split='four training probes only' if a.smoke_steps else 'selection dev')
        (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        (out/'plan.json').write_text(json.dumps(plan,separators=(',',':'))+'\n')
        cache=None
        if a.cache_gold:cache=GoldCache(out/'gold_cache',identity,write=True)
        elif a.mode in ['reference','mean','permuted']:
            ref=runs/a.reference_run
            summary=json.loads((ref/'summary.json').read_text())
            if summary['config']['mode']!='ce' or summary['status']!='COMPLETE':raise ValueError('reference not independent completed CE')
            cache=GoldCache(ref/'gold_cache',identity)
        def getbatch(data,ids,limit=None):
            result={k:torch.from_numpy(np.array(v[ids],dtype=np.int64)).cuda() for k,v in data.items()}
            if limit is not None:
                flat=result['labels'].flatten();pos=(flat!=-100).nonzero().flatten()
                flat[pos[limit:]]=-100
            return result
        def fwd(m,b):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                return m(input_ids=b['input_ids'],attention_mask=b['attention_mask'],use_cache=False).logits
        def valid(z,b):
            labels=b['labels'][:,1:];mask=labels!=-100
            return z[:,:-1][mask],labels[mask]
        def splitbatch(b,i):return {k:v[i:i+a.micro] for k,v in b.items()}
        @torch.no_grad()
        def evaluate(m,tag):
            was=m.training;m.eval();s=0.;count=0;rows=[]
            data=dev if not a.smoke_steps else {k:v[:4] for k,v in train.items()}
            for i in range(0,len(data['input_ids']),a.micro):
                b=getbatch(data,list(range(i,min(i+a.micro,len(data['input_ids'])))))
                z=fwd(m,b);lab=b['labels'][:,1:]
                loss=torch.nn.functional.cross_entropy(z[:,:-1].float().reshape(-1,z.size(-1)),lab.reshape(-1),ignore_index=-100,reduction='none').reshape(lab.shape)
                sums=loss.sum(1);counts=(lab!=-100).sum(1);s+=float(sums.sum());count+=int(counts.sum())
                rows.extend(dict(index=i+j,nll_sum=float(v),targets=int(c)) for j,(v,c) in enumerate(zip(sums,counts)))
            m.train(was);(out/f'dev_{tag}.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
            return s/count
        best=evaluate(model,'0');initial=best;best_step=0;selected=[dict(step=0,targets=0,dev_nll=best)]
        if not a.smoke_steps:save_weights(out/'best.pt')
        teacher_dev=evaluate(teacher,'teacher') if teacher is not None else None
        trained=0;clips=0;norm_sum=0.;update_sum=0.;nonfinite=0;next_eval=a.eval_targets;update_seconds=[]
        torch.cuda.reset_peak_memory_stats()
        def get_rng():return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
        with (out/'metrics.jsonl').open('w') as log:
            for step,entry in enumerate(plan):
                step_start=time.monotonic()
                budget.check();b=getbatch(train,entry['ids'],entry['targets']);ntarget=entry['targets']
                lr=a.lr*((step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/(total-warmup))))
                for group in opt.param_groups:group['lr']=lr
                rp=None
                if a.cache_gold or a.mode=='current':
                    model.eval();parts=[]
                    with torch.no_grad():
                        for i in range(0,len(entry['ids']),a.micro):
                            bb=splitbatch(b,i);z=fwd(model,bb);lab=bb['labels'][:,1:]
                            vals=z[:,:-1].float().log_softmax(-1).gather(-1,lab.clamp_min(0)[...,None]).squeeze(-1)
                            parts.append(vals)
                    rp=torch.cat(parts)
                    if a.cache_gold:cache.put(step,entry['ids'],b['labels'][:,1:],rp)
                elif cache is not None:rp=cache.get(step,entry['ids'],b['labels'][:,1:]).cuda()
                if step==0 and teacher is not None and a.mode in ['kd','reference','current','todi','wclip']:
                    saved_rng=get_rng();norms={};bb=splitbatch(b,0)
                    for component in ['ce','kd']:
                        random.setstate(saved_rng['python']);np.random.set_state(saved_rng['numpy'])
                        torch.set_rng_state(saved_rng['cpu']);torch.cuda.set_rng_state(saved_rng['cuda'])
                        model.train();opt.zero_grad(set_to_none=True)
                        sf,y=valid(fwd(model,bb),bb)
                        with torch.no_grad():tf=valid(fwd(teacher,bb),bb)[0]
                        mask=bb['labels'][:,1:]!=-100
                        probe_loss,stats=token_loss(sf,tf,y,mode=a.mode,lam=a.lam,temperature=a.temperature,
                            reference_logp=rp[:a.micro][mask] if rp is not None else None)
                        stats[component].mean().backward()
                        norms[component]=float(sum(v.grad.float().square().sum() for v in params if v.grad is not None).sqrt())
                        del sf,tf,stats,probe_loss
                    random.setstate(saved_rng['python']);np.random.set_state(saved_rng['numpy'])
                    torch.set_rng_state(saved_rng['cpu']);torch.cuda.set_rng_state(saved_rng['cuda'])
                    (out/'initial_gradient_ratio.json').write_text(json.dumps(dict(
                        scope='first microbatch valid response tokens; raw parameter gradients before clipping',
                        **norms,unweighted_kd_over_ce=norms['kd']/max(norms['ce'],1e-30),
                        weighted_kd_over_ce=a.lam*norms['kd']/max(norms['ce'],1e-30)),indent=2)+'\n')
                    opt.zero_grad(set_to_none=True)
                mean_beta=None;hard=None;weights=None
                if a.mode in ['mean','atkd']:
                    vals=[]
                    with torch.no_grad():
                        for i in range(0,len(entry['ids']),a.micro):
                            bb=splitbatch(b,i);tf,y=valid(fwd(teacher,bb),bb)
                            mask=bb['labels'][:,1:]!=-100
                            vals.append(project(tf,y,rp[i:i+a.micro][mask])[1] if a.mode=='mean' else
                                1-(tf.float()/a.temperature).softmax(-1).gather(1,y[:,None]).squeeze(1))
                    values=torch.cat(vals)
                    if a.mode=='mean':mean_beta=values.mean()
                    else:
                        hard=values>=values.median();weights=torch.where(hard,.8*len(values)/hard.sum().clamp_min(1),.2*len(values)/(~hard).sum().clamp_min(1))
                model.train();opt.zero_grad(set_to_none=True)
                # CPU copies avoid adding a full extra FP32 model to GPU memory.
                before=[v.detach().cpu().clone() for v in params]
                sums=dict(loss=0.,ce=0.,kd=0.,beta=0.);offset=0
                for i in range(0,len(entry['ids']),a.micro):
                    bb=splitbatch(b,i);zs=fwd(model,bb);sf,y=valid(zs,bb);mask=bb['labels'][:,1:]!=-100
                    with torch.no_grad():tf=valid(fwd(teacher,bb),bb)[0] if teacher is not None else None
                    rr=rp[i:i+a.micro][mask] if rp is not None else None
                    # Exact full-vocabulary KL, chunked by valid response positions.
                    pieces=[]
                    def loss_chunk(ss,tt,yy,rrr,hh,ww):
                        ll,st=token_loss(ss,tt,yy,mode=a.mode,lam=a.lam,temperature=a.temperature,
                            reference_logp=rrr,beta_mean=mean_beta,atkd_weights=(hh,ww) if hh is not None else None)
                        return ll,st['ce'],st['kd'],st['beta']
                    for j in range(0,len(y),32):
                        sl=slice(j,j+32);gs=slice(offset+j,offset+min(j+32,len(y)))
                        loss,ce,kd,beta=checkpoint(loss_chunk,sf[sl],tf[sl] if tf is not None else None,y[sl],
                            rr[sl] if rr is not None else None,hard[gs] if hard is not None else None,
                            weights[gs] if weights is not None else None,use_reentrant=False)
                        pieces.append(loss.sum()/ntarget)
                        sums['loss']+=float(loss.detach().sum())
                        for k,val in [('ce',ce),('kd',kd),('beta',beta)]:sums[k]+=float(val.detach().sum())
                    if pieces:torch.stack(pieces).sum().backward()
                    offset+=len(y)
                    del zs,sf,tf
                if offset!=ntarget:raise ValueError('response mask/target denominator mismatch')
                norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
                if not math.isfinite(norm):raise FloatingPointError('nonfinite modern gradient')
                opt.step();update=0.
                for v,old in zip(params,before):update+=float((v.detach()-old.to(v.device)).float().square().sum())
                update=math.sqrt(update);del before
                if not math.isfinite(update):raise FloatingPointError('nonfinite modern update')
                clips+=norm>1;norm_sum+=norm;update_sum+=update;trained+=ntarget
                duration=time.monotonic()-step_start;update_seconds.append(duration)
                row=dict(step=step+1,targets=trained,step_targets=ntarget,lr=lr,lam=a.lam,temperature=a.temperature,
                    preclip_norm=norm,clipped=norm>1,update_norm=update,nonfinite=0,
                    elapsed_seconds=time.monotonic()-start,update_seconds=duration,**{k:v/ntarget for k,v in sums.items()})
                log.write(json.dumps(row)+'\n')
                if (step+1)%20==0:log.flush();print(json.dumps({'name':a.name,**row}),flush=True)
                if a.save_states and step+1 in {warmup,total//2,max(1,total-32)} and not a.smoke_steps:
                    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=get_rng(),
                        scheduler=dict(completed_steps=step+1,total_steps=total,warmup=warmup,peak_lr=a.lr),
                        cursor=step+1,accumulation_position=0,scaler=None,config=config),out/f'state_{step+1}.pt')
                if trained>=next_eval or step+1==total:
                    val=evaluate(model,str(step+1));selected.append(dict(step=step+1,targets=trained,dev_nll=val))
                    if val<best:
                        best=val;best_step=step+1
                        if not a.smoke_steps:save_weights(out/'best.pt')
                    next_eval+=a.eval_targets
                    (out/'selection.json').write_text(json.dumps(selected,indent=2)+'\n')
                    print(json.dumps({'name':a.name,'selected_dev':best,'last_dev':val,'targets':trained}),flush=True)
        if a.role=='preparation' and not a.smoke_steps:save_weights(out/'s0.pt')
        wall=time.monotonic()-start
        summary=dict(status='COMPLETE',config=config,smoke=bool(a.smoke_steps),s0_dev_nll=initial,teacher_dev_nll=teacher_dev,
            teacher_residual=initial-teacher_dev if teacher_dev is not None else None,selected_dev_nll=best,
            final_dev_nll=selected[-1]['dev_nll'],selected_step=best_step,steps=total,targets=trained,
            clip_fraction=clips/total,preclip_norm_mean=norm_sum/total,update_norm_mean=update_sum/total,
            nonfinite_fraction=0.,wall_seconds=wall,gpu_hours=wall/3600,response_targets_per_second=trained/wall,
            update_seconds_mean=float(np.mean(update_seconds)),
            steady_response_targets_per_second=sum(x['targets'] for x in plan[len(plan)//2:])/sum(update_seconds[len(plan)//2:]),
            peak_memory_GiB=torch.cuda.max_memory_allocated()/2**30,unique_training_targets=int(nt.sum()),
            repeated_target_equivalents=trained/int(nt.sum()),warning='CLIP_SATURATION_WARNING' if clips/total>=.95 else '',
            selected_sha256=sha(out/'best.pt') if not a.smoke_steps else None,test_evaluated=False)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps({k:v for k,v in summary.items() if k!='config'}),flush=True)


if __name__=='__main__':main()
