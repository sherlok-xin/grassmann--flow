"""Final response NLL and fixed greedy ROUGE-L, only after a checksum test lock.

The lock is an explicit artifact of completed development selection. This file
never changes training configuration, selects a checkpoint or trains a model.
"""
import argparse
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[4]
PHASE=ROOT/'research/experiments/phase5_ce_reference_distillation'


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--run',required=True)
    p.add_argument('--checkpoint',choices=['best.pt','s0.pt'],default='best.pt');a=p.parse_args()
    from resources import wait_idle
    from prepare_modern import sha
    lock=json.loads((PHASE/'modern_test_lock.json').read_text())
    key=a.run+'/'+a.checkpoint
    if key not in lock['checkpoints']:raise ValueError('checkpoint is not in the frozen development-selected test lock')
    runs=ROOT/'outputs/phase5/modern_runs';path=runs/a.run/a.checkpoint
    if sha(path)!=lock['checkpoints'][key]:raise ValueError('checkpoint hash mismatch')
    wait_idle(a.gpu);os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import LoraConfig,get_peft_model
    from modern_checks import load_adapter_exact
    from rouge_score import rouge_scorer
    from budget import Budget
    torch.set_num_threads(4)
    out=ROOT/'outputs/phase5/modern_evaluation'/(a.run+'_'+a.checkpoint.replace('.pt',''))
    out.mkdir(parents=True,exist_ok=False)
    with Budget('eval_'+out.name,a.gpu,2.) as budget:
        start=time.monotonic();c=json.loads((runs/a.run/'config.json').read_text())
        data=json.loads((PHASE/'modern_data.json').read_text());assets=json.loads((PHASE/'modern_assets.json').read_text())
        ent=data['files']['test']
        if sha(ROOT/ent['path'])!=ent['sha256'] or sha(ROOT/ent['records_path'])!=ent['records_sha256']:
            raise ValueError('test artifact changed')
        with np.load(ROOT/ent['path']) as archive:arrays={k:archive[k] for k in archive.files}
        docs=[json.loads(line) for line in (ROOT/ent['records_path']).read_text().splitlines()]
        repo='Qwen/Qwen3-1.7B-Base' if c['role']=='teacher' else 'Qwen/Qwen3-0.6B-Base'
        use_lora=c['role']=='teacher' or c['student_lora']
        model=AutoModelForCausalLM.from_pretrained(ROOT/assets['models'][repo]['local_dir'],local_files_only=True,
            torch_dtype=torch.bfloat16 if use_lora else torch.float32,attn_implementation='sdpa').cuda()
        weights=torch.load(path,map_location='cpu',weights_only=True)
        if use_lora:
            model=get_peft_model(model,LoraConfig(**c['lora']));load_adapter_exact(model,weights)
        else:model.load_state_dict(weights,strict=True)
        del weights;model.eval().requires_grad_(False);model.config.use_cache=True
        tok=AutoTokenizer.from_pretrained(ROOT/assets['models'][repo]['local_dir'],local_files_only=True)
        rows=[];total=0.;targets=0
        with torch.inference_mode():
            for i,doc in enumerate(docs):
                budget.check();n=int(arrays['attention_mask'][i].sum())
                # Keep the same padded 512 shape and precision as dev evaluation.
                x=torch.tensor(arrays['input_ids'][i:i+1],device='cuda',dtype=torch.long)
                att=torch.tensor(arrays['attention_mask'][i:i+1],device='cuda',dtype=torch.long)
                labels=torch.tensor(arrays['labels'][i:i+1,1:],device='cuda',dtype=torch.long)
                with torch.autocast('cuda',dtype=torch.bfloat16):z=model(input_ids=x,attention_mask=att,use_cache=False).logits
                loss=float(torch.nn.functional.cross_entropy(z[:,:-1].float().reshape(-1,z.size(-1)),labels.reshape(-1),ignore_index=-100,reduction='sum'))
                count=int((labels!=-100).sum());total+=loss;targets+=count
                rows.append(dict(document_hash=doc['document_hash'],nll_sum=loss,targets=count))
        (out/'test_nll_documents.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
        scorer=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True);scores=[]
        torch.manual_seed(20261125);torch.cuda.manual_seed_all(20261125)
        with (out/'generations.jsonl').open('w') as stream,torch.inference_mode():
            for i,doc in enumerate(docs[:512]):
                budget.check();k=doc['prompt_length']
                x=torch.tensor(arrays['input_ids'][i:i+1,:k],device='cuda',dtype=torch.long)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    ids=model.generate(input_ids=x,attention_mask=torch.ones_like(x),do_sample=False,num_beams=1,
                        max_new_tokens=128,eos_token_id=[data['response_end_token_id'],tok.eos_token_id],pad_token_id=tok.pad_token_id)
                text=tok.decode(ids[0,k:],skip_special_tokens=True).strip()
                score=scorer.score(doc['reference_response'],text)['rougeL'].fmeasure;scores.append(score)
                stream.write(json.dumps(dict(document_hash=doc['document_hash'],prediction=text,rougeL=score),ensure_ascii=False)+'\n')
        result=dict(run=a.run,checkpoint=a.checkpoint,checkpoint_sha256=lock['checkpoints'][key],test_nll=total/targets,
            response_targets=targets,test_documents=len(rows),rougeL=float(np.mean(scores)),rougeL_scale='0 to 1',
            generation_documents=len(scores),metric='rouge-score 0.1.2 rougeL Fmeasure with Porter stemming',
            decoding='greedy 128 max new tokens; im_end or endoftext stop',
            reference_responses='synthetic; ROUGE is not factual correctness or general instruction ability',
            wall_seconds=time.monotonic()-start,training_or_selection_performed=False,test_lock_sha256=sha(PHASE/'modern_test_lock.json'))
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
