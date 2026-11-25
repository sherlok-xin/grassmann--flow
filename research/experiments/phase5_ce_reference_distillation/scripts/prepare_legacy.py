"""Derive Phase5 token files; never writes historical dataset/cache directories."""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
PHASE=ROOT/'research/experiments/phase5_ce_reference_distillation'
sys.path.insert(0,str(ROOT))

def htext(s): return hashlib.sha256(s.encode()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--domain',choices=['wt2','ts'],required=True);a=p.parse_args()
    os.environ['HF_DATASETS_OFFLINE']='1';os.environ['HF_HUB_OFFLINE']='1'
    from datasets import load_from_disk,disable_caching
    from transformers import GPT2Tokenizer
    import train_exp4_ddp as old
    disable_caching() # Any transient split indices go to temp, never the source dataset.
    out=ROOT/'outputs/phase5/data'/a.domain
    out.mkdir(parents=True,exist_ok=False)
    with open(ROOT/'research/experiments/phase3a_safe_lambda_diagnostic/condition_inventory.csv') as f:
        row=next(r for r in csv.DictReader(f) if r['condition_id']==('WT2_J_S42' if a.domain=='wt2' else 'TS_F_S42'))
    tok=GPT2Tokenizer.from_pretrained(ROOT/'gpt2_local',local_files_only=True)
    common=dict(tokenizer=tok,max_seq_len=256,dataset_name=row['dataset'],dataset_path=row['dataset_path'],
                text_field=row['text_field'],max_lines=int(row['max_lines']),encode_chars_per_batch=200000,
                tinystories_val_frac=.02,split_seed=42)
    train=old.TextDataset('train',**common)
    arr=np.asarray(train.tokens,dtype=np.int32).reshape(-1,256)
    np.save(out/'train.npy',arr)
    ds=load_from_disk(row['dataset_path'])
    val=ds['validation'] if a.domain=='wt2' else old.get_tinystories_split(ds,'validation')
    documents=[]
    if a.domain=='wt2':
        current=[]
        for text in val['text']:
            s=(text or '').strip()
            if re.match(r'^= [^=].* =$',s) and current:
                documents.append('\n'.join(current));current=[]
            if s: current.append(s)
        if current:documents.append('\n'.join(current))
    else:documents=[str(s).strip() for s in val['text'] if s and str(s).strip()]
    groups={'control':[],'dev':[]}; manifest=[];seen=set()
    for i,d in enumerate(documents):
        key=htext(d)
        if key in seen:continue
        seen.add(key)
        split='control' if int(htext('phase5:'+key)[:8],16)%5==0 else 'dev'
        ids=tok.encode(d,add_special_tokens=False)
        chunks=[ids[k:k+256] for k in range(0,len(ids)-255,256)]
        groups[split].extend(chunks)
        manifest.append(dict(document_id=i,text_sha256=key,split=split,chunks=len(chunks)))
    for split,chunks in groups.items(): np.save(out/f'{split}.npy',np.asarray(chunks,dtype=np.int32))
    from legacy import sha
    (out/'documents.json').write_text(json.dumps(manifest,separators=(',',':'))+'\n')
    payload=dict(domain=a.domain,train_stats=train.stats,validation_grouping='WT2 article headings / TinyStories complete stories',
                 control_dev_disjoint_documents=True,historical_S0_used_original_validation=True,
                 new_test_access=False,seq_len=256,files={s:dict(path=str((out/f'{s}.npy').relative_to(ROOT)),
                    sha256=sha(out/f'{s}.npy'),chunks=len(arr) if s=='train' else len(groups[s])) for s in ['train','control','dev']},
                 document_manifest=dict(path=str((out/'documents.json').relative_to(ROOT)),sha256=sha(out/'documents.json'),documents=len(manifest)))
    (PHASE/f'data_{a.domain}.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k!='documents'},indent=2),flush=True)

if __name__=='__main__':main()
