"""Freeze a single summarization task, tokenizer semantics and document groups.

No model or held-out prediction is evaluated. Full responses must fit; no silent
label truncation. Test records are written once and never used by the trainer.
"""
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
PHASE=ROOT/'research/experiments/phase5_ce_reference_distillation'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def normalized(text):
    return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',text).lower()))


def main():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    assets=json.loads((PHASE/'modern_assets.json').read_text())
    names=['Qwen/Qwen3-0.6B-Base','Qwen/Qwen3-1.7B-Base']
    toks=[AutoTokenizer.from_pretrained(ROOT/assets['models'][n]['local_dir'],local_files_only=True) for n in names]
    if toks[0].get_vocab()!=toks[1].get_vocab():raise ValueError('token ID semantics mismatch')
    if toks[0].special_tokens_map!=toks[1].special_tokens_map:raise ValueError('special tokens mismatch')
    for text in ['A short summary.','中文标点，数学 x² + y²。','<|im_start|>assistant\n', '\n  x\t42🦊']:
        if toks[0].encode(text)!=toks[1].encode(text):raise ValueError('encoding mismatch')
    tok=toks[0];end=tok.convert_tokens_to_ids('<|im_end|>')
    if end==tok.unk_token_id:raise ValueError('missing assistant end token')
    pad=tok.pad_token_id
    files=[ROOT/assets['data']['local_dir']/f for f in assets['data']['files']]
    raw=[]
    for f in files:raw.extend(pq.read_table(f).to_pylist())
    counts=collections.Counter();records=[];exact={};parent=[];sets=[];buckets=collections.defaultdict(list)
    rng=np.random.default_rng(20261125)
    # Fixed 32-hash, 8-band lexical near-duplicate screen; not a semantic guarantee.
    aa=rng.integers(1,2**31,size=32,dtype=np.uint64);bb=rng.integers(0,2**31,size=32,dtype=np.uint64)
    prime=np.uint64(4294967291)
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        i,j=find(i),find(j)
        if i!=j:parent[max(i,j)]=min(i,j)
    for rowid,row in enumerate(raw):
        msgs=row['messages'];kw=row.get('chat_template_kwargs') or {}
        if kw.get('enable_thinking') is not False:
            counts['excluded_not_explicit_no_think']+=1;continue
        if len(msgs)!=2 or [m['role'] for m in msgs]!=['user','assistant']:
            counts['excluded_non_single_turn']+=1;continue
        instruction=kw.get('custom_instructions') or 'Summarize the input text concisely.'
        user,answer=msgs[0]['content'].strip(),msgs[1]['content'].strip()
        if not user or not answer or '<think>' in answer:
            counts['excluded_empty_or_thinking']+=1;continue
        prompt=f'<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n'
        prefix=tok.encode(prompt,add_special_tokens=False);response=tok.encode(answer,add_special_tokens=False)+[end]
        if toks[1].encode(prompt,add_special_tokens=False)!=prefix or toks[1].encode(answer,add_special_tokens=False)+[end]!=response:
            raise ValueError('actual example encodings differ')
        if len(prefix)+len(response)>512 or len(response)<4:
            counts['excluded_length']+=1;continue
        canonical=normalized(user);key=hashlib.sha256(canonical.encode()).hexdigest()
        if key in exact:counts['exact_duplicate_dropped']+=1;continue
        i=len(records);exact[key]=i;parent.append(i)
        words=canonical.split();sh={hashlib.blake2b(' '.join(words[j:j+5]).encode(),digest_size=8).digest() for j in range(max(1,len(words)-4))}
        vals=np.array([int.from_bytes(s,'little')%int(prime) for s in sh],dtype=np.uint64)
        signature=((vals[:,None]*aa[None,:]+bb[None,:])%prime).min(0)
        candidates=set()
        keys=[]
        for b in range(8):
            k=(b,tuple(signature[b*4:b*4+4].tolist()));keys.append(k);candidates.update(buckets[k])
        for j in candidates:
            if len(sh&sets[j])/max(1,len(sh|sets[j]))>=.8:union(i,j);counts['near_duplicate_edges']+=1
        for k in keys:buckets[k].append(i)
        sets.append(sh)
        records.append(dict(row_id=rowid,document_hash=key,prompt=prompt,reference_response=answer,
                            tokens=prefix+response,prompt_length=len(prefix),targets=len(response)))
    groups=collections.defaultdict(list)
    for i in range(len(records)):groups[find(i)].append(i)
    splits={k:[] for k in ['preparation','continuation','control','dev','test']};group_manifest=[]
    for indices in groups.values():
        key=min(records[i]['document_hash'] for i in indices)
        v=int(hashlib.sha256(('phase5-sft-split:'+key).encode()).hexdigest()[:16],16)%1000
        split='preparation' if v<200 else 'continuation' if v<850 else 'control' if v<900 else 'dev' if v<950 else 'test'
        for i in indices:
            rec=records[i];rec['group_hash']=key;splits[split].append(rec)
        group_manifest.append(dict(group_hash=key,split=split,rows=[records[i]['row_id'] for i in indices]))
    out=ROOT/'outputs/phase5/modern_data';out.mkdir(parents=True,exist_ok=False)
    filemeta={}
    for split,recs in splits.items():
        recs.sort(key=lambda r:r['document_hash'])
        x=np.full((len(recs),512),pad,dtype=np.int32);labels=np.full_like(x,-100);attention=np.zeros_like(x,dtype=np.uint8)
        with (out/f'{split}.jsonl').open('w') as f:
            for i,r in enumerate(recs):
                n=len(r['tokens']);x[i,:n]=r['tokens'];attention[i,:n]=1
                labels[i,r['prompt_length']:n]=r['tokens'][r['prompt_length']:]
                f.write(json.dumps({k:v for k,v in r.items() if k!='tokens'},ensure_ascii=False)+'\n')
        np.savez(out/f'{split}.npz',input_ids=x,labels=labels,attention_mask=attention)
        filemeta[split]=dict(path=str((out/f'{split}.npz').relative_to(ROOT)),sha256=sha(out/f'{split}.npz'),
            records_path=str((out/f'{split}.jsonl').relative_to(ROOT)),records_sha256=sha(out/f'{split}.jsonl'),
            examples=len(recs),unique_response_targets=int((labels!=-100).sum()),groups=len({r['group_hash'] for r in recs}))
    (out/'groups.json').write_text(json.dumps(group_manifest,indent=2)+'\n')
    result=dict(dataset=assets['data']['repo'],revision=assets['data']['revision'],subset=assets['data']['split'],
        input_files={str(f.relative_to(ROOT)):sha(f) for f in files},raw_rows=len(raw),retained_rows=len(records),
        counts=dict(counts),files=filemeta,groups_sha256=sha(out/'groups.json'),sequence_length=512,
        tokenizer_id_semantics_equal=True,actual_encodings_all_equal=True,
        vocab_sha256=hashlib.sha256(json.dumps(tok.get_vocab(),sort_keys=True).encode()).hexdigest(),
        special_tokens=tok.special_tokens_map,pad_token_id=pad,response_end_token_id=end,
        masking='only full assistant reference response plus im_end; system/user/header/padding ignored',
        truncation='none; exclude examples exceeding 512 including full response',
        grouping='normalized source document plus 5-word shingle Jaccard>=0.8 for fixed MinHash-LSH candidates',
        grouping_limit='lexical screen is not an exhaustive semantic paraphrase detector',
        license='Apache-2.0 upstream Smol-summarize; synthetic reference responses, not guaranteed truths',
        license_source='https://huggingface.co/datasets/HuggingFaceTB/smoltalk/blob/main/README.md',
        test_access='preprocessing only; no model scores or selection',no_think_only=True)
    (PHASE/'modern_data.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
