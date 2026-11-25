"""Read-only compatibility bridge to historical models, not their training loop."""
import csv
import hashlib
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def context(domain='wt2',seed=42,teacher_needed=True):
    import train_distill_hybrid_lite_from_latefusion_teacher_v2 as old
    from transformers import GPT2Tokenizer
    cond=f'WT2_J_S{seed}' if domain=='wt2' else f'TS_F_S{seed}'
    with open(ROOT/'research/experiments/phase3a_safe_lambda_diagnostic/condition_inventory.csv') as f:
        row=next(x for x in csv.DictReader(f) if x['condition_id']==cond)
    tokenizer=GPT2Tokenizer.from_pretrained(ROOT/'gpt2_local',local_files_only=True)
    roots=old.build_search_roots([])
    teacher=None
    if teacher_needed:
        td=Path(old.resolve_migrated_path(row['teacher_run_dir'],search_roots=roots,remaps=[],must_exist=True))
        teacher,ctx=old.load_teacher_and_context(td,len(tokenizer),roots,[],torch.device('cuda'),teacher_type=row['teacher_type'])
        old.apply_teacher_alpha_override(teacher,float(row['teacher_alpha_override']))
        teacher.eval().requires_grad_(False)
        teacher_hash=sha(ctx['hybrid_ckpt'])
    else: teacher_hash='not_loaded'
    args=SimpleNamespace(model_dim=224,num_layers=6,num_heads=8,reduced_dim=56,
                         window_sizes='1,2,4',dropout=.1,student_late_k=1)
    student=old.instantiate_student_from_args('hybrid_lite',len(tokenizer),256,args).cuda()
    info=old.maybe_load_student_init(student,row['student_run_dir'],roots,[],torch.device('cuda'),
                                   expected_sha256=row['expected_student_sha256'])
    dp=old.resolve_migrated_path(row['dataset_path'],search_roots=roots,remaps=[],must_exist=True)
    common=dict(tokenizer=tokenizer,max_seq_len=256,dataset_name=row['dataset'],dataset_path=dp,
                text_field=row['text_field'],max_lines=int(row['max_lines']),encode_chars_per_batch=200000,
                tinystories_val_frac=.02,split_seed=int(row['split_seed']))
    return student,teacher,old,common,dict(row,teacher_sha256=teacher_hash,student_checkpoint=info['student_init_checkpoint'])


def logits(model,x):
    return model(x,labels=None)[0]


def flat(logit,y):
    return logit[:,:-1].reshape(-1,logit.size(-1)),y[:,1:].reshape(-1)


def batch(dataset,indices):
    return torch.stack([dataset[int(i)][0] for i in indices]).cuda()
