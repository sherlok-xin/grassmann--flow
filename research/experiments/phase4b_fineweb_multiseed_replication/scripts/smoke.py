"""Disposable full-batch FineWeb KD smoke; no state saving or test access."""
import json
import time
import numpy as np
import torch
from train import ART, OUT, load, sha, stream
from core import chunk_losses, packed_batch, schedule

torch.set_num_threads(4)
torch.manual_seed(123)
teacher_state = OUT / 'runs/fineweb/teacher/selected.pt'
before = sha(teacher_state)
model = load('SmolLM2-135M').train()
teacher = load('SmolLM2-360M', teacher_state).eval().requires_grad_(False)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(.9,.95), weight_decay=.01, eps=1e-8)
tokens = stream('fineweb', 'train')
order = np.random.default_rng(4242).permutation((len(tokens)-1)//256)
started = time.perf_counter()
records = []
torch.cuda.reset_peak_memory_stats()
for step in range(2):
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = 5e-5 * schedule(step, 1221)
    ce_total, kl_total = 0., 0.
    for begin in range(0,32,4):
        x,y = packed_batch(tokens, order[step*32+begin:step*32+begin+4])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids=x.cuda(), use_cache=False).logits
            with torch.no_grad():
                target = teacher(input_ids=x.cuda(), use_cache=False).logits
            ce,kl,n = chunk_losses(logits,y.cuda(),target)
        loss = (ce+5*kl)/8
        assert torch.isfinite(loss)
        loss.backward()
        ce_total += ce.item()/8
        kl_total += kl.item()/8
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True).item()
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    records.append({'step':step+1,'ce':ce_total,'scaled_kl_T2':kl_total,'gradient_norm_preclip':norm})
assert sha(teacher_state) == before
result = {'status':'PASS','formal_result':False,'saved_model_state':False,'test_access':False,
          'teacher_sha256':before,'records':records,
          'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
          'tokens_per_second':16384/(time.perf_counter()-started)}
(ART/'smoke_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
