import json
import time
import torch
from core import chunk_losses, packed_batch
from train import ART, load, stream

torch.set_num_threads(4)
torch.manual_seed(42)
model = load('SmolLM2-135M').train()
teacher = load('SmolLM2-360M').eval().requires_grad_(False)
tokens = stream('tinystories', 'train')
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.95), weight_decay=0.01)
torch.cuda.reset_peak_memory_stats()
started = time.perf_counter()
records = []
for step in range(3):
    optimizer.zero_grad(set_to_none=True)
    ce_total, kl_total, count = 0., 0., 0
    for begin in range(0, 32, 4):
        x, y = packed_batch(tokens, range(step * 32 + begin, step * 32 + begin + 4))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(input_ids=x.cuda(), use_cache=False).logits
            with torch.no_grad():
                target = teacher(input_ids=x.cuda(), use_cache=False).logits
            ce, kl, n = chunk_losses(output, y.cuda(), target)
        ((ce + 5 * kl) / 8).backward()
        ce_total += ce.item() / 8
        kl_total += kl.item() / 8
        count += n
    assert count == 8192
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True).item()
    optimizer.step()
    records.append({'step': step + 1, 'tokens': count, 'ce': ce_total, 'scaled_kl': kl_total, 'gradient_norm_preclip': norm})
torch.cuda.synchronize()
result = {'decision': 'FULLBATCH_SMOKE_PASS', 'formal_results': False, 'test_access': False,
          'saved_model_state': False, 'updates': records, 'global_batch': 32, 'microbatch': 4,
          'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
          'tokens_per_second': 24576 / (time.perf_counter() - started)}
(ART / 'fullbatch_smoke.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
