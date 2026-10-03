import hashlib
import json
from pathlib import Path
import time
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from core import chunk_losses

ART = Path(__file__).resolve().parents[1]
ROOT = ART.parents[2]


def run():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    manifest = json.loads((ART / 'model_download_manifest.json').read_text())
    paths = {name: ROOT / 'outputs/phase4a_modern/model_cache' / name / v['revision'] for name, v in manifest.items()}
    ts = {name: AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True) for name, path in paths.items()}
    a, b = ts.values()
    assert a.is_fast and b.is_fast
    assert a.get_vocab() == b.get_vocab()
    assert a.special_tokens_map == b.special_tokens_map
    assert a.backend_tokenizer.to_str() == b.backend_tokenizer.to_str()
    audit = {'torch': torch.__version__, 'transformers': transformers.__version__,
             'bf16_supported': torch.cuda.is_bf16_supported(),
             'gpu': torch.cuda.get_device_name(), 'tokenizer_size': len(a),
             'tokenizer_sha256': hashlib.sha256(a.backend_tokenizer.to_str().encode()).hexdigest(),
             'special_tokens_map': a.special_tokens_map, 'cases': {}}
    student = AutoModelForCausalLM.from_pretrained(paths['SmolLM2-135M'], local_files_only=True, torch_dtype=torch.float32, attn_implementation='sdpa').cuda()
    teacher = AutoModelForCausalLM.from_pretrained(paths['SmolLM2-360M'], local_files_only=True, torch_dtype=torch.float32, attn_implementation='sdpa').cuda()
    audit['parameters'] = {'student': sum(p.numel() for p in student.parameters()), 'teacher': sum(p.numel() for p in teacher.parameters())}
    assert student.config.vocab_size == teacher.config.vocab_size == len(a)
    ids = a.encode('Once upon a time, a small bird learned to fly. The sky was blue.', add_special_tokens=False)
    x = torch.tensor((ids * 30)[:257], device='cuda').unsqueeze(0).repeat(4, 1)
    inputs, labels = x[:, :-1], x[:, 1:]
    for name, model, frozen in [('teacher_ce', teacher, None), ('student_kd_lambda5', student, teacher)]:
        model.train()
        if frozen is not None:
            frozen.eval()
            frozen.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.95), weight_decay=0.01)
        torch.cuda.reset_peak_memory_stats()
        timings, norms = [], []
        for step in range(4):
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(input_ids=inputs, use_cache=False).logits
                with torch.no_grad():
                    target = None if frozen is None else frozen(input_ids=inputs, use_cache=False).logits
                ce, kl, n = chunk_losses(output, labels, target)
                loss = ce + (5 * kl if target is not None else 0)
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step()
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
            norms.append(norm.item())
        audit['cases'][name] = {'microbatch': 4, 'seq_len': 256, 'updates': 4,
                               'ce': ce.item(), 'scaled_kl': kl.item(), 'gradient_norms': norms,
                               'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                               'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                               'tokens_per_second_after_warmup': n * 3 / sum(timings[1:]),
                               'seconds': timings}
        del optimizer
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    audit['decision'] = 'STAGE0_PASS'
    (ART / 'stage0_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    run()
