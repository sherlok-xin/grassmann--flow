"""Phase4C exact-S0 continuation; CE gate before any KD execution."""
import argparse
import importlib.util
import json
import math
import sys
import time
import numpy as np
import torch
from protocol import *

sys.path.insert(0, str(BASE / 'scripts'))
spec = importlib.util.spec_from_file_location('phase4a_train', BASE / 'scripts/train.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
from core import chunk_losses, packed_batch, schedule


def run(args):
    m, audit = verify_inputs()
    lr = args.lr
    kd_lambda = {'kd025': .25, 'kd1': 1.0}.get(args.arm, 0.)
    if kd_lambda:
        frozen = json.loads((ART / 'kd_protocol_frozen.json').read_text())
        assert frozen['ce_gate_passed'] and lr == frozen['selected_lr']
        assert frozen['s0_sha256'] == S0_SHA
        assert frozen['manifest_sha256'] == audit['continuation_manifest_sha256']
        assert args.freeze_commit and len(args.freeze_commit) == 40
    else:
        assert args.arm in CE_ARMS and lr == LRS[CE_ARMS.index(args.arm)]
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = base.load('SmolLM2-135M', S0)
    teacher = None if not kd_lambda else base.load('SmolLM2-360M', TEACHER).eval().requires_grad_(False)
    run_dir = OUT / 'runs' / args.arm
    if not args.smoke:
        run_dir.mkdir(parents=True, exist_ok=False)
    tokens, validation = base.stream('fineweb', 'train'), base.stream('fineweb', 'validation')
    order = np.array(m['ordered_chunk_ids'], dtype='<i8')
    assert len(order) * 256 == BUDGET
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9,.95), weight_decay=.01, eps=1e-8)
    initial = base.evaluate(model, validation)
    assert abs(initial - audit['s0_validation_nll_recorded']) < 1e-7
    summary = {'seed': 42, 'arm': args.arm, 'lambda': kd_lambda, 'budget_target_tokens': BUDGET,
               'steps': STEPS, 'seq_len': 256, 'global_batch': 32, 'microbatch': 4,
               'optimizer': {'lr': lr, 'weight_decay': .01, 'betas': [.9,.95], 'eps': 1e-8},
               'scheduler': 'Phase4A 5% linear warmup + cosine; 977 steps; warmup49',
               'temperature': 2, 'kd_reduction': 'token_mean forward KL times T squared',
               'gradient_clip': 1.0, 'precision': 'FP32 weights / BF16 autocast / FP32 CE-KL / no GradScaler',
               'attention': 'sdpa', 'checkpoint_schedule': CHECK_STEPS,
               'selection': 'minimum validation NLL including fixed S0 step0; ties earlier',
               'initial_validation_nll': initial, 's0_sha256': S0_SHA,
               'teacher_sha256': TEACHER_SHA if teacher is not None else None,
               'manifest_sha256': audit['continuation_manifest_sha256'],
               'order_sha256': m['order_sha256_int64_le'], 'freeze_commit': args.freeze_commit,
               'framework': {'torch': torch.__version__}, 'test_evaluated': False}
    selected_nll, best_step, best_tokens = initial, 0, 0
    if not args.smoke:
        (run_dir / 'config.json').write_text(json.dumps(summary, indent=2) + '\n')
        log = (run_dir / 'metrics.jsonl').open('x')
        log.write(json.dumps({'step': 0, 'tokens': 0, 'validation_nll': initial, 'eligible_for_selection': True}) + '\n')
        log.flush()
    norms, ce_values, kl_values = [], [], []
    consumed, position, clipped = 0, 0, 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(2 if args.smoke else STEPS):
        model.train()
        remaining = min(8192, BUDGET - consumed)
        rows = math.ceil(remaining / 256)
        inds = order[position:position + rows]
        position += rows
        optimizer.zero_grad(set_to_none=True)
        factor = schedule(step, STEPS)
        for group in optimizer.param_groups:
            group['lr'] = lr * factor
        counted, step_ce, step_kl = 0, 0., 0.
        for begin in range(0, rows, 4):
            x, y = packed_batch(tokens, inds[begin:begin+4], valid_limit=remaining-counted)
            x, y = x.cuda(), y.cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=x, use_cache=False).logits
                with torch.no_grad():
                    target = None if teacher is None else teacher(input_ids=x, use_cache=False).logits
                ce, kl, n = chunk_losses(logits, y, target)
                loss = (ce + kd_lambda * kl) * n / remaining
            if not torch.isfinite(loss):
                raise RuntimeError('NONFINITE_LOSS_STOP_NO_TUNING')
            loss.backward()
            counted += n
            step_ce += ce.item() * n / remaining
            step_kl += kl.item() * n / remaining
        assert counted == remaining
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True).item()
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError('NONFINITE_PARAMETER_STOP_NO_TUNING')
        consumed += counted
        clipped += norm > 1.
        norms.append(norm)
        ce_values.append(step_ce)
        kl_values.append(step_kl)
        row = {'step': step+1, 'tokens': consumed, 'ce': step_ce, 'scaled_kl': step_kl,
               'gradient_norm_preclip': norm, 'clipped': norm > 1., 'lr': lr * factor,
               'nonfinite': False}
        if not args.smoke and step+1 in CHECK_STEPS:
            row['validation_nll'] = base.evaluate(model, validation)
            if row['validation_nll'] < selected_nll:
                selected_nll, best_step, best_tokens = row['validation_nll'], step+1, consumed
                base.save_selected(model, run_dir / 'selected.pt')
        if not args.smoke:
            log.write(json.dumps(row) + '\n')
            log.flush()
        if args.smoke or (step+1) % 32 == 0 or 'validation_nll' in row:
            print(args.arm, json.dumps(row), flush=True)
    summary.update({'status': 'SMOKE_PASS' if args.smoke else 'TRAIN_COMPLETE_VALIDATION_SELECTED',
                    'consumed_target_tokens': consumed, 'best_validation_nll': selected_nll,
                    'best_validation_step': best_step, 'best_validation_tokens': best_tokens,
                    'selected_path': str((S0 if best_step == 0 else run_dir / 'selected.pt').relative_to(ROOT)),
                    'clip_fraction': clipped / len(norms), 'nonfinite_fraction': 0., 'overflow_fraction': 0.,
                    'gradient_norm_mean': float(np.mean(norms)), 'gradient_norm_max': max(norms),
                    'ce_mean': float(np.mean(ce_values)), 'scaled_kl_mean': float(np.mean(kl_values)),
                    'wall_seconds': time.perf_counter()-started,
                    'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                    'warning': 'CLIP_SATURATION_WARNING' if clipped/len(norms) >= .95 else '',
                    'overflow_interpretation': 'No GradScaler; losses, preclip norms, parameters checked finite every update.'})
    assert sha(S0) == S0_SHA and sha(TEACHER) == TEACHER_SHA
    if args.smoke:
        assert not kd_lambda, 'CE-only smoke before gate'
        (ART / 'ce_smoke_audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    else:
        log.close()
        assert consumed == BUDGET and position == len(order)
        summary['selected_sha256'] = sha(ROOT / summary['selected_path'])
        (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('arm', choices=CE_ARMS+['kd025','kd1'])
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--freeze-commit')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / ('FAILURE_'+args.arm+'.json')).write_text(json.dumps({'arm':args.arm,'error':repr(exc),'no_retry_or_tuning':True}, indent=2)+'\n')
        raise
