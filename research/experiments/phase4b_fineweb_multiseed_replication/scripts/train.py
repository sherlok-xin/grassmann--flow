"""Phase 4B seed-aware copy of the frozen Phase 4A training loop."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM

ART = Path(__file__).resolve().parents[1]
ROOT = ART.parents[2]
OUT = ROOT / 'outputs/phase4a_modern'
P4A = ROOT / 'research/experiments/phase4a_modern_generalization_pilot'
RUNS = ROOT / 'outputs/phase4b_fineweb/runs'
sys.path.insert(0, str(P4A / 'scripts'))
from core import chunk_losses, packed_batch, schedule


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def load(model_name, state=None):
    manifest = json.loads((P4A / 'model_download_manifest.json').read_text())
    path = OUT / 'model_cache' / model_name / manifest[model_name]['revision']
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.float32, attn_implementation='sdpa').cuda()
    if state is not None:
        model.load_state_dict(torch.load(state, map_location='cpu', weights_only=True), strict=True)
    return model


def stream(dataset, split):
    return np.memmap(OUT / 'data' / dataset / (split + '.bin'), mode='r', dtype='<u4')


@torch.no_grad()
def evaluate(model, tokens, micro=4):
    model.eval()
    total, count = 0.0, 0
    for start in range(0, math.ceil((len(tokens) - 1) / 256), micro):
        inds = list(range(start, min(start + micro, math.ceil((len(tokens) - 1) / 256))))
        x, y = packed_batch(tokens, inds)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids=x.cuda(), use_cache=False).logits
            ce, _, n = chunk_losses(logits, y.cuda())
        if not torch.isfinite(ce):
            raise RuntimeError('NONFINITE_EVALUATION')
        total += ce.item() * n
        count += n
    assert count == len(tokens) - 1
    return total / count


def save_selected(model, path):
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path.with_suffix('.tmp'))
    path.with_suffix('.tmp').replace(path)


def train(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    budget = 10_000_000
    global_batch, micro = 32, 4
    steps = math.ceil(budget / (global_batch * 256))
    run = RUNS / str(args.seed) / args.arm
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'summary.json').exists() or (run / 'selected.pt').exists():
        raise RuntimeError('Refuse to overwrite an existing formal run')
    continuation = args.arm in ('ce', 'kd1', 'kd5')
    kd_lambda = {'kd1': 1.0, 'kd5': 5.0}.get(args.arm, 0.0)
    student_state = RUNS / str(args.seed) / 's0/selected.pt'
    teacher_state = OUT / 'runs' / args.dataset / 'teacher/selected.pt'
    model_name = 'SmolLM2-360M' if args.arm == 'teacher' else 'SmolLM2-135M'
    model = load(model_name, student_state if continuation else None)
    teacher = None if kd_lambda == 0 else load('SmolLM2-360M', teacher_state)
    if teacher is not None:
        teacher.eval().requires_grad_(False)
    train_tokens, validation = stream(args.dataset, 'train'), stream(args.dataset, 'validation')
    data_manifest = json.loads((OUT / 'data' / args.dataset / 'manifest.json').read_text())
    assert sha(OUT / 'data' / args.dataset / 'train.bin') == data_manifest['sha256']['train']
    assert sha(OUT / 'data' / args.dataset / 'validation.bin') == data_manifest['sha256']['validation']
    assert len(train_tokens) == 20_000_000 and len(validation) == 1_000_000
    order_seed = 4242 if continuation else args.seed
    order = np.random.default_rng(order_seed).permutation((len(train_tokens) - 1) // 256)
    summary = {'dataset': args.dataset, 'seed': args.seed, 'arm': args.arm, 'lambda': kd_lambda,
               'budget_target_tokens': budget, 'steps': steps, 'global_batch': global_batch,
               'microbatch': micro, 'seq_len': 256, 'order_seed': order_seed,
               'order_sha256': hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
               's0_sha256': sha(student_state) if continuation else None,
               'teacher_sha256': sha(teacher_state) if teacher is not None else None,
               'train_sha256': data_manifest['sha256']['train'],
               'validation_sha256': data_manifest['sha256']['validation'],
               'model_parameters': sum(p.numel() for p in model.parameters()),
               'optimizer': {'lr': 5e-5, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'eps': 1e-8},
               'precision': 'FP32 weights / BF16 autocast / FP32 CE-KL / no GradScaler',
               'attention': 'sdpa', 'temperature': 2, 'gradient_clip': 1.0,
               'selection': 'minimum full validation NLL among trained evaluation steps; ties choose earlier step',
               'framework': {'torch': torch.__version__}, 'nonfinite_steps': 0, 'overflow_steps': 0}
    (run / 'config.json').write_text(json.dumps(summary, indent=2) + '\n')
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.95), weight_decay=0.01, eps=1e-8)
    selected = run / 'selected.pt'
    initial_nll = evaluate(model, validation)
    selected_nll = math.inf
    best_step, best_tokens = 0, 0
    summary['initial_validation_nll'] = initial_nll
    log = (run / 'metrics.jsonl').open('x')
    log.write(json.dumps({'step': 0, 'tokens': 0, 'validation_nll': initial_nll, 'eligible_for_selection': False}) + '\n')
    log.flush()
    consumed, position, clipped = 0, 0, 0
    norms, losses_ce, losses_kl = [], [], []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        model.train()
        remaining = min(global_batch * 256, budget - consumed)
        rows = math.ceil(remaining / 256)
        inds = order[position:position + rows]
        assert len(inds) == rows
        position += rows
        optimizer.zero_grad(set_to_none=True)
        factor = schedule(step, steps)
        for group in optimizer.param_groups:
            group['lr'] = 5e-5 * factor
        step_ce, step_kl, counted = 0.0, 0.0, 0
        for begin in range(0, rows, micro):
            x, y = packed_batch(train_tokens, inds[begin:begin + micro], valid_limit=remaining - counted)
            x, y = x.cuda(), y.cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=x, use_cache=False).logits
                with torch.no_grad():
                    target = None if teacher is None else teacher(input_ids=x, use_cache=False).logits
                ce, kl, n = chunk_losses(logits, y, target)
                loss = (ce + kd_lambda * kl) * n / remaining
            if not torch.isfinite(loss):
                summary['nonfinite_steps'] += 1
                (run / 'FAILED.json').write_text(json.dumps({'status': 'NONFINITE_LOSS', 'step': step, **summary}, indent=2))
                raise RuntimeError('NONFINITE_LOSS_STOP_NO_TUNING')
            loss.backward()
            step_ce += ce.item() * n / remaining
            step_kl += kl.item() * n / remaining
            counted += n
        assert counted == remaining
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True).item()
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError('NONFINITE_PARAMETER_STOP_NO_TUNING')
        consumed += counted
        clipped += norm > 1.0
        norms.append(norm)
        losses_ce.append(step_ce)
        losses_kl.append(step_kl)
        row = {'step': step + 1, 'tokens': consumed, 'ce': step_ce, 'scaled_kl': step_kl,
               'gradient_norm_preclip': norm, 'clipped': norm > 1, 'lr': 5e-5 * factor}
        if (step + 1) % 256 == 0 or consumed == budget:
            row['validation_nll'] = evaluate(model, validation)
            if row['validation_nll'] < selected_nll:
                selected_nll = row['validation_nll']
                best_step, best_tokens = step + 1, consumed
                save_selected(model, selected)
        log.write(json.dumps(row) + '\n')
        log.flush()
        if (step + 1) % 32 == 0 or 'validation_nll' in row:
            print(args.dataset, args.arm, json.dumps(row), flush=True)
    log.close()
    assert consumed == budget
    summary.update({'status': 'TRAIN_COMPLETE_VALIDATION_SELECTED', 'consumed_target_tokens': consumed,
                    'best_validation_nll': selected_nll, 'best_validation_step': best_step,
                    'best_validation_tokens': best_tokens, 'selected_sha256': sha(selected),
                    'clip_fraction': clipped / steps, 'nonfinite_fraction': 0.0, 'overflow_fraction': 0.0,
                    'gradient_norm_mean': float(np.mean(norms)), 'gradient_norm_max': max(norms),
                    'ce_mean': float(np.mean(losses_ce)), 'scaled_kl_mean': float(np.mean(losses_kl)),
                    'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                    'wall_seconds': time.perf_counter() - started,
                    'overflow_interpretation': 'BF16 has no GradScaler; finite losses, unclipped grad norms and parameters checked each step',
                    'test_evaluated': False})
    if continuation:
        assert sha(student_state) == summary['s0_sha256']
    if teacher is not None:
        assert sha(teacher_state) == summary['teacher_sha256']
    (run / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset', choices=['fineweb'])
    p.add_argument('arm', choices=['s0', 'ce', 'kd1', 'kd5'])
    p.add_argument('--seed', required=True, type=int, choices=[123, 456])
    train(p.parse_args())
