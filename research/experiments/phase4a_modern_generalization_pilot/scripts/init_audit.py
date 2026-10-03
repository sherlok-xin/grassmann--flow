import argparse
import json
from pathlib import Path
import numpy as np
import torch
from core import chunk_losses, packed_batch
from train import ART, OUT, load, sha, stream


def main(dataset):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    s0 = OUT / 'runs' / dataset / 's0/selected.pt'
    tstate = OUT / 'runs' / dataset / 'teacher/selected.pt'
    model = load('SmolLM2-135M', s0)
    teacher = load('SmolLM2-360M', tstate).eval().requires_grad_(False)
    model.train()
    params = list(model.parameters())
    gs = [[torch.zeros_like(p, device='cpu') for p in params] for _ in range(2)]
    tokens = stream(dataset, 'train')
    order = np.random.default_rng(4242).permutation((len(tokens) - 1) // 256)[:32]
    ce_value, kl_value = 0., 0.
    for begin in range(0, 32, 4):
        x, y = packed_batch(tokens, order[begin:begin + 4])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids=x.cuda(), use_cache=False).logits
            with torch.no_grad():
                target = teacher(input_ids=x.cuda(), use_cache=False).logits
            ce, kl, _ = chunk_losses(logits, y.cuda(), target)
        for i, loss in enumerate([ce, kl]):
            grads = torch.autograd.grad(loss / 8, params, retain_graph=i == 0)
            for acc, g in zip(gs[i], grads):
                acc.add_(g.detach().cpu())
            del grads
        ce_value += ce.item() / 8
        kl_value += kl.item() / 8
    nc, nk = [sum(float(g.double().square().sum()) for g in values) ** 0.5 for values in gs]
    assert np.isfinite(nc) and np.isfinite(nk)
    sc = json.loads((OUT / 'runs' / dataset / 's0/summary.json').read_text())
    tc = json.loads((OUT / 'runs' / dataset / 'teacher/summary.json').read_text())
    audit = {'dataset': dataset, 'seed': 42, 's0_sha256': sha(s0), 'teacher_sha256': sha(tstate),
             's0_validation_nll': sc['best_validation_nll'], 'teacher_validation_nll': tc['best_validation_nll'],
             'teacher_residual_advantage_over_s0': sc['best_validation_nll'] - tc['best_validation_nll'],
             'probe': 'first matched continuation global batch, 32 x 256 targets, full student parameter gradients',
             'ce': ce_value, 'scaled_kl_T2': kl_value, 'gradient_norm_ce': nc, 'gradient_norm_kd_unweighted': nk,
             'gradient_ratio_kd_over_ce': nk / nc,
             'gradient_ratio_lambda1_kd_over_ce': nk / nc,
             'gradient_ratio_lambda5_kd_over_ce': 5 * nk / nc,
             'gradient_ratio_ce_over_lambda1_kd': nc / nk,
             'gradient_ratio_ce_over_lambda5_kd': nc / (5 * nk),
             'no_optimizer_step': True, 'test_access': False}
    (OUT / 'runs' / dataset / 'initialization_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset', choices=['tinystories', 'fineweb'])
    main(p.parse_args().dataset)
