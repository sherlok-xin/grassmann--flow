"""Exact target-token accounting and memory-bounded forward-KL, no model changes."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def chunk_losses(student, labels, teacher=None, temperature=2.0, chunk=128):
    s = student.reshape(-1, student.shape[-1])
    y = labels.reshape(-1)
    t = None if teacher is None else teacher.detach().reshape_as(s)
    n = int((y != -100).sum())
    if n == 0:
        raise ValueError('No valid targets')
    ce, kl = s.new_zeros((), dtype=torch.float32), s.new_zeros((), dtype=torch.float32)

    def compute(sc, yc, tc):
        valid = yc != -100
        sc = sc.float()
        c = F.cross_entropy(sc, yc, ignore_index=-100, reduction='sum')
        if tc.numel() == 0:
            return c, c * 0
        log_s = F.log_softmax(sc / temperature, dim=-1)
        log_t = F.log_softmax(tc.float() / temperature, dim=-1)
        k = (log_t.exp() * (log_t - log_s)).sum(-1)
        return c, (k * valid).sum() * temperature ** 2

    for start in range(0, len(y), chunk):
        sc, yc = s[start:start + chunk], y[start:start + chunk]
        tc = sc.new_empty(0) if t is None else t[start:start + chunk]
        if torch.is_grad_enabled() and sc.requires_grad:
            c, k = checkpoint(compute, sc, yc, tc, use_reentrant=False)
        else:
            c, k = compute(sc, yc, tc)
        ce, kl = ce + c, kl + k
    return ce / n, kl / n, n


def schedule(step, steps, warmup_fraction=0.05):
    warm = max(1, math.ceil(steps * warmup_fraction))
    if step < warm:
        return (step + 1) / warm
    progress = (step - warm) / max(1, steps - warm - 1)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def packed_batch(tokens, indices, seq_len=256, valid_limit=None):
    """Stride seq_len, one context token overlap; every target counted once."""
    x = np.zeros((len(indices), seq_len), dtype=np.int64)
    y = np.full_like(x, -100)
    remaining = valid_limit
    for row, index in enumerate(indices):
        offset = int(index) * seq_len
        count = min(seq_len, len(tokens) - offset - 1)
        if remaining is not None:
            count = min(count, remaining)
            remaining -= count
        if count <= 0:
            continue
        x[row, :count] = tokens[offset:offset + count]
        y[row, :count] = tokens[offset + 1:offset + count + 1]
    return torch.from_numpy(x), torch.from_numpy(y)
