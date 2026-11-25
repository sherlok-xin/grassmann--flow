"""Phase 5 numerical core. Does not import or alter legacy training losses."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def log1mexp(x):
    """log(1-exp(x)), x<=0; includes the exact x=0 boundary."""
    if bool((x > 0).any()):
        raise ValueError('log probability must be <=0')
    return torch.where(x < -0.6931471805599453, torch.log1p(-x.exp()), torch.log(-torch.expm1(x)))


@torch.no_grad()
def project(teacher_logits, gold, reference_logp):
    """FP32 closed form, [tokens,vocab], [tokens], [tokens]. Targets detached."""
    z = teacher_logits.detach().float()
    if not torch.isfinite(z).all() or not torch.isfinite(reference_logp).all():
        raise FloatingPointError('nonfinite teacher/reference input')
    lp = z.log_softmax(-1)
    ty = lp.gather(-1, gold[:, None]).squeeze(-1)
    tail_z = z.clone().scatter_(-1, gold[:, None], -torch.inf)
    tail_lse = tail_z.logsumexp(-1)
    log_tail = tail_lse - z.logsumexp(-1)
    r = reference_logp.detach().float()
    if r.shape != ty.shape or bool((r > 0).any()):
        raise ValueError('reference probabilities have invalid shape/range')
    active = r > ty
    qy = torch.maximum(ty, r)
    qtail = torch.where(active, log1mexp(r), log_tail)
    q = tail_z - tail_lse[:, None] + qtail[:, None]
    q.scatter_(-1, gold[:, None], qy[:, None])
    beta = torch.where(active, -torch.expm1(qtail - log_tail), torch.zeros_like(r)).clamp(0, 1)
    return q, beta


@torch.no_grad()
def permute_tail(logq, gold, seed=20261125):
    """A fixed seeded permutation of non-gold vocabulary ranks, per gold ID."""
    n, v = logq.shape
    perm = torch.randperm(v-1, generator=torch.Generator().manual_seed(seed)).to(logq.device)
    rank = torch.arange(v, device=logq.device)[None, :].expand(n, -1)
    non_gold_rank = (rank - (rank > gold[:, None]).long()).clamp(max=v-2)
    source_rank = perm[non_gold_rank]
    source_id = source_rank + (source_rank >= gold[:, None])
    source_id = torch.where(rank == gold[:, None], gold[:, None], source_id)
    return logq.gather(-1, source_id)


def kl_from_logtarget(logs, logq):
    # xlogy handles the allowed q=0 boundary without 0 * -inf NaNs.
    q = logq.detach().exp()
    return (torch.special.xlogy(q, q) - q * logs).sum(-1)


def token_loss(student_logits, teacher_logits, gold, *, mode='kd', lam=1., temperature=1.,
               reference_logp=None, beta_mean=None, smoothing=.1, atkd_weights=None):
    """Already aligned valid tokens. Exact token sums, caller owns denominator."""
    if lam < 0 or temperature <= 0:
        raise ValueError('invalid KD coefficient/temperature')
    z = student_logits.float()
    logs = z.log_softmax(-1)
    ce = -logs.gather(-1, gold[:, None]).squeeze(-1)
    zero = torch.zeros_like(ce)
    if mode == 'ce' or lam == 0:
        return ce, {'ce': ce, 'kd': zero, 'beta': zero}
    if mode == 'ls':
        loss = (1-smoothing)*ce - smoothing*logs.mean(-1)
        return loss, {'ce': ce, 'kd': zero, 'beta': zero}
    logt = (teacher_logits.detach().float()/temperature).log_softmax(-1)
    logs_t = (z/temperature).log_softmax(-1)
    beta = zero
    if mode == 'kd':
        kd = kl_from_logtarget(logs_t, logt) * temperature**2
    elif mode == 'todi':
        # Author implementation, various_divergence.py, compute_ToDi, beta=1.
        weight = torch.sigmoid(logt-logs_t).detach()
        kd = (weight*logt.exp()*(logt-logs_t)+(1-weight)*logs_t.exp()*(logs_t-logt)).sum(-1)*temperature**2
    elif mode == 'wclip':
        # CUD Eq.5-9 component ONLY; no DUS retraining, not full CUD reproduction.
        p = teacher_logits.detach().float().softmax(-1)
        top, ix = p.max(-1); py = p.gather(1,gold[:,None]).squeeze(-1)
        delta = torch.minimum(.5*top,.7*(top-py)).clamp_min(0)
        p.scatter_add_(1,ix[:,None],-delta[:,None]); p.scatter_add_(1,gold[:,None],delta[:,None])
        lq=(p.log()/temperature).log_softmax(-1)
        kd=kl_from_logtarget(logs_t,lq)*temperature**2
    elif mode == 'atkd':
        if atkd_weights is None: raise ValueError('ATKD needs global-batch partition weights')
        nt_s=(z/temperature).scatter(1,gold[:,None],-torch.inf).log_softmax(-1)
        nt_t=(teacher_logits.detach().float()/temperature).scatter(1,gold[:,None],-torch.inf).log_softmax(-1)
        # Explicitly remove the masked gold coordinate before KL (avoid 0*inf).
        mask=torch.ones_like(nt_s,dtype=torch.bool).scatter_(1,gold[:,None],False)
        nckd=kl_from_logtarget(nt_s[mask].reshape(len(gold),-1),nt_t[mask].reshape(len(gold),-1))
        sy=logs_t.gather(1,gold[:,None]).squeeze(-1)
        ty=logt.gather(1,gold[:,None]).squeeze(-1)
        stail=(z/temperature).scatter(1,gold[:,None],-torch.inf).logsumexp(-1)-(z/temperature).logsumexp(-1)
        ttail=(teacher_logits.detach().float()/temperature).scatter(1,gold[:,None],-torch.inf).logsumexp(-1)-(teacher_logits.detach().float()/temperature).logsumexp(-1)
        tckd=kl_from_logtarget(torch.stack([sy,stail],-1),torch.stack([ty,ttail],-1))
        hard,scale=atkd_weights
        kd=scale*(nckd+hard*tckd)*temperature**2
    elif mode in {'reference', 'current', 'fixed', 'mean', 'permuted'}:
        if temperature != 1:
            raise ValueError('projection equivalence is restricted to temperature one')
        if mode == 'current':
            reference_logp = logs.detach().gather(-1, gold[:, None]).squeeze(-1)
        if reference_logp is None:
            raise ValueError('missing pre-update reference probabilities')
        logq, beta = project(teacher_logits, gold, reference_logp)
        if mode == 'mean':
            if beta_mean is None:
                raise ValueError('mean beta must be computed over the entire optimizer batch')
            b = torch.as_tensor(beta_mean, device=z.device).detach()
            if b.numel()!=1 or not 0 <= b.item() <= 1:
                raise ValueError('invalid optimizer-batch beta')
            beta = torch.ones_like(beta)*b
            kd = kl_from_logtarget(logs, logt)
            loss = (1+lam*beta)*ce+lam*(1-beta)*kd
            return loss, {'ce':ce, 'kd':kd, 'beta':beta}
        if mode == 'permuted':
            logq = permute_tail(logq, gold)
        kd = kl_from_logtarget(logs, logq)
    else:
        raise ValueError(f'unimplemented method {mode}')
    return ce+lam*kd, {'ce':ce, 'kd':kd, 'beta':beta}


class GoldCache:
    """Immutable per-step cache. No silent mismatch, interpolation or future lookup."""
    def __init__(self, directory, identity, write=False):
        self.path = Path(directory)
        self.identity = identity
        self.key = digest(identity)
        self.write = write
        if write:
            self.path.mkdir(parents=True, exist_ok=False)
            (self.path/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
        elif json.loads((self.path/'identity.json').read_text()) != identity:
            raise ValueError('reference cache initialization/data/protocol identity mismatch')

    def put(self, step, sample_ids, labels, logp):
        if not self.write:
            raise PermissionError('read-only reference cache')
        path = self.path/f'{step:07d}.pt'
        if path.exists():
            raise FileExistsError(path)
        torch.save({'identity':self.key, 'step':step, 'sample_ids':list(sample_ids),
                    'labels':labels.cpu(), 'logp':logp.detach().float().cpu()}, path)

    def get(self, step, sample_ids, labels):
        row = torch.load(self.path/f'{step:07d}.pt', map_location='cpu', weights_only=True)
        if (row['identity'] != self.key or row['step'] != step or row['sample_ids'] != list(sample_ids)
                or not torch.equal(row['labels'], labels.cpu()) or row['logp'].shape != labels.shape):
            raise ValueError('reference cache step/sample/token alignment mismatch')
        return row['logp']
