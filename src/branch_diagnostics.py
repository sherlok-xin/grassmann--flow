from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _shift_and_mask(logits: torch.Tensor, labels: torch.Tensor):
    if logits.ndim != 3:
        raise ValueError(f"Expected logits with shape [batch, sequence, vocab], got {tuple(logits.shape)}")
    if labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError(
            f"Expected labels with shape {tuple(logits.shape[:2])}, got {tuple(labels.shape)}"
        )
    shifted = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    valid = shifted_labels.ne(-100)
    return shifted, shifted_labels, valid


def per_token_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted, shifted_labels, valid = _shift_and_mask(logits, labels)
    flat_logits = shifted[valid].float()
    flat_labels = shifted_labels[valid]
    if flat_labels.numel() == 0:
        return torch.empty(0, dtype=torch.float32, device=logits.device)
    return F.cross_entropy(flat_logits, flat_labels, reduction="none")


def token_branch_diagnostics(
    student_logits: torch.Tensor,
    teacher_fused_logits: torch.Tensor,
    teacher_transformer_logits: torch.Tensor,
    teacher_grassmann_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 2.0,
    chunk_tokens: int = 64,
    ce_endpoint_logits: Optional[torch.Tensor] = None,
    kd_endpoint_logits: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute compact valid-token diagnostics without retaining raw logits."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")

    tensors = [
        teacher_fused_logits,
        teacher_transformer_logits,
        teacher_grassmann_logits,
    ]
    if any(t.shape != student_logits.shape for t in tensors):
        raise ValueError("student and teacher logits must have identical shapes")

    s_shift, shifted_labels, valid = _shift_and_mask(student_logits, labels)
    f_shift, _, _ = _shift_and_mask(teacher_fused_logits, labels)
    t_shift, _, _ = _shift_and_mask(teacher_transformer_logits, labels)
    g_shift, _, _ = _shift_and_mask(teacher_grassmann_logits, labels)

    valid_indices = valid.nonzero(as_tuple=False)
    flat_labels = shifted_labels[valid]
    outputs = {
        "batch_index": valid_indices[:, 0].to(torch.int64),
        "position": (valid_indices[:, 1] + 1).to(torch.int64),
    }
    value_keys = [
        "branch_jsd",
        "teacher_entropy",
        "teacher_nll",
        "transformer_nll",
        "grassmann_nll",
        "teacher_correct",
        "transformer_correct",
        "grassmann_correct",
        "grad_dot",
        "grad_cosine",
    ]
    collected = {key: [] for key in value_keys}

    s_valid = s_shift[valid]
    f_valid = f_shift[valid]
    t_valid = t_shift[valid]
    g_valid = g_shift[valid]

    eps = torch.finfo(torch.float32).eps
    for start in range(0, flat_labels.numel(), chunk_tokens):
        end = min(start + chunk_tokens, flat_labels.numel())
        y = flat_labels[start:end]
        s = s_valid[start:end].float()
        f = f_valid[start:end].float()
        t = t_valid[start:end].float()
        g = g_valid[start:end].float()

        log_p_t = F.log_softmax(t, dim=-1)
        log_p_g = F.log_softmax(g, dim=-1)
        p_t = log_p_t.exp()
        p_g = log_p_g.exp()
        mixture = 0.5 * (p_t + p_g)
        log_mixture = mixture.clamp_min(eps).log()
        jsd = 0.5 * (
            (p_t * (log_p_t - log_mixture)).sum(dim=-1)
            + (p_g * (log_p_g - log_mixture)).sum(dim=-1)
        )
        jsd = jsd.clamp_min(0.0)

        log_p_f = F.log_softmax(f, dim=-1)
        p_f = log_p_f.exp()
        teacher_entropy = -(p_f * log_p_f).sum(dim=-1)

        log_p_s = F.log_softmax(s, dim=-1)
        p_s = log_p_s.exp()
        p_s_temp = F.softmax(s / temperature, dim=-1)
        p_f_temp = F.softmax(f / temperature, dim=-1)
        grad_kd = temperature * (p_s_temp - p_f_temp)

        ce_grad = p_s.clone()
        ce_grad[torch.arange(y.numel(), device=y.device), y] -= 1.0
        grad_dot = (ce_grad * grad_kd).sum(dim=-1)
        ce_norm_sq = ce_grad.square().sum(dim=-1)
        kd_norm_sq = grad_kd.square().sum(dim=-1)
        grad_cosine = grad_dot / (
            ce_norm_sq.clamp_min(eps).sqrt() * kd_norm_sq.clamp_min(eps).sqrt()
        )
        grad_cosine = grad_cosine.clamp(-1.0, 1.0)

        collected["branch_jsd"].append(jsd)
        collected["teacher_entropy"].append(teacher_entropy)
        collected["teacher_nll"].append(-log_p_f.gather(1, y[:, None]).squeeze(1))
        collected["transformer_nll"].append(-log_p_t.gather(1, y[:, None]).squeeze(1))
        collected["grassmann_nll"].append(-log_p_g.gather(1, y[:, None]).squeeze(1))
        collected["teacher_correct"].append(f.argmax(dim=-1).eq(y).float())
        collected["transformer_correct"].append(t.argmax(dim=-1).eq(y).float())
        collected["grassmann_correct"].append(g.argmax(dim=-1).eq(y).float())
        collected["grad_dot"].append(grad_dot)
        collected["grad_cosine"].append(grad_cosine)

    for key in value_keys:
        outputs[key] = torch.cat(collected[key], dim=0) if collected[key] else torch.empty(0, device=labels.device)

    if (ce_endpoint_logits is None) != (kd_endpoint_logits is None):
        raise ValueError("ce_endpoint_logits and kd_endpoint_logits must be provided together")
    if ce_endpoint_logits is not None:
        ce_nll = per_token_nll(ce_endpoint_logits, labels)
        kd_nll = per_token_nll(kd_endpoint_logits, labels)
        outputs["ce_endpoint_nll"] = ce_nll
        outputs["kd_endpoint_nll"] = kd_nll
        outputs["delta_nll"] = kd_nll - ce_nll

    return outputs
