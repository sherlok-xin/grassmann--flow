"""Token-normalized losses for consensus-routed branch distillation.

The historical fixed fused-logit objective remains implemented in
``kd_losses.py``.  This module is opt-in and requires fused, Transformer, and
Grassmann logits from both the teacher and the student.
"""

from __future__ import annotations

import math
from typing import Mapping, NamedTuple

import torch
import torch.nn.functional as F


CRBD_STRATEGIES = {
    "entropy_gated",
    "disagreement_suppressed",
    "branch_only",
    "crbd",
    "crbd_shuffled",
    "crbd_swapped",
}


class CRBDLossOutput(NamedTuple):
    total: torch.Tensor
    ce: torch.Tensor
    fused_kl_token_mean: torch.Tensor
    branch_kl_token_mean: torch.Tensor
    teacher_jsd_mean: torch.Tensor
    teacher_entropy_mean: torch.Tensor
    fused_route_mean: torch.Tensor
    branch_route_mean: torch.Tensor
    valid_tokens: torch.Tensor


def _validate_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_branches: Mapping[str, torch.Tensor],
    teacher_branches: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
) -> None:
    tensors = [
        student_logits,
        teacher_logits,
        student_branches.get("transformer"),
        student_branches.get("grassmann"),
        teacher_branches.get("transformer"),
        teacher_branches.get("grassmann"),
    ]
    if any(item is None for item in tensors):
        raise ValueError("CRBD requires transformer and grassmann branch logits for teacher and student.")
    if student_logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected logits [batch, sequence, vocabulary] and labels [batch, sequence].")
    expected = student_logits.shape
    if any(item.shape != expected for item in tensors):
        raise ValueError("All fused and branch logits must have the same shape.")
    if expected[:2] != labels.shape:
        raise ValueError("The batch and sequence dimensions of logits and labels must match.")


def _selected_shifted(logits: torch.Tensor, valid_indices: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.size(-1)
    return logits[:, :-1, :].reshape(-1, vocab_size).index_select(0, valid_indices)


def _teacher_routing_statistics(
    teacher_logits: torch.Tensor,
    teacher_branches: Mapping[str, torch.Tensor],
    valid_indices: torch.Tensor,
    temperature: float,
    chunk_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    jsd_parts = []
    entropy_parts = []
    vocab_size = teacher_logits.size(-1)
    log_vocab = math.log(vocab_size)

    with torch.no_grad():
        for start in range(0, valid_indices.numel(), chunk_tokens):
            indices = valid_indices[start : start + chunk_tokens]
            fused = _selected_shifted(teacher_logits, indices).float() / temperature
            transformer = _selected_shifted(teacher_branches["transformer"], indices).float() / temperature
            grassmann = _selected_shifted(teacher_branches["grassmann"], indices).float() / temperature

            fused_logp = F.log_softmax(fused, dim=-1)
            fused_p = fused_logp.exp()
            entropy = -(fused_p * fused_logp).sum(dim=-1) / log_vocab

            t_logp = F.log_softmax(transformer, dim=-1)
            g_logp = F.log_softmax(grassmann, dim=-1)
            t_p = t_logp.exp()
            g_p = g_logp.exp()
            mixture_logp = torch.logaddexp(t_logp, g_logp) - math.log(2.0)
            jsd = 0.5 * (
                (t_p * (t_logp - mixture_logp)).sum(dim=-1)
                + (g_p * (g_logp - mixture_logp)).sum(dim=-1)
            ) / math.log(2.0)
            jsd_parts.append(jsd.clamp_(0.0, 1.0))
            entropy_parts.append(entropy.clamp_(0.0, 1.0))

    return torch.cat(jsd_parts), torch.cat(entropy_parts)


def _mean_normalize(weights: torch.Tensor) -> torch.Tensor:
    return weights / weights.mean().clamp_min(1e-6)


def causal_lm_crbd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_branches: Mapping[str, torch.Tensor],
    teacher_branches: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    strategy: str,
    temperature: float = 2.0,
    fused_lambda: float = 5.0,
    branch_lambda: float = 2.5,
    routing_tau: float = 0.25,
    ignore_index: int = -100,
    chunk_tokens: int = 256,
    shuffle_seed: int = 0,
) -> CRBDLossOutput:
    """Compute an opt-in adaptive or branch-aligned distillation objective.

    Active token weights are detached and normalized to mean one.  Therefore,
    each coefficient retains its token-mean interpretation across datasets.
    """
    _validate_logits(student_logits, teacher_logits, student_branches, teacher_branches, labels)
    if strategy not in CRBD_STRATEGIES:
        raise ValueError(f"Unknown CRBD strategy: {strategy}")
    if temperature <= 0 or routing_tau <= 0:
        raise ValueError("temperature and routing_tau must be positive")
    if fused_lambda < 0 or branch_lambda < 0:
        raise ValueError("distillation coefficients must be non-negative")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")

    shifted_student = student_logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    ce = F.cross_entropy(
        shifted_student.view(-1, shifted_student.size(-1)),
        shifted_labels.view(-1),
        ignore_index=ignore_index,
    )
    valid_indices = shifted_labels.reshape(-1).ne(ignore_index).nonzero(as_tuple=False).squeeze(-1)
    valid_tokens = torch.tensor(valid_indices.numel(), device=student_logits.device, dtype=torch.long)
    if valid_indices.numel() == 0:
        raise ValueError("No valid prediction token remains after shifting labels.")

    jsd, entropy = _teacher_routing_statistics(
        teacher_logits,
        teacher_branches,
        valid_indices,
        temperature,
        chunk_tokens,
    )
    consensus = torch.exp(-jsd / routing_tau)
    confidence = torch.exp(-entropy / routing_tau)

    use_fused = strategy in {
        "entropy_gated",
        "disagreement_suppressed",
        "crbd",
        "crbd_shuffled",
        "crbd_swapped",
    }
    use_branch = strategy in {"branch_only", "crbd", "crbd_shuffled", "crbd_swapped"}

    if strategy == "entropy_gated":
        fused_raw = confidence
    else:
        fused_raw = consensus
    branch_raw = 1.0 - consensus

    if strategy == "crbd_shuffled":
        generator = torch.Generator(device=consensus.device)
        generator.manual_seed(int(shuffle_seed))
        order = torch.randperm(consensus.numel(), generator=generator, device=consensus.device)
        fused_raw = fused_raw.index_select(0, order)
        branch_raw = branch_raw.index_select(0, order)

    fused_weights = _mean_normalize(fused_raw) if use_fused else torch.zeros_like(fused_raw)
    branch_weights = _mean_normalize(branch_raw) if use_branch else torch.zeros_like(branch_raw)

    fused_sum = student_logits.new_zeros((), dtype=torch.float32)
    branch_sum = student_logits.new_zeros((), dtype=torch.float32)
    scale = temperature**2
    swapped = strategy == "crbd_swapped"

    for start in range(0, valid_indices.numel(), chunk_tokens):
        stop = min(start + chunk_tokens, valid_indices.numel())
        indices = valid_indices[start:stop]
        if use_fused:
            student_fused_logp = F.log_softmax(
                _selected_shifted(student_logits, indices).float() / temperature,
                dim=-1,
            )
            teacher_fused_p = F.softmax(
                _selected_shifted(teacher_logits, indices).float() / temperature,
                dim=-1,
            )
            token_kl = F.kl_div(student_fused_logp, teacher_fused_p, reduction="none").sum(dim=-1) * scale
            fused_sum = fused_sum + (fused_weights[start:stop] * token_kl).sum()

        if use_branch:
            student_t_key = "grassmann" if swapped else "transformer"
            student_g_key = "transformer" if swapped else "grassmann"
            student_t_logp = F.log_softmax(
                _selected_shifted(student_branches[student_t_key], indices).float() / temperature,
                dim=-1,
            )
            student_g_logp = F.log_softmax(
                _selected_shifted(student_branches[student_g_key], indices).float() / temperature,
                dim=-1,
            )
            teacher_t_p = F.softmax(
                _selected_shifted(teacher_branches["transformer"], indices).float() / temperature,
                dim=-1,
            )
            teacher_g_p = F.softmax(
                _selected_shifted(teacher_branches["grassmann"], indices).float() / temperature,
                dim=-1,
            )
            token_t = F.kl_div(student_t_logp, teacher_t_p, reduction="none").sum(dim=-1) * scale
            token_g = F.kl_div(student_g_logp, teacher_g_p, reduction="none").sum(dim=-1) * scale
            branch_sum = branch_sum + (branch_weights[start:stop] * 0.5 * (token_t + token_g)).sum()

    denom = valid_tokens.to(dtype=torch.float32)
    fused_kl = fused_sum / denom
    branch_kl = branch_sum / denom
    total = ce + fused_lambda * fused_kl + branch_lambda * branch_kl
    return CRBDLossOutput(
        total=total,
        ce=ce,
        fused_kl_token_mean=fused_kl,
        branch_kl_token_mean=branch_kl,
        teacher_jsd_mean=jsd.mean(),
        teacher_entropy_mean=entropy.mean(),
        fused_route_mean=fused_raw.mean(),
        branch_route_mean=branch_raw.mean(),
        valid_tokens=valid_tokens,
    )
