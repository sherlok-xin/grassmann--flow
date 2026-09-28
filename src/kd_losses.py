"""Knowledge-distillation losses shared by causal language-model trainers."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


class KDLossOutput(NamedTuple):
    total: torch.Tensor
    ce: torch.Tensor
    kl_batchmean: torch.Tensor
    kl_token_mean: torch.Tensor
    valid_tokens: torch.Tensor


def _masked_token_kl_sum(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float,
    chunk_tokens: int,
) -> torch.Tensor:
    """Compute masked KL without materializing a full token-by-vocabulary copy."""
    vocab_size = student_logits.size(-1)
    student_flat = student_logits.reshape(-1, vocab_size)
    teacher_flat = teacher_logits.reshape(-1, vocab_size)
    valid_indices = valid_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    total = student_logits.new_zeros(())
    for start in range(0, valid_indices.numel(), chunk_tokens):
        indices = valid_indices[start : start + chunk_tokens]
        student_log_probs = F.log_softmax(student_flat.index_select(0, indices) / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_flat.index_select(0, indices) / temperature, dim=-1)
        total = total + F.kl_div(student_log_probs, teacher_probs, reduction="sum")
    return total * (temperature**2)


def causal_lm_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 2.0,
    mode: str = "legacy_batchmean",
    alpha: float = 0.7,
    kd_lambda: float = 1.0,
    ignore_index: int = -100,
    chunk_tokens: int = 1024,
) -> KDLossOutput:
    """Return CE and both legacy/token-normalized teacher-to-student KL.

    ``legacy_batchmean`` exactly preserves the historical objective
    ``(1-alpha) * CE + alpha * KL_batchmean``. ``token_mean`` uses the more
    interpretable objective ``CE + kd_lambda * KL_token_mean``.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Student/teacher logits must have identical shapes, got "
            f"{tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
        )
    if student_logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected logits [batch, sequence, vocabulary] and labels [batch, sequence].")
    if student_logits.shape[:2] != labels.shape:
        raise ValueError("The batch and sequence dimensions of logits and labels must match.")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if kd_lambda < 0:
        raise ValueError("kd_lambda must be non-negative")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if mode not in {"legacy_batchmean", "token_mean"}:
        raise ValueError(f"Unknown KD loss mode: {mode}")

    shifted_student = student_logits[:, :-1, :].contiguous()
    shifted_teacher = teacher_logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()

    ce = F.cross_entropy(
        shifted_student.view(-1, shifted_student.size(-1)),
        shifted_labels.view(-1),
        ignore_index=ignore_index,
    )

    student_log_probs = F.log_softmax(shifted_student / temperature, dim=-1)
    teacher_probs = F.softmax(shifted_teacher / temperature, dim=-1)
    kl_batchmean = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (temperature**2)

    valid_mask = shifted_labels.ne(ignore_index)
    valid_tokens = valid_mask.sum()
    if valid_tokens.item() == 0:
        raise ValueError("No valid prediction token remains after shifting labels.")

    if bool(valid_mask.all()):
        kl_token_mean = kl_batchmean * shifted_student.size(0) / valid_tokens
    else:
        masked_sum = _masked_token_kl_sum(
            shifted_student,
            shifted_teacher,
            valid_mask,
            temperature,
            chunk_tokens,
        )
        kl_token_mean = masked_sum / valid_tokens

    if mode == "legacy_batchmean":
        total = (1.0 - alpha) * ce + alpha * kl_batchmean
    else:
        total = ce + kd_lambda * kl_token_mean

    return KDLossOutput(total, ce, kl_batchmean, kl_token_mean, valid_tokens)

