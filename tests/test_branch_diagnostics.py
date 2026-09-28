import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.branch_diagnostics import per_token_nll, token_branch_diagnostics
from train_distill_hybrid_lite_from_latefusion_teacher_v2 import HybridLateFusionAlphaModel


class _DummyBranch(nn.Module):
    def __init__(self, vocab_size=7, model_dim=4, seq_len=5):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(seq_len, model_dim)
        self.embedding_dropout = nn.Identity()
        self.blocks = nn.ModuleList([nn.Identity()])
        self.ln_f = nn.Identity()
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)


def test_optional_branch_outputs_preserve_default_forward():
    torch.manual_seed(3)
    model = HybridLateFusionAlphaModel(_DummyBranch(), _DummyBranch(), init_alpha=0.4, late_k=1)
    input_ids = torch.randint(0, 7, (2, 5))
    default_logits, default_loss = model(input_ids)
    diagnostic_logits, diagnostic_loss, branches = model(input_ids, return_branches=True)
    assert default_loss is None and diagnostic_loss is None
    assert torch.equal(default_logits, diagnostic_logits)
    expected = model.alpha()[0] * branches["transformer"] + (1.0 - model.alpha()[0]) * branches["grassmann"]
    assert torch.allclose(diagnostic_logits, expected)


def test_identical_teacher_branches_have_zero_jsd():
    torch.manual_seed(0)
    student = torch.randn(2, 4, 7)
    teacher = torch.randn(2, 4, 7)
    labels = torch.randint(0, 7, (2, 4))
    result = token_branch_diagnostics(student, teacher, teacher, teacher, labels, chunk_tokens=2)
    assert result["branch_jsd"].shape == (6,)
    assert torch.allclose(result["branch_jsd"], torch.zeros(6), atol=1e-6)


def test_ignore_mask_and_endpoint_delta_nll():
    student = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]])
    teacher_t = student.clone()
    teacher_g = torch.flip(student, dims=[-1])
    teacher_f = 0.5 * (teacher_t + teacher_g)
    labels = torch.tensor([[0, 0, -100]])
    ce_endpoint = student.clone()
    kd_endpoint = torch.flip(student, dims=[-1])
    result = token_branch_diagnostics(
        student,
        teacher_f,
        teacher_t,
        teacher_g,
        labels,
        ce_endpoint_logits=ce_endpoint,
        kd_endpoint_logits=kd_endpoint,
    )
    assert result["position"].tolist() == [1]
    assert result["delta_nll"].item() > 0
    assert per_token_nll(student, labels).shape == (1,)


def test_gradient_statistics_are_finite():
    torch.manual_seed(1)
    student = torch.randn(1, 5, 11)
    teacher_f = torch.randn(1, 5, 11)
    teacher_t = torch.randn(1, 5, 11)
    teacher_g = torch.randn(1, 5, 11)
    labels = torch.randint(0, 11, (1, 5))
    result = token_branch_diagnostics(student, teacher_f, teacher_t, teacher_g, labels)
    assert torch.isfinite(result["grad_dot"]).all()
    assert torch.isfinite(result["grad_cosine"]).all()
    assert (result["grad_cosine"].abs() <= 1.00001).all()


def test_saturated_logits_preserve_metric_bounds():
    student = torch.tensor([[[40.0, -40.0], [0.0, 0.0]]])
    teacher_t = torch.tensor([[[39.0, -39.0], [0.0, 0.0]]])
    teacher_g = torch.tensor([[[38.0, -38.0], [0.0, 0.0]]])
    teacher_f = 0.5 * (teacher_t + teacher_g)
    labels = torch.tensor([[0, 0]])
    result = token_branch_diagnostics(student, teacher_f, teacher_t, teacher_g, labels)
    assert (result["branch_jsd"] >= 0).all()
    assert (result["grad_cosine"].abs() <= 1.0).all()
