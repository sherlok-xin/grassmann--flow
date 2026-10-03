from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safe_lambda_core import candidate_safe_lambda, quadratic_delta


def test_quadratic_expression_matches_direct_taylor_difference():
    dtype = torch.float64
    theta = torch.tensor([0.7, -1.1], dtype=dtype)
    hessian = torch.tensor([[3.0, 0.4], [0.4, 1.5]], dtype=dtype)
    g_c = torch.tensor([0.2, -0.3], dtype=dtype)
    g_d = torch.tensor([-0.5, 0.8], dtype=dtype)
    g_v = hessian @ theta
    eta = 1e-2
    lam = 2.5

    ce_update = theta - eta * g_c
    kd_update = theta - eta * (g_c + lam * g_d)
    value = lambda point: 0.5 * point @ hessian @ point
    direct = float(value(kd_update) - value(ce_update))
    a = float(g_v @ g_d)
    b = float(g_c @ hessian @ g_d)
    c = float(g_d @ hessian @ g_d)
    predicted = quadratic_delta(eta=eta, lam=lam, a=a, b=b, c=c)
    assert math.isclose(predicted, direct, rel_tol=1e-12, abs_tol=1e-12)


def test_safe_lambda_requires_beneficial_slope_and_positive_curvature():
    good = candidate_safe_lambda(eta=0.1, a=1.0, b=2.0, c=4.0)
    assert good.status == "finite_positive_upper_crossing"
    assert math.isclose(good.value, 4.0)

    bad_slope = candidate_safe_lambda(eta=0.1, a=0.1, b=2.0, c=4.0)
    assert bad_slope.value is None
    assert bad_slope.status == "no_positive_local_safe_interval"

    indefinite = candidate_safe_lambda(eta=0.1, a=1.0, b=2.0, c=-1.0)
    assert indefinite.value is None
    assert indefinite.status == "no_finite_upper_bound_nonpositive_curvature"


def test_autograd_hvp_matches_explicit_hessian_finite_difference_and_symmetry():
    dtype = torch.float64
    hessian = torch.tensor([[2.0, -0.3, 0.2], [-0.3, 1.7, 0.5], [0.2, 0.5, 3.2]], dtype=dtype)
    theta = torch.tensor([0.4, -0.8, 1.2], dtype=dtype, requires_grad=True)
    u = torch.tensor([0.7, -0.2, 0.4], dtype=dtype)
    v = torch.tensor([-0.1, 0.9, 0.3], dtype=dtype)

    loss = 0.5 * theta @ hessian @ theta
    grad = torch.autograd.grad(loss, theta, create_graph=True)[0]
    hvp_v = torch.autograd.grad(grad @ v, theta, retain_graph=True)[0]
    hvp_u = torch.autograd.grad(grad @ u, theta)[0]
    assert torch.allclose(hvp_v, hessian @ v, atol=1e-12, rtol=1e-12)
    assert torch.allclose(u @ hvp_v, v @ hvp_u, atol=1e-12, rtol=1e-12)

    epsilon = 1e-5
    grad_plus = hessian @ (theta.detach() + epsilon * v)
    grad_minus = hessian @ (theta.detach() - epsilon * v)
    finite_difference = (grad_plus - grad_minus) / (2.0 * epsilon)
    assert torch.allclose(hvp_v, finite_difference, atol=1e-9, rtol=1e-9)
