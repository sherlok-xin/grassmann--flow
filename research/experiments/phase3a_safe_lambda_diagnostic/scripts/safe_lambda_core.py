"""Pure numerical helpers for the Phase 3A offline diagnostic."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SafeLambdaResult:
    value: float | None
    status: str


def quadratic_delta(*, eta: float, lam: float, a: float, b: float, c: float) -> float:
    """Second-order validation-loss difference between KD and CE updates."""
    return -eta * lam * a + eta * eta * lam * b + 0.5 * eta * eta * lam * lam * c


def candidate_safe_lambda(*, eta: float, a: float, b: float, c: float) -> SafeLambdaResult:
    """Return a finite positive upper crossing only when its assumptions hold."""
    values = (eta, a, b, c)
    if eta <= 0 or not all(math.isfinite(value) for value in values):
        return SafeLambdaResult(None, "nonfinite_or_invalid_input")
    numerator_term = a / eta - b
    if numerator_term <= 0:
        return SafeLambdaResult(None, "no_positive_local_safe_interval")
    if c <= 0:
        return SafeLambdaResult(None, "no_finite_upper_bound_nonpositive_curvature")
    value = 2.0 * numerator_term / c
    if not math.isfinite(value) or value <= 0:
        return SafeLambdaResult(None, "nonfinite_or_nonpositive_crossing")
    return SafeLambdaResult(value, "finite_positive_upper_crossing")


def sign_label(value: float, *, beneficial_when_positive: bool = True, tolerance: float = 0.0) -> str:
    if not math.isfinite(value):
        return "nonfinite"
    oriented = value if beneficial_when_positive else -value
    if oriented > tolerance:
        return "beneficial"
    if oriented < -tolerance:
        return "harmful"
    return "tie"
