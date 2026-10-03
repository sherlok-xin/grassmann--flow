"""Validation-only step-zero diagnostic and preregistered sign decisions."""
import math


def including_s0(s0_validation_nll, continuation_validation_nll, trained_step):
    if not all(math.isfinite(x) for x in [s0_validation_nll, continuation_validation_nll]):
        raise ValueError('Nonfinite selection candidate')
    # Equal NLL chooses the earlier candidate, consistent with primary selection.
    return 0 if s0_validation_nll <= continuation_validation_nll else trained_step


def decisions(gains1, gains5, directions5):
    assert len(gains1) == len(gains5) == len(directions5) == 3
    l5 = all(x < 0 for x in gains5) and all(directions5)
    l1 = all(x < 0 for x in gains1)
    flags = []
    if l5:
        flags.append('FINEWEB_L5_NEGATIVE_REPLICATED')
        flags.append('FINEWEB_L1_L5_NEGATIVE_REPLICATED' if l1 else 'FINEWEB_STRENGTH_DEPENDENT')
    else:
        flags.append('MODERN_NEGATIVE_TRANSFER_NOT_ROBUST')
    return flags
