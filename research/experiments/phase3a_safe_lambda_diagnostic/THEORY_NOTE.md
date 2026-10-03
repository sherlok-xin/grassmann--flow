# Phase 3A Theory Note

## Status and scope

Phase 3A is an explicitly authorized post-freeze, offline diagnostic extension. It does not train a KD endpoint, tune a coefficient, modify an existing checkpoint, or change the frozen manuscript evidence. The question is retrospective: can quantities measured at the warm-start checkpoint predict the sign and relative magnitude of the already observed fixed-strength KD transfer?

Let `C(theta)` denote CE on a deterministic training probe, `D(theta)` denote token-mean KD on the same probe, and `V(theta)` denote CE on a held-out validation calibration subset. Define `gC`, `gD`, and `gV` as their full-parameter gradients at S0. For the local SGD-style update

`theta_lambda+ = theta - eta * (gC + lambda * gD)`,

the difference from the matched CE-only update is approximated by

`DeltaV(lambda) = -eta * lambda * a + eta^2 * lambda * b + 0.5 * eta^2 * lambda^2 * c`,

where `a = gV^T gD`, `b = gC^T H_V gD`, and `c = gD^T H_V gD`. Negative `DeltaV` predicts that KD improves calibration CE relative to CE-only. For reporting, `predicted_gain_quad = -DeltaV`, so its sign follows the endpoint convention `Gain_KD = NLL_CE - NLL_KD`.

The nominal local step size is fixed at `eta = 1e-4`, matching the configured base learning rate of all primary endpoint protocols. This is an SGD-style diagnostic, not a claim that AdamW follows the same quadratic trajectory.

## Candidate safe strength

For positive `lambda`, write the nonzero factor of the quadratic as

`q(lambda) = -eta * a + eta^2 * b + 0.5 * eta^2 * lambda * c`.

The candidate upper crossing is

`lambda_safe = 2 * (a / eta - b) / c`.

It is reported only when the initial local slope predicts benefit (`a / eta - b > 0`), the curvature is positive (`c > 0`), and the resulting value is finite and positive. A non-positive curvature is not coerced into a bound; it is labeled `no_finite_upper_bound_nonpositive_curvature` when the initial slope is beneficial. A non-beneficial initial slope is labeled `no_positive_local_safe_interval`. Non-finite quantities are labeled explicitly.

## Controls and interpretation boundary

The first-order validation control is `a` together with `cosine(gV, gD)`. The optimizer-aware control applies the frozen grid `lambda in {0, 0.5, 1, 2.5, 5, 10}` to one full-batch gradient computed at S0, clips the combined gradient at norm 1, and uses a fresh AdamW state for every lambda. The literal formal scheduler initializes the first optimizer learning rate to zero; this exact control is retained and expected to be a no-op. A separately labeled nominal-LR sensitivity uses the configured `1e-4` learning rate without the zero-LR scheduler initialization. Neither virtual state is saved.

The approximation is evaluated as a diagnostic only. A useful retrospective score does not establish a new training method, a causal mechanism, or a universally safe coefficient.
