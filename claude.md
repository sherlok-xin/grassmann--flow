# Research Objective

This project explores attention-free sequence modeling
through Grassmann manifold dynamics.

The core hypothesis is that manifold-constrained
state evolution can preserve long-range contextual
geometry more efficiently and stably than
Euclidean recurrent dynamics.

# Constraints

- Fully attention-free
- Causal architecture only
- Linear memory complexity
- Preserve orthogonality
- Avoid hidden-state collapse

# Primary Evaluation

- Perplexity
- Long-context stability
- Throughput
- Memory usage
- Spectral stability
- Effective hidden rank

# Research Priorities

Priority order:

1. Novelty
2. Mechanism interpretability
3. Stability
4. Performance
5. Efficiency

# Important Research Questions

- Why does Grassmann flow help?
- Does manifold evolution improve memory retention?
- Is orthogonality the true source of improvement?
- Does the model suffer subspace collapse?
- How does spectrum evolve during long sequence propagation?

# Reviewer Concerns

Always proactively evaluate:

- Is the gain merely from normalization?
- Is the manifold formulation necessary?
- Is the comparison against Mamba/RWKV fair?
- Are ablations sufficient?
- Is the mechanism theoretically justified?

# Agent Behaviors

## Hypothesis Agent

Generate hypotheses ONLY if:
- grounded in observed failure
- theoretically motivated
- experimentally verifiable

Avoid random architectural complexity.

## Experiment Agent

Every experiment must:
- define a single clear variable
- log all metrics
- save reproducibility config
- include spectral statistics

## Mechanism Agent

Always analyze:
- singular value evolution
- hidden state geometry
- orthogonality preservation
- effective rank
- long-range memory retention

## Reviewer Agent

Aggressively criticize:
- weak novelty
- missing controls
- unfair baselines
- insufficient theory
- unclear mechanism

Never assume improvements are meaningful
without statistical and mechanistic support.