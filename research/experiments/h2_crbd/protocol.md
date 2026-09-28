# Stage-C CRBD Prototype Protocol

Date frozen: 2026-09-17

## Scope and decision basis

Stage B passed the preregistered directional gate on PTB and TinyStories, but the incremental predictive value of branch JSD on PTB was weak. Stage C is therefore a bounded prototype rather than a confirmatory method evaluation. Historical-output review found no prior branch-disagreement gating, branch-aligned distillation, shuffled-routing control, or swapped-pairing control in this repository.

The existing `fixed_fused` path remains the default and must reproduce the current loss implementation. Every new strategy requires a Hybrid-Lite student, token-normalized KL, and explicit command-line activation. Student inference architecture and parameter count remain unchanged.

## Frozen objectives

All arms retain supervised CE. C1 uses fused token-mean KL with `lambda_f=5`. C2 weights fused KL by normalized teacher confidence `exp(-H(p_F)/(log(V) tau))`. C3 weights fused KL by normalized consensus `exp(-JSD(p_T,p_G)/(log(2) tau))`. C4 uses the mean of architecture-aligned Transformer and Grassmann branch KL terms. C5 combines consensus-weighted fused KL and complementary disagreement-weighted branch KL. C6 shuffles the paired routing weights across valid tokens using an independent deterministic seed. C7 keeps the C5 weights but swaps the Transformer and Grassmann teacher--student branch pairing. Active weights are detached and normalized to mean one, so the coefficients retain a token-mean interpretation.

## Validation sequence

First, remote unit tests must cover zero disagreement, identical teacher/student logits, branch-only decomposition, ignored labels, deterministic shuffling, and swapped pairing. A PTB one-epoch smoke then uses seed 42, `max_lines=2000`, batch size 8, `lambda_f=5`, `lambda_b=2.5`, `tau=0.25`, and the frozen PTB teacher and initialization. The smoke passes only if config, metrics, summary, and reloadable checkpoint are produced with finite losses and no persistent AMP overflow.

The reduced TinyStories prototype uses seed 42, `max_lines=20000`, two epochs, batch size 8, learning rate `1e-4`, temperature 2, and validation PPL for selection. It first evaluates C0--C7 at `lambda_b=2.5`, `tau=0.25`. If C5 is finite, at most four additional C5 settings are selected without test access: `(lambda_b,tau)` in `{(1,0.25), (5,0.25), (2.5,0.1), (2.5,0.5)}`. The best C5 setting is selected by validation PPL and then frozen. Test PPL is descriptive only after selection.

## Prototype gate

C5 must be no more than 0.3 validation PPL worse than C1 on the PTB smoke, reduce at least half of the C1-versus-C0 TinyStories excess validation PPL, and outperform C6 and C7 in the predicted direction. Failure stops full-data CRBD experiments. Passing authorizes Stage D but does not itself support a paper-level effectiveness claim.

All runs preserve historical outputs and record teacher, student initialization, dataset statistics, strategy, coefficients, routing temperature, routing seed, gradient diagnostics, validation selection, and final checkpoint. Training runs only on `10.42.0.197`.
