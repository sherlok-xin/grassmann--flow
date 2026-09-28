# H0 Protocol: KD Loss Normalization

Status: PTB smoke completed; confirmatory cross-domain runs pending

## Question

Does the apparent cross-domain sensitivity of KD remain after KL is normalized by valid prediction tokens and all controls are matched?

## Mechanistic prediction

The legacy implementation sums KL across sequence positions, so the effective distillation pressure depends on sequence length and teacher distribution. Token normalization should make the weight interpretable and reduce accidental domain-specific scaling. It may change the optimal coefficient and may explain part, but not necessarily all, of the TinyStories degradation.

## Required implementation

Keep `legacy_batchmean` unchanged for checkpoint and result reproduction. Add `token_mean`, computed as the sum of per-token KL over valid shifted labels divided by the number of valid tokens, followed by the usual temperature-squared factor. Record both raw and normalized KL. Use the transparent objective `CE + lambda_kd * KL_token` for new runs.

## Smoke experiment

Use the existing PTB teacher and the same 224x56, 6-layer warm-start checkpoint. Fix seed 42, sequence length 256, batch size 32, learning rate 1e-4, temperature 2, data order and one epoch. Compare CE-only, legacy alpha 0.02, and token-normalized lambda values 2.5, 5 and 10.

The smoke passes only if legacy loss matches the old scale within normal nondeterminism, all new losses remain finite, gradient clipping is not saturated for most steps, all checkpoints reload, and config/summary include the new provenance fields.

## Confirmatory experiment

After smoke, run PTB and TinyStories for the original 10-epoch budget with seeds 42, 123 and 456. Preselect one token-normalized lambda from PTB validation in the smoke or a bounded pilot; do not select separately on test. Report paired seed differences against CE-only.

## Outcomes

Support for H0 requires materially smaller cross-domain variation in the scale of normalized KL and stable optimization. Whether normalized KD improves PPL is a separate empirical result. If TinyStories remains negative, proceed to H1. If the negative transfer disappears, the main contribution becomes a rigorous implementation and evaluation finding rather than an adaptive method.

## Smoke decision

The PTB smoke selected `lambda_kd=5` for confirmatory runs. With 255 prediction positions per sequence, legacy `alpha=0.02` is scale-equivalent, up to a positive global objective factor, to `lambda_kd = 255 * 0.02 / 0.98 = 5.204`. The observed validation and test PPL values of these two modes differ by less than 0.05. Although `lambda_kd=10` produced the lowest one-epoch validation PPL, its gradient clipping fraction was 0.547, so it failed the stability criterion and was not selected. Detailed results are recorded in `analysis.md`.
