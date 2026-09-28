# H0 PTB Smoke Analysis

Date: 2026-09-16

## Scope

This smoke test validates the new token-normalized KD implementation on PTB. It is not a final performance comparison. All arms use the same late-fusion teacher, hybrid-lite warm-start checkpoint, seed 42, sequence length 256, batch size 32, learning rate `1e-4`, temperature 2, and one training epoch.

## Results

| Objective | Val PPL | Test PPL | CE | KL/token | Gradient clip fraction |
|---|---:|---:|---:|---:|---:|
| CE only (`lambda_kd=0`) | 69.3132 | 59.0448 | 3.5110 | 0.399078 | 0.000 |
| Legacy (`alpha=0.02`) | 61.6915 | 53.2614 | 3.5932 | 0.257855 | 0.094 |
| Token mean (`lambda_kd=2.5`) | 62.8843 | 54.0773 | 3.5571 | 0.279369 | 0.029 |
| Token mean (`lambda_kd=5`) | 61.7313 | 53.2860 | 3.5911 | 0.258781 | 0.086 |
| Token mean (`lambda_kd=10`) | 61.3501 | 53.1319 | 3.6291 | 0.242961 | 0.547 |

All loss, validation, and test metrics are finite. Every completed checkpoint was reloaded by the training program before test evaluation, and each config and summary contains the loss mode and coefficient provenance. The original GPU 0 legacy attempt failed before the first optimizer step because an unrelated process occupied 3.7 GiB; the rerun on GPU 1 completed, and the failed config-only directory was retained.

## Equivalence and stability

For this unpadded PTB loader, every sequence has 255 prediction positions. The legacy objective is

```text
0.98 * CE + 0.02 * KL_batchmean
= 0.98 * (CE + 5.2040816 * KL_token_mean).
```

The `lambda_kd=5` result therefore provides an implementation-level check against legacy `alpha=0.02`. Their validation PPL values are 61.6915 and 61.7313, and their test PPL values are 53.2614 and 53.2860. A diagnostic rerun that separated finite gradients from AMP overflow found finite mean gradient norms of 0.7874 and 0.7872, clipping fractions of 0.101 and 0.094, and identical nonfinite/AMP-overflow fractions of 0.014. This supports numerical parity within the expected difference between 5 and 5.204.

`lambda_kd=10` had the best one-epoch validation PPL, but 54.7% of its steps exceeded the clipping threshold or were nonfinite before dynamic loss-scale recovery. It therefore fails the preregistered stability criterion. `lambda_kd=5` is selected for the confirmatory experiment because it is stable, close to the analytically equivalent legacy strength, and stronger than `lambda_kd=2.5` on validation PPL.

## Interpretation and next test

The PTB smoke validates the loss API and shows that token normalization can reproduce the legacy behavior when its coefficient is converted correctly. It does not yet establish the cross-domain claim of H0. The next experiment must compare CE-only, legacy `alpha=0.02`, and token-normalized `lambda_kd=5` on PTB and TinyStories with matched seeds 42, 123, and 456. Model selection must use validation PPL; the test split remains report-only.
