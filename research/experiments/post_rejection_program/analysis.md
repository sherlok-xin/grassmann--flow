# A1/A2 confirmation and Stage-B diagnostics

## Completed experiments

The confirmation used matched CE and token-normalized KD runs from the same seed-specific initialization. Every run used a 31,434,257-parameter Hybrid-Lite student, temperature 2, ten epochs, batch size 32, learning rate `1e-4`, and validation PPL for checkpoint selection. The KD arm used `CE + 5 KL_token_mean`.

On PTB, the three CE runs obtained test PPL values of 59.1175, 59.6566, and 59.0819 for seeds 42, 123, and 456. The corresponding KD values were 51.2771, 51.3745, and 51.2061. Mean test PPL decreased from 59.2853 (sample SD 0.3220) to 51.2859 (sample SD 0.0846). The paired change was -7.9994 PPL on average (sample SD 0.2454), equivalent to a 13.49% reduction from the CE mean. Every seed had the same effect direction.

On TinyStories seed 42, CE reached 4.8611 test PPL, whereas KD reached 5.4127. The paired change was +0.5516 PPL, or an 11.35% degradation. The warm-start checkpoint itself had 5.0654 test PPL and the teacher had 4.9691, so CE surpassed both while fixed KD finished worse than both.

## Stability

PTB CE had no clipping, non-finite gradients, or AMP overflows. PTB KD had first-epoch maximum clip fractions of 0.187--0.201 and a 0.0144 non-finite/overflow fraction, followed by zero in later epochs. TinyStories KD had substantially stronger clipping, starting at 0.588 and decreasing to 0.074 by epoch 10. Its non-finite/overflow fraction remained below 0.0007 and was also present at a smaller scale in CE, so the degradation is not explained by a failed run. The high clipping rate nevertheless indicates that `lambda=5` is aggressive on TinyStories.

## PTB mechanism diagnostic

The corrected diagnostic covered all 350 PTB validation chunks and 89,250 valid tokens. Mean endpoint `delta_nll = NLL(KD)-NLL(CE)` was -0.1555, although 40.34% of tokens had positive delta NLL. Negative CE--KD gradient cosine occurred for 36.91% of tokens.

After controlling for fused-teacher entropy and NLL, the standardized JSD coefficient for gradient cosine was -0.0212 with a 95% chunk-bootstrap interval of [-0.0274, -0.0147]. The endpoint delta-NLL coefficient was 0.0067 with an interval of [-0.00005, 0.0127]. Adding JSD changed held-out conflict AUROC by -0.00032 and endpoint R2 by +0.00053. JSD is therefore associated with conflict on PTB, but adds almost no predictive power over teacher entropy and NLL. JSD-bin behavior is also non-monotonic.

## TinyStories mechanism diagnostic

The diagnostic used 2,048 deterministically selected validation chunks and 522,240 valid tokens. Mean endpoint delta NLL was +0.1069, 55.05% of tokens had positive delta NLL, and 45.35% had negative CE--KD gradient cosine. These token-level directions agree with the sequence-level negative-transfer result.

After controlling for fused-teacher entropy and NLL, the standardized JSD coefficient for gradient cosine was -0.0148 with a 95% chunk-bootstrap interval of [-0.0170, -0.0127]. The endpoint delta-NLL coefficient was +0.00615 with an interval of [0.00386, 0.00836]. Adding JSD increased held-out conflict AUROC from 0.5471 to 0.5580 (+0.0109) and endpoint R2 from 0.0422 to 0.0452 (+0.00306). Across JSD deciles, harmful endpoint frequency rose from 37.97% in the lowest bin to 62.26% in the highest bin; mean delta NLL generally increased with disagreement. The association is therefore clearer in the negative-transfer domain than on PTB.

## Interpretation and decision

Token-normalized KD is effective and highly reproducible on PTB, but the same frozen coefficient causes clear negative transfer on TinyStories. The sign aligns with relative teacher quality: the PTB teacher is much stronger than the CE student, whereas the TinyStories CE student surpasses its teacher. The results support domain-dependent or teacher-quality-aware KD, not a universal fixed KD coefficient.

The Stage-B result passes the literal preregistered gate: the JSD coefficients have the predicted harmful directions in both domains, and adding JSD improves at least one held-out metric in each domain. The strength of evidence differs substantially, however. TinyStories provides consistent coefficient, predictive, and decile evidence, whereas PTB provides a stable gradient-conflict coefficient but only a +0.00053 R2 gain and a slightly negative AUROC change. This justifies only the bounded Stage-C prototype, not a claim that CRBD is already effective. Stage C should retain teacher-quality/confidence gating and branch-aligned KD as strong controls because relative teacher quality remains a simpler explanation of the domain-level sign change.
