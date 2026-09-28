# Stage-C CRBD Prototype Analysis

Date: 2026-09-18

## Comparability and completion

C0, C1, C2, C4, C5, C6, and C7 completed with summary files. They use the same seed-42 TinyStories warm start, frozen teacher, 20,000-line dataset budget, 4,473,074 pre-trim training tokens, 17,472 training chunks, batch size 8, two epochs, learning rate `1e-4`, and validation-PPL checkpoint selection. C3 did not complete: both attempts were externally stopped while GPU 3 was locked near 210 MHz. C3 is therefore missing rather than negative.

## Results

CE-only obtained validation/test PPL 5.0183/5.0195. Fixed fused KD degraded this to 5.3041/5.3018, reproducing the TinyStories negative-transfer boundary on the reduced dataset. Entropy-gated fused KD was effectively unchanged at 5.3027/5.3008. Branch-only KD was worse at 5.6135/5.6100.

CRBD obtained 5.5104/5.5067, while shuffled routing obtained 5.5045/5.5010 and swapped branch pairing obtained 6.3668/6.3616. Relative to CE, fixed KD added 0.2858 validation PPL and CRBD added 0.4921. The repair fraction `(C1-C5)/(C1-C0)` is -0.7220: rather than removing 50% of the fixed-KD excess, CRBD increased it by 72.2%. CRBD was also 0.0059 PPL worse than shuffled routing, although it was 0.8563 better than swapped pairing.

The PTB smoke gate alone passed because C5 validation PPL was 69.0009 versus 69.2466 for C1, a difference of -0.2458. The complete Stage-C gate does not pass: the TinyStories repair criterion fails, and C5 does not outperform C6. C5 does outperform C7, but this single control is insufficient. Stage D must not start from the current formulation.

## Optimization diagnosis

The runs are finite: accumulated non-finite/AMP-overflow fractions are below 0.23%. However, every KD arm had a gradient clipping fraction of 1.0 in both epochs, while CE clipping averaged 0.0135. The prototype therefore operates in a fully clip-saturated regime.

For C5 at epoch 2, token-mean fused KL was 0.5538 and branch KL was 1.4134. With coefficients 5 and 2.5, their weighted contributions were approximately 2.77 and 3.53, compared with CE 1.73. Separate mean-one normalization of fused and branch routing weights preserves the full global strength of both losses; it does not make them a fixed-budget mixture. The current implementation consequently adds a large branch objective on top of fixed fused KD. C4 shows that branch supervision at this strength is itself harmful, and the near-tie between C5 and C6 gives no evidence that measured disagreement improves routing over a shuffle.

C7 confirms that swapped pairing is worse, but its branch KL is roughly twice the aligned C5 branch KL. This control therefore demonstrates a severe mismatch, not yet a loss-scale-matched causal benefit of architecture alignment.

## Decision

Do not launch full-data or multi-seed Stage D. The current CRBD formulation is rejected at the bounded-prototype gate. If work continues, the next experiment should first complete C3, then compare fixed-budget routing in which fused and branch terms share one total KD budget. Any follow-up must match initial weighted KD magnitude or gradient norm to C1 and avoid 100% clipping. Running the remaining `lambda_b` grid under the current separate mean-one normalization is low value because it tunes an identified scale confound rather than testing routing cleanly.
