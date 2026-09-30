# Phase 2F: Homogeneous Ensemble Control

## Question

Does the downstream KD advantage of the Transformer+Grassmann (TG) teacher remain when the control teacher is a homogeneous Transformer+Transformer (TT) ensemble?

This phase tests architectural heterogeneity and TG complementarity only. It cannot identify Plucker geometry as a cause. No new loss, routing method, architecture, or manuscript edit is allowed.

## Frozen sources

T1 is the validation-selected Transformer checkpoint from `outputs/experiments/20260317_114514_wt2_baseline_both`. T2 is trained independently with seed 123 using the same `train_exp.py` architecture, tokenizer, raw WikiText-2 data, sequence length 256, batch size 32, AdamW optimizer, learning rate `3e-4`, weight decay 0.01, cosine schedule, dropout 0.1, and 20-epoch budget. T2 checkpoint selection uses validation NLL only.

The recovered historical `wikitext-2-raw-v1` cache exactly reproduces all T1 recorded preprocessing statistics: train/validation/test non-empty lines are 23,767/2,461/2,891, character counts are 10,916,756/1,144,610/1,288,512, token counts are 2,391,884/247,289/283,287, and 256-token chunk counts are 9,343/965/1,106. The cache is stored outside the protected project `datasets/` directory.

The TG comparator is the existing joint Teacher J at `outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint`. TG KD endpoints are reused from Phase 2E and are not rerun.

## TT teacher training

The TT late-logit teacher uses T1 and T2, joint mode, 10 epochs, one frozen-branch epoch, branch learning rate `1e-5`, alpha learning rate `1e-2`, initial alpha 0.5, `late_k=1`, batch size 32, weight decay 0.01, AMP, and validation-NLL checkpoint selection. Teacher fine-tuning uses the same saved WikiText-2 dataset path and preprocessing used by Teacher J. Test data are evaluated only after validation selection.

The historical Teacher J configuration records alpha learning rate `5e-3`, whereas the authorized Phase 2F TT protocol explicitly fixes `1e-2`. This is a preregistered teacher-training mismatch and must remain in the final confound assessment. No hyperparameter is changed after seeing teacher or KD endpoints.

All TT/TG comparison and TT KD evaluation forces effective alpha to exactly 0.5, independent of the learned checkpoint alpha.

## Diagnostics

For TG and TT, record total parameter count, fixed-alpha validation NLL/PPL, both branch validation NLL/PPL, branch JSD, top-1 agreement, and fusion gain relative to the better branch. Record the parameter-count difference explicitly. Do not call the teachers parameter-matched unless counts are identical.

## KD matrix

Reuse the three independent Phase 2C student S0 checkpoints for seeds 42, 123, and 456. For TT use token-mean KD, lambda 5, temperature 2, 10 epochs, batch size 32, learning rate `1e-4`, weight decay 0.01, warmup ratio 0.05, cosine schedule, sequence length 256, AMP, clipping value 1.0, and validation-NLL checkpoint selection. Only the TT arms are new.

For each seed compute `Gain_TG = NLL_C0 - NLL_TG`, `Gain_TT = NLL_C0 - NLL_TT`, and `H = Gain_TG - Gain_TT = NLL_TT - NLL_TG`. Report the mean, sample standard deviation, and sign consistency of each quantity.

## Interpretation and stopping

If `H>0` for all seeds, the allowed statement is that architectural heterogeneity or TG complementarity provides additional transferable signal under this protocol. If TG and TT are approximately equal, attribute the Phase 2E advantage primarily to generic ensemble diversity and reduce the Grassmann-specific claim. If TT is better, stop the Grassmann-specific KD mechanism claim. A large teacher-quality difference must be treated as a confound.

Any NaN, failed formal run, clearly elevated non-finite/overflow rate, or anomalous endpoint stops interpretation rather than triggering hyperparameter changes. Phase 2G is not launched automatically.
