# H1 Protocol: Branch Disagreement and Consensus-Routed KD

Status: exploratory plan; execute only after author approval and the matched H0 confirmation

## Question

Can disagreement between the Transformer and Grassmann teacher branches identify tokens for which fused-logit KD is harmful, and can it determine when supervision should switch from fused logits to architecture-matched branches?

## Mechanistic prediction

Low branch disagreement indicates that two structurally different predictors support a similar target distribution. High disagreement indicates uncertain or architecture-specific evidence, where forcing a small student toward the fused distribution may conflict with the observed token. KD gain should therefore decrease as teacher branch disagreement increases, especially on TinyStories. If the evidence is architecture-specific rather than merely unreliable, routing it to the matching student branch should be stronger than suppressing KD altogether.

## Analysis before training

On fixed validation subsets from PTB and TinyStories, save only compact token statistics: teacher branch Jensen--Shannon divergence, fused-teacher entropy and NLL, teacher correctness, analytic CE--KD logit-gradient cosine and dot product, and the endpoint NLL difference between matched CE and fixed-KD students. Raw logits and raw text are not retained. Use chunk-grouped cross-validation and chunk bootstrap rather than treating tokens as independent. Test whether disagreement adds held-out predictive value beyond entropy and teacher NLL.

## Candidate method

Use a detached token weight derived from normalized branch disagreement. Low-disagreement tokens receive token-normalized fused-logit KD. High-disagreement tokens receive Transformer-to-Transformer and Grassmann-to-Grassmann branch KD. CE remains unchanged. Parameter count and inference cost remain identical because teacher statistics and branch losses are used only during training. The complete loss and the staged parameter-selection rule are frozen in `research/experiments/post_rejection_program/experiment_plan.md`.

## Controls

Compare CE-only, fixed token-normalized KD, entropy-gated fused KD, disagreement-suppressed fused KD, branch-aligned-only KD, consensus-routed KD, a shuffled-routing control, and a swapped-branch control. Use the same warm-start checkpoint and seed for every method. PTB and TinyStories are the initial domains; other domains are held out until the direction is established.

## Success criterion

Before method training, disagreement must improve held-out prediction of gradient conflict or endpoint KD harm over entropy/NLL controls in both domains. The final method must remain within 0.3 PPL of fixed KD on PTB, remove at least half of the fixed-KD excess PPL on TinyStories, and beat shuffled-routing and swapped-branch controls in the predicted direction. Otherwise the routed H1 is rejected; branch-aligned-only KD may then be evaluated as a separate, weaker hypothesis.
