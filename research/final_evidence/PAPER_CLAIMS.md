# Final Paper Claim Set

## One-sentence contribution

This paper provides a controlled empirical account of warm-start language-model distillation, showing reproducible positive and negative transfer boundaries and demonstrating that teacher training state and predictive behavior, rather than architectural heterogeneity alone, govern the tested outcomes.

## Central claim 1: Warm-start KD exhibits reproducible positive and negative transfer across text domains

Under the same token-mean KD form (`lambda=5`, temperature 2) and matched seed-specific warm starts, KD improves PTB and WikiText-2 but harms TinyStories. The mean paired NLL gains are `+0.144937` on PTB, `+0.155665` on WikiText-2, and `-0.107219` on TinyStories, with consistent signs for all three independent student seeds in every dataset.

Supporting experiments: PTB A1/A2 confirmation, Phase 2C C0/C1, and Phase 2G. The claim is about observed cross-domain behavior, not a causal claim that dataset identity determines the sign. The single-seed CodeParrot result is excluded from this central claim.

## Central claim 2: Teacher training state and predictive quality strongly modulate transfer under controlled composition

At fixed Transformer/Grassmann architecture and forced alpha 0.5, the jointly trained Teacher J improves every student, whereas the alpha-only Teacher A harms every student. The mean paired contrast `Q=NLL_A-NLL_J` is `+0.183319` NLL with sample SD `0.001375` and 3/3 positive signs. Mean KD gains are `+0.155665` for Teacher J and `-0.027654` for Teacher A.

Supporting experiment: Phase 2D. The claim must refer jointly to teacher training state and resulting predictive quality. It must not state that global teacher NLL is the only causal variable, because joint training changes both branch representations.

## Central claim 3: Global teacher likelihood alone is insufficient to determine distillability

The pure-Grassmann alpha-0.0 teacher is globally worse than the matched C0 student for every WikiText-2 seed, with residual advantages `-0.266520`, `-0.253492`, and `-0.251006`, yet it still gives positive KD gains `+0.046771`, `+0.033598`, and `+0.042604`. Conversely, the much weaker fixed-composition Teacher A produces negative transfer in Phase 2D. These results reject the universal rule that a teacher worse than the student must cause negative transfer.

Supporting experiments: Phase 2C and Phase 2D. Hard-token utility provides a possible explanation, but it remains observational and must not be presented as the established mechanism.

## Central claim 4: Ensemble supervision helps, but TG heterogeneity has no advantage over the homogeneous control

Within the fixed TG checkpoint, fused supervision gives larger mean KD gain than Transformer-only and Grassmann-only supervision (`0.155665` versus `0.134868` and `0.040991`). The fused-versus-Transformer difference is small (`+0.020797` NLL) and carries a permanent Transformer-only clipping warning. In the decisive homogeneous control, TT outperforms TG for all three students: `H=Gain_TG-Gain_TT` has mean `-0.012460` NLL and sample SD `0.001348`.

Supporting experiments: Phase 2E and Phase 2F. The allowed interpretation is that ensemble supervision is useful in the tested setup, while no evidence supports a uniquely transferable Grassmann signal. Teacher quality, parameter count, and teacher-training differences remain disclosed confounds in TT versus TG.

## Secondary result: PTB quality--efficiency operating points

The frozen PTB teacher, quality student, and efficiency student have `36.264M`, `31.434M`, and `23.565M` parameters. Their PPL values are `50.1125`, `51.8339±0.0554`, and `53.9139±0.0692`, while batch-32 latency is `55.984`, `48.460`, and `34.959 ms` on one RTX 3090 at sequence length 256. This result belongs in a secondary deployment section, not in the central causal KD claims.

## Claims excluded from the paper

The manuscript must not claim that Grassmann-specific KD is superior, that Plücker geometry causes transfer gains, that heterogeneous ensembles outperform homogeneous ensembles, that branch disagreement strongly predicts transfer, that CRBD succeeds, that a globally weaker teacher necessarily causes negative transfer, that warm-starting is necessary, or that the Hybrid-lite architecture has a unique efficiency advantage over matched non-Grassmann students.
