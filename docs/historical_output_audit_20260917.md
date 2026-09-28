# Historical Output Audit (2026-09-17)

The `outputs/` tree currently contains 203 run directories represented by 205 normalized result rows: 174 complete, 27 config-only, and 2 incomplete. Completed evidence spans PTB, WikiText-2, TinyStories, and Code, but it was produced by several generations of runners and is not a single matched benchmark.

The strongest reusable result is the PTB quality--efficiency suite. The late-fusion teacher has 36.264M parameters and test PPL 50.1125. The 224x56 six-layer student has 31.434M parameters and three-seed test PPL 51.8431, 51.8842, and 51.7745. The 192x48 four-layer student has 23.565M parameters and three-seed test PPL 53.8719, 53.9937, and 53.8761. At batch size 32, measured latency is 55.98 ms for the teacher, 48.46 ms for the quality student, and 34.96 ms for the efficiency student. At batch size 1, however, the quality student is slightly slower than the teacher (16.09 versus 15.48 ms), so latency claims must state batch size.

The historical PTB legacy optimum (`alpha=0.02`) reaches 51.2740 test PPL, essentially identical to the new token-normalized seed-42 result of 51.2771. This supports implementation parity and clearer loss provenance, not a new accuracy gain from normalization. The new matched three-seed experiment is the stronger causal evidence.

TinyStories consistently shows the opposite coefficient direction in the historical seed-42 sweep: CE obtains 4.8611, while every positive legacy coefficient is worse, from 5.3127 at `alpha=0.01` to 5.5008 at `alpha=0.05`. The new token-normalized `lambda=5` result of 5.4127 confirms negative transfer but remains single-seed for this domain.

WikiText-2 has a matched single-seed legacy series with CE 71.5748 and best KD 60.8349 at `alpha=0.02`. Code has a usable common-5k-file single-seed series with CE 7.2454 and best KD 6.0018 at `alpha=0.01`. Earlier Code runs with `max_lines=-1` use PTB teacher/initialization provenance and 181M training tokens, whereas the later 5k series uses the Code teacher/initialization and 22.3M training tokens. These generations must not be compared in one effect estimate.

Historical random-initialization controls are not suitable for causal claims because initialization checkpoints, learning rates, epochs, or data budgets differ from their nominal controls. The subspace-analysis directories also contain an initial hook-bug generation with all-zero spectral metrics and a later nonzero generation, while the verification scripts encode conflicting rank-direction hypotheses. These artifacts are exploratory only until one definition and a checkpoint-matched protocol are frozen.

No historical run name or configuration contains branch disagreement gating, confidence routing, branch-aligned KD, CRBD, shuffled routing, or swapped branch pairing. A bounded Stage-C prototype therefore tests a genuinely new mechanism within the repository rather than repeating prior experiments.
