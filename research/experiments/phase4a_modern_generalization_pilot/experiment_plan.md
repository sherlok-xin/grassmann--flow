# Phase 4A — Modern-family external-validity pilot

Authorized parent: c7dd579a1e2b0f30586aacaad3cce0a40d194987. Seed 42 only.
Question: do positive and negative warm-start KD transfer persist with base
SmolLM2-135M students and base SmolLM2-360M teachers?

Stage 0 verifies exact official revisions, identical tokenizer semantics,
licenses, installed Transformers compatibility, FP32 token-mean CE/KL values
and gradients, GPU memory, throughput, and a short disposable smoke. A failure
that requires major framework changes or prevents reliable fitting terminates
with MODERN_PILOT_INFRA_BLOCKED. Models may be downloaded locally into the NFS
output namespace because direct remote Hugging Face access is unavailable;
all model computation runs on 10.42.0.197.

The intended formal budget is 10,000,000 predicted target tokens per run:
teacher CE adaptation, student CE preparation (S0), and three continuations
from the exact validation-selected S0 (CE, CE + KL at lambda 1, and CE + KL at
lambda 5). No other coefficients. Teacher is the validation-selected adapted
360M snapshot, frozen throughout all continuations. T=2, KL(teacher || student)
is averaged over valid next-token targets and multiplied by T squared.

Formal optimization and data packing will be locked in a second protocol
commit after Stage 0, before any endpoint training. The planned optimizer is
AdamW, lr 5e-5, weight decay 0.01, betas (0.9, 0.95), epsilon 1e-8; 5% linear
warmup and cosine decay; gradient clip 1.0; FP32 trainable weights with BF16
autocast; no GradScaler. Same sequence length and global batch for every arm
and domain. Stage 0 determines a fitting microbatch without inspecting gains.

TinyStories reuses the existing disk corpus and the original split function:
2% of original train held out by seed 42, original validation used as test.
FineWeb-Edu uses a pinned official sample-10BT revision, deterministic source
order, disjoint document splits, 20M train / 1M validation / 1M test tokens.
Only compact subsets are materialized under outputs/phase4a_modern; existing
datasets/ and checkpoints/ remain read-only. FineWeb-Edu starts only after
TinyStories completes cleanly. No test values enter checkpoint selection.

Before test evaluation, the replication criterion is fixed: absolute paired
CE-minus-KD test NLL gain >= 0.02, validation/test directions agree, and no
major optimization failure. At least one qualifying dataset/strength yields
MODERN_REPLICATION_WORTHWHILE; otherwise STOP_MODERN_REPLICATION. Single-seed
effects are pilots, not independent-initialization confirmation. Any numerical
training failure stops the affected protocol without tuning coefficients.

FineWeb-Edu appeared in SmolLM2 pretraining, and the two model sizes have
different pretraining token budgets. This limits claims about unseen data and
pure size effects. Do not attribute results to Grassmann geometry.

Preserve research/final_evidence and the manuscript. Do not launch Phase 3B,
Phase 2H, or seeds 123/456. Save compact audits, metrics and provenance, then
commit and STOP.
