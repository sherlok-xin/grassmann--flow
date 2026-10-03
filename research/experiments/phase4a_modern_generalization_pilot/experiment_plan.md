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

Formal optimization and data packing are locked before endpoint training.
Stage 0 passed (stage0_audit.json and fullbatch_smoke.json). The optimizer is
AdamW, lr 5e-5, weight decay 0.01, betas (0.9, 0.95), epsilon 1e-8; 5% linear
warmup and cosine decay; gradient clip 1.0; FP32 trainable weights with BF16
autocast; no GradScaler. Sequence length 256, global batch 32, microbatch 4
with eight accumulation passes. Use isolated Transformers 4.46.3 and
tokenizers 0.20.3; installed Transformers 4.57.6 does not import against this
NVIDIA PyTorch build. No framework source patch or global package change.
The first learning rate is positive (base lr / 62 warmup steps); total
optimizer updates are 1,221, with the final batch masked to reach exactly 10M
valid target tokens. CE/KD loss for each microbatch is weighted by its share
of the current global target count, including the partial final batch.

Teacher and S0 preparation use the same seed-42 permutation of training
chunks; all continuations use the same seed-4242 permutation. Chunks use
256 inputs, 256 next-token targets and one context token overlap. Training
orders can overlap between preparation and continuation, as in repeated
target adaptation; there is no claim of disjoint stage data. Validation/test
contain exactly 1M encoded tokens each (999,999 predicted targets). Evaluate
full validation at trained steps 256, 512, 768, 1024 and 1221; choose minimum
NLL, ties earlier. Step-zero NLL is recorded but is not selection-eligible.
All five runs finish and their selected hashes are fixed before test access.
The continuation teacher and S0 are selected from target-adapted snapshots.

TinyStories reuses the existing disk corpus and the original split function:
2% of original train held out by seed 42, original validation used as test.
FineWeb-Edu uses revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9,
official sample-10BT, deterministic source shard/row order, and document ID
SHA256 modulo 22 (0..19 train, 20 validation, 21 test). Both domains use
20M train / 1M validation / 1M test encoded tokens, exact-text deduplication
across splits, one EOS per document, no BOS, no tokenizer vocabulary edits.
When a split fills, its last document is truncated to the fixed count.
TinyStories source revision is historically unknown; local Arrow hashes,
fingerprints, original row IDs and exact token hashes identify the snapshot.
Only compact subsets are materialized under outputs/phase4a_modern; existing
datasets/ and checkpoints/ remain read-only. FineWeb-Edu starts only after
TinyStories completes cleanly. No test values enter checkpoint selection.

Before test evaluation, the replication criterion is fixed: absolute paired
CE-minus-KD test NLL gain >= 0.02, validation/test directions agree, and no
major optimization failure. A nonfinite loss/gradient/parameter, OOM, failed
run, incomplete budget or broken checkpoint provenance invalidates a run.
Clipping is measured and disclosed; sustained clipping alone is not proof
of a numerical failure and cannot be tuned away. At least one qualifying dataset/strength yields
MODERN_REPLICATION_WORTHWHILE; otherwise STOP_MODERN_REPLICATION. Single-seed
effects are pilots, not independent-initialization confirmation. Any numerical
training failure stops the affected protocol without tuning coefficients.

FineWeb-Edu appeared in SmolLM2 pretraining, and the two model sizes have
different pretraining token budgets. This limits claims about unseen data and
pure size effects. Do not attribute results to Grassmann geometry.

Preserve research/final_evidence and the manuscript. Do not launch Phase 3B,
Phase 2H, or seeds 123/456. Save compact audits, metrics and provenance, then
commit and STOP.
