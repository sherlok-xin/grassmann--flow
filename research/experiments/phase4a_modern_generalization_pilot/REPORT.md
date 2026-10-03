# Phase 4A — Modern warm-start KD external-validity pilot

## Decision

`DECISION = MODERN_REPLICATION_WORTHWHILE`

The qualifying result is negative transfer on FineWeb-Edu at lambda=5:
test NLL gain is -0.046403663516 and validation gain is -0.046727056123.
All formal optimization is finite, and the test effect exceeds the
preregistered absolute 0.02 threshold with matching validation direction.
This is a seed-42 pilot under sustained clipping, not multiseed confirmation
or modern external validation of beneficial KD. Independent student seeds
123/456 are recommended for research-lead review only. No replication has
been launched, and Phase 4A stops here.

TinyStories has small, opposite-sign gains at the two fixed strengths, neither
meeting the clear-effect threshold. Its earlier strong negative-transfer
boundary is not clearly reproduced in this modern pretrained setting.

## Frozen setting and execution

Each domain has five completed seed-42 runs: 360M teacher CE adaptation,
135M student CE preparation, and CE/KD1/KD5 continuations from the exact same
validation-selected S0. Every run processed exactly 10,000,000 target tokens
in 1,221 updates; total formal budget is 100M targets. Token files contain
20M train / 1M validation / 1M test encoded tokens per domain. Evaluation
averages 999,999 next-token targets per held-out split.

The loss is CE + lambda times T-squared KL(teacher || student), with T=2,
token_mean reduction, lambda in {1,5}, a frozen adapted teacher, sequence
length 256, global batch 32 and microbatch 4. AdamW uses lr=5e-5, weight
decay=0.01, betas=(0.9,0.95), epsilon=1e-8, 5% warmup and cosine decay.
FP32 trainable weights use BF16 autocast and FP32 CE/KL; clipping remains 1.
Checkpoint selection uses full validation NLL at trained steps 256, 512,
768, 1024 and 1221. Step zero is recorded but not selection-eligible.
The complete protocol was committed before endpoint training; see
experiment_plan.md and provenance.md.

FineWeb S0 selected step 256, corresponding to 2,097,152 targets, although
its preparation run completed the full 10M budget. Every other selected
snapshot, including all six continuations, is at step 1221. This selection
difference is disclosed; the final unselected S0 was not substituted.
Both held-out splits and all selected hashes were frozen before final test
evaluation. No test result selected a teacher, checkpoint or coefficient.

## Primary paired effects

Positive gain means KD improves the matched CE continuation. One independent
student preparation seed was run per domain; sample SD and 3/3 consistency
are undefined and are not reported. No unmatched legacy endpoint is included.

| Dataset | Lambda | Validation gain | Test gain | Direction agrees | Clear effect |
|---|---:|---:|---:|---|---|
| TinyStories | 1 | +0.004337605834 | +0.004948632001 | yes | no |
| TinyStories | 5 | -0.005984952475 | -0.004797389422 | yes | no |
| FineWeb-Edu | 1 | -0.015706368649 | -0.015718598299 | yes | no |
| FineWeb-Edu | 5 | -0.046727056123 | -0.046403663516 | yes | yes |

FineWeb-Edu lambda=5 raises test PPL from 19.104049 to 20.011437. The
lambda=1 arm is also harmful, but its effect is below the replication
threshold. TinyStories lambda=1 provides a small benefit and lambda=5 a
small harm; these directions suggest strength sensitivity without establishing
a robust modern TinyStories boundary. The FineWeb outcome is consistent with
setting-dependent KD and does not supply modern evidence of beneficial transfer.

## Selected endpoints

| Dataset | Arm | Validation NLL | Test NLL | Test PPL | Selected step |
|---|---|---:|---:|---:|---:|
| TinyStories | Teacher | 1.404917 | 1.452421 | 4.273448 | 1221 |
| TinyStories | S0 | 1.578774 | 1.619276 | 5.049434 | 1221 |
| TinyStories | CE | 1.534506 | 1.579429 | 4.852187 | 1221 |
| TinyStories | KD1 | 1.530168 | 1.574481 | 4.828234 | 1221 |
| TinyStories | KD5 | 1.540491 | 1.584227 | 4.875521 | 1221 |
| FineWeb-Edu | Teacher | 2.747486 | 2.726017 | 15.271938 | 1221 |
| FineWeb-Edu | S0 | 2.967112 | 2.944338 | 18.998085 | 256 |
| FineWeb-Edu | CE | 2.973155 | 2.949900 | 19.104049 | 1221 |
| FineWeb-Edu | KD1 | 2.988861 | 2.965619 | 19.406711 | 1221 |
| FineWeb-Edu | KD5 | 3.019882 | 2.996304 | 20.011437 | 1221 |

FineWeb CE is itself slightly worse than S0. Both KD arms further degrade
it, so their negative paired gains are not improvements hidden by a stronger
CE baseline. Validation-selected continuation comparisons remain the
preregistered primary endpoint; no post-hoc selection of S0 as the CE endpoint
was performed.

## Teacher residual and optimization audit

| Dataset | Teacher residual advantage over S0 (validation) | Initial CE gradient norm | Initial unweighted KD gradient norm | KD/CE ratio at lambda=1 | KD/CE ratio at lambda=5 |
|---|---:|---:|---:|---:|---:|
| TinyStories | +0.173856356061 | 0.774641 | 4.333653 | 5.594400 | 27.972002 |
| FineWeb-Edu | +0.219626011745 | 1.058419 | 2.276195 | 2.150561 | 10.752805 |

Initialization probes use the same first continuation global training batch
and the complete student parameter set, without an optimizer update or
test access. Initial T-squared KL is 0.953603 on TinyStories and 1.365737 on
FineWeb-Edu. Reciprocal CE/KD ratios and state hashes are in the raw
initialization audit JSON files.

| Dataset | Arm | Clip fraction | Mean preclip gradient norm | Mean T-squared KL | Nonfinite fraction |
|---|---|---:|---:|---:|---:|
| TinyStories | CE | 0.004914 | 0.841764 | 0 | 0 |
| TinyStories | KD1 | 0.479934 | 1.050601 | 0.660231 | 0 |
| TinyStories | KD5 | 1.000000 | 2.614984 | 0.616611 | 0 |
| FineWeb-Edu | CE | 0.058968 | 0.929848 | 0 | 0 |
| FineWeb-Edu | KD1 | 0.850123 | 1.084600 | 1.092807 | 0 |
| FineWeb-Edu | KD5 | 1.000000 | 2.557157 | 1.043843 | 0 |

`CLIP_SATURATION_WARNING` applies permanently to both lambda=5 arms. Their
gradients trigger clipping at every optimizer step. Lambda=1 clipping is also
substantial, particularly on FineWeb-Edu. These results concern the specified
clipped optimization protocol; they do not isolate teacher information from
optimization constraints. Clipping is a possible confound, not an experimentally
established cause of harm. No coefficient or clipping threshold was changed.
Under the preregistered validity rule, clipping alone is a disclosed constraint
rather than a nonfinite/OOM/incomplete-run failure.

All losses, preclip gradient norms and updated parameters remained finite.
Observed overflow/nonfinite fractions are zero. BF16 uses no GradScaler, so
the overflow field records no observed nonfinite event, not an FP16 scaler
skip-rate comparison. Norm/KL means above are arithmetic means over optimizer
steps; held-out NLL uses exact target-token weighting. Formal peak allocated
memory is 8.133 GiB for teacher adaptation, 3.559 GiB for S0/CE and 5.070 GiB
for KD. Optimization_audit.csv contains all ten run records, including
preparation telemetry and wall times (excluding loading and initial validation).

## Infrastructure and data limits

Official base SmolLM2-135M/360M snapshots have 134,515,008 / 361,821,120
unique parameters and exactly matching 49,152-entry tokenizers. Weight files
match official LFS hashes. No instruct model or model substitution was used.
Stage 0 passed four numerical/accounting tests and disposable model/fullbatch
smokes. The installed Transformers4.57.6 failed a NVIDIA-PyTorch internal
import; an isolated Transformers4.46.3/tokenizers0.20.3 directory resolves this
without framework source patches or changes to the shared virtualenv.

TinyStories retains the original train/validation/test document split mapping,
but uses a fixed compact subset and the SmolLM2 tokenizer. Its historical
upstream commit is unknown; the saved Arrow hashes and document IDs identify
the actual source. Modern absolute NLL is not compared with GPT2-tokenized
legacy scores. FineWeb-Edu uses pinned official sample-10BT first-shard data,
deterministic document-ID hashing and exact-text deduplication. Both domains
have disjoint train/validation/test IDs and text hashes within the materialized
subset. Document-order manifests and token hashes are committed; tokens and
large model states are not.

The model cards include FineWeb-Edu in pretraining. Exact document overlap is
unknown, and the model sizes have different pretraining budgets (2T/4T).
Thus this is a modern pretrained-family/target-corpus pilot, not validation on
guaranteed unseen corpus content or a pure architecture-size intervention.
The deterministic subset, short 256-token context, finite adaptation budget,
early selected FineWeb S0 and single student seed further limit generalization.
See model_and_data_audit.md for licenses, pinned revisions and official sources.

## Claims and stop boundary

The allowed descriptive conclusion is that lambda=5 warm-start KD harms the
modern SmolLM2 student on the fixed FineWeb-Edu subset in this finite, persistently
clipped pilot, despite a better teacher likelihood. This warrants independent
student replication under the same frozen protocol. It does not establish a
universal harmful effect, explain its mechanism, or demonstrate reproducibility.
TinyStories effects remain below the clear-effect threshold, and no setting
in this pilot provides a qualifying positive-transfer result.

No Grassmann-specific KD, Plucker causality, heterogeneous-ensemble superiority,
new loss, routing or Safe-KD claim follows from these runs. The frozen final
evidence package and manuscript remain unchanged. The recommended next step
is research-lead review of an unchanged FineWeb CE/KD1/KD5 replication with
independently prepared student seeds123/456 and the same frozen seed42 teacher.
That is a recommendation only. Phase 4A is complete and STOPPED; no new
training is launched.
