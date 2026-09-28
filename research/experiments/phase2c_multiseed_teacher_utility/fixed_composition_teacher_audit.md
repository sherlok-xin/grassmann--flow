# Fixed-Composition Teacher-Quality Audit

## Audit criterion

A valid candidate must preserve teacher architecture, the two source branch identities, fusion semantics, and the evaluation alpha while changing teacher quality through a naturally existing training state. Phase 2C does not train, corrupt, or perturb a teacher.

All validation NLL values below use the same deterministic 512-chunk WikiText-2 subset with selection seed 20260920 and selection-index SHA256 `236633612db1e3144b8f3187a5bfb1795f558a50f26368252a8adf3f0b49413f`. For the valid comparison, both checkpoints are evaluated with alpha fixed to 0.5, independent of the alpha stored in each checkpoint.

| Candidate | Checkpoint | Architecture | Source branches | Evaluation alpha | Training stage | Validation NLL | Fixed-composition validity |
|---|---|---|---|---:|---|---:|---|
| Joint-trained teacher | `outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint/checkpoints/hybrid_best.pt` | Same late-logit scalar-fusion teacher, 37,749,633 parameters | Grassmann `20260317_165317_wt2_ws124_rd64`; Transformer `20260317_114514_wt2_baseline_both` | 0.5 | Epoch-10 validation-selected joint continuation; branches trainable after epoch 1 | 4.300646 | YES |
| Alpha-only teacher | `outputs/hybrid_experiments/20260328_084319_wt2_v1_hybrid_alpha_only/checkpoints/hybrid_best.pt` | Same late-logit scalar-fusion teacher, 37,749,633 parameters | Same two source runs and checkpoints | 0.5 override | Epoch-3 validation-selected alpha-only training; source branch weights frozen | 6.376223 | YES |
| Joint checkpoint `.orig` | `outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint/checkpoints/hybrid_best.pt.orig` | Same as joint-trained teacher | Same | 0.5 | Serialization backup of the joint checkpoint | 4.300646 by tensor identity | NO: not an independent quality condition |
| Alpha sweep within joint teacher | Joint-trained checkpoint with alpha 0.0--1.0 | Same | Same | Changes | Same trained checkpoint | See Phase 2A landscape | NO: changes Transformer/Grassmann composition |
| Source branch checkpoints | Individual Transformer or Grassmann runs | Single branch rather than fused teacher | One source branch only | Endpoint composition | Independently trained sources | See Phase 2A branch metrics | NO: changes architecture/composition |

## Integrity checks

The joint and alpha-only configurations reference exactly the same source branch run directories. Their state dictionaries contain the same 191 keys and compatible tensor shapes after reshaping the legacy scalar `logit_alpha` to the current length-one representation. At fixed alpha 0.5, architecture and fusion composition are therefore matched.

Checkpoint SHA256 values are `a5ad284f6895c0b4ed516c855c1b149e6636b945173aa2e842b8132299582694` for the joint teacher and `43d288db1dba826d8ad1bb3fcbd09b9720bdde24c198e858e00ce4c4658df012` for the alpha-only teacher. Both configurations resolve the same source checkpoint hashes: `6916246499dc2e6066c21760421571c76d9f63f1006fab7de838534ef2f733f4` for Grassmann and `4c166d20245bd923dbe126b3f9be39dbb1947d568358fd6cc152c7b9faaafe6c` for Transformer.

The branch tensors are not identical: 179 of 191 state tensors differ between the joint and alpha-only checkpoints because the joint run updates branch parameters after the first epoch, while the alpha-only run keeps them frozen. This is the intended quality intervention. It represents naturally existing teacher training stage or branch continuation, not artificial corruption.

The `.orig` joint checkpoint is not a separate condition. All 191 tensors are numerically identical to the current joint checkpoint; only the saved alpha tensor shape differs between scalar and length-one formats.

No usable intermediate per-epoch teacher checkpoints exist. The JSONL files contain metrics but not recoverable teacher states. Other WikiText-2 hybrid artifacts either change model size, source checkpoints, student architecture, or fusion composition and are not valid fixed-composition teacher-quality interventions.

## Phase 2D feasibility decision

The repository contains a valid two-condition fixed-composition intervention. The minimum teacher conditions are the alpha-only and joint-trained checkpoints above, both evaluated and distilled at fixed alpha 0.5. Their 2.075577-NLL validation separation is large, so a future pilot would be informative, although it would compare frozen-source versus jointly continued branch weights rather than finely spaced teacher quality.

No Phase 2D training is launched in this audit.
