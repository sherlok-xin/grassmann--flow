# Grassmann Flows: Project Overview for AI Review

This file is the shortest reliable entry point for reviewing the repository without access to local checkpoints, datasets, or historical chat context. The repository began as a Grassmann-flow language-model reproduction and later became an empirical study of knowledge distillation from a heterogeneous Grassmann--Transformer teacher into smaller Hybrid-lite students.

## Current research question

The strongest current question is not whether Grassmann mixing replaces attention. It is whether output-level KD from a frozen Grassmann--Transformer teacher transfers reliably to a smaller hybrid student, how teacher quality changes that transfer, and where negative transfer occurs. The current evidence supports a teacher-quality effect on WikiText-2 and a domain-dependent KD boundary, but it does not establish a new distillation method or a Grassmann-specific causal advantage.

## Active architecture and objective

The teacher contains independent Transformer and Grassmann branches. Their final logits are combined as `alpha * Transformer + (1 - alpha) * Grassmann`. The Hybrid-lite student has the same two-branch topology at a smaller scale. Current controlled experiments warm-start the student from a CE checkpoint and optimize `CE + lambda_kd * KL_token_mean` with the teacher frozen.

The main implementation files are:

- `src/models/grassmann_v4.py`: causal multi-window Grassmann mixer with normalized Plucker coordinates and a feature-wise gate.
- `train_exp4_ddp.py`: shared dataset and Transformer definitions used by later scripts.
- `train_hybrid_latefusion_alpha_ddp_v1.py`: late-logit fusion teacher training.
- `train_hybrid_lite_latefusion_baseline_v2.py`: Hybrid-lite CE baseline training.
- `train_distill_hybrid_lite_from_latefusion_teacher_v2.py`: current warm-start KD entry point.
- `src/kd_losses.py`: legacy batch-normalized and current token-normalized KD losses.
- `src/crbd_losses.py`: bounded experimental routing objectives; current prototype did not pass its gate.
- `benchmark_ptb_efficiency_v4.py`: PPL, latency, throughput, memory, and parameter evaluation.

## Most reliable experimental results

The WikiText-2 Phase 2C experiment is the newest controlled result. It uses three independently trained student initializations, the same frozen teacher branches, token-normalized KD, temperature 2, lambda 5, and validation-selected checkpoints. T_a05 denotes fusion alpha 0.5, and T_a00 denotes alpha 0.0.

| Quantity | Seed 42 | Seed 123 | Seed 456 | Mean +/- sample SD |
|---|---:|---:|---:|---:|
| KD gain with T_a05 | 0.162501 | 0.147337 | 0.157157 | 0.155665 +/- 0.007692 |
| KD gain with T_a00 | 0.046771 | 0.033598 | 0.042604 | 0.040991 +/- 0.006733 |
| Paired contrast D | 0.115731 | 0.113739 | 0.114553 | 0.114674 +/- 0.001001 |

Here KD gain is `test_NLL_C0 - test_NLL_KD`, and `D = gain_a05 - gain_a00`. Every value is positive. Teacher quality therefore changes the magnitude of transfer very consistently across these three student seeds. However, the globally worse T_a00 teacher still improves the student, so negative residual teacher advantage is not a deterministic negative-transfer law.

Offline token-level analysis shows a limited complementarity signal. For T_a00, mean gold-token utility is globally negative for all three students, but becomes positive in the highest student-loss quintile: 0.273620, 0.165971, and 0.200348 for seeds 42, 123, and 456. The positive-utility fraction in that quintile is 0.5746, 0.5460, and 0.5567. In contrast, mean utility over all student-wrong tokens remains negative and top-1 rescue is small. This is evidence of concentrated conditional gold-token information, not a proven correction mechanism.

Earlier matched evidence also shows that token-normalized warm-start KD improves PTB across three seeds, while the same fixed-strength protocol harms TinyStories for the tested seed. The bounded CRBD prototype did not repair that negative transfer and should not be presented as a successful method.

## Evidence boundaries

- Do not claim that a teacher worse than the student necessarily causes negative transfer.
- Do not claim that CRBD is validated; its current bounded prototype failed.
- Do not claim a Grassmann-specific KD advantage without parameter-matched Transformer-only and Grassmann-only student controls.
- Do not treat historical random-init, hidden-fusion, or cross-domain runs as fully matched causal comparisons.
- Do not compare legacy KD coefficients across datasets without accounting for token normalization.
- The active manuscript predates the newest controlled evidence and is an archival revision, not the definitive description of the latest experiments.

## Reading order

1. `AI_RESEARCH_HANDOFF.md` for the evidence audit and limitations.
2. `docs/project_map.md` for code and data flow.
3. `research/experiments/phase2c_multiseed_teacher_utility/REPORT.md` for the newest controlled result.
4. `research/experiments/phase2b_controlled_teacher/REPORT.md` for the seed-42 intervention that motivated Phase 2C.
5. `research/experiments/phase2_teacher_transfer_audit/REPORT.md` for the cross-domain teacher landscape.
6. `research/experiments/post_rejection_program/analysis.md` and `research/experiments/h2_crbd/analysis.md` for matched PTB/TinyStories evidence and the failed method prototype.
7. `论文投稿/cac/conference_101719.tex` for the archived CAC manuscript source.

## Repository policy

Datasets, model checkpoints, full outputs, local tokenizers, raw high-volume token diagnostics, and training logs are intentionally excluded from GitHub. Compact configs, summaries, hashes, result tables, plots, experiment plans, provenance records, analysis code, and manuscript sources are included. Heavy experiments must run on server `10.42.0.197`; local edits operate on the shared NFS checkout.
