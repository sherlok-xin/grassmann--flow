# Grassmann Flows: Project Overview for AI Review

This file is the shortest reliable entry point for reviewing the repository without access to local checkpoints, datasets, or historical chat context. The repository began as a Grassmann-flow language-model reproduction and later became an empirical study of knowledge distillation from a heterogeneous Grassmann--Transformer teacher into smaller Hybrid-lite students.

## Current research question

The strongest current question is not whether Grassmann mixing replaces attention. It is whether output-level KD from a frozen Grassmann--Transformer teacher transfers reliably to a smaller hybrid student, how teacher quality and branch source change that transfer, and where negative transfer occurs. Current evidence supports a teacher-quality effect and a small fused-versus-Transformer transfer advantage on WikiText-2, plus a domain-dependent KD boundary. It does not establish a new distillation method or a Grassmann-specific causal advantage.

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

The WikiText-2 Phase 2C experiment supplies the reusable three-seed C0, fused, and Grassmann-only baselines used by the newer Phase 2E result below. It uses three independently trained student initializations, the same frozen teacher branches, token-normalized KD, temperature 2, lambda 5, and validation-selected checkpoints. T_a05 denotes fusion alpha 0.5, and T_a00 denotes alpha 0.0.

| Quantity | Seed 42 | Seed 123 | Seed 456 | Mean +/- sample SD |
|---|---:|---:|---:|---:|
| KD gain with T_a05 | 0.162501 | 0.147337 | 0.157157 | 0.155665 +/- 0.007692 |
| KD gain with T_a00 | 0.046771 | 0.033598 | 0.042604 | 0.040991 +/- 0.006733 |
| Paired contrast D | 0.115731 | 0.113739 | 0.114553 | 0.114674 +/- 0.001001 |

Here KD gain is `test_NLL_C0 - test_NLL_KD`, and `D = gain_a05 - gain_a00`. Every value is positive. Teacher quality therefore changes the magnitude of transfer very consistently across these three student seeds. However, the globally worse T_a00 teacher still improves the student, so negative residual teacher advantage is not a deterministic negative-transfer law.

Offline token-level analysis shows a limited complementarity signal. For T_a00, mean gold-token utility is globally negative for all three students, but becomes positive in the highest student-loss quintile: 0.273620, 0.165971, and 0.200348 for seeds 42, 123, and 456. The positive-utility fraction in that quintile is 0.5746, 0.5460, and 0.5567. In contrast, mean utility over all student-wrong tokens remains negative and top-1 rescue is small. This is evidence of concentrated conditional gold-token information, not a proven correction mechanism.

Earlier matched evidence also shows that token-normalized warm-start KD improves PTB across three seeds, while the same fixed-strength protocol harms TinyStories for the tested seed. The bounded CRBD prototype did not repair that negative transfer and should not be presented as a successful method.

## Current Phase 2D status

Phase 2D is complete. Teacher J and Teacher A preserve architecture, source branches, fusion semantics, and effective alpha 0.5; 179 of 191 state tensors differ because joint training updated branch parameters. Their validation NLL values on the same 512 chunks are 4.300644 and 6.376220. Full-parameter KD-gradient ratios `G_A/G_J` are 1.102792, 1.148522, and 1.093056, so no relative scale mismatch was detected. An explicitly authorized matched full-data J/A stability gate passed before formal training.

For seeds 42/123/456, reused Teacher-J gain is +0.162501/+0.147337/+0.157157, whereas new Teacher-A gain is -0.022394/-0.035362/-0.025206. The primary paired contrast `Q=NLL_A-NLL_J` is +0.184895/+0.182699/+0.182364, with mean 0.183319 and sample SD 0.001375. Teacher training state and resulting quality therefore strongly modulate KD under fixed composition, but global NLL is still not a sufficient general rule because Phase 2C observed positive transfer from another globally weak teacher condition.

## Current Phase 2E status

Phase 2E is complete. It uses the same joint teacher checkpoint and compares fused alpha 0.5, Transformer-only alpha 1.0, and Grassmann-only alpha 0.0 under the matched three-seed continuation protocol. Only the three missing Transformer-only arms were trained; all other endpoints were reused from Phase 2C.

For seeds 42/123/456, `Gain_F` is 0.162501/0.147337/0.157157, `Gain_T` is 0.142785/0.125226/0.136593, and `Gain_G` is 0.046771/0.033598/0.042604. Fused > Transformer-only > Grassmann-only > C0 for every seed. The primary paired contrast `C_FT=NLL_T-NLL_F` is 0.019716/0.022110/0.020564, mean 0.020797 and sample SD 0.001214. This is 3/3 positive and narrowly exceeds the preregistered 0.02 practical threshold.

`CLIP_SATURATION_WARNING` is mandatory: Transformer-only clipping was 1.0 in the full-data gate and epoch 1 of every formal run. The research lead authorized continuation before endpoints because gradients were finite, T/F preflight ratios remained in range, and overflow/non-finite fractions stayed at 0.013605. Clipping declined strongly later, but early optimization constraints may partly affect the small `C_FT` gap.

The evidence supports only the bounded claim that adding the Grassmann branch yields a small, stable extra KD gain over Transformer-only supervision under this tested protocol. It does not isolate Grassmann geometry from generic ensemble diversity. The selected next experiment is a homogeneous Transformer+Transformer ensemble teacher control versus the Transformer+Grassmann teacher; it has not been launched.

## Evidence boundaries

- Do not claim that a teacher worse than the student necessarily causes negative transfer.
- Do not claim that CRBD is validated; its current bounded prototype failed.
- Do not claim a Grassmann-specific KD advantage without parameter-matched Transformer-only and Grassmann-only student controls.
- Do not attribute the Phase 2E fused-versus-Transformer gap specifically to Grassmann geometry; no homogeneous ensemble teacher control exists.
- Preserve `CLIP_SATURATION_WARNING` whenever reporting Phase 2E Transformer-only endpoints.
- Do not treat historical random-init, hidden-fusion, or cross-domain runs as fully matched causal comparisons.
- Do not compare legacy KD coefficients across datasets without accounting for token normalization.
- The active manuscript predates the newest controlled evidence and is an archival revision, not the definitive description of the latest experiments.

## Reading order

1. `AI_RESEARCH_HANDOFF.md` for the evidence audit and limitations.
2. `docs/project_map.md` for code and data flow.
3. `research/experiments/phase2e_teacher_branch_ablation/REPORT.md` for the newest controlled teacher-source result.
4. `research/experiments/phase2d_fixed_composition_teacher_quality/REPORT.md` for the fixed-composition teacher-state intervention.
5. `research/experiments/phase2c_multiseed_teacher_utility/REPORT.md` for the reused C0/fused/Grassmann-only endpoints.
6. `research/experiments/phase2b_controlled_teacher/REPORT.md` for the seed-42 intervention that motivated Phase 2C.
7. `research/experiments/phase2_teacher_transfer_audit/REPORT.md` for the cross-domain teacher landscape.
8. `research/experiments/post_rejection_program/analysis.md` and `research/experiments/h2_crbd/analysis.md` for matched PTB/TinyStories evidence and the failed method prototype.
9. `论文投稿/cac/conference_101719.tex` for the archived CAC manuscript source.

## Repository policy

Datasets, model checkpoints, full outputs, local tokenizers, raw high-volume token diagnostics, and training logs are intentionally excluded from GitHub. Compact configs, summaries, hashes, result tables, plots, experiment plans, provenance records, analysis code, and manuscript sources are included. Heavy experiments must run on server `10.42.0.197`; local edits operate on the shared NFS checkout.
