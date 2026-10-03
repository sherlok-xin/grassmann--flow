# Grassmann Flows: Final Project Overview for AI Review

## Authoritative entry point

The authorized Phase 4A modern external-validity pilot is complete and STOPPED.
Read `research/experiments/phase4a_modern_generalization_pilot/GPT_HANDOFF.md`
and `REPORT.md` for the latest result: FineWeb-Edu lambda5 has seed42 test
gain -0.046403663516, meeting the pilot replication gate, with permanent
CLIP_SATURATION_WARNING. TinyStories effects are below the clear-effect
threshold. No independent-seed replication is authorized. Phase 3A remains
terminal STOP_NEW_METHOD_DIAGNOSTIC_FAILED. Do not run new training, tune
coefficients, start Phase 3B/2H, edit the manuscript, or infer claims from
historical filenames.

For the frozen Phase 2A--2G manuscript evidence, read
`research/final_evidence/GPT_MANUSCRIPT_HANDOFF.md`. That package remains
unchanged by the post-freeze extensions. The authoritative claim ledger is
`research/final_evidence/FINAL_EVIDENCE_LEDGER.md`; the conservative claim set
and paper structure are in `PAPER_CLAIMS.md` and `MANUSCRIPT_BLUEPRINT.md`.
The CAC source is an archival manuscript and does not describe the final evidence.

## Final research question

The project asks when output-level warm-start KD transfers useful information to a smaller language model and when it causes negative transfer. The Grassmann--Transformer teacher is an experimental test bed that enables controlled teacher-state, branch-source, and ensemble comparisons. The final contribution is an empirical account of teacher distillability, not a new Grassmann KD method.

## Confirmed transfer boundaries

All confirmatory comparisons use independent seed-specific S0 checkpoints, matched WS+CE controls, token-mean KD with `lambda=5` and temperature 2, validation-NLL checkpoint selection, and test-only final reporting.

| Dataset | Seeds | Mean KD gain in NLL | Sample SD | Signs | Decision |
|---|---:|---:|---:|---:|---|
| PTB | 3 | +0.144937 | 0.003938 | 3/3 positive | positive transfer confirmed |
| WikiText-2 | 3 | +0.155665 | 0.007692 | 3/3 positive | positive transfer confirmed |
| TinyStories | 3 | -0.107219 | 0.000415 | 3/3 negative | negative transfer confirmed |

CodeParrot common-5k has a positive one-seed legacy result but is excluded from the confirmatory table because it uses the old loss normalization and winner selection.

## Controlled teacher evidence

Phase 2D compares two WikiText-2 TG teachers at fixed architecture, source branches, fusion semantics, and effective alpha 0.5. Teacher J improves all three students, whereas Teacher A harms all three. `Q=NLL_A-NLL_J` has mean `+0.183319` NLL and sample SD `0.001375`. This confirms a teacher training-state/resulting-quality effect, not scalar-NLL-only causality.

Phase 2C provides the key counterexample to a universal quality law. The alpha-0.0 teacher is globally worse than the matched student for every seed but still yields mean gain `+0.040991` NLL with 3/3 positive signs. Global teacher likelihood is informative but insufficient to determine distillability.

## Ensemble and Grassmann claim boundary

Phase 2E finds mean KD gains of `0.155665`, `0.134868`, and `0.040991` for fused, Transformer-only, and Grassmann-only supervision. The fused-versus-Transformer difference is `+0.020797` NLL but carries a permanent `CLIP_SATURATION_WARNING` because the Transformer-only gate and epoch 1 clipped every step.

Phase 2F is decisive for the architecture narrative. The homogeneous TT teacher outperforms TG for all three students: `H=Gain_TG-Gain_TT` is `-0.012460±0.001348` NLL. TG has higher branch JSD and fusion gain, but those quantities do not yield better downstream transfer. Teacher quality, parameter count, and alpha-learning-rate differences remain confounds, so the result rejects TG superiority without proving intrinsic TT superiority.

The manuscript must not claim a Grassmann-specific KD advantage, Plücker-geometry causality, superiority of architectural heterogeneity, a strong branch-JSD mechanism, or successful CRBD. Stage C CRBD fails its bounded prototype gate and is worse than shuffled routing under a clip-saturated setup.

## Efficiency result

The PTB teacher, quality student, and efficiency student have `36.264M`, `31.434M`, and `23.565M` parameters and PPL `50.1125`, `51.8339±0.0554`, and `53.9139±0.0692`. Batch-32 latency is `55.984`, `48.460`, and `34.959 ms`. Latency uses one frozen checkpoint per model on one RTX 3090 at sequence length 256. The quality student is not faster at batch sizes 1 or 8; the efficiency student is faster at all three measured batch sizes.

## Main implementation files

- `src/models/grassmann_v4.py`: causal multi-window Grassmann mixer and feature-wise gate.
- `train_hybrid_latefusion_alpha_ddp_v1.py`: TG teacher training.
- `train_hybrid_tt_latefusion_alpha_ddp_v1.py`: TT teacher training.
- `train_hybrid_lite_latefusion_baseline_v2.py`: Hybrid-lite S0 training.
- `train_distill_hybrid_lite_from_latefusion_teacher_v2.py`: warm-start KD entry point.
- `src/kd_losses.py`: legacy and token-normalized KD losses.
- `src/crbd_losses.py`: failed bounded routing prototype, not a validated method.
- `benchmark_ptb_efficiency_v4.py`: quality and efficiency benchmark.

## Reading order

For the latest pilot, read the Phase 4A handoff/report and then
`AI_RESEARCH_HANDOFF.md`. The following order concerns the immutable legacy
manuscript-evidence package only; do not mix its confirmatory endpoints with
the modern single-seed pilot.

1. `research/final_evidence/GPT_MANUSCRIPT_HANDOFF.md`
2. `research/final_evidence/FINAL_EVIDENCE_LEDGER.md`
3. `research/final_evidence/PAPER_CLAIMS.md`
4. `research/final_evidence/MANUSCRIPT_BLUEPRINT.md`
5. `research/final_evidence/OLD_MANUSCRIPT_AUDIT.md`
6. Phase 2C--2G reports for exact experimental provenance
7. `AI_RESEARCH_HANDOFF.md` for the complete historical chain

## Repository policy

Datasets, checkpoints, full outputs, local tokenizers, training logs, and high-volume diagnostics are excluded from GitHub. Compact configs, summaries, hashes, tables, figures, protocols, provenance, and manuscript-planning artifacts are included. The old CAC source remains unchanged until full manuscript reconstruction is explicitly authorized.
