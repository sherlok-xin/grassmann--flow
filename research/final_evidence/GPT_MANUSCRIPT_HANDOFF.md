# GPT Manuscript Reconstruction Handoff

## Authoritative state

Experiments are frozen after Phase 2G. Do not launch Phase 2H, run new training, tune coefficients, alter checkpoints, or develop a new method. The next authorized activity is manuscript reconstruction, but the full manuscript has not yet been written.

Start with these files, in order:

1. `FINAL_EVIDENCE_LEDGER.md`
2. `PAPER_CLAIMS.md`
3. `MANUSCRIPT_BLUEPRINT.md`
4. `OLD_MANUSCRIPT_AUDIT.md`
5. the six final table families in this directory

The old source `论文投稿/cac/conference_101719.tex` remains unchanged and is not the authoritative claim set.

## Recommended paper identity

One-sentence contribution: this paper provides a controlled empirical account of warm-start language-model distillation, showing reproducible positive and negative transfer boundaries and demonstrating that teacher training state and predictive behavior, rather than architectural heterogeneity alone, govern the tested outcomes.

The paper is an empirical study of teacher distillability and negative transfer. It is not a new Grassmann KD method paper. The Grassmann--Transformer teacher is the experimental test bed and supplies useful interventions, but Phase 2F removes any Grassmann-specific transfer claim.

## Four central claims

1. PTB and WikiText-2 show three-seed positive transfer, while TinyStories shows three-seed negative transfer under the frozen token-mean warm-start protocol. Mean paired NLL gains are `+0.144937`, `+0.155665`, and `-0.107219`.
2. Under fixed 0.5/0.5 composition, Teacher J transfers substantially better than Teacher A: mean `Q=+0.183319` NLL with 3/3 positive signs. State this as a teacher training-state/resulting-quality effect, not NLL-only causality.
3. Global teacher likelihood is insufficient. The globally weaker alpha-0.0 teacher still gives mean gain `+0.040991` with 3/3 positive signs, contradicting “worse teacher implies negative transfer.”
4. Fused supervision beats the TG components, but homogeneous TT beats TG by mean `0.012460` NLL with 3/3 consistent signs. Ensemble supervision is defensible; Grassmann-specific advantage is not.

## Red lines

Never claim that Plücker geometry causes better KD, TG heterogeneity is superior to TT, branch JSD is a strong transfer mechanism, CRBD succeeds, teacher residual quality is a universal sign law, warm-starting is necessary, dataset identity causally determines transfer, or Hybrid-lite has a uniquely Grassmann-specific efficiency advantage.

Do not mix CodeParrot or old batch-mean sweep winners into the confirmatory table. CodeParrot common-5k is a one-seed legacy observation only. Do not remove the Phase 2E `CLIP_SATURATION_WARNING`. Do not call TT and TG strictly parameter- or training-matched.

## Final table map

- `table_cross_domain_transfer`: main three-domain confirmatory result; legacy CodeParrot is separate.
- `table_teacher_quality_intervention`: Phase 2D fixed-composition Teacher J/A result.
- `table_teacher_source_ablation`: Phase 2E F/T/G result with clipping warning.
- `table_tt_vs_tg`: Phase 2F homogeneous control and confounds.
- `table_efficiency`: secondary PTB deployment operating points.
- `table_failed_or_negative_controls`: JSD, CRBD, TinyStories, Teacher A, and rejected claims.

Each table has CSV and Markdown forms. Exact per-seed source results remain in the corresponding Phase 2 experiment directories.

## Required writing discipline

Use effect sizes, sample SD, and sign consistency. With only three independent student seeds, do not make significance claims. Distinguish student-seed replication from fixed-teacher evidence. Separate confirmatory, partial, contradicted, and exploratory evidence. Treat conditional hard-token utility as observational. Describe latency as a one-RTX-3090, sequence-length-256 measurement using one checkpoint per model.

No new citation should be added from memory. Verify every reused or new reference programmatically before it enters the reconstructed manuscript.

## Next action

When explicitly authorized, create a new manuscript source from the selected venue template and reconstruct it according to `MANUSCRIPT_BLUEPRINT.md`. Do not incrementally polish the CAC text or overwrite it. Preserve the old manuscript as an archival record.
