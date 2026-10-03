# Audit of `论文投稿/cac/conference_101719.tex`

## Overall judgment

The old manuscript cannot be revised by incremental polishing. Its main evidence comes from single-seed legacy coefficient sweeps and frames the study around a heterogeneous Grassmann--Transformer teacher. Phase 2A--2G changed both the evidence depth and the defensible contribution. The replacement manuscript should focus on reproducible transfer boundaries and teacher distillability, retain the Grassmann--Transformer model as the experimental setting, and delete every implication of a Grassmann-specific KD mechanism.

The classifications below refer to the current 518-line file. `KEEP` means that the content remains factually useful with only venue-level editing. `REWRITE` means that the topic remains but the evidence, definitions, or emphasis must change. `DELETE` means that the paragraph or claim must not enter the future narrative. `MOVE TO APPENDIX` means that the content remains useful for reproducibility or transparency but is not part of the main argument.

## Front matter

| Lines | Existing content | Decision | Required action |
|---:|---|---|---|
| 20--21 | Title centered on “Grassmann--Transformer Hybrids” and an empirical cross-domain study | REWRITE | Use a title centered on warm-start KD transfer boundaries or teacher distillability. Do not imply a new Grassmann KD method. |
| 23--33 | Authors and affiliations | KEEP | Preserve when migrating to the eventual target template; anonymize if the venue requires double-blind review. |
| 37--53 | Abstract based on four single-seed coefficient sweeps and PTB efficiency | DELETE | Replace completely with the three-seed PTB/WT2/TinyStories evidence, Phase 2D intervention, and Phase 2F negative control. CodeParrot must not appear as confirmatory evidence. |
| 55--58 | Keywords include Grassmann manifold and domain sensitivity | REWRITE | Prefer knowledge distillation, negative transfer, teacher quality, warm start, and language modeling. Retain Grassmann only if needed to describe the experimental teacher. |

## Introduction

| Lines | Existing content | Decision | Required action |
|---:|---|---|---|
| 62--67 | General KD motivation and speculative difficulty of learning from heterogeneous teachers | REWRITE | Motivate the unresolved question of teacher distillability and negative transfer. Remove the unsupported implication that heterogeneity itself makes transfer difficult. |
| 69--75 | Description of Transformer and Grassmann branches and fused teacher | REWRITE | Keep only enough architecture context to define the test bed. Do not imply that Plücker structure is the source of transferable knowledge. |
| 77--86 | Four-domain single-seed sweep summary and unmatched random-init caveat | DELETE | Replace with the matched three-seed positive/negative transfer matrix. Move the random-init mismatch to limitations or appendix. |
| 88--95 | Claimed contribution: four-dataset matched evaluation, loss reduction, and PTB deployment | REWRITE | Replace with the four final claims in `PAPER_CLAIMS.md`. The code result becomes legacy appendix evidence; teacher-state and TT controls become central. |

## Related work

| Lines | Existing content | Decision | Required action |
|---:|---|---|---|
| 99--115 | KD objectives, autoregressive KD, initialization, and same-capacity students | REWRITE | Retain the thematic structure, verify every citation before reuse, and connect prior work to teacher quality, negative transfer, and distillability rather than coefficient sweeps. |
| 117--128 | Heterogeneous sequence models and cross-architecture transfer | REWRITE | Keep as context, but state that the paper tests rather than assumes benefits of heterogeneous supervision. Add homogeneous ensemble controls to the narrative. |
| 130--134 | Grassmann geometry motivation | MOVE TO APPENDIX | Retain as architecture provenance. The main related-work section should not present geometry as a contribution or explanatory mechanism. |

## Method and experimental object

| Lines | Existing content | Decision | Required action |
|---:|---|---|---|
| 138--175 | Detailed causal Grassmann mixing and feature-wise gate | MOVE TO APPENDIX | The equations appear consistent with the implementation and remain useful for reproducibility, but they are not central to the empirical contribution. Recheck against code during manuscript writing. |
| 177--191 | Late-fusion teacher construction | KEEP | Preserve the scalar-logit fusion definition and joint-training schedule, while adding Teacher J/A and TT construction distinctions. |
| 193--200 | Student capacities and warm-start definitions | REWRITE | Retain definitions; distinguish the confirmatory 31.434M student from historical efficiency checkpoints and make seed-specific S0 independence explicit. |
| 202--221 | Legacy convex-weighted batch-mean KD objective | DELETE | The confirmatory paper must define token-mean KD as `CE + lambda * T^2 KL_token_mean`, with `lambda=5` and `T=2`. Legacy batch-mean sweeps belong only in historical context. |
| 223--237 | Architecture figure and detailed caption | MOVE TO APPENDIX | Replace the main Figure 1 with the experimental logic and key transfer boundaries. The architecture figure may be retained in the appendix after visual verification. |

## Experimental protocol

| Lines | Existing content | Decision | Required action |
|---:|---|---|---|
| 243--251 | Shared optimizer and hardware summary | REWRITE | Split confirmatory protocols by PTB/WT2/TinyStories and distinguish reused versus newly trained arms. Preserve validation selection and RTX 3090 execution facts. |
| 253--259 | Four datasets and CodeParrot subset | REWRITE | The main confirmatory setting contains PTB, WT2, and TinyStories. Label CodeParrot common-5k as a single-seed legacy observation in the appendix. |
| 261--281 | Dataset/model-size table | REWRITE | Replace with confirmatory data/protocol information. Do not imply compression on CodeParrot or a shared teacher size across controls. |
| 283--291 | Single-seed evidence disclaimer and unmatched random initialization | MOVE TO APPENDIX | Retain as a historical-evidence boundary. The main paper now has three seeds for PTB, WT2, and TinyStories. |

## Old RQ1: teacher quality

| Lines | Existing content | Decision | Required action |
|---:|---|---|---|
| 293--301 | Teacher compared with CE-from-scratch student across four datasets | DELETE | Replace with seed-matched teacher residual advantage relative to WS+CE and the controlled Phase 2D intervention. Scratch CE is not the correct transfer reference. |
| 303--320 | Four-domain teacher-versus-scratch table | DELETE | Replace with `table_teacher_quality_intervention` and the alpha-0.0 counterexample. |

## Old RQ2: cross-domain KD

| Lines | Existing content | Decision | Required action |
|---:|---|---|---|
| 322--331 | Single-seed best-coefficient conclusions | DELETE | Replace with `table_cross_domain_transfer`, using the fixed token-mean protocol and three independent seeds per dataset. |
| 333--352 | Four-domain warm-start table mixing legacy winners | DELETE | Do not mix CodeParrot or legacy winner selection with the confirmatory matrix. A separate appendix table may report the CodeParrot observation. |
| 354--364 | Legacy coefficient-sweep figure | MOVE TO APPENDIX | It may document the project history, but it must not be a primary result or imply comparable coefficient scales across reductions. |
| 366--372 | Claim that the teacher is better than TinyStories CE scratch while KD harms | DELETE | The matched Phase 2G result is stronger and different: WS+CE is better than the teacher for all three seeds. Replace with the exact residual advantages. |

## Old RQ3: efficiency

| Lines | Existing content | Decision | Required action |
|---:|---|---|---|
| 374--383 | PTB parameter/PPL/latency trade-off | KEEP | Retain as a secondary result with the three-seed PPL means and single-checkpoint latency scope. State that these are historical legacy KD operating points. |
| 385--401 | PTB operating-point table | REWRITE | Replace with `table_efficiency`, including batch 1/8/32 behavior rather than only batch 32. |
| 403--409 | Three-seed PPL and online-inference caveat | KEEP | Preserve. Use exact means and sample SDs: `51.8339±0.0554` and `53.9139±0.0692`. |
| 411--420 | Batch latency figure | KEEP | Retain if the target venue has space; caption must state one RTX 3090, sequence length 256, and one checkpoint per model. |

## Discussion and exploratory evidence

| Lines | Existing content | Decision | Required action |
|---:|---|---|---|
| 424--431 | Domain-varying sweep minima and untested geometry explanation | REWRITE | Use confirmed cross-domain sign variation. Retain the explicit statement that geometry is untested, strengthened by the Phase 2F contradiction. |
| 433--438 | Batch-mean loss-scale explanation | MOVE TO APPENDIX | Keep as provenance for legacy sweeps. The main method uses token-mean KD. |
| 440--445 | Teacher PPL does not predict benefit, based on scratch comparisons | REWRITE | Replace with the stronger Phase 2C/2D conclusion: global likelihood orders some effects but is not sufficient for distillability. |
| 447--457 | Unmatched random-initialization runs | MOVE TO APPENDIX | Preserve only as an explicit non-result. Do not claim warm-start necessity. |
| 459--470 | Unmatched PTB-to-code stress test | MOVE TO APPENDIX | Keep only if space permits as a provenance warning; it does not support the final paper claims. |

## Limitations and conclusion

| Lines | Existing content | Decision | Required action |
|---:|---|---|---|
| 472--488 | Limitations centered on single-seed sweeps and legacy normalization | REWRITE | Replace with remaining limitations: three student seeds but fixed teachers, teacher-quality/training confounds, TT/TG mismatch, Transformer clipping warning, one-hardware latency, small models, and observational hard-token diagnostics. |
| 490--497 | Old future-experiment TODO comments | DELETE | Experiments are frozen. Do not include proposals that imply missing required experiments; document remaining limitations without scheduling Phase 2H. |
| 499--513 | Conclusion claiming positive transfer on PTB/WT2/code and negative TinyStories | DELETE | Replace with the final four-claim synthesis. Code must be labeled legacy, and the conclusion must state that homogeneous TT outperforms TG. |
| 515--516 | Bibliography commands | KEEP | Reuse only after all BibTeX records and attributed claims are programmatically verified. |

## Claims that must be removed from the future narrative

The future manuscript must delete or explicitly reject the following propositions: Grassmann-specific KD advantage; Plücker geometry as the cause of better transfer; superiority of heterogeneous TG ensembles over homogeneous TT ensembles; branch disagreement as a strong transfer mechanism; CRBD as a successful method; a universal rule that a globally worse teacher causes negative transfer; warm-start necessity; unique Grassmann-specific efficiency; and causal attribution of the transfer sign to dataset identity.

## Material that can survive

The most reusable components are the factual architecture definition, late-fusion equation, teacher construction procedure, warm-start definition, PTB efficiency benchmark, citation inventory, and explicit caveats around unmatched random-init and cross-domain runs. These components must be embedded in the new teacher-distillability narrative rather than used to preserve the old Grassmann-centered story.
