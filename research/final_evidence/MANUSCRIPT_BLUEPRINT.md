# Manuscript Blueprint

## Proposed narrative

The manuscript should be an empirical study of teacher distillability under warm-start language-model KD. Its central contribution is not a new loss or a Grassmann-specific mechanism. The paper shows that a fixed KD protocol can transfer positively or negatively across text domains, isolates a large teacher-state effect under fixed composition, demonstrates that global teacher likelihood is informative but insufficient, and uses a homogeneous TT control to reject an architectural-heterogeneity advantage.

A suitable working title is: “When Does Warm-Start Distillation Transfer? Controlled Evidence from Teacher Quality and Ensemble Composition.” The final title should be selected only when the target venue is known.

## 1. Introduction

Main argument: KD success cannot be inferred from the existence of a trained teacher or from aggregate teacher quality alone. The project provides a matched empirical map of positive and negative warm-start transfer and then tests candidate explanations through controlled teacher interventions and ensemble controls.

Exact experiments: PTB A1/A2, Phase 2C C0/C1, Phase 2G, Phase 2D, and Phase 2F.

Required figure/table: Figure 1 should summarize the experimental logic and show the three-domain gain signs plus the Teacher J/A and TT/TG outcomes. Cite `table_cross_domain_transfer`, `table_teacher_quality_intervention`, and `table_tt_vs_tg` as the numerical anchors.

Claims that must NOT be made: a new successful KD method; Grassmann-specific transfer; causal benefit from Plücker geometry; domain identity as the cause; universal teacher-quality law.

## 2. Related Work

Main argument: position the paper among logit KD for autoregressive models, negative transfer and teacher/student compatibility, teacher quality and same-capacity distillation, cross-architecture transfer, and ensemble distillation. The gap is controlled evidence about when a warm-start student benefits from or is harmed by a fixed teacher signal.

Exact experiments used: none as evidence; later sections establish the contribution.

Required figure/table: none. Reuse and verify relevant citations from the old bibliography; add no citation without programmatic verification.

Claims that must NOT be made: novelty of the Grassmann operator; comprehensive coverage of all KD objectives; priority claims not established by verified literature search.

## 3. Experimental Setting and Warm-Start KD

Main argument: define the teacher/student pipeline, S0 warm start, matched WS+CE control, token-mean KL, `lambda=5`, temperature 2, validation-based selection, independent student seeds, and gain conventions. Explain that the TG teacher is the principal test bed rather than the claimed innovation.

Exact experiments: confirmatory PTB, Phase 2C, and Phase 2G protocols; teacher construction provenance from Phase 2A and Phase 2D.

Required figure/table: a compact pipeline schematic showing S0 branching into matched CE and KD continuations; a protocol table in the appendix. The detailed Grassmann architecture figure and equations move to the appendix.

Claims that must NOT be made: protocol identity across datasets when data and teacher checkpoints differ; random-init comparisons; equivalence between legacy batch-mean coefficients and token-mean coefficients without conversion.

## 4. Teacher Construction and Controlled Interventions

Main argument: explain Teacher J, Teacher A, fused/branch-only deployment, and TT construction. Separate fixed-composition teacher-state interventions from teacher-source and teacher-architecture interventions.

Exact experiments: Phase 2B preflight, Phase 2D integrity audit and formal endpoints, Phase 2E branch deployment, and Phase 2F TT construction.

Required figure/table: intervention diagram with axes for parameter state, source composition, and ensemble architecture; `table_teacher_quality_intervention`.

Claims that must NOT be made: Teacher J/A differs only in scalar NLL; TT and TG are strictly parameter-matched; alpha-learning-rate equality; causal isolation of architecture in Phase 2F.

## 5. Positive and Negative Transfer Across Domains

Main argument: the same token-mean coefficient gives reproducible positive transfer on PTB and WT2 and reproducible negative transfer on TinyStories. Emphasize effect size, sample SD, and 3/3 sign consistency rather than significance claims from only three seeds.

Exact experiments: PTB A1/A2, Phase 2C C0/C1, and Phase 2G.

Required figure/table: `table_cross_domain_transfer`; a paired or forest-style figure of per-seed NLL gains with zero line. CodeParrot appears only in a separately labeled appendix legacy table.

Claims that must NOT be made: CodeParrot confirmation; domain identity causes the sign; universal optimality or failure of `lambda=5`; broad large-model generalization.

## 6. What Determines Teacher Distillability?

Main argument: teacher training state and quality exert a large controlled effect, but global teacher likelihood is not sufficient. The Phase 2C alpha-0.0 condition falsifies the universal rule “worse teacher implies harmful KD,” while Phase 2D shows that a severely degraded fixed-composition teacher can cross into negative transfer.

Exact experiments: Phase 2C residual ordering and hard-token audit; Phase 2D Teacher J/A.

Required figure/table: `table_teacher_quality_intervention`; optional appendix plot of conditional utility. A main-text conceptual panel can contrast global residual quality with endpoint gain without fitting an underpowered regression.

Claims that must NOT be made: global NLL is irrelevant; hardest-quintile utility causally produces the gain; a deterministic residual-quality threshold; teacher quality explains every domain.

## 7. Ensemble and Branch Ablations

Main argument: fused TG supervision outperforms either component in the fixed checkpoint, but the homogeneous TT ensemble outperforms TG. Greater TG JSD and fusion gain therefore do not translate into greater downstream transfer.

Exact experiments: Phase 2E and Phase 2F; Stage B JSD diagnostic as a negative mechanism test.

Required figure/table: `table_teacher_source_ablation` and `table_tt_vs_tg`; reuse the Phase 2F paired-gain figure if visually suitable. Put `CLIP_SATURATION_WARNING` next to the F/T comparison.

Claims that must NOT be made: Grassmann branch uniquely improves KD; architectural heterogeneity is superior; TT is intrinsically superior independent of teacher quality; branch JSD is a strong mechanism; Plücker geometry is causal.

## 8. Efficiency / Deployment Trade-offs

Main argument: the frozen PTB checkpoints illustrate two modest student operating points. One retains closer PPL with limited compression and only large-batch speedup; the smaller student gives consistent latency reductions at all measured batch sizes with a larger PPL cost.

Exact experiments: historical three-seed PTB endpoints and `outputs/benchmark_reports/ptb_final_*.json`.

Required figure/table: `table_efficiency` and the existing batch-latency figure after caption revision.

Claims that must NOT be made: broad deployment superiority; online speedup for the quality student; hardware-independent latency; large-scale compression; unique Grassmann efficiency without matched architecture controls.

## 9. Discussion and Limitations

Main argument: synthesize teacher distillability as a joint property of teacher state, predictive distribution, student state, and optimization. Explain that the frozen evidence rules out several simple mechanisms while leaving token-conditional utility open.

Exact experiments: Stage B JSD, Stage C CRBD failure, Phase 2C conditional utility, Phase 2E clipping audit, Phase 2F confounds, and Phase 2G optimization audit.

Required figure/table: `table_failed_or_negative_controls`; detailed CRBD and legacy results may move to the appendix.

Claims that must NOT be made: CRBD success; proof that routing cannot work; causal explanation from clipping; generalization beyond small models, three datasets, fixed teachers, and one principal hardware setting.

Required limitations: only three student seeds and fixed teacher seeds; small models; WT2-centered mechanism controls; TT/TG teacher-quality and alpha-LR mismatch; Transformer-only clipping warning; observational conditional utility; one-GPU latency; legacy CodeParrot and efficiency protocols; no matched random-init study.

## 10. Conclusion

Main argument: warm-start KD has reproducible transfer boundaries; teacher state strongly modulates the outcome; global likelihood is insufficient; and generic ensemble supervision, not Grassmann-specific heterogeneity, explains the defensible ensemble result.

Exact experiments: concise synthesis of Phases 2C, 2D, 2F, and 2G plus PTB confirmation.

Required figure/table: none.

Claims that must NOT be made: any stronger excluded claim from `FINAL_EVIDENCE_LEDGER.md`; any promise that additional experiments were run; any presentation of exploratory CodeParrot, random-init, JSD, or CRBD evidence as confirmation.

## Appendix plan

The appendix should contain the detailed Grassmann operator, architecture diagram, complete protocols and checkpoint identities, legacy CodeParrot observation, batch-mean/token-mean conversion history, unmatched random-init and cross-domain runs as explicit non-results, JSD diagnostics, CRBD failure analysis, optimization and clipping traces, per-seed endpoint tables, and hardware benchmark configuration.
