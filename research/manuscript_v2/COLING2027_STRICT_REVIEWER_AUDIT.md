# Executive decision

READY TO SUBMIT after the minor editorial corrections documented below. This is manuscript-level readiness, not an acceptance prediction, certification of submission-system compliance, or confirmation of upload. The credible remaining risk is significance and external validity, not a detected false endpoint or invalid paired comparison.

Reviewed on 2026-10-11 from authoritative commit `ab149eb20da37bdbe0d5862e67c9d6e1a8777bd6`. Source: `论文投稿/coling2027_arr/`. The three reviewer perspectives independently read the canonical 14-page compiled PDF in actual order, including References and Appendices A–D, and inspected every page, figure and table before proposing changes. The lead editor also read and inspected the complete input and final PDFs. The unrelated 15-page xdvipdfmx file was not used.

Input PDF SHA256: `0408342b232c6e3ba86deae8129cec552ed9e3e5a5c7b3ac4060a27a28792e20`. Final PDF SHA256: `700490be24483fef4bfd69a367ade958981bbe7e05e4804c3c14efa40cde7255`.

The contribution remains a controlled empirical evidence chain for warm-start KD, not a new objective, selection algorithm, compatibility score, Grassmann method, universal theory, or causal factorization. Experiments remain permanently frozen. No training, inference, checkpoint loading, endpoint recomputation, experimental diagnostic, baseline, or statistical test was performed. Editorial QA reads existing records and compiled documents only.

Scores use the requested approximate 1–5 scale: 1 strong reject, 2 reject, 3 borderline, 4 accept, 5 strong accept. The dimension scores express the corresponding quality assessment. These are not a transcription of the official review form. Soundness is distinguished from excitement and overall recommendation, consistent with [ARR reviewer guidance](https://aclrollingreview.org/reviewerguidelines). Scores refer to the input paper and were not raised to reward polishing.

| Reviewer | Soundness | Significance | Novelty | Clarity | Reproducibility | Overall | Confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: empirical NLP / KD | 4 | 3 | 3 | 4 | 3 | 3 | 4 |
| B: methods / causality | 4 | 3 | 2 | 4 | 4 | 3 | 4 |
| C: modern NLP / significance | 4 | 2 | 3 | 4 | 4 | 2 | 4 |

# Reviewer A

## Summary

The paper measures whether frozen-teacher supervision improves an already-trained LM over matched CE continuation from exactly the same student checkpoint. Independent validation selection precedes test reporting. Three student preparations reproduce positive transfer on PTB and WikiText-2 and negative transfer on TinyStories. A fixed-composition teacher-state comparison reverses the transfer sign; a likelihood-weaker source still transfers positively; a homogeneous ensemble exceeds the heterogeneous source. The contribution is the connected empirical investigation and its explanatory boundaries.

## Strengths

The comparator is meaningful and unusually precise. Same-seed student weights, fresh optimizers, matched continuation data and budgets, and independent checkpoint selection avoid attributing unrelated preparation or budget differences to KD. The primary effects have identical directions across three student preparations. TinyStories is especially informative because CE improves over S0, so negative gain is measured relative to a productive continuation. The J/A reversal is stronger evidence about a teacher-state bundle than the cross-corpus signs. Unfavorable TT and diagnostic results are retained rather than used to rescue the original Grassmann explanation.

## Weaknesses

The individual principles have precedents: capacity mismatch, harmful KD, useful weaker teachers and fidelity–utility differences are not new. Novelty rests on their combination under matched warm-start continuation. The primary evidence concerns one compact hybrid student architecture, fixed corpus splits and fixed teachers. Student seeds do not replicate teacher preparation or model-family coverage. The modern extension cannot supply clean confirmation because S0 wins all nine diagnostic selections and the CE headroom gate fails. In the input PDF, primary student identity and PTB/TinyStories teacher configurations were under-specified. The broadest likelihood/statistic wording could be read as rejecting every scalar predictor, rather than the tested likelihood-ordering assumption and local diagnostics.

## Questions for authors

What is the intended generalization unit: the frozen configurations or warm-start LM KD in general? Can the appendix identify the primary teachers using existing configurations? Does the weaker-source result reject the need for a likelihood advantage over CE, or claim to disprove every likelihood-based predictor? The final edits answer the architecture/configuration questions and make the narrower comparator conclusion explicit; broader generalization remains unresolved.

## Scores and reasons

Soundness 4: measured paired comparisons are credible and confounds are disclosed. Significance 3: the operational evaluation question is useful but evidence coverage is narrow. Novelty 3: the integrated controls add empirical value, not a new phenomenon or algorithm. Clarity 4: gain direction and comparator scopes are clear. Reproducibility 3: the input PDF lacked some primary teacher metadata despite strong repository provenance; that reporting gap is now reduced. Overall 3: borderline, primarily on novelty and scope. Confidence 4: full PDF, visuals and selected frozen provenance were reviewed, without independent execution.

# Reviewer B

## Summary

This is a sound bounded study of CE-relative KD endpoints, not an identified causal decomposition. Fixed teachers, reused CE/J/TG endpoints, J/A training differences, TT/TG quality/count/history differences, clipping and modern checkpoint eligibility are already disclosed. The main acceptance uncertainty is the significance of the empirical combination.

## Strengths

Exact S0 matching and independent validation selection make the endpoint interpretable. Sample SD and 3/3 signs are not mislabeled as confidence intervals or significance tests. J/A fixes deployed architecture, source identities and composition while transfer reverses, but the paper correctly describes a teacher-state/history/quality bundle. The weaker G comparison names the matched selected CE endpoint and preserves the validation-subset versus test-endpoint distinction. Shared controls and unsuccessful modern/diagnostic work are visible.

## Weaknesses

The teacher-state NLL gap is large and the training modes and budgets differ, so the experiment cannot isolate likelihood as a cause. TT already has better full-validation teacher NLL, fewer parameters and different training/fusion history; its higher gain cannot identify intrinsic homogeneous superiority. T clips every early update, TT also clips heavily initially, and modern KD1/KD5 are strongly constrained by clipping. These are interpretation limits, not evidence that clipping caused the ordering. The offline probe collection is small, dependent and retrospective. Its failure does not establish that gradient-based selection or adaptive KD is impossible. The modern study fails to establish useful continuation headroom. The input PDF called FineWeb provenance pinned without giving its exact revision.

## Questions for authors

Will an anonymous reproducibility package expose exact source/model/data revisions? What scope should be assigned to probes using the first 32 training chunks and two validation chunks? How should the trained-checkpoint comparison be interpreted when step 0 would win? Is there independent evidence of scalar-quality causality beyond J/A? Can TT/TG establish more than the tested homogeneous alternative matching/exceeding TG? The current paper answers the endpoint and causal questions conservatively; sampling, factorial identification and coverage cannot be repaired editorially.

## Scores and reasons

Soundness 4: bounded claims fit the paired evidence. Significance 3: comparator discipline and transparent negative controls are useful. Novelty 2: most individual phenomena are established and there is no new identified mechanism. Clarity 4: scopes and dependence are explicit, although some terminology resembles internal auditing. Reproducibility 4: detailed settings, endpoints and provenance exist; exact FineWeb metadata is now added. Overall 3: borderline on incremental insight rather than scientific invalidity. Confidence 4: entire manuscript and selected provenance reviewed, with no independent experiment.

# Reviewer C

## Summary

The evidence chain is more informative than an isolated statement that better teachers need not teach better. However, confirmatory results use one 31.43M hybrid student, PTB/WikiText-2 and a controlled TinyStories subset. SmolLM2/FineWeb provides bounded stress evidence rather than clean contemporary confirmation. This restricts COLING significance without invalidating the measured effects.

## Strengths

The matched CE comparator defines an operational decision. TinyStories demonstrates harm relative to useful CE continuation. Reporting TT above TG, weaker disagreement predictiveness, clipping and the failed modern gate increases confidence that attractive explanations were not preserved at the expense of results. Related Work accurately distinguishes forward-KL corpus continuation from reverse-KL, on-policy and adaptive methods. The paper does not claim to outperform algorithms it did not evaluate.

## Weaknesses

The primary scale and corpora limit relevance to current deployment. PTB and WikiText-2 are useful controlled test beds, not evidence of broad contemporary generative-model performance. A small unusual hybrid architecture supplies less architecture coverage than an ordinary Transformer multi-family study. The modern result does not remedy this limitation. Capacity mismatch, useful weak teachers and negative gradient-influence findings already have close antecedents. In particular, GRACE reports weak validation-influence/LESS teacher-ranking baselines, not only a spectral teacher-response score. The input Related Work omitted this closer overlap. Lack of a new method is not itself a soundness defect, but the empirical chain must provide enough new insight to meet the significance bar.

## Questions for authors

What decision does the chain support beyond running matched CE/KD evaluation? Does the primary student itself contain a Grassmann branch? How do the local failures differ from GRACE's negative influence baselines? How much modern confirmation remains once S0 is eligible? The defensible answers are evaluation discipline, an explicitly hybrid primary student, paired corpus-likelihood endpoints rather than generated-response teacher ranking, and limited stress evidence with clean modern confirmation unresolved.

## Scores and reasons

Soundness 4: precise comparator and internally consistent evidence; limited coverage is not treated as a hidden numerical flaw. Significance 2: direct modern-NLP relevance remains limited. Novelty 3: the combined evidence chain is distinctive although individual principles are familiar. Clarity 4: readable figures and bounded conclusions, with two useful scope clarifications now made. Reproducibility 4: detailed objective, protocols, selected endpoints and telemetry; aggregate compute accounting remains unavailable. Overall 2: lean reject on significance/external validity, not invalidity. Confidence 4: complete manuscript, visuals, literature records and relevant cached primary-method text checked.

# Mock meta-review

The strongest acceptance argument is a coherent empirical contribution built around an exact student-state comparator. It connects reproducible directions, teacher-state reversal, a likelihood-ordering counterexample, a homogeneous ensemble alternative and unsuccessful diagnostics, while retaining results unfavorable to the original explanation. A new loss is not required for this to be legitimate empirical research.

The strongest rejection argument is limited incremental explanatory knowledge and coverage. The work revisits established KD limitations in a small hybrid test bed; its modern extension is degenerate and the attempted headroom control fails. The likely borderline question is whether the connected controls and honest comparator discipline supply enough additional insight for COLING. A plausible consensus is scientifically credible but borderline, with a strict modern-NLP reviewer leaning reject. These three simulations are not independent human reviewer samples and do not support an acceptance-probability estimate.

Fixable without experiments: precise likelihood/statistic wording, explicit primary architecture, existing teacher/data provenance and acknowledgment of the closest diagnostic overlap. Not fixable by prose: independent teacher replications, broader architectures/tasks, quality/count/history-matched TT/TG, factorial identification, clipping controls, adaptive baselines or clean modern continuation confirmation. None was run or promised in this pass.

# Claim-evidence audit

Statuses below assess the final bounded interpretation. Locations use section/figure/table identifiers so they remain stable after reflow.

| Claim | Evidence | Main-text location | Required caveat | Status |
|---|---|---|---|---|
| C1: PTB/WT2 positive; TinyStories negative | Frozen cross-domain table and selected endpoints; three preparations per corpus, means +0.144937/+0.155665/−0.107219 | Abstract; Introduction; §5; Figure 2; Table 1 | One primary hybrid architecture; fixed teachers/splits; differing corpus histories; not corpus-identity causality | SUPPORTED WITH CAVEAT |
| C2: teacher state modulates transfer at fixed deployed composition | J +0.155665, A −0.027654; paired Q +0.183319; Table 2 preparation differences | §6.1; Figure 3; Discussion | Deployed architecture/source identities/alpha fixed; mode, budget, branch states and error structure change; not scalar-likelihood causality | SUPPORTED WITH CAVEAT |
| C3: a likelihood advantage over matched CE is unnecessary for positive transfer | G common-subset NLL 4.652064; residuals −0.266520/−0.253492/−0.251006; test gains +0.046771/+0.033598/+0.042604 | Introduction; §6.2; Figure 3; Appendix C | Compare teacher/selected CE on the same validation subset; transfer uses selected test endpoints; does not disprove every scalar predictor | SUPPORTED |
| C4: ensemble benefit does not establish heterogeneous/Grassmann superiority | F exceeds T/G; TT gain +0.168125 versus TG +0.155665; H −0.012460, 3/3 negative | §7.1–7.2; Figure 4; Discussion | T early clipping; TT/TG quality/count/history/fusion-rate differences; no intrinsic TT superiority either | SUPPORTED WITH CAVEAT |
| Tested local diagnostics are not reliable predictors of these endpoints | Existing first-/second-order AUROC 0.644444, accuracy 0.571429, family misses, calibration instability; Table 9 | §7.3; Appendix D.1; Figure 1 evidence map | Small dependent retrospective collection; some discrimination remains; not a test/refutation of GRACE or adaptive KD generally | SUPPORTED WITH CAVEAT |
| Modern result is bounded stress evidence | Negative trained-checkpoint gains; all nine step-0 choices retain S0; CE-only best improvement 0.004849 < 0.01 | §8; Discussion; Appendix D.2–D.3; Tables 10–12 | Degenerate continuation, clipping, shared pretrained base, overlap unknown; conditional KD/new test not run | SUPPORTED WITH CAVEAT |
| Configuration-dependent distillability | Joint interpretation of the above contrasts | Abstract; Discussion; Conclusion | Protocol-specific interpretation, not a factorization or universal teacher-selection theory | SUPPORTED WITH CAVEAT |

The input abstract's final “rather than a single teacher-quality or compatibility statistic” was AMBIGUOUS: if read as excluding every possible scalar predictor, it would be TOO STRONG. The input Introduction's “cannot determine distillability” likewise allowed that broader reading. Both were narrowed without weakening the actual counterexample. The short “likelihood is insufficient” wording elsewhere is followed by that exact counterexample and necessary causal boundaries; no global defensive rewrite was justified.

No final claim supports Grassmann-specific KD advantage, geometry as a cause, heterogeneous superiority, a successful CRBD method, strong disagreement causality or a universal “weaker teacher means negative transfer” rule. These claims remain excluded.

# Numerical/comparator audit

The existing original 62-check audit and all 62 corresponding source/rendered-PDF checks pass. Manual checks covered the compiled prose, equations, figures and tables rather than only the LaTeX source. No endpoint or new statistic was computed. Figures use stated rounding, not independently modified values.

| Quantity | Frozen/rendered value | Checked locations and interpretation |
|---|---|---|
| PTB gain | +0.144937 ± 0.003938 | Table 1/Figure 2/§5/Appendix C; 3/3 positive |
| WT2 J/TG gain | +0.155665 ± 0.007692 | Table 1/Figure 2/§6–7/Appendix C; same reused J/TG endpoint |
| TinyStories gain | −0.107219 ± 0.000415 | Table 1/Figure 2/§5/Table 7; 3/3 negative; CE improves S0 |
| WT2 A gain | −0.027654 ± 0.006822 | §6.1 and per-seed table; 3/3 negative |
| Q = NLL_KD,A − NLL_KD,J | +0.183319 ± 0.001375 | Equation 4 and Appendix C; positive favors J |
| G gain | +0.040991 ± 0.006733 | §6.2/§7/Figure 4/Table 4; Figure rounds to +0.0410 |
| T gain | +0.134868 ± 0.008906 | §7/Appendix C/Figure 4; Figure rounds to +0.1349 |
| F−T gain contrast | +0.020797 ± 0.001214 | §7.1/Appendix C; clipping-limited interpretation |
| F−G gain contrast | +0.114674 ± 0.001001 | §7.1/Appendix C; composition/quality coupled |
| TT gain | +0.168125 ± 0.008948 | §7.2/Appendix C/Figure 4; rounds to +0.1681 |
| H = Gain_TG − Gain_TT = NLL_TT − NLL_TG | −0.012460 ± 0.001348 | Equation 5; all three negative means TT student endpoint lower |
| Modern lambda 1 gain | −0.013550 ± 0.003542 | §8/Table 10; trained-checkpoint primary, 3/3 negative |
| Modern lambda 5 gain | −0.043924 ± 0.004017 | §8/Table 10; 3/3 negative, full saturation warning |
| CE headroom gate | +0.004849; threshold 0.01 | §8/Table 12; positive but insufficient, not zero headroom |

Throughout, Gain = selected test NLL_CE − selected test NLL_KD. Positive favors KD over CE, not necessarily over S0. Q and H have distinct explicitly stated signs. SD is sample SD over three student preparations, not a standard error, confidence interval or significance claim.

For G, teacher likelihood is compared to the validation-selected CE checkpoint on the same 512-chunk validation subset, while gains use selected test endpoints. The comparator is CE, not an arbitrary initialization or automatically S0. J/A/source subset results are separate from full-validation TT/TG metrics; repeated J values 4.300644 and 4.300646 retain their original precision rather than being pooled.

Replication/dependence passes: §3 identifies independent student preparations and fixed teacher/corpus sampling; §4 and Appendix C identify shared CE/J/TG endpoints; Appendix D.1 distinguishes token/chunk bootstrap uncertainty from model replications; Appendix D.2 says modern seeds adapt one pretrained base. Seven diagnostic families and 21 conditions are not 21 independent teacher experiments, nor are three calibration subsets extra student replications.

J/A fixes deployed architecture and alpha but changes mode/budget/state. Main §6.1 expressly denies scalar-likelihood causality, with five alpha-only versus ten joint epochs in Table 2. No added defensive paragraph is needed. TT/TG full-validation NLLs 4.293589/4.303697, parameter counts 35,340,801/37,749,633, fusion rates and histories are disclosed. Higher TG JSD 0.182087 versus TT 0.099793 accompanies lower transfer, not an identified diversity effect.

Optimization caveats are proportionate. Main §7 flags T gate/early full clipping and F–T caution. Appendix B gives T mean epoch clip fractions 0.4990/0.4605/0.4398 and TT means 0.170068/0.137755/0.127551. TinyStories KD means are 0.236891/0.283344/0.240090, not persistent full saturation; accepted endpoints are finite with recorded nonzero AMP skips. Modern KD1 clips approximately 0.80–0.85 of updates, KD5 clips all; recorded overflow/nonfinite fractions are zero. The finite CE456 norm spike warning and failed-gate finite spike remain visible. None establishes clipping causality or requires exclusion of completed endpoints.

# Novelty audit

The novelty is the combination, not the first occurrence of any single KD failure. Introduction and Related Work already say this explicitly. The six links are exact warm-start CE matching; replicated +/+/- outcomes; fixed-composition state reversal; the weaker-source counterexample; the tested homogeneous alternative; and unsuccessful local-diagnostic prediction. They constrain different explanations and are not interchangeable independent effects.

| Prior line | Closest verified references | Accurate distinction / remaining overlap |
|---|---|---|
| Capacity mismatch / harmful teacher training | Cho and Hariharan; Park et al. | Not a novel capacity-mismatch principle or student-friendly teacher method; fixed warm-start continuation is the evaluated decision |
| Statistical utility / useful weak teachers | Menon et al.; Kaplun et al.; Stanton et al. | No general theory; a likelihood-ordering counterexample with explicit CE comparator, not classification assumptions transferred as proof |
| Harmful autoregressive KD / adaptive token guidance | ATKD; selective NMT KD | Harmful LM KD is already reported; this work evaluates exact CE-relative continuation, not superiority over adaptive fixes |
| Reverse-KL and on-policy LM KD | MiniLLM; GKD; entropy-aware on-policy KD | Different objective/context-generation protocols; no generation-quality comparison or baseline superiority claim |
| Teacher selection / adaptive ensemble supervision | Yuan et al.; MetaDistil; Du et al.; AdaKD | No selector/routing/compatibility method; tested frozen sources and observations only |
| Distillation scaling | Busbridge et al. | No scaling-law replacement; protocol-specific small-model controls rather than compute-optimal allocation |
| GRACE teacher compatibility | Panigrahi et al. | Spectral student response-gradient ranking is not the present fixed-logit probe. GRACE also reports weak validation-influence/LESS baselines; the final text now acknowledges this direct negative-diagnostic overlap |

The 24 current bibliography entries were checked against `FULL_TEXT_REFERENCE_REVIEW.md`, `RELATED_WORK_GAPS.md` and `full_text_reference_verification.json`; no obvious missing directly competing reference within this verified set was found. This is not a claim of exhaustive new literature discovery. No BibTeX was added or altered. Existing [ATKD](https://aclanthology.org/2024.acl-long.587/) and [scaling-law](https://proceedings.mlr.press/v267/busbridge25a.html) records were checked against primary publication pages. GRACE's overlap was verified in cached primary full text, Section 2.1 and Appendix C.1/C.2, not inferred from a summary. The added sentence uses its existing citation and does not claim to evaluate or refute GRACE.

# Main-text clarity audit

Abstract, Introduction, Related Work, Discussion and Conclusion consistently frame an empirical evidence chain. No new loss, universal rule, causality of likelihood, first-of-kind claim or geometric mechanism appears. The final abstract gives a practical matched-continuation assessment rather than rejecting every possible compatibility statistic. The revised Introduction states the exact likelihood-advantage counterexample. Main §4 now identifies the primary hybrid Transformer–Grassmann student so ordinary Transformer-only coverage is not silently implied.

The important explanations were not fatally compressed during conversion: §3 explains exact S0 matching, fresh optimizers and independent validation selection; §6.1 explains why a fixed deployed state contrast is more controlled than corpus signs; §6.2 specifies the weaker-teacher comparator; §7.2 distinguishes the TT ordering from intrinsic architectural superiority; §8 gives the modern degeneracy and failed gate. Do not fill page 8. No expansion of these already accurate paragraphs was needed.

Discussion provides genuine synthesis: controls address distinct alternatives; predictive diversity need not be transferable diversity; local quantities need not summarize optimization and validation selection; checkpoint eligibility changes what a negative gain means. These are bounded interpretations, not new mechanisms. “What Determines Distillability?” is a question heading, not a claim of factorial identification. “Homogeneous ensemble falsification” is acceptable when read as falsifying the tested practical TG advantage; the immediately adjacent text denies intrinsic TT superiority. Renaming these headings is optional and was not done.

Modern recommendation: KEEP AS IS in the main text. Its three compact paragraphs distinguish harm relative to useful CE from poorer outcomes when both continuations are unnecessary. It reports negative primary gains, 9/9 diagnostic S0 choices, clipping, +0.004849 versus 0.01, and unperformed conditional KD/test. Moving more detail would save little while obscuring a useful boundary; adding telemetry to the main text would distract. It remains secondary, not modern confirmation.

Limitations covers all requested boundaries: small models, three preparations, fixed teachers, reused controls, no factorial decomposition, TT/TG mismatches, clipping/AMP, no adaptive baseline, retrospective diagnostics, modern degeneracy, and unknown pretraining overlap. Shared-base adaptation and unresolved larger-model/generation scope are also stated. It does not call the measured comparisons invalid. No softening or additional self-defeating caveats were justified.

Terminology and grammar checks found no need for a global rewrite, lists or new emphasis. Internal “authorized” language and permanent warning identifiers remain somewhat log-like, but warnings are required and their factual context is useful. No mechanical “AI-style” rewrite was performed. Two nonblocking visual issues remain: a one-word “statistic.” continuation at the top of page 2's right column, and loose justification around the long source identifier on page 13. Neither clips content; changing wording merely to reshape these lines would be optional and was not implemented.

# Appendix utility audit

No appendix text or table was deleted. The following are recommendations, not executed changes. The approved float directives are unchanged; natural reflow moves Table 2 to page 11 and Table 9 to page 14 without a defect.

| Subsection/item | Classification | Exact recommendation |
|---|---|---|
| A: student identity/dimensions/tokenizer | ESSENTIAL | Keep; explicit hybrid identity and link to teacher configurations added |
| A: causal Plücker equations, gate and Figure A1 | USEFUL | Keep for reconstruction of unusual test bed; do not promote to geometric KD evidence |
| B.1: splits, preparation, objective, subset scopes | ESSENTIAL | Keep in full; targets and shared validation scopes matter |
| B.2: teacher configurations and Table 2 | ESSENTIAL | Keep; add authenticated PTB/TinyStories architectures, histories, learned alpha versus forced WT2 alpha |
| B.2: reduced-data smoke and exact eligibility ratios | USEFUL / RESEARCH-LOG-LIKE | If subsequently shortening, compress the paragraph starting “Joint training changes 179...” to the changed-state fact, full-data finite gate, registered ratios and AMP level. Do not hide a protocol amendment or warning |
| B.2: T and TT clipping trajectories | ESSENTIAL caveat / USEFUL telemetry | Keep the full-clipping amendment and trajectory summary; exact per-seed means may live in an anonymous reproducibility supplement only if their interpretation remains in the PDF |
| B.3: TinyStories stability and accepted retry | USEFUL | Keep completed-run stability and pre-endpoint interruption/restart disclosure; do not expand into execution narrative |
| C: per-seed endpoints, shared controls, weaker comparator | ESSENTIAL | Keep scopes and endpoint tables; mean-summary repetition could be compressed later but is not a blocker |
| D.1: probe definition/calibration/Table 9 | ESSENTIAL for secondary claim | Keep actual score definitions, dependence, instability and family misses; no universal method conclusion |
| D.1: literal zero-LR first step and virtual-update details | USEFUL | Keep compactly to prevent a no-op being mistaken for a safety result |
| D.2: modern selection and step-0 interpretation | ESSENTIAL | Keep distinction between frozen primary and no-training diagnostic |
| D.2: full telemetry and spike warning | USEFUL | Keep warning and selected-step relationship; do not speculate on cause |
| D.3: failed headroom gate and missing-by-design KD/test | ESSENTIAL | Keep +0.004849 below 0.01 and conditional not-run decision |
| D.3: literal terminal decision identifier | RESEARCH-LOG-LIKE / USEFUL provenance | Retain under current frozen-warning requirements; no need to repeat it in the main text |

| Table | Classification | Exact recommendation |
|---|---|---|
| 2: WT2 teacher histories | ESSENTIAL | Keep mode/budget/LR differences visible |
| 3: TinyStories telemetry | USEFUL | Keep; residual comparator and temperature scaling in caption prevent ambiguity |
| 4: per-seed paired gains | ESSENTIAL | Keep seven-family overview; it overlaps endpoint tables usefully |
| 5: PTB selected validation/test/epochs | ESSENTIAL | Keep independent checkpoint-selection evidence |
| 6: WT2 selected student NLL | ESSENTIAL | Keep all source/state/control endpoints and reused J/TG caption |
| 7: TinyStories S0/CE/KD and validation direction | ESSENTIAL | Keep proof of productive CE headroom and direction agreement |
| 8: TT/TG teacher metrics | ESSENTIAL | Keep quality/diversity confounds and full-validation scope |
| 9: pooled predictors | USEFUL | Keep to avoid cherry-picking failed scores; not 21 independent experiments |
| 10: modern selected endpoints | ESSENTIAL for §8 | Keep primary-versus-step-0 caption; not primary-paper confirmation |
| 11: modern optimization audit | USEFUL | Keep selected steps and saturation/spike transparency; exact norm columns are the first candidate for supplement-only placement if later space is constrained |
| 12: CE-only failed gate | ESSENTIAL | Keep explicit stop rationale and unperformed KD/test |

If a future submission requires a shorter supplement, the first candidates are redundant mean summaries in Appendix C, exact reduced-data gate ratios in B.2 and Table 11 norm columns, not comparator definitions, negative controls, retry exclusions or missing-by-design outcomes. No such deletion is authorized or performed here.

# Figure/table audit

All existing SVG/PDF figure masters and all twelve table sources are byte-identical to the input. No approved vector was redesigned. Figures were read both in context and as figure/caption pairs; there is no newly identified scientific ambiguity requiring vector changes.

| Item | Decision | Reason |
|---|---|---|
| Figure 1 | KEEP | Same S0 branches to CE/KD, independent selection, fixed Gain sign; evidence map does not promise a predictor |
| Figure 2 | KEEP | Seed markers, mean/sample SD, zero line and sign immediately readable; values match Table 1 |
| Figure 3 | KEEP | State reversal and weaker-source counterexample clear; caption explicitly excludes scalar-likelihood causality and geometric advantage |
| Figure 4 | KEEP | Exact rounded gains and Positive transfer retain measured ranking; caption prominently gives T clipping and unmatched TT/TG quality/training |
| Figure A1 | KEEP | Causal local flow and generic free alpha retained; caption states architecture context, not geometric KD superiority |
| Main Table 1 | KEEP | Exact means/SD/sign consistency complement visual Figure 2; redundancy serves reviewers checking numbers |

Figure 4 alone could be overread as an architecture ranking, but the title denies heterogeneity superiority and the adjacent caption gives the confound. Repeating additional caveats inside approved panels is unnecessary. Figure 3's shorthand “quality matters” is constrained by its caption and state/quality wording, not a causal quality manipulation.

# Changes actually made

All changes are clearly justified SHOULD FIX corrections. Reviewer A proposed treating two broad clauses as MUST FIX if read universally; editorial synthesis treats them as resolvable ambiguity, not a fatal defect. No numerical/methodological MUST FIX was detected. The three initial reviews were completed before source editing.

The machine-readable exact before/after whitelist is in `coling2027_strict_review_qa.json` and `scripts/audit_coling_strict_review.py`. It verifies all eight substitutions/additions against the authoritative input, leaving 36 other submission files byte-preserved.

## E1: abstract concluding sentence

Before: “Together, these results support configuration-dependent distillability: whether KD helps varies with the teacher, warm-start student, and training configuration, rather than a single teacher-quality or compatibility statistic.”

After: “Together, these results support configuration-dependent evaluation of warm-start KD against matched CE continuation, rather than inferring transfer from teacher likelihood ordering or the tested local diagnostics.”

Reason: A/C concern that an arbitrary untested scalar compatibility measure appeared ruled out. Retains the practical contribution and limits inference to tested evidence; no method claim or new factorization.

## E2: Introduction counterexample sentence

Before: “Global teacher likelihood alone therefore cannot determine distillability.”

After: “A teacher likelihood advantage over the matched CE comparator is therefore not necessary for positive transfer.”

Reason: A concern about universal prediction language. States the strong exact counterexample rather than claiming every likelihood-based predictor fails.

## E3: main student identity

Before: “The 31.43M-parameter student uses a local GPT-2 tokenizer...”

After: “The 31.43M-parameter hybrid Transformer--Grassmann student uses a local GPT-2 tokenizer...”

Reason: A/C scope and reproducibility concern; confirms the existing Hybrid-lite architecture without new novelty or coverage. Archived Phase 2C/2G primary configurations support this identity.

## E4: existing GRACE overlap

Before: the sentence ending “unlike our fixed-logit validation-gradient probes.” directly precedes “We do not evaluate these adaptive algorithms.”

After: insert “It also reports weak validation-influence baselines for generative teacher selection; our probes target paired CE-relative corpus-likelihood endpoints.” between those sentences.

Reason: C identified the closer overlap in authenticated GRACE full text, Section 2.1 and Appendix C.1/C.2. Uses the existing citation; no new reference, claim of algorithm comparison or inferred failure of GRACE.

## E5–E6: appendix identity and teacher pointer

Before: “The compact primary student...” and “Teacher dimensions and states follow their source configurations, not the student dimensions.”

After: “The compact primary hybrid student...” and “Primary teacher configurations are specified in Appendix~\ref{app:protocols}.”

Reason: A/C architecture clarity and A paper-level source-configuration concern. The previously vague pointer now resolves to concrete existing metadata.

## E7: primary teacher metadata

Before: Appendix B.2 begins immediately with the WT2 Table 2 configuration paragraph, without explicit PTB/TinyStories teacher metadata.

After: insert the following paragraph, with no result or parameter change:

> PTB and TinyStories teachers each combine a six-layer Transformer of width 256 with a six-layer Grassmann branch of width 240, reduced width 64, and lags {1,2,4}; both use dropout 0.1. Source preparation uses seed 42, 20 epochs, batch 32, sequence length 256, and learning rate 3×10−4; all four sources select epoch 20 by validation. Fused teachers use ten joint epochs, freezing branches in epoch 1, branch learning rate 10−5, weight decay 0.01, AMP, initial alpha=0.5, and late-k=1. Fusion-weight learning rates are 5×10−3 and 10−2, respectively. Validation selects fused epochs 8 and 10, retaining learned alpha=0.470293 and 0.599406 for primary continuations; forced alpha=0.5 applies to the WikiText-2 controls.

Reason: A reproducibility concern. Values were checked in four historical source configs/summaries, two fused configs/summaries, the frozen PTB teacher audit and TinyStories effective-alpha metadata. Exact paths/hashes are in the QA JSON. Source optimizer defaults, head counts and unrecorded weight decay were not guessed from current constructor defaults. Learned primary alpha is distinguished from WT2 forced alpha; no sweep or endpoint was added.

## E8: FineWeb source identity and natural break

Before: “FineWeb-Edu uses the first shard of the pinned sample-10BT revision.”

After: “FineWeb-Edu uses shard sample/10BT/000_00000.parquet of sample-10BT at revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9.”

Reason: B reproducibility concern; exact existing `data_manifest_fineweb.json` source metadata, not a new data choice. The 40-character identifier has a discretionary natural break between two monospaced spans. An initial 0.68pt overfull line was eliminated this way, not by changing margins, fonts, spacing or template.

# Optional changes deliberately not made

No optional heading renames, global de-AI rewrite, added defensive prose, new references, expanded background, float reorganization, appendix deletion or figure redesign was implemented. Section 6's question heading, Section 7.2's tested falsification heading and Section 7.3's bounded diagnostic heading remain. “Authorized” execution-history language remains where factual amendments are reported. Warning identifiers remain required provenance.

The one-word page-2 continuation and page-13 identifier justification were judged nonblocking after visual inspection. No sentence was shortened solely to consume spare space or reshape an otherwise readable column. Approved figure fonts, scientific labels, paths, positions and caption caveats remain unchanged. No extra eighth main page was filled.

# Remaining unavoidable reviewer risks

The strongest unresolved risk is significance: a strict reviewer may find the chain a careful consolidation of known caveats rather than a sufficiently substantial advance. One compact hybrid student family and old/synthetic corpora limit generalization; teachers and splits are fixed. Three independent student preparations support repeatability conditional on that setting, not all teachers or model architectures.

J/A identifies a changed state/history/quality bundle, not likelihood-only causality. TT/TG is an informative practical control but not an architecture-only intervention. Clipping and nonzero primary AMP skips constrain mechanistic interpretation. Local diagnostics are dependent and retrospective, not a prospectively evaluated selection method. No adaptive-KD comparison, generation-quality evidence, clean modern continuation confirmation or full compute accounting is supplied. Manuscript editing cannot create those results.

PDF anonymity passes, but the public identified GitHub repository is a separate operational anonymity risk if submitted or linked as author-identifying supplementary material. An anonymous artifact package, author profile/service eligibility, disclosure/checklist answers, licensing and actual upload remain human submission tasks. This pass does not perform or certify them. The official review rulers and page numbers are intentional, not anomalous numeric watermarks.

# Final technical QA

Canonical build used the existing Makefile with its documented output override:

```sh
make -C 论文投稿/coling2027_arr BUILD_DIR=../../outputs/coling2027_arr
```

Build completed successfully with official pdfLaTeX/ACL, 7 main-content pages and 14 total pages. Limitations follows Conclusion on page 7; References occupy pages 7–8; appendices pages 9–14. All 14 final pages were rendered at 150dpi and inspected by the lead editor and an independent post-edit checker. Main Figures 1–4 remain on pages 2/5/5/6, A1 on page 10. All figure/table captions and mathematical signs are present; no clipping, overlapping labels or missing content was found. Natural reflow, not float-directive editing, accounts for changed table positions.

Original frozen numerical audit: PASS 62/62. Canonical source and rendered-PDF inventory: PASS 62/62. Exact eight-change whitelist, 36 untouched submission files, frozen-path preservation, official template hashes, 24 unchanged verified bibliography records, references/labels, anonymity text/metadata/links and vector checks: PASS. All five figure PDFs and the whole manuscript contain zero embedded raster images. Minimum figure text/script sizes at actual ACL width remain 6.213/7.133/6.520/6.633/6.136pt; fine print remains small and this is not physical-print certification.

Final LaTeX has zero overfull boxes, undefined references/citations, duplicate labels/destinations or BibTeX warnings. It retains 18 underfull hbox and 2 underfull vbox notices; these are disclosed, not suppressed. Abstract is 185 whitespace-delimited words under the same audit convention. Neither official style file was modified, and no layout-gain command was added.

Official aclpubcheck raw output is NOT a zero-error PASS: 912 margin findings, zero warnings, no other error category. Read-only classification accounts for 465 left review numbers + 433 right review numbers + 14 footer page numbers. Independent body-word and vector bounds pass; no non-review margin finding remains. The increased count relative to 890 in the input reflects additional typeset text/rulers, not a newly hidden production defect. The checker and review marks were not changed; external name-check upload was disabled.

Final canonical PDF: `outputs/coling2027_arr/main.pdf`. SHA256: `700490be24483fef4bfd69a367ade958981bbe7e05e4804c3c14efa40cde7255`.

Authoritative new QA: `coling2027_strict_review_qa.json`; reproducible read-only command: `python3 research/manuscript_v2/scripts/audit_coling_strict_review.py`. Earlier conversion/final-layout exact-diff reports remain historical; none of their assertions was relaxed to accommodate this new editorial pass. Build logs, checker output and page renders remain under `outputs/coling2027_arr/` and are not included in the compact commit.

No blocking scientific or PDF-production defect remains within this task's scope. Readiness is conditional on the separate human submission requirements above. Commit only the compact seven changed submission source files and new audit/QA/script artifacts, preserving pre-existing `.gitignore` and project-document changes outside staging. No new experiment is authorized. STOP.
