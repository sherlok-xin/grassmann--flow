# Phase 2A — Teacher–Student Transfer Landscape Audit

## 1. Scope

This audit tests whether the existing positive and negative KD outcomes are more consistent with teacher residual quality, teacher–student distribution or gradient compatibility, Transformer–Grassmann branch heterogeneity, or dataset identity. It uses frozen checkpoints only. No model was trained, no checkpoint or dataset was modified, CRBD was not extended, and the manuscript was not edited.

The source and environment state was frozen before adding the analysis code under `research/snapshots/20260920/`. Full paths, checkpoint identities, data fingerprints, selection indices, metric definitions, and limitations are recorded in `provenance.md`.

## 2. Definitions

The primary outcome is NLL. Teacher residual advantage is

`Delta_teacher = NLL_WS_CE - NLL_teacher`,

where a positive value means that the teacher is better than the matched continued-CE student. KD gain is

`Delta_KD = NLL_WS_CE - NLL_WS_KD`,

where a positive value means that KD improves over matched CE continuation. PPL is reported only as `exp(NLL)` for readability.

CE scratch and the S0 warm-start checkpoint refer to the same trained Hybrid-lite checkpoint in the current runs, but their roles remain explicit: CE scratch describes how that checkpoint was obtained, whereas S0 identifies the parameter state from which WS+CE and WS+KD continuation begins.

## 3. Evaluation protocol

The branch and alpha analyses use validation data. PTB uses its full validation set of 350 chunks and 89,250 next-token targets. WikiText-2, TinyStories, and CodeParrot common-5k each use 512 chunks and 130,560 targets selected deterministically with seed 20260920. All sequences contain 256 GPT-2 tokens. TinyStories uses the historical train-derived validation split; CodeParrot retains the frozen first-5,000-record preprocessing.

The teacher implementation was verified directly. For all four last-1 teachers,

`fused_logits = alpha * transformer_logits + (1 - alpha) * grassmann_logits`.

The offline reconstruction agrees with the model output up to the expected mixed-precision rounding, with maximum absolute logit differences from 0.015625 to 0.03125. Alpha therefore denotes the Transformer weight, not the Grassmann weight.

Gradient compatibility is measured at the logit level with temperature 2. The CE gradient is `p_student - one_hot(y)`, and the KD gradient is `T * (p_student,T - p_teacher,T)`. These values are descriptive and are not full parameter-gradient measurements.

## 4. Existing four-domain transfer outcomes

The table reports exact test NLL and PPL from the original run summaries. PTB values are means over three matched seeds; the remaining domains are seed-42 results.

| Domain | Teacher NLL / PPL | S0 warm-start NLL / PPL | WS+CE NLL / PPL | WS+KD NLL / PPL | Delta teacher | Delta KD | Evidence status |
|---|---:|---:|---:|---:|---:|---:|---|
| PTB | 3.914270 / 50.1125 | 4.072201 / 58.6865 | 4.082352 / 59.2853 | 3.937415 / 51.2859 | +0.168082 | +0.144937 | matched token-normalized KD, three seeds |
| WikiText-2 | 4.202696 / 66.8663 | 4.250827 / 70.1634 | 4.270742 / 71.5748 | 4.108163 / 60.8349 | +0.068047 | +0.162579 | legacy batch-normalized KD, one seed |
| TinyStories | 1.603229 / 4.9691 | 1.622437 / 5.0654 | 1.581269 / 4.8611 | 1.688747 / 5.4127 | -0.021960 | -0.107479 | matched token-normalized KD, one seed |
| CodeParrot common-5k | 1.737313 / 5.6821 | 2.037176 / 7.6689 | 1.980364 / 7.2454 | 1.792055 / 6.0018 | +0.243051 | +0.188309 | legacy batch-normalized KD, one seed |

The sign pattern is exact in these four observations: the teacher is better than WS+CE and KD helps on PTB, WikiText-2, and CodeParrot, whereas WS+CE is better than the teacher and KD hurts on TinyStories. This pattern supports teacher residual advantage as a candidate explanation. It does not establish causality because dataset, protocol generation, teacher checkpoint, and evidence depth vary together. PTB is the only multi-seed result, and the WikiText-2 and CodeParrot outcomes come from legacy sweeps rather than the corrected token-normalized protocol.

Continued CE also behaves differently by domain. Relative to S0, WS+CE worsens test NLL on PTB and WikiText-2 but improves it on TinyStories and CodeParrot. Consequently, scratch CE cannot be substituted for the matched WS+CE endpoint when defining either residual advantage or KD gain.

## 5. Teacher branches and learned fusion

| Domain | Transformer NLL | Grassmann NLL | Fused NLL | Improvement over best branch | Learned alpha | Branch JSD | Branch top-1 agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| PTB | 4.241051 | 4.200023 | 4.053753 | 0.146271 | 0.470293 | 0.130924 | 0.587978 |
| WikiText-2 | 4.445656 | 4.652055 | 4.299172 | 0.146483 | 0.506482 | 0.181289 | 0.508464 |
| TinyStories | 1.785048 | 2.112059 | 1.600880 | 0.184168 | 0.599406 | 0.177059 | 0.586451 |
| CodeParrot common-5k | 2.089248 | 2.680898 | 1.784318 | 0.304930 | 0.572315 | 0.239257 | 0.568436 |

The learned fusion improves validation NLL over the better individual branch in every evaluated domain. This is direct predictive-quality evidence for a branch complementarity signal. It does not by itself show that Grassmann geometry is the cause, because the branches differ in architecture and were not parameter-matched controls.

The best point on the 0.1 alpha grid is 0.5 for PTB and 0.6 for the other three domains. The learned alpha is at or close to this validation optimum in PTB, TinyStories, and CodeParrot. WikiText-2 has a learned alpha of 0.506, whereas the coarse-grid optimum is 0.6; its NLL difference is 0.01208.

## 6. Offline alpha landscapes

All four NLL landscapes are U-shaped over the frozen branches: intermediate fusion outperforms both endpoints. This result is strongest on CodeParrot, where learned fusion lowers NLL by 0.30493 relative to the Transformer branch, and smallest but still material on PTB and WikiText-2, where the improvement is approximately 0.146.

Teacher–student KL is also generally minimized at an intermediate alpha rather than at the best individual branch. Relative to the branch with the lower NLL, learned fusion reduces S0 KL by 0.09877 on PTB, 0.37473 on WikiText-2, 0.19942 on TinyStories, and 0.65332 on CodeParrot. Teacher quality and proximity to the student therefore move together in these landscapes, making either quantity difficult to isolate observationally.

The gradient landscapes do not yield a domain-independent rule. At S0, the canonical mean CE–KD cosine is +0.04009 on PTB, -0.07410 on WikiText-2, +0.04375 on TinyStories, and +0.08130 on CodeParrot. WikiText-2 therefore has a negative mean cosine despite positive legacy KD gain, while TinyStories has a positive S0 mean cosine despite negative matched KD gain. At S1, the TinyStories cosine becomes negative (-0.04040), but that post-continuation state cannot retroactively explain the initial training direction.

The complete landscapes are in `alpha_landscape.csv` and `gradient_compatibility.csv`. The figures under `figures/` show NLL, KL, and gradient cosine against alpha. Neighboring alpha values reuse identical branches, students, and validation tokens; they are not independent experiments.

## 7. Candidate explanations P1–P7

The n=4 descriptive Pearson and Spearman values are:

| Predictor | Pearson r with Delta KD | Spearman rho | Direct reading |
|---|---:|---:|---|
| P1 teacher residual advantage | 0.820 | 0.800 | Correct sign separation in all four observed domains |
| P2 teacher–student KL at S0 | 0.890 | 1.000 | Larger KL co-occurs with larger gains here, contrary to a simple incompatibility interpretation |
| P3 mean gradient cosine at S0 | -0.161 | 0.200 | Does not separate positive from negative transfer |
| P4 negative-gradient fraction at S0 | -0.808 | -0.200 | Pearson is dominated by four heterogeneous points; rank relation is weak |
| P5 teacher entropy | 0.489 | 0.400 | Entropy scale is strongly domain-dependent |
| P6 branch JSD | 0.204 | 0.800 | No stable magnitude relation; evidence tiers differ |
| P7 branch top-1 disagreement | 0.434 | 0.600 | PTB and TinyStories have almost identical disagreement but opposite KD signs |

These values are exploratory only. Four domains are not statistical replicates, and the outcome mixes matched token-normalized evidence with legacy sweep winners. In particular, the large P2 correlation must not be interpreted as evidence that larger teacher–student KL causes KD improvement. Teacher quality, data domain, and checkpoint history confound it.

P1 currently gives the cleanest qualitative account because its sign matches the outcome sign in all four observations. P3 and P4 do not support a simple rule in which average initial gradient alignment determines whether KD helps. P6 and P7 show branch heterogeneity but do not explain the existing transfer sign.

## 8. Controlled candidate teachers

Candidates were selected from validation data only, using an operational near-teacher margin of absolute NLL difference at most 0.025 nat/token from the same-subset S1 student. The full values are in `candidate_teachers.md`.

For PTB, alpha 0.5 is T_good with NLL 4.053784 and residual advantage +0.183495, while alpha 1.0 is T_near with NLL 4.241051 and residual advantage -0.003772. No alpha in [0,1] is clearly worse than S1 under the declared margin, so T_bad is unavailable.

For TinyStories, alpha 0.6 is T_near with NLL 1.600853 and residual advantage -0.020675, while alpha 0.0 is T_bad with NLL 2.112059 and residual advantage -0.531881. No alpha in [0,1] is better than S1, so T_good is unavailable.

The existing PTB and TinyStories frozen branches therefore cannot produce complete good/near/bad triplets. No missing condition was fabricated. A future controlled-quality experiment would need either a different domain whose frozen alpha landscape spans all three regimes or an independently justified teacher-strength intervention.

## 9. Grassmann-specific aggregate diagnostic

Learned fusion improves over the better branch on all four validation evaluations. It also reduces KL to S0 in all four. Relative to the better-quality branch, fusion improves mean S0 gradient cosine by +0.01284 on PTB, +0.01182 on WikiText-2, and +0.02288 on TinyStories, but reduces it by -0.02354 on CodeParrot. Predictive fusion benefit is therefore consistent, whereas gradient-alignment benefit is not.

The token aggregates provide a concrete but non-causal complementarity signal. On PTB, learned fusion corrects 3,760 Transformer errors on tokens where the Grassmann branch gives the gold token higher probability, corresponding to 6.18% of Transformer errors; it harms 1,667 correct Transformer predictions under the converse probability condition, or 5.86% of Transformer-correct tokens. The corresponding correction/harm counts are 4,621/3,196 on WikiText-2, 7,080/4,260 on TinyStories, and 7,033/3,670 on CodeParrot. Correction rates among Transformer errors are 5.21%, 12.54%, and 15.44%, while harm rates among Transformer-correct tokens are 7.62%, 5.75%, and 4.32%, respectively.

These counts establish that both beneficial and harmful Grassmann contributions occur. They do not establish that Plucker geometry, rather than model diversity more generally, causes the corrections. Parameter-matched architecture controls remain absent.

## 10. Relation to Stage-B disagreement results

Stage B found that token-level branch JSD had essentially no predictive value for KD harm: PTB JSD AUROC delta was -0.000323 with grouped-CV R2 gain +0.000533, and TinyStories JSD AUROC delta was +0.01091 with grouped-CV R2 gain +0.003061. Phase 2A does not overturn that result. It shows that intermediate branch fusion improves aggregate teacher quality, but it does not show that per-token disagreement identifies where KD will help or hurt.

This distinction is important. Aggregate branch complementarity is supported as a predictive-quality observation. Disagreement-based routing or a disagreement mechanism remains weakly supported and should not be inferred from the fusion gains.

## 11. Limitations

The central limitation is evidence mismatch. Only PTB has three matched seeds. TinyStories has one matched seed, and WikiText-2 and CodeParrot use older loss normalization and winner selection. Validation branch metrics for the latter three domains use fixed 512-chunk subsets, whereas transfer outcomes use exact test summaries. The P1–P7 comparison therefore generates hypotheses but cannot rank causal mechanisms.

Logit-gradient cosine compresses a high-dimensional, token-dependent optimization signal to one mean and one negative fraction. It omits parameter Jacobians, optimizer state, gradient accumulation across tokens, and changes during KD training. The fusion sweep also changes teacher quality, entropy, KL, and gradients simultaneously. These covariates cannot be separated without a controlled training intervention.

Finally, the observed fusion benefit is not yet specifically attributable to the Grassmann construction. A heterogeneous pair can benefit from ensembling even when neither architecture contributes a unique mechanism.

## 12. Artifact index

`transfer_table.csv` contains seed-level and aggregate transfer outcomes. `teacher_branch_metrics.csv` contains branch and canonical fusion validation metrics. `alpha_landscape.csv` contains all grid, learned-alpha, and historical-alpha evaluations. `gradient_compatibility.csv` provides separate S0 and S1 gradient rows. `candidate_teachers.md` records the controlled candidates and unavailable regimes. `grassmann_specific.csv` and `predictor_comparison.csv` provide compact secondary tables. Raw manifests and summaries are under `raw/`, execution logs are under `logs/`, and plotting code and exported figures are included in this directory.

## Decision 1 — Is “domain dependence” currently justified?

PARTIAL

The observed KD sign differs across domains, with positive gains on PTB, WikiText-2, and CodeParrot and a negative result on TinyStories. However, domain identity is confounded with teacher residual quality, protocol generation, and replication depth. The evidence supports domain-conditioned observations, not domain identity as the established cause.

## Decision 2 — Is teacher residual advantage a plausible explanation?

SUPPORTED AS HYPOTHESIS

The sign of teacher residual advantage matches the sign of KD gain in all four existing domain observations, and TinyStories is the only domain in which WS+CE surpasses the teacher. The n=4, mixed-protocol evidence is insufficient for a causal law, but residual advantage is the strongest current hypothesis.

## Decision 3 — Does Transformer–Grassmann heterogeneity add measurable value beyond the best branch?

YES

Predictive-quality evidence is consistent: learned fusion reduces validation NLL relative to the better branch by 0.146–0.305 nat/token in all four domains and produces both aggregate correction and harm signals. Gradient-compatibility evidence is only partial: fusion improves S0 cosine relative to the better branch on PTB, WikiText-2, and TinyStories but worsens it on CodeParrot. This does not yet isolate a Grassmann-specific geometric mechanism from generic ensemble diversity.

## Decision 4 — Is branch disagreement worth pursuing as a mechanism?

WEAK

Branch diversity is useful for aggregate fusion quality, but neither the prior Stage-B token analysis nor the four-domain descriptive comparison shows that JSD or top-1 disagreement reliably explains KD harm. Additional routing or disagreement-loss development is not justified by the current evidence.

## Decision 5 — What is the single most informative next TRAINING experiment?

D. controlled teacher-quality alpha experiment

This experiment most directly tests the leading residual-advantage hypothesis while holding the two frozen branch checkpoints and teacher architecture fixed. It should use validation-selected good, near, and bad alpha conditions on one domain whose landscape spans all three regimes, an identical S0 tensor, the corrected token-normalized KD objective, and a matched WS+CE control. PTB and TinyStories do not individually span all three regimes with the current frozen branches, so the domain and alpha triplet must be verified before launch. This choice is based on its ability to break the current domain/teacher-quality confound, not on its likelihood of producing a positive KD result.
