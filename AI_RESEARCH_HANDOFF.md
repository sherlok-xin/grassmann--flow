# Grassmann / Transformer / Distillation Research Handoff

## 2026-10-03 Phase 4A complete — modern negative-transfer pilot / STOP

The research lead authorized a bounded new phase using official base SmolLM2
135M/360M on TinyStories and FineWeb-Edu, seed 42 only. Stage 0 passed with
matching 49,152-entry tokenizers, verified official weight hashes, finite CE/KL
and gradients, four numerical tests and a fullbatch smoke. An isolated
Transformers 4.46.3/tokenizers 0.20.3 installation resolves an existing
4.57.6/NVIDIA-PyTorch import incompatibility without framework source edits.

Both domains are complete: ten formal runs of exactly 10M predicted targets,
with validation-only selection followed by fixed-state test evaluation.
The seed42 CE-relative test gains at lambda1/lambda5 are
+0.004948632001/-0.004797389422 on TinyStories and
-0.015718598299/-0.046403663516 on FineWeb-Edu. Only the negative FineWeb
lambda5 effect exceeds abs(gain)>=0.02 with matching validation direction.
Decision: MODERN_REPLICATION_WORTHWHILE, a review-only recommendation.

CLIP_SATURATION_WARNING applies to both lambda5 arms (clip fraction1.0).
All optimization remained finite, with no observed overflow/nonfinite event.
This is a single-student-seed, clipping-constrained negative pilot, not robust
confirmation or modern evidence of beneficial KD. TinyStories does not clearly
reproduce the old strong negative boundary. FineWeb pretraining overlap,
different model pretraining budgets and early validation-selected S0 are
additional confounds. No Grassmann-specific or new-method claim is supported.

Read research/experiments/phase4a_modern_generalization_pilot/REPORT.md and
GPT_HANDOFF.md first, then results_seed42.csv, optimization_audit.csv and
completed_run_audit.json (PASS). The final evidence and manuscript remain
immutable; Phase 3A remains terminal. No seed123/456 replication is authorized.
Project status: modern_pilot_complete_replication_review_next. STOP; await
research-lead review before any additional training or manuscript work.

## 2026-10-03 Authorized Post-Freeze Extension: Phase 3A

Phase 3A, the offline Safe-Distillation diagnostic, is complete. It did not launch KD training, use test data for score construction, write checkpoints, alter endpoints, edit the manuscript, or modify `research/final_evidence/`.

The terminal decision is `STOP_NEW_METHOD_DIAGNOSTIC_FAILED`. Across 21 seed-level conditions and three deterministic calibration subsets, the second-order quadratic score and first-order validation-gradient dot product have identical signs on all 63 rows and identical primary classification metrics. They fail to retain established positive WT2 Teacher J/TG, Grassmann-only, and TT families. Only 11/21 seed-level conditions are sign-stable across calibration subsets. The nominal-LR virtual AdamW control identifies harmful rows but falsely predicts harm for 9/15 beneficial seed rows. The literal first scheduled AdamW update is a verified zero-LR no-op.

The authoritative Phase 3A entry points are `research/experiments/phase3a_safe_lambda_diagnostic/REPORT.md`, `GPT_HANDOFF.md`, `diagnostics_blinded.csv`, `diagnostics_with_endpoints.csv`, and `condition_summary.csv`. The blind-table SHA256 is `594ac6d441fbe19776280225a0d652eb38baa7402dbf7a7295e065f7530fd9ad`. Do not start Phase 3B or turn this failed diagnostic into a method claim.

Audit date: 2026-09-19

Scope: repository, experiment artifacts, logs, current CAC manuscript, and post-rejection work in /home/xin/fuwuqi/grassmann-flows.

This document is an evidence audit, not a paper draft or a new research proposal. No training was launched during this audit. The actual code and saved artifacts take precedence over older documentation and manuscript descriptions.

Evidence labels used below:

- CONFIRMED: directly supported by current code plus a complete config/summary or benchmark artifact.
- PARTIAL: an artifact exists, but the comparison is single-seed, confounded, incomplete, or otherwise insufficient for a strong claim.
- UNVERIFIED: stated in documentation or hypothesized, but not supported by an adequate current artifact chain.
- IMPLEMENTED: code exists and is reachable behind the stated entry point.
- RUN: at least one execution artifact exists.

## 1. Project Objective

### 1.1 Original objective

The repository began as an independent reproduction of the Grassmann-flow language-modeling method described in “Attention Is Not What You Need.” The original question was whether the Grassmann sequence mixer could approach a size-matched Transformer on WikiText-2 while offering a different sequence-modeling operator. README.md still presents this reproduction as the main project and reports a 22.6% performance gap.

The currently recoverable baseline artifact is outputs/experiments/20260317_114514_wt2_baseline_both. It contains a 17.695M-parameter Grassmann model with test PPL 244.411 and a 17.670M-parameter Transformer with test PPL 197.645, a 23.66% relative PPL gap. These values do not exactly match the README table values 242.94 and 198.17. The README statement is therefore historical and not tied to the currently identified canonical summary.

### 1.2 Current research question

The active question after the CAC rejection is:

> Under what conditions does output-level KD from a heterogeneous Grassmann–Transformer teacher improve or harm a smaller hybrid student, and can the negative-transfer boundary be explained or reduced without increasing inference cost?

The current evidence is strongest for the narrower statement that fixed-strength warm-start KD is beneficial on PTB but harmful on TinyStories under a matched training protocol. The broader cross-domain statement involving WikiText-2 and CodeParrot remains single-seed legacy evidence.

### 1.3 Claimed contributions and current evidence

| Candidate contribution or claim | Status | Evidence boundary |
|---|---|---|
| Causal Grassmann mixing based on projected token pairs and normalized Plücker coordinates | IMPLEMENTED | src/models/grassmann_v4.py |
| A heterogeneous teacher formed by late fusion of Transformer and Grassmann logits | IMPLEMENTED and RUN | train_hybrid_latefusion_alpha_ddp_v1.py and complete teacher runs |
| Warm-start KD substantially improves a Hybrid-lite student on PTB | CONFIRMED | Three matched seeds with token-normalized KD; mean PPL 59.2853 to 51.2859 |
| The same fixed KD strength harms TinyStories | PARTIAL but strong single-seed evidence | Matched seed 42; PPL 4.8611 to 5.4127 |
| KD behavior is generally domain-dependent | PARTIAL | Four-domain legacy sweep is mostly seed 42; domain, teacher quality, and data scale are confounded |
| Token normalization itself improves accuracy | NOT SUPPORTED | PTB legacy alpha 0.02 and token lambda 5 give essentially the same result at fixed length; normalization improves interpretation and provenance, not observed accuracy |
| Teacher-branch disagreement explains negative transfer | PARTIAL | Directional associations exist, but incremental held-out value is tiny on PTB and small on TinyStories |
| Consensus-Routed Branch Distillation is effective | REJECTED BY CURRENT PROTOTYPE | Stage C failed its preregistered TinyStories and shuffled-control gates |
| Hybrid-lite is a large compression result | NOT SUPPORTED | The 31.434M student is only 13.3% smaller than the 36.264M PTB teacher; the 23.565M student provides the clearer 35.0% reduction |
| The current project establishes a Grassmann-specific KD advantage | NOT SUPPORTED | Matched Transformer-only, Grassmann-only, and Hybrid-lite student controls are not complete |

### 1.4 Paper/code mismatch summary

The current manuscript is 论文投稿/cac/conference_101719.tex. Its title is “Warm-Start Knowledge Distillation for Grassmann--Transformer Hybrids: An Empirical Study Across Text Domains,” not the earlier submitted title “Bridging the Gap: Warm-start Hybrid-lite Distillation for Grassmann Flow Sequence Models.”

The manuscript accurately describes the current Grassmann pair order as (z_t, z_{t-\Delta}), the normalized Plücker representation, feature-wise internal gate, and final-logit fusion. However:

- It reports the historical legacy objective based on batchmean KL and the single-seed coefficient sweeps.
- It does not contain the September 2026 three-seed PTB token-normalized confirmation.
- It does not contain the September 2026 branch-disagreement diagnostic or failed Stage-C prototype.
- It describes only the historical objective L=(1-lambda)CE+lambda KL. The recent confirmation uses CE+lambda_kd KL_token_mean with lambda_kd=5.
- Its strongest cross-domain table is still single-seed outside PTB.
- It cannot be used as the definitive description of the latest code because the CRBD strategies and token-normalized path were implemented after the manuscript.

## 2. Repository Structure

### 2.1 Active code

| Path | Current role |
|---|---|
| src/models/grassmann_v4.py | Current Grassmann GPT, Plücker encoder, causal multi-window mixer, feature-wise gate |
| train_exp4_ddp.py | Shared SmallTransformer and TextDataset implementation; early single-branch training |
| train_hybrid_latefusion_alpha_ddp_v1.py | Main late-fusion teacher construction |
| train_hybrid_lite_latefusion_baseline_v2.py | CE-from-scratch Hybrid-lite student training |
| train_distill_hybrid_lite_from_latefusion_teacher_v2.py | Current KD entry point for random-init or warm-start students; fixed and opt-in adaptive strategies |
| src/kd_losses.py | Legacy batchmean and token-normalized KD objectives |
| src/crbd_losses.py | Opt-in Stage-C entropy, disagreement, branch, CRBD, shuffled, and swapped objectives |
| src/branch_diagnostics.py | Token-level disagreement and CE–KD conflict statistics |
| analyze_branch_disagreement.py | Extracts compact diagnostic statistics from frozen endpoints |
| analyze_branch_diagnostic_statistics.py | Chunk-grouped cross-validation and bootstrap analysis |
| benchmark_ptb_efficiency_v4.py | PPL, latency, throughput, memory, and parameter benchmark |
| tools/build_experiment_registry.py | Normalizes heterogeneous config/summary schemas into an audit registry |

### 2.2 Experiment launchers

The shell suites in the repository encode many actual configurations more reliably than configs.py. Important examples are run_ptb_latefusion_suite.sh, run_ptb_hybrid_lite_baseline_suite_v2.sh, run_ptb_next_stage_warmstart_suite.sh, run_wt2_distill_suite.sh, run_ts_distill_suite.sh, run_code_distill.sh, run_h0_kd_normalization_smoke.sh, run_h0_kd_gradient_diagnostic.sh, run_stage_c_ptb_smoke.sh, run_stage_c_tinystories_prototype.sh, and run_stage_c_ts_gpu3_retry.sh.

### 2.3 Artifacts

| Path | Content |
|---|---|
| outputs/experiments | Single-branch and early paired runs |
| outputs/hybrid_experiments | Fused teachers and CE Hybrid-lite baselines |
| outputs/distill_experiments | KD runs, including H0/A1/A2 and Stage C |
| outputs/transformer_experiments | Three historical two-stage Transformer artifacts |
| outputs/subspace_analysis | Twelve experimental directories outside the current registry schema |
| outputs/benchmark_reports | PTB quality/latency benchmark JSON files |
| logs | Launcher logs, including failed and retried runs |
| research/experiments | Frozen protocols, result tables, and diagnostic analyses |
| docs/generated | Generated registry, checkpoint manifest, and loss-scale audit; generated on 2026-09-16 and stale with respect to Stage C |
| 论文投稿/cac | Current CAC manuscript and figures |

The current read-only scan found 216 run directories in the four registry families and 218 normalized result rows: 185 complete directories, 29 config-only directories, and 2 incomplete directories. The extra two rows arise because a paired summary can expose both Grassmann and Transformer result blocks. There are also 12 subspace-analysis directories that are outside the registry families. Checkpoint subdirectories contain 196 .pt files totaling 19.448 GiB; including weights stored directly in other output directories gives 205 .pt files totaling 20.011 GiB.

## 3. Current Architecture

### 3.1 Transformer branch

The Transformer used by the main experiments is SmallTransformer in train_exp4_ddp.py. It has learned token and position embeddings, pre-normalized residual blocks, PyTorch MultiheadAttention with a causal mask, a GELU feed-forward block of width 4d, a final layer normalization, and a tied output head by default. The language-model loss shifts logits by one token and applies token-level cross entropy.

The primary Hybrid-lite configuration is:

| Field | Value |
|---|---:|
| Vocabulary | 50,257 |
| Sequence length | 256 |
| Model dimension | 224 |
| Layers per branch | 6 |
| Transformer heads | 8 |
| FFN dimension | 896 |
| Dropout | 0.1 |
| Student late_k | 1 |

The efficiency student uses d=192, r=48, four layers per branch. Historical source teachers can use different branch dimensions inherited from their source checkpoints; the PTB teacher has 36.264M parameters in total.

### 3.2 Grassmann branch

For hidden states h_t in R^d, the current implementation first projects to z_t in R^r. For each causal offset delta in the configured window set, it forms antisymmetric coordinates

p_ij(t,delta) = z_t,i z_(t-delta),j - z_t,j z_(t-delta),i.

This confirms that the implementation uses the ordered causal pair (z_t, z_{t-\Delta}), not (z_{t-\Delta}, z_t), not adjacent future states, and not a learned two-token matrix. In src/models/grassmann_v4.py, z_current is z[:, delta:, :] and z_past is z[:, :-delta, :].

The upper-triangular i<j coordinates have dimension r(r-1)/2. Each Plücker vector is L2-normalized with epsilon 1e-8, projected back to d, accumulated at its current position, and averaged over the valid offsets at that position. Positions without a valid offset receive a zero Grassmann feature. The primary student uses offsets {1,2,4}; the class default remains {1,2,4,8,12,16}.

A feature-wise sigmoid gate is computed from [h_t;g_t], producing a vector alpha_t in (0,1)^d. The mixer returns alpha_t elementwise times h_t plus (1-alpha_t) elementwise times g_t, followed by internal layer normalization and dropout. GrassmannBlock then wraps this mixer and an FFN in pre-normalized residual connections.

### 3.3 What is and is not shared

The current teacher and Hybrid-lite student both contain a full Transformer branch and a full Grassmann branch. The branches have separate embeddings, blocks, final normalizations, and output heads. They do not exchange hidden states in the main late-fusion topology. The only top-level interaction is a learned scalar fusion of branch logits at the selected final layer. Consequently, “hybrid” in the main result means two independent networks plus logit fusion, not alternating Transformer and Grassmann layers.

## 4. Fusion Mechanisms

### 4.1 Implemented mechanisms

| Mechanism | Implementation | Actual logic | Run status |
|---|---|---|---|
| Offline fixed alpha sweep | eval_fusion*.py | Evaluate alpha z_T +(1-alpha) z_G without updating branch weights | RUN; historical |
| Scalar learned logit fusion | train_hybrid_alpha_ddp.py | One learned scalar after final branch logits | RUN on PTB and WT2 |
| “Layerwise” v1 | train_hybrid_layerwise_alpha_ddp_v1.py | Despite the filename, this is the same one-scalar final-logit model | RUN; naming is misleading |
| True layerwise hidden fusion | train_hybrid_layerwise_alpha_ddp_v2.py | After each matched layer, mix hidden states with alpha_l and feed the same mixture to both next layers | RUN on PTB same-config source pair |
| Late-k logit fusion | train_hybrid_latefusion_alpha_ddp_v1.py | Produce logits from the last k hidden states of each independent branch, fuse each pair, then average across k | RUN; main teacher uses k=1 |
| Hybrid-lite late fusion | train_hybrid_lite_latefusion_baseline_v2.py | Same k=1 final-logit topology, but both smaller branches are trained jointly from scratch | RUN on four datasets |

For joint teacher training, both branch families are frozen for freeze_branch_epochs, normally one epoch, and then unfrozen. Alpha has a separate optimizer group. For alpha_only, branches remain frozen.

### 4.2 Fusion results and validity

| Dataset / comparison | Test PPL | Interpretation |
|---|---:|---|
| PTB offline fixed alpha, best alpha 0.48 | 50.778 | Static fusion improves over either source branch in that source pair |
| PTB scalar alpha-only | 50.779 | Essentially reproduces static fusion |
| PTB scalar joint, 10 epochs | 50.104 | Strong teacher result |
| PTB scalar joint, 20 epochs | 50.098 | Only a negligible gain over 10 epochs |
| PTB late-k=1 alpha-only | 50.779 | Same expected frozen-branch behavior |
| PTB late-k=1 joint | 50.112 | Canonical PTB teacher |
| PTB late-k=2 joint | 50.686 | Worse than k=1 |
| PTB true layerwise hidden alpha-only | 64.471 | Different source configuration from the main late-fusion teacher |
| PTB true layerwise hidden joint | 63.932 | Cannot isolate fusion position against late fusion because source checkpoints/configuration differ |
| WT2 scalar joint | 66.866 | Canonical manuscript teacher, but source/data-generation provenance is not fully matched to early branch runs |
| TinyStories late-k=1 joint | 4.969 | Single-seed teacher |
| Code late-k=1 joint, common 5k | 5.682 | Same total parameter count as the Hybrid-lite code student |

The fusion comparison is not a clean architectural ablation. The true layerwise runs use a same-config source pair from outputs/experiments/20260318_094720_ptb_baseline_both, whereas the strongest late-fusion run uses a different Grassmann source configuration. The v1 “layerwise” runs do not implement hidden fusion at all. Therefore, current artifacts support selection of a useful teacher, not a causal claim that late fusion is superior to hidden fusion.

## 5. Distillation Pipeline

### 5.1 Teacher

The current teacher is loaded from a completed HybridLateFusionAlphaModel run. The loader reconstructs both source branches from the run metadata, loads the fused checkpoint, freezes the teacher, and produces final fused logits. For Stage B/C, the same forward implementation can additionally expose Transformer and Grassmann branch logits.

Canonical teachers:

| Dataset | Teacher run | Params | Test PPL |
|---|---|---:|---:|
| PTB | outputs/hybrid_experiments/20260328_095917_ptb_latefusion_last1_joint | 36.264M | 50.1125 |
| WikiText-2 | outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint | 37.750M | 66.8663 |
| TinyStories | outputs/hybrid_experiments/20260515_022710_ts_latefusion_last1_joint_e10 | 36.264M | 4.9691 |
| Code common-5k | outputs/hybrid_experiments/20260529_120718_code_teacher_last1_joint | 31.434M | 5.6821 |

### 5.2 Student modes

The current KD entry point supports student_type values that reconstruct a Grassmann-only, Transformer-only, or Hybrid-lite student. The paper-facing results use Hybrid-lite.

The four experimental modes should be interpreted as follows:

| Mode | Initialization | Objective | Current evidence |
|---|---|---|---|
| CE scratch | Random | CE only, normally through train_hybrid_lite_latefusion_baseline_v2.py | Complete baselines; PTB has three seeds |
| RI+KD | Random | CE plus KD in the distillation runner | Historical runs exist, but most are not matched to CE scratch in learning rate or epochs |
| WS+CE | Load CE Hybrid-lite checkpoint | CE continuation, KD coefficient zero | Matched control in recent PTB and TinyStories confirmation |
| WS+KD | Load the same CE checkpoint | CE plus positive KD | Strongest current evidence |

### 5.3 Loss definitions

The historical legacy objective is:

L_legacy = (1-alpha) CE + alpha KL_batchmean,

where PyTorch batchmean divides by batch size but sums over the 255 shifted prediction positions. The direction is KL(p_teacher || p_student), computed by passing student log probabilities and teacher probabilities to F.kl_div. Temperature scaling multiplies the KL by T squared.

The current token-normalized objective is:

L_token = CE + lambda_kd KL_token_mean.

At fixed length 256 with no ignored tokens, KL_batchmean is 255 times KL_token_mean. Apart from a positive global scale on the full objective, legacy alpha and token lambda correspond through:

lambda_kd = 255 alpha / (1-alpha).

Thus legacy alpha=0.02 corresponds to lambda_kd approximately 5.204. The H0 smoke showed that token lambda=5 closely reproduces the legacy alpha=0.02 result. This is an implementation/provenance correction, not an accuracy innovation.

### 5.4 Optimization

The recent confirmation uses AdamW, learning rate 1e-4, weight decay 0.01, warmup ratio 0.05, cosine decay, gradient clipping at norm 1, AMP, and validation-loss checkpoint selection. Test is evaluated from the best validation checkpoint. There is no early stopping. The baseline runner uses learning rate 2e-4 and no warmup. Earlier base train_exp4_ddp.py enables AMP whenever the device is CUDA, regardless of its --amp flag.

## 6. Dataset Status

The common tokenizer is the local GPT-2 tokenizer with vocabulary size 50,257. Text is filtered for empty records, concatenated with newline separators, tokenized without added special tokens, partitioned into non-overlapping fixed 256-token chunks, and the final remainder is dropped. Inputs and labels are the same tensor; the model shifts internally. There is no dynamic padding in the main experiments.

| Dataset | Saved path on server | Text field | Canonical train budget | Tokens before trim / chunks | Status |
|---|---|---|---|---:|---|
| PTB | /workspace/grassmannflows/datasets/ptb_text_only_saved | sentence | Full saved train split | 1,136,446 / 4,439 | Best-audited dataset; multi-seed KD evidence |
| WikiText-2 | /workspace/grassmannflows/datasets/wikitext2_v1_saved | text | Full saved train split | 2,401,115 / 9,379 in teacher metadata | Single-seed current teacher/student/KD chain; early baseline used 2,391,884 tokens and 9,343 chunks |
| TinyStories | /workspace/grassmannflows/datasets/tinystories_saved | text | max_lines=300,000 for full experiments | 67,081,279 / 262,036 | Full-data matched CE/KD only seed 42 |
| CodeParrot common subset | /workspace/grassmannflows/datasets/codeparrot_saved | text | max_lines=5,000 per split | 22,293,130 / 87,082 train chunks | Usable single-seed same-data chain |
| CodeParrot historical full route | Same saved path | text | max_lines unset; about 181M train tokens in audited runs | Varies | Confounded with PTB teacher/init provenance; not comparable to common-5k |

TinyStories handling is special in TextDataset for both disk and online loading: the original training split is divided deterministically into train and validation with split_seed, while the original validation split is used as test. The saved dataset artifacts should be treated as authoritative for completed runs, but no content hash or immutable dataset manifest is recorded.

Code metadata is unreliable in older configs. The common-5k teacher records dataset_name=ptb even though dataset_path is codeparrot_saved and text_field=text. Dataset identity must be inferred from path and token counts, not from dataset_name alone.

No current main-chain experiment uses a separate “code” loader choice. Code experiments reuse the generic disk loader with an inherited dataset label. Other datasets mentioned by the base runner, including WikiText-103, have no current paper-level evidence chain.

## 7. Experiment Ledger

The generated registry at docs/generated/experiment_registry.csv is useful but stale as of 2026-09-16. The current scan reports 216 directories in its four families, whereas docs/generated/experiment_inventory.md still reflects an earlier count. The table below records the research-relevant completed chains and explicitly marks weaker evidence. Duplicate failed launch directories and config-only retries are not treated as results.

| Experiment family | Dataset / seeds | Main configuration | Result | Evidence | Artifact |
|---|---|---|---|---|---|
| Early reproduction baseline | WT2 / 42 | 17.7M branches, 20 epochs | Grassmann 244.411; Transformer 197.645 | CONFIRMED artifact, historical protocol | outputs/experiments/20260317_114514_wt2_baseline_both |
| Early PTB branches | PTB / 42 | 17.7M branches, 20 epochs | Grassmann 63.609; Transformer 57.165 | CONFIRMED artifact | outputs/experiments/20260318_094720_ptb_baseline_both |
| Offline PTB logit sweep | PTB / 42 | fixed alpha grid | alpha 0.48; test 50.778 | PARTIAL | fusion_result.json |
| PTB learned scalar/late fusion | PTB / 42 | joint 10 epochs | scalar 50.104; late-k1 50.112; late-k2 50.686 | CONFIRMED artifacts; architectural comparison PARTIAL | outputs/hybrid_experiments/20260328_* |
| PTB hidden layerwise fusion | PTB / 42 | true per-layer alpha | joint test 63.932 | CONFIRMED run; comparison to main teacher confounded | outputs/hybrid_experiments/20260328_093917_ptb_layerwise_alpha_joint_samecfg |
| PTB CE Hybrid-lite baseline | PTB / 42,123,456 | 224x56x6, 20 epochs | 58.533, 59.043, 58.484 | CONFIRMED | outputs/hybrid_experiments/20260328_142654* and 20260402_140126/140326* |
| PTB smaller CE baseline | PTB / 42,123,456 | 192x48x4, 20 epochs | 60.625, 60.565, 60.401 | CONFIRMED | outputs/hybrid_experiments/20260402_130614/140739/143306* |
| PTB legacy WS+KD quality student | PTB / 42,123,456 | 224x56x6, alpha 0.05 | 51.843, 51.884, 51.774 | CONFIRMED repeated endpoint; not matched against CE continuation at every historical seed | outputs/distill_experiments/20260402_112236/142019/142442* |
| PTB legacy WS+KD efficiency student | PTB / 42,123,456 | 192x48x4, alpha 0.05 | 53.872, 53.994, 53.876 | CONFIRMED repeated endpoint | outputs/distill_experiments/20260402_131852/142405/144420* |
| PTB legacy alpha sweep | PTB / 42 | alpha 0 to 0.2 | best later point alpha 0.02: 51.274; CE continuation 59.078 | PARTIAL single-seed sweep | outputs/distill_experiments/20260329_* and 20260515_12* |
| WT2 teacher / CE / legacy WS+KD | WT2 / 42 | 224x56x6 student | teacher 66.866; CE scratch 70.163; WS+CE 71.575; best alpha 0.02 60.835 | PARTIAL single seed | outputs/hybrid_experiments/20260328_080104*, 20260515_004119*; outputs/distill_experiments/20260515_* |
| TinyStories teacher / CE / legacy WS+KD | TS / 42 | 300k stories | teacher 4.969; CE scratch 5.065; WS+CE 4.861; best positive alpha 0.01 5.313 | PARTIAL single seed, negative-transfer direction consistent | outputs/hybrid_experiments/20260515_*; outputs/distill_experiments/20260516_* |
| Code common-5k teacher / CE / legacy WS+KD | Code / 42 | batch 16, 10 epochs | teacher 5.682; CE scratch 7.669; WS+CE 7.245; best alpha 0.01 6.002 | PARTIAL single seed; teacher and student equal size | outputs/hybrid_experiments/20260529_120718*, 20260527_074518*; outputs/distill_experiments/20260529_154901*, 20260530_* |
| H0 normalization smoke | PTB / 42 | one epoch | legacy alpha 0.02 and token lambda 5 differ by less than 0.05 validation/test PPL; lambda 10 clip fraction 54.7% | CONFIRMED smoke only | outputs/distill_experiments/20260916_034* |
| A1 matched token KD | PTB / 42,123,456 | WS+CE versus WS+KD lambda 5, 10 epochs | CE mean 59.2853; KD mean 51.2859; paired delta -7.9994 | CONFIRMED strongest causal result | research/experiments/post_rejection_program/results.csv |
| A2 matched token KD | TS / 42 | same init and schedule | CE 4.8611; KD 5.4127; delta +0.5516 | CONFIRMED run, PARTIAL generality | same result table |
| Stage B disagreement diagnostic | PTB / 42 endpoints | all 350 validation chunks | JSD adds AUROC -0.00032 and R2 +0.00053 | CONFIRMED diagnostic; weak mechanism value | research/experiments/h1_branch_disagreement/ptb_full_seed42 |
| Stage B disagreement diagnostic | TS / 42 endpoints | 2,048 chunks | JSD adds AUROC +0.01091 and R2 +0.00306 | CONFIRMED diagnostic; small effect | research/experiments/h1_branch_disagreement/tinystories_full_seed42_2048 |
| Stage C bounded prototype | PTB smoke and TS 20k / 42 | C0,C1,C2,C4,C5,C6,C7 | CRBD TS val 5.5104 versus fixed 5.3041 and shuffled 5.5045; gate failed | CONFIRMED negative result | research/experiments/h2_crbd/results.csv |
| PTB batch latency | PTB / fixed checkpoints | bs 1,8,32; RTX 3090 | See Section 9 | CONFIRMED on one hardware/software setting | outputs/benchmark_reports/ptb_final_*.json |
| Subspace analyses | PTB and derived snapshots | hooks/SVD variants | Some generations all zero; later scripts encode conflicting rank expectations | UNVERIFIED | outputs/subspace_analysis and run_subspace_verify.py |
| CUDA speed claim | historical | custom extension | README states 2x, but current correctness/import chain is not closed | UNVERIFIED for current code | README.md, logs/build_grassmann_cuda.log, logs/test_cuda_kernels.log |

## 8. Recently Completed Codex Experiments

### 8.1 H0: KD normalization smoke

Motivation: determine whether historical batchmean scaling invalidated coefficient interpretation and select a token-normalized coefficient without claiming a new method.

Code changes: src/kd_losses.py added legacy_batchmean and token_mean modes. train_distill_hybrid_lite_from_latefusion_teacher_v2.py added explicit loss-mode, coefficient, chunking, gradient-clipping, non-finite, and AMP-overflow metadata. Unit tests were added under tests/test_kd_losses.py.

Exact launcher: run_h0_kd_normalization_smoke.sh. The recorded remote execution pattern was:

    ~/fuwuqi/agent-tools/exec_grassmann.sh 'cd /workspace/grassmannflows/grassmann-flows && bash run_h0_kd_normalization_smoke.sh'

Configuration: PTB teacher 20260328_095917, seed-42 Hybrid-lite warm start 20260328_142654, one epoch, batch 32, lr 1e-4, T=2, sequence length 256. Arms were CE only, legacy alpha 0.02, and token lambda 2.5, 5, and 10.

Result: token lambda 5 approximately reproduced legacy alpha 0.02, as expected from the 255-token scale relation. Lambda 10 was clip-heavy. Interpretation: token normalization is necessary for interpretable coefficients and future variable-length data, but it does not explain the TinyStories sign reversal.

Problems: the first legacy launch OOMed on GPU 0 because an external process occupied memory. The failed directory/log was preserved and the run was repeated on a free GPU.

### 8.2 A1: PTB matched three-seed CE versus token KD

Motivation: replace the single-seed legacy sweep with a matched causal comparison.

Command source: the fully expanded command and per-seed substitutions are frozen in research/experiments/post_rejection_program/experiment_plan.md. Each run used train_distill_hybrid_lite_from_latefusion_teacher_v2.py with teacher 20260328_095917, the seed-matched Hybrid-lite checkpoint, token_mean, lambda 0 or 5, T=2, 10 epochs, batch 32, lr 1e-4, and PTB full saved data. Remote execution used exec_grassmann.sh. Logs are logs/h0_ptb_confirm_token_l{0,5}_seed*.log.

| Seed | CE test PPL | KD test PPL | Paired delta |
|---:|---:|---:|---:|
| 42 | 59.1175 | 51.2771 | -7.8405 |
| 123 | 59.6566 | 51.3745 | -8.2821 |
| 456 | 59.0819 | 51.2061 | -7.8758 |
| Mean | 59.2853 | 51.2859 | -7.9994 |

Sample SD is 0.3220 for CE, 0.0846 for KD, and 0.2454 for the paired delta. All seeds have the same direction. The mean relative reduction is 13.49%.

Problems: the initial seed-42 command referenced a nonexistent initialization directory and failed before training. The correct directory is outputs/hybrid_experiments/20260328_142654_ptb_hybrid_lite_baseline_mild_e20. The 30-minute limit was too short for a ten-epoch PTB run and was extended to 50 minutes with author approval. Failed artifacts were retained.

### 8.3 A2: TinyStories matched CE versus token KD

Motivation: confirm whether the historical negative-transfer direction survives token normalization and a matched endpoint.

Command source: research/experiments/post_rejection_program/experiment_plan.md. Logs are logs/h0_ts_confirm_token_l0_seed42.log and logs/h0_ts_confirm_token_l5_seed42.log.

Configuration: seed 42, 300,000-story budget, batch 32, 10 epochs, lr 1e-4, T=2, token lambda 0 or 5, same warm-start checkpoint and teacher.

Result: CE test PPL 4.8611; KD test PPL 5.4127; delta +0.5516 or +11.35%. The CE continuation surpassed both the 5.0654 warm-start checkpoint and the 4.9691 teacher. The KD endpoint was worse than both.

Problems: KD had first-epoch gradient clipping fraction 0.588, declining to 0.074 by epoch 10. Non-finite/overflow fraction stayed below 0.0007. The run is valid, but lambda 5 is aggressive on this domain.

### 8.4 Stage B: branch-disagreement diagnostic

Motivation: test whether teacher-branch JSD predicts CE–KD gradient conflict and endpoint harm beyond teacher entropy and NLL.

Entry points: analyze_branch_disagreement.py and analyze_branch_diagnostic_statistics.py. The latter was run with five chunk-grouped folds, 2,000 chunk-bootstrap resamples, and seed 20260916. Exact inputs and selection hashes are stored in each output manifest. Logs are logs/h1_ptb_branch_diagnostic_*.log and logs/h1_tinystories_branch_diagnostic_*.log.

PTB result: 350 chunks and 89,250 tokens. Mean endpoint delta NLL was -0.1555, but 40.34% of tokens were harmed and 36.91% had negative CE–KD gradient cosine. Conditional JSD coefficient for gradient cosine was -0.0212 with 95% interval [-0.0274,-0.0147]. JSD changed held-out conflict AUROC by -0.00032 and endpoint R2 by +0.00053.

TinyStories result: 2,048 chunks and 522,240 tokens. Mean delta NLL was +0.1069; 55.05% of tokens were harmed and 45.35% had negative gradient cosine. JSD changed held-out conflict AUROC by +0.01091 and R2 by +0.00306. Harm frequency rose from 37.97% in the lowest JSD decile to 62.26% in the highest.

Interpretation: the preregistered directional gate passed literally, but the independent predictive value is weak, especially on PTB. This supported only a bounded prototype, not a mechanism claim.

Problems: the first diagnostic formula suffered floating-point cancellation in a closed-form CE norm near saturated probabilities. It was replaced with an explicit CE gradient vector, theoretical bounds were enforced, and the corrected tests passed remotely. Earlier smoke outputs remain historical and should not be mixed with the corrected full outputs.

### 8.5 Stage C: bounded CRBD prototype

Motivation: test whether routing low-disagreement tokens to fused KD and high-disagreement tokens to architecture-matched branch KD repairs TinyStories negative transfer while preserving PTB.

Code changes: src/crbd_losses.py and opt-in distill_strategy flags were added. The default fixed_fused path remained unchanged. Tests cover identical distributions, ignored labels, branch decomposition, deterministic shuffling, and swapped pairing. The reported remote test suite passed 15/15.

Exact launchers:

    ~/fuwuqi/agent-tools/exec_grassmann.sh 'cd /workspace/grassmannflows/grassmann-flows && bash run_stage_c_ptb_smoke.sh'
    ~/fuwuqi/agent-tools/exec_grassmann.sh 'cd /workspace/grassmannflows/grassmann-flows && bash run_stage_c_tinystories_prototype.sh'

The TinyStories prototype used max_lines=20,000, 4,473,074 pre-trim train tokens, 17,472 train chunks, seed 42, batch 8, two epochs, lr 1e-4, T=2, fused lambda 5, branch lambda 2.5, and tau 0.25.

| Arm | Strategy | Validation / test PPL | Status |
|---|---|---:|---|
| C0 | CE only | 5.0183 / 5.0195 | Complete |
| C1 | Fixed fused KD | 5.3041 / 5.3018 | Complete |
| C2 | Entropy-gated fused KD | 5.3027 / 5.3008 | Complete |
| C3 | Disagreement-suppressed fused KD | — | Incomplete; infrastructure stop |
| C4 | Branch-only KD | 5.6135 / 5.6100 | Complete |
| C5 | CRBD | 5.5104 / 5.5067 | Complete |
| C6 | Shuffled routing | 5.5045 / 5.5010 | Complete |
| C7 | Swapped branch pairing | 6.3668 / 6.3616 | Complete |

The PTB smoke condition passed, but the full gate failed. CRBD increased rather than repaired the fixed-KD excess and was 0.0059 validation PPL worse than shuffled routing. It did outperform swapped pairing, but that control had a much larger branch KL and does not isolate architecture alignment at a matched loss scale.

All completed KD arms clipped gradients on every step in both epochs. In C5 epoch 2, weighted fused and branch contributions were about 2.77 and 3.53, compared with CE 1.73. Separate mean-one normalization preserved a full budget for each KD component, creating a loss-scale confound. Stage D was correctly stopped.

C3 failed twice because GPU 3 was locked near 210 MHz under concurrent load and the tasks were externally terminated. It is missing, not a negative method result.

## 9. Known Results

### 9.1 CONFIRMED

The strongest scientific result is the PTB matched A1 experiment: token-normalized warm-start KD with lambda 5 improves test PPL by about 8 points for each of three paired seeds, reducing the mean from 59.2853 to 51.2859.

The strongest negative result is the matched TinyStories seed-42 experiment: the same training protocol changes PPL from 4.8611 to 5.4127. A teacher can be better than the original CE checkpoint while still becoming worse than a continued CE student; fixed KD then prevents the student from surpassing the teacher.

The best historical PTB deployment points remain:

| Model | Params | PPL | bs1 latency | bs8 latency | bs32 latency |
|---|---:|---:|---:|---:|---:|
| Teacher | 36.264M | 50.1125 | 15.482 ms | 19.145 ms | 55.984 ms |
| Quality student | 31.434M | 51.8431 | 16.089 ms | 19.332 ms | 48.460 ms |
| Efficiency student | 23.565M | 53.8719 | 11.502 ms | 16.559 ms | 34.959 ms |

The quality student is not faster at batch sizes 1 or 8. Its latency benefit appears only at batch size 32. The efficiency student is faster at all three measured batch sizes. These measurements are specific to a single RTX 3090 software/hardware setting.

The Stage-C formulation failed its preregistered gate and should not be expanded in its current form.

### 9.2 PARTIAL

The legacy single-seed sweeps show improvements over matched warm-start CE on WikiText-2 and common-5k code, and degradation on TinyStories. They support hypothesis generation, not a universal domain taxonomy.

The TinyStories negative-transfer result is matched but has only one KD-stage seed and only one independently trained warm-start checkpoint.

Teacher-branch JSD has a statistically stable conditional association with conflict under chunk bootstrap, but its incremental prediction over entropy and NLL is negligible on PTB and small on TinyStories.

The late-fusion teacher is empirically strong, but the current fusion variants do not form a checkpoint- and budget-matched architectural ablation.

### 9.3 UNVERIFIED

The README claim of a current 2x CUDA inference speedup is not supported by a closed correctness and import chain for the present repository state.

The subspace/rank outputs do not establish a link between Grassmann geometry and language-modeling gains. Some outputs come from a hook bug that produced zeros, and later verification code contains conflicting directional expectations.

No artifact establishes that the observed KD benefit is specifically caused by the Grassmann branch rather than by extra parameters, ensembling, joint continuation, or a two-branch student.

No artifact establishes CRBD novelty or effectiveness. The candidate overlaps with disagreement weighting, adaptive teaching, and branch-aligned heterogeneous KD, and its only bounded prototype failed.

## 10. Failed and Negative Experiments

| Item | What happened | Scientific meaning |
|---|---|---|
| Early random-init distillation | PTB PPL often 83–172; branch-only two-stage runs can exceed 200 | Mostly optimization/protocol failures; controls are unmatched and cannot prove warm start is necessary |
| PTB CE continuation | Historical 59.078 and recent mean 59.285, around or worse than CE scratch | Additional CE steps under the KD schedule do not explain the KD gain |
| Large legacy KD coefficients | PTB alpha 0.1–0.2 around 53.1–53.7, worse than alpha 0.02–0.05 | Fixed strength is sensitive; not monotonic |
| TinyStories positive KD | All tested positive legacy points and token lambda 5 are worse than CE continuation | Genuine negative-transfer boundary for the tested seed/protocol |
| Late-k=2 teacher | PTB 50.686 versus 50.112 for k=1 | Averaging more late-layer logits did not help |
| True layerwise hidden fusion | PTB 63.932 in the available joint run | Negative result for that source/configuration, not a fair late-vs-hidden conclusion |
| Code first late-fusion teacher | PPL 18.300 | Wrong or weak source checkpoint generation; superseded by common-5k teacher 5.682 |
| Stage-C branch-only | TinyStories test 5.610 | Harmful at branch lambda 2.5 under saturated clipping |
| Stage-C entropy gate | TinyStories test 5.301, essentially fixed KD | Did not repair negative transfer in the reduced prototype |
| Stage-C CRBD | TinyStories test 5.507; worse than fixed and shuffled | Failed the bounded gate; formulation has a known loss-budget confound |
| Stage-C C3 | Two terminated attempts on clock-limited GPU 3 | Missing result, not a method failure |
| Initial branch diagnostics | Closed-form norm cancellation near saturated probabilities | Superseded by corrected explicit-gradient diagnostics |
| Initial subspace generation | All-zero spectral metrics from a hook bug | Invalid evidence |
| CUDA supplement | State-dict mismatch and circular/import problems were recorded | Current system claim remains unverified |
| Several TS teacher launches | Multiple config-only/incomplete directories before 20260515_022710 | Infrastructure/launch failures; no result should be inferred |

## 11. Reproducibility Problems

### Critical

1. Git provenance is not usable for reproducing the active project. The repository has only the upstream initial commit 67efbc1, while nearly all experiment runners, research code, documents, and the manuscript are untracked. There is no commit corresponding to any reported experiment.

2. The working tree is highly dirty: 26 tracked entries are modified and 98 top-level status entries are untracked. Most tracked changes are file-mode changes, but the active code itself is predominantly untracked, so git diff cannot reconstruct the experimental state.

3. Dataset artifacts lack immutable content hashes and sample manifests. max_lines and token counts are recorded, but they do not prove that another saved dataset directory contains the same examples in the same order.

### High

4. Historical metadata schemas are inconsistent. Code runs can be labeled PTB or WikiText-2; many configs omit script name, data stats, or resolved checkpoint paths. Automatic comparison by experiment name is unsafe.

5. Historical batchmean KD coefficients depend on the number of valid tokens. They are only comparable in the current fixed-length, unpadded setting.

6. Several comparison families are not matched in checkpoints, training budget, learning rate, or epochs. This affects random-init controls, fusion-position comparisons, and full-data versus 5k code experiments.

7. The generated registry is stale after the September 17 Stage-C runs. Current counts are 216 directories and 218 normalized rows in the four registered families, not the earlier generated inventory count.

8. Remote execution paths have changed across generations: /root/songxin, /songxin, and /workspace/grassmannflows appear in summaries. Path remapping code helps load artifacts but can conceal missing provenance.

### Medium

9. The base train_exp4_ddp.py AMP switch ignores the command-line flag and enables AMP on any CUDA device. The current distillation runner handles the flag explicitly.

10. Stage-C logs show PyTorch warning that scheduler.step can precede optimizer.step after AMP overflow, skipping a learning-rate value. This is not shown to change conclusions, but it weakens bit-level reproducibility.

11. Random seeds are set, but deterministic algorithms and deterministic CUDA kernels are not enforced. Dataloader and GPU-level nondeterminism remain.

12. Benchmark results record the measured values but some JSON fields such as model_name, device, batch_size, and sequence length are absent or null; the filenames and launcher are needed to recover the condition.

13. Local Python cannot currently serve as the canonical environment because the local torch/libtorch_python setup is broken. The known environment is the remote grassmann_lab container with Python 3.12.3 and PyTorch 2.6.0a0+cu126.

## 12. Current Bottlenecks

1. Evidence breadth: only PTB has a matched three-seed KD comparison. TinyStories, WikiText-2, and code do not have equivalent replication depth.

2. Causal attribution: dataset domain, data volume, relative teacher quality, branch quality, and loss scale vary together across the four datasets.

3. Method status: the only bounded new-method prototype failed its gate, and its objective operated in a fully clip-saturated regime.

4. Grassmann-specific contribution: there is no matched set of Transformer-only, Grassmann-only, and hybrid students under the current token-normalized protocol.

5. Fusion evidence: the available late, scalar, and hidden-fusion runs do not all start from the same branch checkpoints and training budgets.

6. Code data provenance: multiple generations use different file budgets and teacher/init routes, while configs contain incorrect dataset labels.

7. Geometry mechanism: Plücker rank/subspace analyses are not linked to predictive improvement and include invalid historical generations.

8. Deployment generality: latency is measured on one GPU family and only at sequence length 256; online, memory-limited, and other-hardware conclusions are absent.

9. Research identity: the project spans reproduction, model fusion, compression, domain-sensitive KD, and adaptive routing without a single currently validated novel-method claim.

10. Software provenance: the active state cannot be recovered from git, and historical runs do not store a source snapshot or diff hash.

## 13. Open Scientific Questions

1. Is the sign of KD gain primarily determined by whether continued CE can outperform the teacher, rather than by semantic “domain”?

2. Does the TinyStories negative-transfer direction persist across independently trained warm-start checkpoints for seeds 123 and 456?

3. Under a fully matched schedule and initialization tensor, does random-init KD underperform random-init CE on PTB and TinyStories?

4. Does a two-branch Hybrid-lite student obtain gains that cannot be matched by parameter- and latency-matched Transformer-only or Grassmann-only students?

5. Is late-logit fusion stronger than scalar final fusion or hidden layerwise fusion when every variant uses the same source checkpoints, number of updates, and parameter budget?

6. Does teacher confidence or per-token teacher excess NLL predict KD harm more reliably than Transformer–Grassmann branch disagreement?

7. Can branch-aligned supervision provide benefit after its total loss magnitude and gradient norm are matched to fixed fused KD?

8. Are the WikiText-2 and common-5k code improvements reproducible under token-normalized KD and multiple paired seeds?

9. Does the PTB quality–efficiency frontier remain favorable at other sequence lengths and on hardware other than an RTX 3090?

10. Do any Grassmann-specific representation statistics predict paired token-level or sequence-level KD gains after controlling for teacher entropy, teacher NLL, frequency, and layer depth?

## 14. Candidate Next Experiments

These are proposals only. None was launched during this audit.

| Priority | Candidate experiment | Hypothesis | Compute | Information gain | Decision outcomes |
|---:|---|---|---|---|---|
| 1 | Complete missing TinyStories C3 with the exact frozen Stage-C config | Disagreement suppression alone may match entropy/fixed KD without branch-loss scale confound | Low–medium | Moderate; closes an incomplete arm | Better than C1 supports suppression as a baseline; equal/worse rejects this route at the tested setting |
| 2 | Fixed-total-budget routing prototype with initial loss or gradient norm matched to C1 | The Stage-C failure was caused by double KD budget rather than routing itself | Low–medium prototype | High for CRBD go/no-go | Passing the frozen repair/shuffle gates justifies confirmation; failure ends routed-KD development |
| 3 | TinyStories paired CE/fixed-KD with independently trained seeds 123 and 456 | Negative transfer is robust to initialization, not a seed-42 artifact | High | Very high for the core claim | Consistent harm supports a real boundary; mixed signs downgrade the claim |
| 4 | Fully matched random-init CE versus KD on PTB and TinyStories | Warm start changes optimization response to teacher supervision | Medium–high | High | A paired interaction supports an initialization claim; otherwise remove it |
| 5 | Teacher-quality/confidence baseline against JSD routing | Teacher reliability predicts harm better than heterogeneous-branch disagreement | Medium | High for mechanism selection | Reliability wins narrows the story to teacher quality; JSD adds value supports branch-aware work |
| 6 | Checkpoint-matched PTB fusion ablation | Late-logit fusion is better than hidden or scalar fusion independently of source quality | Medium | Moderate–high | A clean advantage supports fusion design; no advantage removes fusion novelty |
| 7 | Parameter/latency-matched Transformer-only, Grassmann-only, and Hybrid-lite CE/KD controls | The hybrid student adds value beyond capacity and KD alone | High | Very high for Grassmann relevance | Hybrid-specific gain supports the architecture; otherwise simplify the paper claim |
| 8 | Token-normalized three-seed WikiText-2 and common-5k code confirmation | Historical positive effects are reproducible under the corrected objective | High | High for cross-domain generality | Consistent effects support broader scope; inconsistent effects restrict the paper to PTB/TinyStories |
| 9 | Multi-length PTB evaluation at fixed checkpoints where position limits allow | Quality and latency trade-offs change with context length | Low–medium | Moderate for deployment claims | Stable Pareto order broadens efficiency claims; reversal makes them length-specific |

No large multi-domain or Stage-D run should be scheduled before the research lead selects a primary claim and an acceptable compute budget.

## 15. Git and Code State

Branch: main

HEAD: 67efbc158ad823f7196f0696415f6e32b5e2e2fa

Remote: origin points to https://github.com/Infatoshi/grassmann-flows.git.

The only visible commit is “Initial commit: Grassmann Flows reproduction study.” It predates the active KD project. origin/main and local main point to the same commit.

Current git status summary:

- 26 tracked entries modified.
- 98 untracked status entries.
- Tracked textual diff is only eight inserted and three removed lines; most tracked changes are executable-bit changes.
- README.md and src/__init__.py contain textual changes.
- The active training scripts, KD losses, CRBD losses, diagnostics, tests, tools, research directory, docs directory, and 论文投稿 directory are untracked.
- outputs is ignored by .gitignore, so experiment artifacts are not versioned.

The working tree must be treated as the only existing source snapshot. There is no safe historical commit for the March–September experiments and no basis for reverting or cleaning files automatically.

Current project state recorded in research-state.yaml is stage_c_base_prototype_failed_followup_decision_required. This agrees with the completed Stage-C analysis.

## 16. Questions for the Research Lead

1. Is the intended next paper an empirical study of positive versus negative KD transfer, or must it contain a successful new distillation method?

2. Must Grassmann geometry remain central to the contribution, or is the heterogeneous-teacher setting sufficient?

3. Is a PTB/TinyStories two-domain paper acceptable if WikiText-2 and code cannot receive three-seed confirmation?

4. What is the maximum GPU-hour budget for independently trained TinyStories seed-123 and seed-456 initializations?

5. Should the failed CRBD direction be stopped, or is one fixed-budget redesign authorized after C3 is completed?

6. Is the primary deployment objective quality retention, parameter compression, latency, or a combined Pareto frontier?

7. Which venue and page limit are now targeted, and what level of novelty does that venue require?

8. May the active code and documents be committed into a clean project-owned repository so future results can record a source commit?

9. Is the common-5k CodeParrot subset the definitive code benchmark, and can an immutable file/sample manifest be created for it?

10. Should the current CAC manuscript remain an archival record, or should a future manuscript be rebuilt around the matched token-normalized evidence?

End of handoff. No further experiment or manuscript modification was performed.

## 17. Phase 2A Teacher–Student Transfer Audit

Phase 2A was completed as a checkpoint-only diagnostic. No model training, CRBD continuation, checkpoint mutation, dataset modification, or manuscript edit was performed. The pre-analysis repository and runtime state is frozen under `research/snapshots/20260920/`, including Git status, HEAD, tracked diff, untracked research-file inventory, active-file SHA256 manifest, local and server-197 environments, and manifests for PTB, WikiText-2, TinyStories, and CodeParrot common-5k.

The complete audit is under `research/experiments/phase2_teacher_transfer_audit/`. Primary artifacts are `REPORT.md`, `transfer_table.csv`, `teacher_branch_metrics.csv`, `alpha_landscape.csv`, `gradient_compatibility.csv`, `candidate_teachers.md`, and `provenance.md`. Additional analysis tables are `predictor_comparison.csv` and `grassmann_specific.csv`; raw JSON, logs, plotting code, and PDF/PNG figures are retained in the same directory.

The exact test-NLL outcomes are: PTB three-seed mean Delta_teacher +0.168082 and Delta_KD +0.144937; WikiText-2 legacy seed-42 +0.068047 and +0.162579; TinyStories matched seed-42 -0.021960 and -0.107479; CodeParrot common-5k legacy seed-42 +0.243051 and +0.188309. The sign of teacher residual advantage therefore matches the existing KD outcome in all four observations, but this remains a hypothesis because domain, protocol generation, and replication depth are confounded.

Validation-only branch evaluation verifies the fusion convention `alpha * Transformer logits + (1-alpha) * Grassmann logits`. Learned fusion improves NLL over the better branch by 0.146271 on PTB, 0.146483 on WikiText-2, 0.184168 on TinyStories, and 0.304930 on CodeParrot. Fusion also reduces S0 teacher-student KL in all four domains. It improves mean S0 CE-KD cosine relative to the better branch in three domains but worsens it on CodeParrot, so predictive complementarity is supported while a universal gradient-alignment explanation is not.

Canonical S0 gradient cosine does not explain the sign of the existing KD outcomes: WikiText-2 has a negative mean cosine despite positive legacy gain, while TinyStories has a positive mean cosine despite matched KD harm. Prior Stage-B results also remain decisive against a strong disagreement-routing claim: token-level JSD added approximately zero AUROC/R2 on PTB and only negligible increments on TinyStories. Phase 2A supports aggregate branch complementarity, not disagreement as a validated mechanism.

Controlled candidate construction is incomplete by design. PTB supplies T_good at alpha 0.5 and T_near at alpha 1.0 but no clearly bad teacher. TinyStories supplies T_near at alpha 0.6 and T_bad at alpha 0.0 but no good teacher. No missing regime was fabricated.

Important unresolved confounders are the lack of multi-seed TinyStories replication, legacy loss normalization and winner selection for WikiText-2 and CodeParrot, validation-subset versus test-outcome comparisons, coupling among teacher quality/KL/entropy/gradient statistics across alpha, and the absence of parameter-matched controls that would separate Grassmann-specific effects from generic ensemble diversity.

The single recommended next training experiment is option D, a controlled teacher-quality alpha experiment. It should use one domain whose frozen validation landscape demonstrably spans good, near, and bad teachers, identical S0 tensors, matched WS+CE, and the corrected token-normalized KD objective. This recommendation is based on information gain for the residual-quality hypothesis. The experiment has not been launched and requires research-lead review.

## 18. Phase 2B Controlled Teacher Intervention

Phase 2B completed the recommended controlled intervention without modifying the manuscript, datasets, historical checkpoints, CRBD, architecture, routing, or loss design. Validation-only preflight found a complete WikiText-2 triplet within the existing frozen branch pair: T_good at alpha 0.5 with `Delta_teacher=+0.084900`, T_near at alpha 0.3 with `Delta_teacher=+0.000330`, and T_bad at alpha 0.0 with `Delta_teacher=-0.266510`. The alpha convention is Transformer weight; T_good/T_near are heterogeneous fusions and T_bad is pure Grassmann.

All arms use seed 42 and the identical S0 checkpoint SHA256 `9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef`. C0 WS+CE obtains validation/test NLL 4.383154/4.270742 and test PPL 71.5748. C1 T_good obtains 4.216686/4.108241 and 60.8396, for `Delta_KD=+0.162501`. C2 T_near obtains 4.250850/4.139916 and 62.7975, for `Delta_KD=+0.130826`. C3 T_bad obtains 4.340190/4.223972 and 68.3042, for `Delta_KD=+0.046771`.

The preregistered ordering holds exactly as C1 > C2 > C3, with a 0.115731-NLL good-to-bad separation. The result is PARTIAL rather than strong support because T_bad still improves over CE and no positive/negative transfer boundary is crossed. It is a single-seed mechanism pilot, not a statistically established law.

Formal epoch-1 initial KD logit-gradient norms are closely matched across C1/C2/C3 at 0.115484/0.115873/0.118931. Weighted KD contributions are 2.544399/2.731717/3.642114, total parameter-gradient norms are 1.373554/1.333842/1.417159, clipping fractions are 0.547619/0.394558/0.585034, and all three have overflow/non-finite fractions 0.013605. No arm is clip-saturated, instability is low and matched, and C3 is not dominated by a much larger total gradient. Decision 3 is therefore UNLIKELY; no formal arm is flagged `LOSS_SCALE_CONFOUNDER`.

Complete artifacts are under `research/experiments/phase2b_controlled_teacher/`, including preflight and preregistration documents, `results.csv`, `REPORT.md`, provenance, copied raw run artifacts, logs, plotting code, and PDF/300-dpi PNG figures. The final decision is to justify a minimum replication of C0/C1/C3 for seeds 123 and 456, corresponding to option A. No replication or subsequent phase has been launched; research-lead review is required.

## 19. Phase 2C Multiseed Teacher Utility Audit

Phase 2C completed the minimum replication authorized by Phase 2B. Seeds 123 and 456 use independently trained WikiText-2 Hybrid-lite S0 checkpoints rather than reusing seed 42. The three S0 SHA256 values are distinct, and configuration comparison shows that architecture, preprocessing, optimizer, 20-epoch budget, AMP, and validation selection are matched; only the training seed and experiment metadata differ.

For seeds 42/123/456, C0 test NLL is 4.270742/4.277520/4.280830, C1 T_a05 is 4.108241/4.130183/4.123673, and C3 T_a00 is 4.223972/4.243922/4.238226. The paired contrast `D=NLL_C3-NLL_C1` is 0.115731/0.113739/0.114553, giving mean 0.114674 and sample standard deviation 0.001001. T_a00 also produces positive KD gain for all seeds, with mean 0.040991. The alpha-0.5 versus alpha-0.0 difference is therefore robust to the tested student stochasticity, while negative teacher residual advantage still does not imply negative transfer.

Seed-matched validation residual advantage remains ordered for all seeds: T_a05 is +0.084898/+0.097926/+0.100412, whereas T_a00 is -0.266520/-0.253492/-0.251006. All formal optimization diagnostics remain below the preregistered mismatch thresholds. No arm is clip-saturated or numerically unstable.

The offline utility audit uses the same 512 WikiText-2 validation chunks for every student. T_a00 has negative mean gold-token utility globally and on the complete student-error partition, but the hardest student-loss quintile has positive mean utility for all seeds (+0.273620/+0.165971/+0.200348) and a positive-utility fraction above 0.54. Because top-1 rescue is nearly absent and no conditional intervention was trained, this is only a concentrated complementarity signal, not a mechanism claim.

The fixed-composition audit identifies a valid future two-condition intervention: alpha-only versus jointly trained teacher checkpoints, both used at alpha 0.5 with the same architecture and source branch identities. This would separate teacher quality from the composition change that remains confounded in Phase 2B/2C. It is the single recommended next experiment, but Phase 2D was not launched.

The final Phase 2C decisions are YES for robust D, YES for positive T_a00 transfer, YES for residual ordering, NO for global NLL sufficiency, PARTIAL for concentrated useful information, YES for an existing fixed-composition intervention, and option A as the next experiment. Complete artifacts are under `research/experiments/phase2c_multiseed_teacher_utility/`, including `REPORT.md`, result tables, plans, provenance, utilities, plots, and compact raw summaries.

## 20. Phase 2D Fixed-Composition Teacher Quality Intervention

Phase 2D was preregistered to compare the jointly trained WikiText-2 teacher J against the alpha-only teacher A while fixing architecture, source branch identities, fusion semantics, and effective alpha at 0.5. The teacher-integrity gate passed: checkpoint and source hashes match, state keys and shapes are compatible, and exactly 179 of 191 tensors differ because joint training updated branch parameters.

On the common 512-chunk validation subset, Teacher J/A fused NLL is 4.300644/6.376220. Transformer branch NLL is 4.445648/6.540640 and Grassmann branch NLL is 4.652064/6.696787. Joint training therefore improved both branches. Teacher J has slightly higher branch JSD and lower top-1 agreement, but its fusion gain over the better branch is slightly smaller; the dominant observed change is branch quality rather than a larger fusion-gain margin.

Teacher A has negative global gold-token utility for all three S0 students but positive hardest-quintile utility of 0.305649/0.183726/0.229669. This remains observational. Full-parameter KD-gradient ratios `G_A/G_J` are 1.102792/1.148522/1.093056, so no `KD_SCALE_MISMATCH` condition was triggered and no D2-scale arm was created.

The original one-epoch, 2,000-line Teacher-A smoke failed with clipping fraction 1.00 and AMP overflow/non-finite fractions 0.16. The research lead explicitly authorized a matched full-data one-epoch J/A stability amendment because the inherited short Teacher-J smoke showed the same 25-step behavior despite stable full-data training. The amended gate passed: J/A clipping was 0.234694/0.275510 and overflow/non-finite was 0.013605 for both. This amendment changed only the observation window, not the objective, endpoint matrix, or interpretation logic.

Formal Teacher-A test NLL for seeds 42/123/456 is 4.293136/4.312882/4.306036. Relative to reused matched C0, `Gain_A` is -0.022394/-0.035362/-0.025206, so Teacher A harms every student. Reused Teacher-J `Gain_J` is +0.162501/+0.147337/+0.157157. The primary contrast `Q=NLL_A-NLL_J` is +0.184895/+0.182699/+0.182364, with mean 0.183319 and sample standard deviation 0.001375. The fixed-composition teacher-state effect is therefore strong and consistent across the three independent S0 checkpoints.

The final decisions are YES for a fixed-composition teacher-state effect, NO for positive Teacher-A transfer, YES for cross-seed robustness, NO for material gradient-scale confounding, and NO for global teacher NLL as a sufficient general explanation. Joint training clearly improves both branches, while complementarity metrics are mixed and fusion gain decreases slightly. The single selected next experiment is Transformer-only versus Grassmann-only versus fused-teacher KD; it was not launched. Compact artifacts are under `research/experiments/phase2d_fixed_composition_teacher_quality/`. No manuscript, dataset, historical checkpoint, or KD objective was modified.

## 21. Phase 2E Teacher Branch Ablation

Phase 2E is complete. It reused the Phase 2C C0, fused alpha-0.5, and Grassmann-only alpha-0.0 endpoints and trained only the missing Transformer-only alpha-1.0 condition for independent S0 seeds 42, 123, and 456. The single fixed teacher checkpoint SHA256 is `a5ad284f6895c0b4ed516c855c1b149e6636b945173aa2e842b8132299582694`; all S0 hashes and formal configurations were revalidated by the collector.

The common 512-chunk teacher NLL values are 4.300646 for fused, 4.445647 for Transformer-only, and 4.652064 for Grassmann-only. Mean residual advantage over the three S0 students is +0.074515, -0.070486, and -0.276903. T/F full-parameter KD-gradient ratios are 1.3164, 1.2788, and 1.2324, all within the preregistered `[0.5, 2.0]` range.

The reduced Transformer smoke reproduced the short-window artifact. The full-data gate retained Transformer-only clip fraction 1.0 but had finite gradients and overflow/non-finite fractions 0.013605. Before any formal endpoint existed, the research lead explicitly authorized unchanged formal training with permanent `CLIP_SATURATION_WARNING`. No alpha, lambda, AMP, clip value, budget, data, S0, or evaluation rule changed.

Transformer-only test NLL is 4.127957/4.152294/4.144237. `Gain_F` is 0.162501/0.147337/0.157157, `Gain_T` is 0.142785/0.125226/0.136593, and `Gain_G` is 0.046771/0.033598/0.042604. Thus fused > Transformer-only > Grassmann-only > C0 for all seeds.

The primary `C_FT=NLL_T-NLL_F` contrast is 0.019716/0.022110/0.020564, mean 0.020797 and sample SD 0.001214, with 3/3 positive signs. This narrowly clears the frozen 0.02 practical threshold. `C_FG` has mean 0.114674 and `C_TG` mean 0.093877, again with 3/3 positive signs. Transformer-only and Grassmann-only each provide positive transfer, but fused provides the best transfer.

`CLIP_SATURATION_WARNING` limits interpretation: all Transformer-only arms clipped every step in epoch 1, although clipping declined to 0.0034 later and overflow/non-finite fractions never exceeded 0.013605. The endpoint is numerically valid, but early optimization constraints may partly contribute to the small fused-versus-Transformer gap.

The bounded conclusion is that adding the Grassmann branch yields a small, stable extra KD gain over Transformer-only supervision under this matched WikiText-2 protocol. Phase 2E does not establish that Grassmann geometry uniquely causes the gain. The single highest-information follow-up is a homogeneous Transformer+Transformer ensemble teacher control against the Transformer+Grassmann teacher. It is selected but not launched. Complete artifacts are under `research/experiments/phase2e_teacher_branch_ablation/`; Phase 2E now stops.

## 22. Phase 2F Homogeneous Ensemble Control

Phase 2F completed the selected Transformer+Transformer control and stopped. T1 is the existing validation-selected WT2 Transformer. T2 was independently trained with seed 123 under the matching 20-epoch protocol and selected epoch 12, obtaining validation/test NLL 5.230674/5.288460. The new loader supports explicit `teacher_type=tg|tt` while preserving the historical TG default and state keys; seven remote compatibility tests passed before smoke and formal training.

The jointly trained TT teacher selected epoch 10. At forced alpha 0.5 on the common full validation set, TT/TG NLL is 4.293589/4.303697. TT has 35,340,801 parameters versus 37,749,633 for TG, so the teachers are not parameter-matched and TT is 6.38% smaller. TT is nevertheless better by 0.010108 validation NLL. TG shows greater branch diversity and fusion gain, but those diagnostic advantages do not produce a better downstream endpoint.

Using the frozen Phase 2C S0 checkpoints and the identical token-mean KD protocol, TT test NLL for seeds 42/123/456 is 4.094289/4.118854/4.111574. Reused TG test NLL is 4.108241/4.130183/4.123673. `Gain_TG` has mean 0.155665 and sample SD 0.007692; `Gain_TT` has mean 0.168125 and sample SD 0.008948. The primary `H=Gain_TG-Gain_TT=NLL_TT-NLL_TG` values are -0.013952/-0.011329/-0.012099, with mean -0.012460, sample SD 0.001348, and 3/3 negative signs. Both ensembles transfer positively, but TT wins every paired comparison.

All TT runs completed without NaNs. Formal epoch-1 clipping is 0.8435--0.9286 and falls to 0.0034; maximum overflow/non-finite fractions remain 0.013605. TT is more strongly clipped than TG but not saturated, so clipping does not provide an evident explanation for its advantage. Teacher quality is a material, direction-aligned confound, and the authorized TT alpha learning rate `1e-2` differs from the historical TG value `5e-3`. The parameter difference is also disclosed, although TT is smaller rather than larger.

The Grassmann-specific KD mechanism claim is contradicted by this control and must be removed. Phase 2E is now interpreted as evidence for beneficial ensemble supervision, not unique Grassmann signal. Phase 2G is not necessary to preserve the rejected claim and was not launched. Complete artifacts are under `research/experiments/phase2f_homogeneous_ensemble_control/`, including `REPORT.md`, `GPT_HANDOFF.md`, exact CSV/JSON results, provenance, and PDF/300-dpi PNG figures.

## 23. Phase 2G TinyStories Negative-Transfer Confirmation

Phase 2G was subsequently authorized as an independent final confirmation of the TinyStories negative-transfer boundary, not as an attempt to preserve the rejected Grassmann-specific claim. Seed 42 reuses the historical matched pair. Seeds 123 and 456 use independently trained 20-epoch S0 checkpoints, followed by matched WS+CE and frozen token-mean WS+KD continuations with `lambda=5` and `T=2`. The three S0 checkpoint hashes are distinct.

For seeds 42/123/456, WS+CE test NLL is 1.581269/1.581807/1.583515 and WS+KD test NLL is 1.688747/1.689246/1.690255. `Delta_KD=NLL_WS_CE-NLL_WS_KD` is -0.107479/-0.107439/-0.106740, with mean -0.107219, sample SD 0.000415, and 3/3 negative signs. Under the preregistered rule, the TinyStories negative-transfer boundary is `CONFIRMED`.

The frozen teacher test NLL is 1.603229. Teacher residual advantage is negative for all three seeds, so the teacher is worse than the matched CE endpoint by 0.019714--0.021960 NLL. This is consistent with the teacher-quality boundary hypothesis but does not establish a causal mechanism.

All endpoints and gradient records are finite. KD mean gradient norm is 0.974--0.981 and mean clipping fraction is 0.237--0.283, both higher than CE; maximum overflow/non-finite fraction is 0.000611. This optimization difference must be disclosed, but the runs are neither clip-saturated nor numerically failed. The incomplete GPU-3 seed-456 attempt was excluded because of a persistent software power cap; after explicit approval, an otherwise identical from-scratch GPU-1 retry completed and is the accepted endpoint.

The allowed claim is limited to stable negative transfer for the frozen TinyStories teacher, warm start, `lambda=5`, and `T=2` protocol. It is not a universal KD claim and does not restore any Grassmann-specific mechanism claim. Complete artifacts are under `research/experiments/phase2g_tinystories_negative_transfer/`. The project state is `experiments_frozen_manuscript_rewrite_next`; no Phase 2H may be launched.

## 24. Final Evidence Package and Manuscript Reconstruction State

No experiment was launched after Phase 2G. The project evidence was consolidated under `research/final_evidence/`, and that directory is now the authoritative bridge from experiments to the future manuscript. `FINAL_EVIDENCE_LEDGER.md` maps every major claim to its experiment, dataset, seed count, exact effect, status, confound, and main-paper permission. `PAPER_CLAIMS.md` freezes four conservative central claims. `OLD_MANUSCRIPT_AUDIT.md` classifies the current CAC source by line range without modifying it, and `MANUSCRIPT_BLUEPRINT.md` specifies the ten-section reconstruction.

The confirmatory cross-domain table contains only matched token-mean three-seed results: PTB gain `+0.144937±0.003938` NLL, WikiText-2 `+0.155665±0.007692`, and TinyStories `-0.107219±0.000415`, all with 3/3 sign consistency. CodeParrot common-5k is separated as a one-seed legacy batch-normalized observation.

The final narrative is teacher distillability under warm-start KD. Phase 2D supports a fixed-composition teacher-state effect (`Q=+0.183319±0.001375`). Phase 2C contradicts the universal rule that a globally worse teacher must transfer negatively, because the alpha-0.0 condition gives mean gain `+0.040991` with 3/3 positive signs. Phase 2F contradicts a Grassmann-specific or heterogeneity-superiority claim because TT beats TG by mean `0.012460` NLL for all three students.

The main manuscript must not claim Plücker-geometry causality, Grassmann-specific KD advantage, heterogeneous-ensemble superiority, strong branch-JSD mechanism, successful CRBD, warm-start necessity, or unique Grassmann efficiency. The PTB efficiency result remains a secondary, hardware-specific operating-point analysis. The old file `论文投稿/cac/conference_101719.tex` is unchanged and must be preserved as an archival source.

Project status is now `manuscript_reconstruction`. The next action requires explicit authorization to write a new full manuscript from the final evidence package. Experiments remain frozen, and Phase 2H is prohibited.
