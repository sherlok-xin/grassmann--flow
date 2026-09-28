# Phase 2C Multiseed Teacher Utility Plan

## Governing evidence and scope

Phase 2B `REPORT.md` is the governing evidence document. Phase 2C tests whether the alpha-0.5 versus alpha-0.0 KD contrast is robust to independently trained student initializations and audits conditional teacher utility. It does not introduce a new loss, architecture, routing rule, teacher, alpha, lambda, or manuscript claim.

## S0 audit

Repository-wide config inspection found no complete WikiText-2 Hybrid-lite S0 checkpoints for seeds 123 or 456. The only valid canonical run is seed 42 at `outputs/hybrid_experiments/20260515_004119_wt2_hybrid_lite_baseline_224x56_l6_e20`, checkpoint SHA256 `9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef`.

Seeds 123 and 456 will therefore be trained independently from scratch with the canonical recipe: Hybrid-lite model dimension 224, six layers, eight Transformer heads, Grassmann reduced dimension 56, windows 1/2/4, dropout 0.1, late-k 1, learnable scalar alpha initialized at 0.5, batch size 32, 20 epochs, AdamW at learning rate 2e-4 and weight decay 0.01, AMP, gradient clipping at 1.0, and validation-NLL checkpoint selection. Dataset preprocessing remains fixed at WikiText-2, sequence length 256, full native splits, encoding batch 200,000 characters, and split seed 42. Only the model/training seed changes.

The canonical baseline script is `train_hybrid_lite_latefusion_baseline_v2.py`, SHA256 `57d4e42920fa51f318e2c03aafac4908e3067bd73fdf63fae46b60b2096ed6e1`. A one-epoch reduced-data smoke must complete for each seed before formal S0 training.

## Frozen replication

The frozen teacher run is `outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint`, checkpoint SHA256 `a5ad284f6895c0b4ed516c855c1b149e6636b945173aa2e842b8132299582694`. The same teacher is used for all student seeds.

For each of seeds 123 and 456, the only continuation arms are C0 WS+CE, C1 WS+KD with neutral teacher name T_a05 and alpha 0.5, and C3 WS+KD with neutral teacher name T_a00 and alpha 0.0. Every arm within a seed must start from the exact same seed-specific S0 checkpoint hash. Continuation uses the Phase 2B protocol: 10 epochs, batch size 32, learning rate 1e-4, weight decay 0.01, 5% warmup, cosine schedule, sequence length 256, temperature 2, token-mean KL, lambda 5 for KD and lambda 0 for C0, AMP, gradient clipping at 1.0, and validation-NLL checkpoint selection. Alpha and lambda are frozen before new-seed outcomes are observed.

Seed-42 C0/C1/C3 results are reused without rerunning. Alpha 0.3 is not trained in Phase 2C.

## Preregistered replication interpretation

For seed s, `Delta_KD_a05(s) = NLL_C0(s) - NLL_C1(s)`, `Delta_KD_a00(s) = NLL_C0(s) - NLL_C3(s)`, and `D(s) = NLL_C3(s) - NLL_C1(s)`. The primary hypothesis is that D remains positive for independently trained students.

Decision 1 is YES only if D is positive for all three seeds. It is PARTIAL if D is positive for two of three seeds, or if all signs are positive but an integrity flag makes one seed qualitatively incomparable. It is NO otherwise. Decision 2 is YES only if `Delta_KD_a00` is positive for all seeds, MIXED if signs differ, and NO if no seed has positive transfer. Decision 3 is YES only if seed-matched `Delta_teacher_a05 > Delta_teacher_a00` for all seeds, PARTIAL if this holds for two seeds or an integrity issue prevents one comparison, and NO otherwise.

No p-value is a primary result. For `Delta_KD_a05`, `Delta_KD_a00`, and D, the report will include every seed value, mean, sample standard deviation, minimum, maximum, and sign consistency.

## Optimization integrity

The same epoch-1 and initial-logit diagnostics used in Phase 2B will be recorded. A new arm is flagged for qualitative optimization mismatch if it is clip-saturated at or above 0.95, has AMP overflow/non-finite fraction at or above 0.05, fails checkpoint integrity, or has an initial KD logit-gradient norm more than twice or less than half the corresponding seed-42 value. Flags are reported rather than averaged away; no lambda retuning is allowed.

## Offline conditional utility

Offline analysis uses the same deterministic WikiText-2 validation subset selected with seed 20260920. For each independently trained S0 and alpha 0.5, 0.3, and 0.0, token utility is `u_i = log p_teacher(y_i) - log p_student(y_i)`. Outputs include global utility summaries, utility quantiles and masses, student-correct/incorrect partitions, student-loss quintiles, rescue and harm rates, teacher-student KL, and initial CE-KD gradient compatibility.

Global teacher NLL is considered insufficient as a complete description if a globally worse teacher still has positive mean utility and a positive-utility majority on student-error tokens, with nontrivial rescue behavior. Such an observation is recorded only as conditional utility or a complementarity signal, not as a causal mechanism.

## Fixed-composition audit

Existing repository artifacts will be searched for teacher checkpoints that vary training quality while preserving architecture, source branches, and alpha semantics. No teacher will be trained or artificially corrupted. A valid Phase 2D candidate must change teacher quality without simultaneously changing Transformer/Grassmann composition.

## Stop rule

After producing the required Phase 2C report, tables, raw artifacts, logs, plots, provenance, and fixed-composition audit, execution stops. Phase 2D is not launched automatically.
