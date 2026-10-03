# Phase 4A — COMPLETE / STOPPED

`DECISION = MODERN_REPLICATION_WORTHWHILE`

This is a recommendation for research-lead review, NOT authorization for
additional training. Only seed 42 was run. Do not launch seeds123/456,
Phase 3B, Phase 2H, new methods, or manuscript reconstruction automatically.
Phase 3A remains terminal `STOP_NEW_METHOD_DIAGNOSTIC_FAILED`.

## Read first

REPORT.md contains the complete interpretation and limitations.
results_seed42.csv and decision.json contain the primary endpoints and gate.
optimization_audit.csv contains all ten formal optimization records.
selected_state_manifest.csv records validation-selected states and hashes.
completed_run_audit.json reports PASS for budgets, selection, finite telemetry,
data disjointness, source-state hashes and paired gains. Raw summaries,
compressed step telemetry and ordered document manifests are in raw/.
experiment_plan.md, model_and_data_audit.md and provenance.md define the
frozen protocol, actual data source and execution history.

## Primary results

Gain = NLL_CE - NLL_KD. Positive means KD helps the matched CE continuation.

| Dataset | Lambda | Validation gain | Test gain | Clear effect |
|---|---:|---:|---:|---|
| TinyStories | 1 | +0.004337605834 | +0.004948632001 | no |
| TinyStories | 5 | -0.005984952475 | -0.004797389422 | no |
| FineWeb-Edu | 1 | -0.015706368649 | -0.015718598299 | no |
| FineWeb-Edu | 5 | -0.046727056123 | -0.046403663516 | yes |

The qualifying effect is NEGATIVE FineWeb-Edu transfer at lambda=5, not
beneficial modern KD. All four validation/test directions agree. TinyStories
does not clearly reproduce its old strong negative boundary: the modern
effects are small and opposite in sign across strengths. No single-seed
sample SD or 3/3 robustness claim is permitted.

The gate was frozen before endpoints: abs(test gain)>=0.02, matching
validation direction and no major optimization failure. Clipping alone was
defined as a disclosed constraint, not a numeric failure. The FineWeb-Edu
negative effect passes that rule but remains optimization-constrained.

## Non-negotiable limitations

`CLIP_SATURATION_WARNING` applies permanently to BOTH lambda=5 arms:
clip_fraction=1.0 throughout formal training. KD1 clip fractions are 0.479934
(TinyStories) and 0.850123 (FineWeb-Edu). All runs remained finite with zero
observed overflow/nonfinite events. Do not attribute harm causally to clipping,
ignore the constraint, or tune lambda/clip thresholds to repair the result.

Teacher validation residual advantage over S0 is +0.173856356061 on TinyStories
and +0.219626011745 on FineWeb-Edu. Initial full-parameter KD/CE gradient
ratios at lambda1/lambda5 are 5.594400/27.972002 and 2.150561/10.752805.
Teacher likelihood superiority does not imply beneficial transfer here.
FineWeb S0 selected step256 (2,097,152 targets) after a complete 10M-token
preparation run; all other selected states are step1221. FineWeb CE itself is
slightly worse than S0. Do not conflate CE-relative KD gain with gain over S0.

Both SmolLM2 model cards list FineWeb-Edu in pretraining; exact overlap is
unknown. Size and pretraining budgets differ (135M/2T vs 360M/4T). This is not
a guaranteed unseen-corpus or pure architecture causal experiment. Subset,
sequence length256, short adaptation budget and single student seed also
limit generalization. No Grassmann/Plucker/heterogeneity/method claim follows.

## Frozen execution and reproducibility

Protocol commit 58f4cd7897858d84d577c3c048cc3fd83294d50c preceded all formal
endpoint training. FineWeb input-hash/transport lock commit
55ca7801f146b2ea05cf95a0664e048cff3b29a7 preceded its endpoint training.
Each dataset completed teacher preparation, independent student preparation,
and exact-S0 CE/KD1/KD5 continuation: ten runs, exactly 100M predicted targets.
Official base models, tokenizer, dataset manifests, token/order hashes,
optimizer, temperature, budgets and selection are fixed. No test-directed
choice was made. Existing datasets/, checkpoints/, manuscript and
research/final_evidence/ remain unchanged.

All heavy computation took place on 10.42.0.197 via exec_grassmann.sh.
Use isolated PYTHONPATH=outputs/phase4a_modern/python_deps for reproduction
(Transformers4.46.3/tokenizers0.20.3), not a shared environment replacement.
Large states/tokens stay in outputs/phase4a_modern and are NOT in Git.
Committed document manifests preserve IDs/order/text hashes without text.
Plots can be regenerated from compact committed results/telemetry only.

## Review-only next step

Recommend reviewing an unchanged FineWeb-Edu CE/KD1/KD5 replication with
independently prepared student seeds123/456 and the same frozen seed42
adapted teacher. Do not run it until explicit research-lead authorization.
Do not modify research/final_evidence/ or the manuscript. STOP.
