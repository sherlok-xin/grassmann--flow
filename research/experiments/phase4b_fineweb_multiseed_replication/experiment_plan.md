# Phase 4B — Frozen FineWeb-Edu independent-student replication

Authorized parent commit: e4b9878f50afb21cc8d05a2e3bfa149897b075f7.
Only FineWeb-Edu; only new student seeds123/456. Reuse seed42 without retraining.
Same frozen Phase4A adapted 360M teacher, SHA256
c7411708b61e6524ef05bd24e9e618f19fcc119e9a3c0664c5f4f169dbf16928.
No teacher training, TinyStories replication, coefficient sweep, clip change,
new method, manuscript edit or final_evidence edit is authorized.

## Exact protocol and independence

Use the Phase4A official base SmolLM2-135M snapshot and existing immutable
FineWeb-Edu token streams. Data revision, tokenizer, packing and split hashes
are identical to data_manifest_fineweb.json in Phase4A. Do not retokenize,
download a new corpus, alter datasets/ or checkpoints/, or substitute models.

Each new student prepares S0 independently from the same official pretrained
base, with its own torch/CUDA seed and preparation chunk permutation seed123
or456. This is independent target-adaptation initialization, not random
pretraining from scratch. All continuations retain the Phase4A fixed chunk
permutation seed4242, identical across arms and students. Torch/CUDA RNG
uses the corresponding student seed in every arm. This changes only the
authorized independent student seed; it does not change the data subset.

Each S0 preparation and each CE/KD1/KD5 continuation processes exactly 10M
predicted targets in1221 updates (eight new runs,80M targets). From the exact
validation-selected S0 run CE, CE+1*T²KL and CE+5*T²KL, with T=2 and forward
KL(teacher||student), token_mean normalization. Teacher is frozen in eval mode.
Same FP32 weights/BF16 autocast/FP32 chunked losses/noGradScaler, AdamW
lr5e-5, weight_decay.01, betas(.9,.95), eps1e-8,5%warmup+cosine,clip1,
global batch32/micro4/accumulation8,seq256. The partial final batch is masked
to the exact budget. Validation schedule256/512/768/1024/1221, minimum full
validation NLL among trained steps, ties earlier. Step0 is NOT eligible for
the primary endpoint. Reset optimizer at continuation as in Phase4A.

All three continuation arms for each student must finish and their selected
hashes must be verified before final test access. No test-directed choices.
Preparation and continuation may revisit target-data chunks, as in Phase4A.
Freeze the protocol commit before formal training; run numerical/adapter tests,
hash preflight and disposable fullbatch smoke first. Model computation remote
only on10.42.0.197 with the existing isolated Transformers4.46.3/tokenizers0.20.3.
GPU assignment changes execution placement only, not science settings.

## Primary quantities and decision

Gain1=NLL_CE-NLL_KD1; Gain5=NLL_CE-NLL_KD5. Report seed42/123/456 individually,
mean, sample SD (ddof1) and sign counts. Also report teacher validation/test
residual over each S0, NLL_CE-NLL_S0 (positive means further CE harms),
validation/test sign agreement, clipping, mean/max preclip norms, nonfinite
and overflow fractions, selected steps and hashes. No pilot magnitude cutoff
is added to the user's three-seed sign decision. Report raw magnitudes so
small effects are not described as large. First-batch full-parameter CE/KD
gradient probes remain observational, with no parameter update or test access.

If all Gain5<0 and all validation/test directions agree, record
FINEWEB_L5_NEGATIVE_REPLICATED. If additionally all Gain1<0, record
FINEWEB_L1_L5_NEGATIVE_REPLICATED. Otherwise, when lambda5 is consistent but
lambda1 is mixed/zero/positive, record FINEWEB_STRENGTH_DEPENDENT. A materially
sign-changing lambda5 or failed direction agreement does not confirm the
negative boundary: MODERN_NEGATIVE_TRANSFER_NOT_ROBUST. Report exactly-zero
effects as zero, not negative. Lambda1 validation directions must also be
shown even though its decision rule explicitly requires test signs.

Any numerical loss/gradient/parameter failure, OOM, incomplete run or broken
provenance stops the affected formal protocol: no tuning or silent retry.
Keep CLIP_SATURATION_WARNING for saturated arms, with exact fractions (1.0
is full saturation). Consistent with Phase4A, >=.95 is flagged as near/full
saturation; distinguish near saturation from every-step clipping in prose.
Clipping itself is a disclosed optimization constraint, not proof of numeric
failure. Observed BF16 overflow fields are not FP16GradScaler skip rates.

## Secondary best_including_S0 — no training

After trained validation selection, choose between the original selected S0
and the primary selected trained checkpoint using ONLY validation NLL.
The S0 candidate is continuation step0 and wins ties. Its checkpoint retains
the original preparation selection (do not select a new S0). Store the choice
and candidate hashes before any new test evaluation, in selection_with_s0.json.
Evaluate only the already-fixed states; reuse existing seed42 test metrics
without rerunning any seed42 model. Report all nine arm-level choices in
best_including_s0.csv. Compute diagnostic CE-minus-KD effects under these
choices, clearly separated from primary gains. S0 winning most/all arms must
be stated as a regime where further CE itself offers little/no improvement.
This diagnostic never changes primary checkpoints, gains or decisions.

## Claims and STOP

If supported, claim only reproducible negative transfer under this frozen
SmolLM2-360M→135M warm-start protocol on the fixed FineWeb-Edu subset despite
better teacher held-out likelihood. Do not claim better teachers generally
harm, FineWeb causes harm, clipping causes harm, or all modern KD fails.
Pretraining overlap unknown, different pretraining budgets, short context,
finite budget, selection and clipping remain confounds. No Grassmann claim.

Write REPORT/GPT_HANDOFF, primary/diagnostic/optimization CSVs, provenance,
compact plots/raw summaries; update research state/handoffs/project docs;
commit/push compact artifacts only and STOP. Do not launch clipping controls
or another modern experiment automatically.
