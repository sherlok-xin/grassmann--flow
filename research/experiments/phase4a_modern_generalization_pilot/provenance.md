# Phase 4A provenance

User-authorized new external-validity phase; Phase 3A remains terminal.
Parent HEAD c7dd579a1e2b0f30586aacaad3cce0a40d194987.
Initial feasibility protocol commit: 3ad6ea6, pushed before real-model smoke.
Formal protocol commit: 58f4cd7897858d84d577c3c048cc3fd83294d50c,
pushed before endpoint training. TinyStories preparation runs launched on
GPU 0 (teacher) and GPU 1 (S0); continuations use GPU 0/1/2 (CE/KD1/KD5).

All model computation runs on 10.42.0.197 through exec_grassmann.sh. Local
editing and official model downloads use the shared NFS mount. Remote
Hugging Face direct access reset connections, so exact official model files
were downloaded locally and verified by LFS hashes. No model substitution.

All new large files are ignored under outputs/phase4a_modern. No existing
datasets/ or checkpoints/ content is changed. TinyStories index creation
explicitly points to new output cache files. An initial empty preparation
attempt failed on a datasets keep_in_memory/cache-name incompatibility;
its empty output files were moved to tinystories_prepare_failed_1. No target
training occurred before this repair. Transient imports while pip was still
installing were retried only after installation finished. No framework source
was patched and the original environment remains intact.

Isolated wheel SHA256: transformers-4.46.3-py3-none-any.whl
a12ef6f52841fd190a3e5602145b542d03507222f2c64ebb7ee92e8788093aef;
tokenizers-0.20.3-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
f2b7cb962564785a83dafbba0144ecb7f579f1d57d8c406cdaa7f32fe32f18ad.
The earlier cp310 wheel downloaded by local pip is not used remotely.

Reproduction: download_models.py runs locally, install these two wheels with
remote pip --no-deps --target outputs/phase4a_modern/python_deps, then set
PYTHONPATH to that directory. Run remote unittest discovery and stage0.py,
prepare_data.py tinystories and smoke_fullbatch.py. The formal entry point is
bash scripts/run_dataset.sh tinystories. Use the same entry point for fineweb
only after Stage 1 completes and deterministic FineWeb tokens are materialized.
Model state files and token streams are not committed; hashes, compact
metrics and ordered document manifests are committed. Plot script reads only
these committed compact artifacts.

Frozen final-evidence aggregate SHA256 before work:
3d172e634db611edad0d566e74ea0f826ed8cdae69922120baa94ebf9e6e2b96.
Pre-existing .gitignore modification belongs to user; not staged by Phase 4A.
Manuscript unchanged. No Phase 3B, 2H, new loss, routing or mechanism work.

Reproducibility records include all model-file hashes, corpus revision/snapshot
hashes, document IDs/order, token hashes, training order hashes, S0/teacher/
selected state hashes, optimization telemetry and validation selection steps.
Test evaluator first verifies all five completed training summaries and the
same S0 and continuation-order hashes across CE/KD, then evaluates only the
fixed selected states. No test values influence model selection.

TinyStories training completion: all five runs consumed exactly 10,000,000
predicted targets. Selected continuation validation NLL was CE 1.5345059863,
KD1 1.5301683804, KD5 1.5404909387, all at step 1221. KD5 clipping fraction
was 1.0: `CLIP_SATURATION_WARNING`. This is a finite but optimization-constrained
arm; its endpoint cannot be interpreted independently of sustained clipping.
Lambda and clip value were not modified. CE/KD1 clipping fractions were
0.0049140049 / 0.4799344799. Test evaluation follows the fixed selection.

Stage 1 completed successfully before Stage 2 preparation. TinyStories final
test CE/KD1/KD5 NLL is 1.5794294928 / 1.5744808608 / 1.5842268822;
paired gains +0.0049486320 / -0.0047973894. Both validation/test directions
agree but neither reaches abs(gain)>=0.02. Its standalone replication gate
is STOP_MODERN_REPLICATION. FineWeb-Edu remains authorized as the second
pilot; no training setting or token budget is changed after these results.

Stage 2 data preparation: the mirror's repository listing returned absolute
main-domain pagination links and timed out before reading any document. Only
the verified owned preparation PID 578401 was terminated; no training or
other user's process was stopped. Its empty token outputs were moved to
fineweb_listing_failed_1. Direct streaming of the pinned official first
sample-10BT Parquet shard then succeeded (byte-range test returned HTTP206
and PAR1). This preserves the same official-config prefix and frozen document
split rule while bypassing listing infrastructure. The 20M/1M/1M subset and
20,727 ordered document records are frozen before FineWeb endpoint training.
No complete large corpus file is materialized or committed.

FineWeb input-hash/transport lock commit was
55ca7801f146b2ea05cf95a0664e048cff3b29a7, pushed before any FineWeb endpoint
training. All five FineWeb runs then completed exactly 10,000,000 predicted
targets each, with unchanged training/core code from the formal protocol.
The selected S0 is step256; the other four states are step1221. All test
evaluation occurred after all five validation-selected summaries existed.
Final test CE/KD1/KD5 NLL is 2.949900313167 / 2.965618911467 /
2.996303976684. Gains are -0.015718598299 / -0.046403663516, with matching
validation signs. Only lambda5 exceeds the preregistered absolute 0.02 gate.

FineWeb KD1/KD5 clip fractions are 0.8501228501 / 1.0. Both domains' KD5
arms permanently carry CLIP_SATURATION_WARNING. All ten formal runs have
zero observed overflow/nonfinite events and finite losses/gradient norms.
No optimization hyperparameter was modified to suppress clipping. The gate
remains valid under its preregistered clipping-warning rule, but the qualifying
negative result cannot isolate teacher information from optimizer constraints.

The final collector copied all ten summaries, compressed step telemetry,
two initialization probes, two final results and both ordered document
manifests into compact raw artifacts. selected_state_manifest.csv records
all selected hashes; results_seed42.csv and optimization_audit.csv preserve
unrounded numeric values. Generated PDF/PNG gain and validation plots were
visually inspected. No test model forward is used by the final read-only
audit script. completed_run_audit.json reports PASS on both datasets,
ten runs and 100,000,000 formal predicted targets, including data-ID/text
disjointness, exact budgets, selection, source-state/order hashes, finite
telemetry, paired gains and the unchanged frozen-evidence aggregate hash.

The model-download helper was subsequently pinned to the same already-used
official commit hashes rather than resolving moving main. This is a future
reproduction safeguard; it changes neither downloaded weights nor run inputs.
No token files, model states, dependency wheels or corpus text are committed.

Final decision: MODERN_REPLICATION_WORTHWHILE, based only on the negative
FineWeb lambda5 pilot. The modern TinyStories effects do not pass the gate.
No qualifying positive transfer was found; only one student preparation seed
was run. Recommend unchanged FineWeb independent-student replication only
for lead review. Phase 4A is COMPLETE and STOPPED. No seeds123/456 or new
training is launched. The manuscript and final_evidence remain immutable.
