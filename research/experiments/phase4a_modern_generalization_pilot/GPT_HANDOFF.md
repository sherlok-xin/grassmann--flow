# Phase 4A — IN PROGRESS

This new phase is authorized in the 2026-10-03 user attachment. Phase 3A is
terminal STOP_NEW_METHOD_DIAGNOSTIC_FAILED. No new method, Phase 3B, Phase 2H,
manuscript edit, final_evidence edit or seed123/456 training is permitted.

Protocol commit 58f4cd7897858d84d577c3c048cc3fd83294d50c was pushed before
endpoint training. Read experiment_plan.md, model_and_data_audit.md and
provenance.md first. Model revisions, data tokenization, target budgets,
optimizer, losses, checkpoint selection and replication criterion are frozen.
All models compute remotely on 10.42.0.197 using exec_grassmann.sh. The isolated
PYTHONPATH is outputs/phase4a_modern/python_deps (Transformers4.46.3,
tokenizers0.20.3). The existing environment remains unchanged.

Stage 0 and real train-data accumulated fullbatch smoke passed. TinyStories
teacher and S0 preparation have completed the exact 10M-token budgets with
finite optimization. Selected validation NLL: teacher 1.4049174157, S0
1.5787737718; teacher residual advantage +0.1738563561. Initial unweighted
KD/CE gradient norm ratio is 5.5944004116 (lambda5 weighted ratio 27.9720020580).

TinyStories CE/KD1/KD5 continuations and selected test evaluation are COMPLETE.
Their common S0 SHA256 is
d8c9fe54b3c22a4996b5e2f095a304b70901fd4d88ac61599f80ceecafe2c48a;
frozen teacher SHA256 is
a9d292a2a0559ea9d102a5a3318297092ffcbdfa8363df3bbaba4989c06dab1f.
TinyStories final test CE/KD1/KD5 NLL: 1.5794294928 / 1.5744808608 /
1.5842268822. Gains +0.0049486320 / -0.0047973894; both below the 0.02
clear-effect threshold. Dataset decision STOP_MODERN_REPLICATION. KD5
clip_fraction=1.0, CLIP_SATURATION_WARNING. All optimization remained finite.
The modern TinyStories pilot does not establish clear negative transfer.

The launcher is scripts/run_dataset.sh; raw output is
outputs/phase4a_modern/runs/tinystories. Poll exact PIDs in continuation.pids;
do not relaunch or overwrite incomplete runs. Endpoint test evaluation runs
only after all five validation-selected summaries exist. A final_result.json
and successful launcher exit establish clean Stage 1 completion.

Stage 1 exited successfully. FineWeb-Edu sample-10BT compact deterministic
subset (20M/1M/1M encoded tokens) is now materialized; hashes are recorded in
data_manifest_fineweb.json. Direct pinned first-shard streaming bypasses a
mirror listing/pagination error; document order/split rules are unchanged.
Next authorized steps: lock these hashes, and run the same five seed42 runs
from official base checkpoints (not TinyStories-adapted states). Finish both
domains, execute scripts/collect.py and figures/gen_fig_pilot.py, inspect figures,
write final REPORT/GPT_HANDOFF, update project state/memory/journal, verify the
frozen evidence/manuscript unchanged, and commit/push compact artifacts.

Clipping frequency must be reported. Do not tune lambda or clip values. No
test-directed selection, budget change or automatic replication. Any invalid
formal run must be reported, not silently retried with different settings.
