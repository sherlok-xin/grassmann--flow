# Phase 4B — RUNNING / NOT YET COMPLETE

Read experiment_plan.md. Only FineWeb student seeds123/456 are authorized.
Reuse Phase4A frozen teacher, data and seed42 endpoints. Eight new10M-target
runs: two independent S0 preparations plus six continuations. No TinyStories,
teacher training, lambda/clip tuning, new method or manuscript/evidence edit.

Primary trained-step selection remains unchanged. best_including_S0 is a
separate validation-only diagnostic and cannot replace the primary endpoint.
Never claim confirmation until all complete summaries, state hashes and
final test endpoints pass integrity checks. No automatic further experiment.

Protocol commit 1a525c379d7b72e809847dfed6ef1bf71fd0fe50 was pushed before
formal runs. The remote launcher PID is587220. Initial S0 PIDs are587237
(seed123, GPU0) and587238 (seed456, GPU1). Exact child PID/seed/arm/GPU
registry is outputs/phase4b_fineweb/process_registry.txt; never match a monitor
by its own seed-bearing command. Raw runs/logs are outputs/phase4b_fineweb.
Do not relaunch, overwrite, or infer endpoints from partial validation logs.

Launcher runs both S0 preparations, then unchanged initialization audits,
then three paired continuations per seed. All eight runs must complete before
finish.py freezes all nine validation-only best_including_S0 choices and
evaluates only the selected new-seed states on test. Seed42 metrics are reused
without new model evaluation. After launch exits successfully, run collect.py,
audit_completed.py and figures/gen_fig_replication.py remotely, inspect plots,
write REPORT/final handoff, update state/project docs, verify protected hashes,
commit/push compact artifacts and STOP. No subsequent training is authorized.
