# Phase 4B provenance

Explicit research-lead authorization follows parent
e4b9878f50afb21cc8d05a2e3bfa149897b075f7. FineWeb only, independent student
seeds123/456, same frozen adapted seed42 teacher and input token streams.
Seed42 is reused, not rerun. The legacy final_evidence and manuscript are
immutable. Existing user .gitignore changes are preserved and excluded.

Phase4B train.py is an exact copy of the frozen Phase4A loop with only the
listed training_adapter_changes.json substitutions: seeds, output paths,
shared Phase4A model/core references, and restricted CLI. A unit test verifies
the complete transformed source and immutable original source/core SHA256.
Preparation permutation varies with student seed; every continuation retains
the original4242 permutation. No training math, optimizer or selection change.
All large new files remain under ignored outputs/phase4b_fineweb.

Preflight audits immutable data/teacher/frozen-evidence hashes. Tests and a
disposable train-data fullbatch BF16 KD smoke precede formal training. Smoke
does not save a state or access test. Protocol is committed/pushed before
formal endpoint runs. All heavy work is on10.42.0.197 via exec_grassmann.sh;
editing/Git remain local NFS operations. The isolated dependency path from
Phase4A is reused without package installation or framework surgery.

Formal protocol commit:1a525c379d7b72e809847dfed6ef1bf71fd0fe50, pushed before
the launcher started. Owned remote launcher PID587220; preparation child
PIDs587237/587238 on GPU0/1. GPU0/1/2/3 were read-only checked idle immediately
before launch. Subsequent exact child PID/arm/GPU identities are appended to
outputs/phase4b_fineweb/process_registry.txt by the launcher. Other-user tasks
are never terminated. Analysis/collection/plot helpers may be added while
training runs, but cannot alter the locked train/core/source behavior.

## Training-completion snapshot, before final test results

All eight new training runs finished the exact10M-target budget. Before any
new test forward, finish.py saved selection_with_s0.json using ONLY the
original S0 and trained validation minima. All nine diagnostic choices are
S0; these do not replace primary trained-step selections. New S0 selected
steps are123→256 and456→1221; CE123/CE456 select1221/256 respectively,
and all four new KD arms select1221. Both new KD5 clipping fractions are1.0,
so CLIP_SATURATION_WARNING remains permanent. Test evaluation is pending;
no three-seed transfer decision has been made from intermediate results.

## Final completion

Final evaluation completed successfully on2026-10-04 after all eight complete
summaries/states were verified. The launcher emitted
PHASE4B_TRAINING_AND_TEST_COMPLETE. All selected student states reproduced
their stored full validation NLL within1e-5 before final test measurement.
Seed42 and teacher endpoints were reused without new model evaluation.

Primary Gain1 is -0.015718598299/-0.015470071017/-0.009462713488;
Gain5 is -0.046403663516/-0.046078763438/-0.039289819652. Both3/3 negative,
all validation/test directions agree. Mean/sample SD are
-0.013550460935/0.003542273400 and -0.043924082202/0.004016675497.
Decision FINEWEB_L1_L5_NEGATIVE_REPLICATED; lambda5-only flag also satisfied.
All9 diagnostic choices return S0, giving zero diagnostic paired gains.
The primary selected checkpoints/metrics were never replaced.

CE456_GRADIENT_SPIKE_WARNING records a finite norm86.482155 at step711,
the only CE456 norm above2. It is clipped and occurs after its selected
step256 state. Cause UNKNOWN; a later selection/trajectory effect is not
isolated. No result was excluded, gate amended, parameter changed or run
retried. This telemetry QA is not a new diagnostic method or training control.

The read-only final audit is PASS on data/teacher hashes, eight exact budgets,
three distinct S0/order hashes, matching continuation order, selected-state
hashes, trained validation minima, finite telemetry, paired gains and secondary
selection. Frozen manuscript SHA256 remains
8b8e3fcc3f04b596058cd4b63ddfe38b6eab91f338c7d4f12db4b602c9982ec8;
final_evidence aggregate remains
3d172e634db611edad0d566e74ea0f826ed8cdae69922120baa94ebf9e6e2b96.
The Phase4A source/artifacts and locked Phase4B train/adapter source remain
unchanged from the parent/protocol commits. Raw summaries, compressed telemetry,
state/order manifests, diagnostic choices and plots are compact; weights,
tokens, corpus text and environment artifacts are excluded from Git.

Two PDF/300-dpi PNG figure pairs were generated from committed-size CSVs and
visually inspected; a long ylabel was shortened to prevent clipping. No
scientific result was changed by formatting. Project state is
experiments_frozen_phase4b_complete. Phase4B COMPLETE / STOPPED; no follow-up
clipping control, modern experiment, TinyStories replication or manuscript
change is launched or authorized.
