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
