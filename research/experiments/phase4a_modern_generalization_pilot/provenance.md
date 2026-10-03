# Phase 4A provenance

User-authorized new external-validity phase; Phase 3A remains terminal.
Parent HEAD c7dd579a1e2b0f30586aacaad3cce0a40d194987.
Initial feasibility protocol commit: 3ad6ea6, pushed before real-model smoke.
Formal protocol commit will precede endpoint training.

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
