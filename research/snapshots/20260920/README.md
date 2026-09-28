# Frozen Research-State Snapshot — 2026-09-20

This directory records the repository and runtime state immediately before Phase 2A analysis code was added. The dirty working tree was preserved as-is; no reset, cleanup, deletion, or remote push was performed.

`HEAD.txt`, `git_status.txt`, and `tracked.diff` record the Git state. `untracked_research_code_files.txt` lists untracked files relevant to the active research code and documentation. `active_files_sha256.txt` contains SHA256 hashes for active Python, shell, configuration, and research files. `local_environment.txt` and `remote_environment.txt` record the local and server-197 software/hardware environments. The canonical evaluation environment is the remote environment because the local PyTorch installation is not usable.

The four `dataset_manifest_*.json` files and the combined `dataset_manifests.json` record resolved server paths, native split fingerprints and record counts, historical preprocessing/token counts, tokenizer settings, sequence length, and hashes of lightweight dataset metadata. Multi-gigabyte Arrow payloads and checkpoint tensors were intentionally not hashed. The dataset manifests were generated after the source-state capture by the Phase 2A provenance script; they describe the same mounted datasets used by the frozen runs.

The active Git commit is `67efbc158ad823f7196f0696415f6e32b5e2e2fa`, but this commit does not contain most active research code. The snapshot manifests, rather than Git alone, are therefore required to identify the audited state.
