# Phase4C provenance

Parent GitHub commit: 6506c79e560efd2bbdacce7b87e7164b21d56180.
Client audit date: 2026-10-04. Edits and Git are local NFS; heavy model
work only on10.42.0.197 via agent-tools/exec_grassmann.sh. Same isolated
Transformers4.46.3/tokenizers0.20.3 packages, NVIDIA Torch2.6.0a0, RTX3090.
No global package or hardware settings changed; existing .gitignore excluded.

Exact fixed seed42 preparation step256 is outputs/phase4a_modern/runs/fineweb/s0/selected.pt,
SHA256 f497cf2310050b163ecb167f75ae4f89c2780d4165aa8f25ff77025715f4bd28.
Archived summary selected step256/tokens2097152 and matching metrics row,
state checksum and preparation permutation checksum confirm provenance.
The historical selected file happens to be the requested time point; no
validation-based search for a different S0 occurred. CE smoke recomputation
gives exactly the archived validation NLL2.9671119410592275.

Frozen teacher SHA256 c7411708b61e6524ef05bd24e9e618f19fcc119e9a3c0664c5f4f169dbf16928.
Same official135M/360M pinned revisions and shared49152-vocabulary tokenizer
as Phase4A; see its model_download_manifest.json/model_and_data_audit.md.
Same FineWeb sample-10BT pinned revision and doc-disjoint train/val/test
streams. Full source split hashes are in continuation_manifest.json.

Manifest file SHA256 b81200c6648a8dfbe2a3d9b432800377152162f490ac7290b3f91fb8492974d1.
Ordered little-endian int64 chunk-array SHA256
c43ec9fc22b28cd14b2586707858039ad8931ce621d4d24fe4b63ef8cce42708.
input_audit.json also fixes original summary/metrics checksums. The manifest
stores explicit excluded and continuation chunk indices, no data text or tokens.
Test stream was checksum-audited only, not forwarded or used for LR selection.

Unchanged Phase4A train/core helpers are source-hash checked before every run.
New Phase4C adapter changes only the authorized fixed S0, disjoint order,
LR candidates, budget and step0-eligible selection. Numerical losses/evaluation/
packing/scheduler/model loader remain the old implementations. New5 protocol
tests and old4 numerical tests PASS; CE-only disposable2-update smoke PASS,
no saved state/test/KD execution. Formal protocol commit precedes CE results;
any conditional KD additionally requires a selected-protocol commit first.

Large weights, caches and full logs remain under ignored outputs/; compact
telemetry summaries and compressed metrics are archived separately on completion.
Manuscript SHA8b8e3fcc3f04b596058cd4b63ddfe38b6eab91f338c7d4f12db4b602c9982ec8;
final_evidence aggregate SHA3d172e634db611edad0d566e74ea0f826ed8cdae69922120baa94ebf9e6e2b96.
No old Phase4A/B endpoint, datasets/ or checkpoints/ write is performed.

## Completion chain

Formal CE protocol commit dabd3f6e31e1a371cdf4a78ac7ca3c6bd9cfd65f was pushed
before launch. Remote launcher608564 owned CE children608566/608567/608568
on GPU0/1/2. All exit0; all3 runs consumed8M targets/977 updates. raw/ archives
registry, completion, original summaries and gzip-compressed full metrics.
ce_lr_gate.csv records selected-state SHA256 and validation-only comparisons;
all selected977, but max improvement0.004849292809 failed.01. No KD settings
commit, KD run or new test evaluation. Gradient ratio/test remain unmeasured.

collect_gate.py independently checks full target accounting, exact validation
schedule/minimum/tie selection, telemetry and state hashes before
archive_gate_failure.py emits explicit NOT_RUN rows. Final audit PASS.
The2e-5 CE finite norm14.602655411 at656 is retained in raw telemetry and
gradient_event_audit.json; cause/impact UNKNOWN, no exclusion or retry.
Old Phase4A/B, datasets/checkpoints, manuscript and final_evidence have no
Git diff against parent6506c79. Existing .gitignore edit is untouched.
Compact artifacts only; STOP_MODERN_CONTINUATION_NOT_ESTABLISHED is terminal.
