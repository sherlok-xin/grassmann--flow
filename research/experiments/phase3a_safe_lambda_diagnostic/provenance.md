# Phase 3A Provenance

## Authorization and repository state

- Authorized scope: post-freeze offline diagnostic only; no KD training, endpoint modification, manuscript edit, or automatic Phase 3B.
- Protocol parent HEAD: `7f12aabd565bbb861557a2d55c636d2432602969`.
- Execution host: `10.42.0.197` through `~/fuwuqi/agent-tools/exec_grassmann.sh`.
- Remote project path: `/workspace/grassmannflows/grassmann-flows`.
- Initial `research/final_evidence/` aggregate SHA256: `3d172e634db611edad0d566e74ea0f826ed8cdae69922120baa94ebf9e6e2b96`.
- Pre-existing unrelated worktree change: `.gitignore`; Phase 3A does not modify or stage it.

## Frozen protocol

The locked scientific and numerical protocol is in `preregistered_gate.md`; the mathematical sign convention and safe-bound failure cases are in `THEORY_NOTE.md`. The 21-row primary set is in `condition_inventory.csv`. Protocol code contains no endpoint reader and does not construct a test split.

The formal runner verifies all S0 hashes, records teacher hashes before and after computation, uses full-parameter FP32 gradients/HVPs, and refuses a last-layer fallback. Raw diagnostic JSON contains an explicit `uses_test_data=false` manifest field. Virtual states are restored after every coefficient and are never saved.

## Blinding chain

`scripts/build_blinded.py` is the only producer of `diagnostics_blinded.csv`. It reads raw diagnostics and the condition inventory only. It writes `raw/blinded_freeze.json` with the complete table SHA256 before `scripts/merge_endpoints.py` is allowed to read historical endpoint tables. The merger aborts on a hash mismatch.

## Execution record

Pending protocol commit, remote unit tests, real-model smoke, and formal diagnostics. Commands, software versions, model hashes, blind-table hash, and any deviations will be appended after execution.
