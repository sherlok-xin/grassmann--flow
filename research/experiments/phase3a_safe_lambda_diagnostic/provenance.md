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

- Protocol commit: `80904d5b067ef998804fa446f0a91db8b9612da0`, pushed before tests and model-level computation.
- Blinded-results commit: `ea86510`, pushed before endpoint merge.
- Remote Python: 3.12.3. Four RTX 3090 GPUs were idle before launch.
- Remote test command: `python -m pytest -q research/experiments/phase3a_safe_lambda_diagnostic/tests`; result: 4 passed.
- Real-model smoke: `WT2_J_S42`, two training chunks, one calibration chunk, one calibration seed; completed full-parameter FP32 HVP without OOM. Smoke output is excluded from all tables.
- Formal command: `bash research/experiments/phase3a_safe_lambda_diagnostic/scripts/run_remote_matrix.sh`.
- Formal completion: 21/21 condition JSON files; no process failure and no checkpoint mutation.
- Formal raw size: approximately 276 KiB including the blind freeze record; no model state is stored.
- Blinded table: 63 rows, 21 conditions, no endpoint column; SHA256 `594ac6d441fbe19776280225a0d652eb38baa7402dbf7a7295e065f7530fd9ad`.
- Endpoint-merged table SHA256: `89fe9e08d1b3190a1c2c9041ef1754efacbde2eda163bc6bd0dcd7ca6974ad91` at initial merge.

The literal optimizer-aware control used the exact source `LambdaLR` initialization. Its first update learning rate is zero for every condition and coefficient. The additional `nominal_lr_sensitivity` is explicitly not called the literal formal first step.

No damping or finite-difference HVP approximation was used in formal model diagnostics. Finite differences are used only in the synthetic numerical test. Hessian symmetry is also tested on the explicit synthetic quadratic. Full-model symmetry was not recomputed with a second HVP because it would double the formal HVP workload without changing the frozen scalar test; this limitation is disclosed rather than replaced by a reduced-parameter approximation.

## Deviations

There is no scientific-protocol deviation. Dataset construction emitted the historical tokenizer warning that concatenated corpus text exceeds the tokenizer model maximum; the existing `TextDataset` then chunks tokens to sequence length 256 exactly as in prior experiments. The warning did not produce an indexing error.

The informative virtual control uses the already computed exact 32-example probe gradients and applies AdamW updates without rerunning model forwards for each lambda. This is algebraically identical to assigning `gC + lambda*gD` at S0 before clipping, and avoids six redundant backward passes. Dropout is deterministically fixed during probe-gradient construction.
