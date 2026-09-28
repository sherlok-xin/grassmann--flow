# Phase 2C Provenance

Phase 2C is governed by `research/experiments/phase2b_controlled_teacher/REPORT.md`. The Phase 2B conclusion is partial: seed-42 KD gain is ordered T_a05 > T_a03 > T_a00, while T_a00 remains positive-transfer.

The WikiText-2 dataset manifest is `research/snapshots/20260920/dataset_manifest_wikitext2.json`, SHA256 `2534df9ffce9a64cd8dc49ad267a1cb549c42fd8fd219b836bf15499de7ab0a5`. The mounted dataset fingerprint and preprocessing statistics are inherited unchanged. Heavy execution occurs only in the server-197 `grassmann_lab` container.

Canonical seed-42 S0 checkpoint:

`outputs/hybrid_experiments/20260515_004119_wt2_hybrid_lite_baseline_224x56_l6_e20/checkpoints/hybrid_best.pt`

SHA256: `9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef`

Frozen teacher checkpoint:

`outputs/hybrid_experiments/20260328_080104_wt2_hybrid_alpha_joint/checkpoints/hybrid_best.pt`

SHA256: `a5ad284f6895c0b4ed516c855c1b149e6636b945173aa2e842b8132299582694`

Source hashes before training are `57d4e42920fa51f318e2c03aafac4908e3067bd73fdf63fae46b60b2096ed6e1` for the S0 trainer, `abd501c5f43a8ea79316dcbe56710bb62afd06f1b24e18d848cd3ad6cbeb2202` for the continuation trainer, and `4a7d002db4f068d299d69e97832ca98c0d03644912c4e1d898ceb4a81592374d` for the KD loss implementation.

Repository-wide configuration inspection found no valid complete seed-123 or seed-456 WikiText-2 Hybrid-lite S0 run. These checkpoints must be trained independently before continuation. Exact commands, run directories, hashes, retries, and integrity checks will be appended as execution proceeds.

## Pre-training gates and infrastructure notes

One-epoch, 2,000-line S0 smoke runs completed for both new seeds. Seed 123 produced validation/test NLL 9.257250/9.229374 and seed 456 produced 9.266918/9.238943; both have 31,434,257 parameters, complete summaries and checkpoints, seed-specific trajectories, and the required split seed 42. These smoke checkpoints are not used as formal S0 states.

The first attempt to launch the seed-456 smoke did not start because shell backgrounding returned the second command to the container default directory before log redirection. The subsequent standalone launch completed. The first formal seed-456 S0 attempt was sent to physical GPU 0 because `--gpu-id 0` overrode the intended `CUDA_VISIBLE_DEVICES=1`; it failed with OOM before the first optimizer step while seed 123 was already using that card. The incomplete directory and logs are retained. Seed 456 was relaunched unchanged except for the corrected physical device assignment to GPU 1.

The utility implementation passed a 32-chunk seed-42 smoke and then reproduced the Phase 2A 512-chunk selection-index SHA256 `236633612db1e3144b8f3187a5bfb1795f558a50f26368252a8adf3f0b49413f`. It stores no raw text or complete logits. The combined utility, alpha-control, checkpoint-hash, legacy-alpha compatibility, and KD-loss test suite passes 10/10 tests in the remote environment.

The first alpha-only fixed-composition evaluation failed before inference because its historical checkpoint stores scalar `logit_alpha`, while the current model stores a length-one vector. A bounded compatibility loader now reshapes only when element counts match, without changing the scalar value. A unit test covers this behavior. The alpha-only evaluation then completed on the same 512 chunks.

The continuation trainer SHA256 after the two compatibility-only changes is `5217f3d4a8b5e35ee4852194346f67c03d00d0a4ae829f0ed11770c865359c41`. These changes add legacy scalar-alpha loading and recognize the existing `student` summary schema; they do not alter the forward pass, objective, optimizer, alpha override, or behavior for the Phase 2B joint teacher checkpoint.

The formal seed-123 and seed-456 S0 configs were compared field by field with the canonical seed-42 config while training was in progress. Apart from experiment metadata, the random seed, and the physical GPU identifier for seed 456, there are no differences. In particular, architecture, dataset preprocessing, optimizer settings, epoch budget, AMP, and validation-selection behavior are identical. The frozen remaining-matrix launcher is `run_remaining_matrix.sh`; it waits for both complete S0 summaries, hashes each selected checkpoint, and then executes only the preregistered C0/C1/C3 arms. Its SHA256 is `9d9fa90dce688aed0736c62128eeed01f5f28e27284520388a07213a2c0bb74c`.

Both formal S0 replications completed the full 20 epochs. Seed 123 selected epoch 13 with validation/test NLL 4.380287/4.261903 and checkpoint SHA256 `a44b64ba5a84d3bd2708a96aad4eae829ff7ab0f00476f0e31146fc314da1f13`. Seed 456 selected epoch 14 with validation/test NLL 4.383104/4.260188 and checkpoint SHA256 `3fbe985d6474c84d1c61fd7937ea906102cf460846829eb3af4c709593b01757`. The canonical seed-42 S0 selected epoch 15 with validation/test NLL 4.364849/4.250827. The three checkpoint hashes are distinct.
