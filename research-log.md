# Research Decision Log

## 2026-09-16: Post-rejection audit and reset

The project was moved from manuscript repair back to empirical research. The repository was audited before starting new training. A normalized registry now covers 186 run directories and identifies 159 complete runs, 25 config-only runs and 2 additional incomplete runs. No historical output was moved or deleted.

The audit found that current KD uses a token-summed `batchmean` KL, making the reported coefficient sequence-length dependent. It also found effective CodeParrot runs whose recorded dataset label remains PTB or WikiText-2. The next confirmatory experiment is therefore loss normalization and matched-control validation, not another legacy alpha sweep.

Remote execution was checked through the project wrapper. Work is compute-blocked because `/songxin` is not mounted on `10.42.0.197` and the SSH user cannot mount it without sudo. Local heavy training was not attempted.

Detailed historical experiments remain in `docs/experiment_journal.md`; this file records only research-level decisions and direction changes.

## 2026-09-16: Remote mount restored

The workspace handoff at `/home/xin/fuwuqi/CODEX_WORKSPACE_HANDOFF.md` was read and adopted. It confirms that code is edited through the local NFS path while all Grassmann execution occurs in the `grassmann_lab` container on `10.42.0.197`, with `/workspace/grassmannflows/grassmann-flows` as the project path.

A read-only check through `agent-tools/exec_grassmann.sh` succeeded after the mount was restored. The container sees 159 summaries and 184 configs, PyTorch 2.6.0a0 with CUDA 12.6, and four RTX 3090 GPUs. No training process was active. The H0 smoke experiment is no longer infrastructure-blocked.

## 2026-09-16: H0 PTB smoke passed

The KD trainer now has a legacy-compatible loss mode and a valid-token-normalized mode. Four unit tests passed remotely. A five-arm, one-epoch PTB smoke showed that token `lambda_kd=5` closely reproduces legacy `alpha=0.02`, as predicted by the exact 255-token conversion (`lambda=5.204`). Their validation PPL values were 61.73 and 61.69, and their test PPL values were 53.29 and 53.26.

The nominally stronger `lambda_kd=10` achieved 61.35 validation PPL but clipped or overflowed on 54.7% of steps, so it was rejected. A repeated gradient diagnostic found only 1.4% AMP overflow for both legacy and token `lambda_kd=5`, with nearly identical finite gradient norms. The fixed coefficient for cross-domain confirmation is therefore `lambda_kd=5`; it will not be tuned separately on TinyStories.

The first legacy launch failed on GPU 0 because an unrelated process occupied 3.7 GiB. The scheduler was changed to use GPUs 1--3, completed arms were reused, and all missing arms finished. The failed artifact remains indexed. H0 is now pending matched PTB/TinyStories runs with seeds 42, 123 and 456.
