# AGENTS

- 代码在本机 NFS 挂载目录编辑（`/home/xin/fuwuqi/grassmann-flows`）。
- 实验统一在 `10.42.0.197` 上运行，不要在本机跑重训练任务。
- 优先使用 `~/fuwuqi/agent-tools/exec_grassmann.sh '<command>'` 执行远程命令。
- 修改前先看日志与最近结果，先做 smoke test，再放大到正式实验。
- 不要改动 `datasets/` 与 `checkpoints/` 目录及其内容。
- 默认维护当前项目自己的 `docs/project_memory.md` 与 `docs/experiment_journal.md`，且不与其他项目混用。
