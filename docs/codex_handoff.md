# Grassmann Flows Codex Handoff

> 2026-09-16 更新：本文件保留 2026-04 的历史交接内容。当前导航、证据分级与实验路线分别见 `docs/project_map.md`、`docs/experiment_catalog.md` 和 `docs/research_roadmap.md`；不要再直接按下方旧优先级启动实验。

更新时间：2026-04-26
适用范围：`grassmann-flows`

## 1. 先看哪些文件
- `docs/project_memory.md`
- `docs/experiment_journal.md`
- `train_hybrid_latefusion_alpha_ddp_v1.py`
- `train_hybrid_lite_latefusion_baseline_v2.py`
- `train_distill_hybrid_lite_from_latefusion_teacher_v2.py`
- `benchmark_ptb_efficiency_v4.py`
- `run_ptb_final_cuda_supplement_suite.sh`
- `run_ptb_final_batch_latency_suite.sh`

## 2. 这个项目现在真正要解决什么
- README 里的早期重点是复现论文版 Grassmann/Transformer 对比。
- 当前真正的主线已经切到 PTB：
- 训练一个效果最强的 `hybrid late-fusion teacher`
- 再训练可部署的 `hybrid-lite student`
- 用 warm-start 蒸馏把 student 的 PPL 压下去
- 最后用统一 benchmark 输出质量/延迟/吞吐/显存/参数量

## 3. 当前最重要的已知结果
- 目前最终三候选来自 2026-04-02 的 final suites。
- Teacher:
- 结构：`hybrid late-fusion`
- 参数量：`36.264M`
- `test_ppl=50.11`
- Student-best:
- 结构：`224x56, l6, kd0.05`
- 参数量：`31.434M`
- `test_ppl=51.84`
- Student-tradeoff:
- 结构：`192x48, l4, kd0.05`
- 参数量：`23.565M`
- `test_ppl=53.87`
- batch latency 结论：
- bs1: teacher `15.48ms`, best `16.09ms`, tradeoff `11.50ms`
- bs8: teacher `19.15ms`, best `19.33ms`, tradeoff `16.56ms`
- bs32: teacher `55.98ms`, best `48.46ms`, tradeoff `34.96ms`

## 4. 已经被验证过的实验判断
- `warm-start + 小 KD 权重` 是有效路线。
- `distill_alpha=0.05` 明显优于 `0.0/0.1/0.15/0.2` 这批已跑点。
- `31M` student 是质量优先候选。
- `23.6M` student 是速度优先候选。
- CE-only warm-start 基本没用。
- 大 KD 权重会拉坏结果。

## 5. 关键代码主链路
- Teacher 训练：
- `train_hybrid_latefusion_alpha_ddp_v1.py`
- Baseline student：
- `train_hybrid_lite_latefusion_baseline_v2.py`
- Warm-start KD student：
- `train_distill_hybrid_lite_from_latefusion_teacher_v2.py`
- 统一 benchmark：
- `benchmark_ptb_efficiency_v4.py`
- 早期与衍生脚本：
- `train_exp4_ddp.py`
- `train_hybrid_alpha_ddp_v5.py`
- `train_distill3_ddp_v5.py`

## 6. 代码结构怎么理解
- `src/models/grassmann_v4.py`
  - 当前 Grassmann GPT 主要实现。
- `src/models/gpt2.py`
  - 基础 GPT2 风格模型部件。
- `train_*`
  - 直接承载具体实验范式，很多配置不只在 `configs.py`，而是写在脚本参数和 suite shell 里。
- `outputs/hybrid_experiments/*/summary.json`
  - teacher/baseline 结果摘要。
- `outputs/distill_experiments/*/summary.json`
  - 蒸馏实验摘要。
- `outputs/benchmark_reports/*.json`
  - 统一 benchmark 汇总。

## 7. 当前最值得继续做的事
- 第一优先级：
- 修 CUDA 校验链路。
- 已知问题：
- `test_cuda_kernels.py` 有 state_dict key mismatch。
- `benchmark_cuda.py` 有 `src` 循环导入问题。
- 第二优先级：
- 在 warm-start 框架下继续细扫 `alpha in {0.03, 0.05, 0.07}`，每个点至少多 seed。
- 第三优先级：
- 如果目标转向论文/交付，就把 teacher / best / tradeoff 三个模型的表格、批大小敏感结论和选型建议固化。

## 8. 这个项目常见误区
- 不要再把注意力放回 README 的 WT2 早期结论上，当前主线不是那个。
- 不要只看单次训练日志，很多结论已经体现在 `summary.json` 和 final benchmark json 里。
- 不要直接在本机跑正式训练。

## 9. 远程工作流
- 先检查环境：
- `./agent-tools/run197_check.sh`
- 提交任务：
- `./agent-tools/run197_submit.sh --name <run_name> -- <command>`
- 看状态：
- `./agent-tools/run197_status.sh <run_name>`
- 跟日志：
- `./agent-tools/run197_logs.sh --follow <run_name>`
- 停任务：
- `./agent-tools/run197_stop.sh <run_name>`

## 10. 对下一个 Codex 的建议
- 先读 `docs/project_memory.md` 和 `docs/experiment_journal.md`，不要重复做已经被证伪的 CE-only / 大 KD 扫描。
- 如果你的目标是“继续实验”，优先补 CUDA correctness 或做 `alpha` 细扫。
- 如果你的目标是“产出结论”，优先整理 final suites 的三模型对照表，而不是再开新大实验。
