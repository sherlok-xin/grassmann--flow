# 项目压缩记忆（project_memory）

## 2026-09-17 Stage-C checkpoint

历史结果再审计见 `docs/historical_output_audit_20260917.md`。当前不重复旧实验的下一项方法实验是 bounded Consensus-Routed Branch Distillation prototype，冻结协议为 `research/experiments/h2_crbd/protocol.md`。

CRBD 已作为显式 opt-in strategy 实现，并使用 token chunk 限制显存；默认 fused KD 路径不变。远端测试当前为 15/15 通过。PTB fixed、CRBD、shuffled-routing、swapped-pairing smoke 已完成。TinyStories reduced prototype 正在运行：CE validation PPL 为 5.0183，fixed-KD validation PPL 为 5.3041，因此预注册的 50% repair 阈值为 5.1612。在 C2--C7 summary 完整并执行 Stage-C gate 前，不启动 Stage D。

四卡同时负载时应避免将 GPU 3 用于限时对比；该卡在训练负载下锁在约 210 MHz。其不完整 C3 尝试是基础设施记录，不是方法失败。C3/C7 应在正常时钟 GPU 上按完全相同配置重跑。

## 2026-09-18 Stage-C result

TinyStories reduced prototype 的 C0/C1/C2/C4/C5/C6/C7 已完成，C3 缺失。validation PPL 依次为 5.0183、5.3041、5.3027、5.6135、5.5104、5.5045、6.3668。C5 相对 C1 又恶化 0.2063，repair fraction 为 -0.7220，并比 shuffled C6 差 0.0059；因此 bounded Stage-C gate 失败，不启动 Stage D。

当前 formulation 的关键问题是 fused 与 branch routing weights 分别归一到 mean one，导致两项完整 KD budget 同时叠加。C5 epoch 2 的 weighted fused/branch loss 约为 2.77/3.53，CE 为 1.73；所有 KD arm 的 gradient clip fraction 均为 1.0。若继续，应先补 C3，并将 routing 改为共享一个 fixed total KD budget，再匹配 C1 的初始 loss/gradient norm。完整结论见 `research/experiments/h2_crbd/analysis.md`。

更新时间：2026-04-12  
适用范围：仅 `grassmann-flows` 当前仓库

## 1. 项目目标
- 复现并扩展 Grassmann Flows 路线，评估其在语言建模中的效果与效率。
- 当前阶段重点不再是 README 中的早期 WT2 对比，而是 PTB 上的 `Hybrid teacher -> Hybrid-lite student` 质量/延迟权衡。
- 目标是得到可交付的最终候选：在尽量接近 teacher 困惑度（PPL）的前提下，显著降低延迟和参数量。

## 2. 当前任务与指标
- 当前任务：确定 PTB 最终候选（teacher、best student、tradeoff student）并给出批大小敏感的延迟结论。
- 核心指标：
  - 质量：`test_ppl`（越低越好）
  - 效率：`avg_batch_latency_ms`（bs=1/8/32）、`tokens_per_sec`、`peak_memory_gb`
  - 成本：`params_m`
- 当前关键结果（2026-04-02 final suite）：
  - Teacher（36.264M）：`test_ppl=50.11`
  - Student-best 224x56 l6 kd0.05（31.434M）：`test_ppl=51.84`
  - Student-tradeoff 192x48 l4 kd0.05（23.565M）：`test_ppl=53.87`

## 3. 关键代码结构
- 训练主入口（当前阶段）：
  - `train_hybrid_latefusion_alpha_ddp_v1.py`：训练 hybrid teacher
  - `train_hybrid_lite_latefusion_baseline_v2.py`：训练 hybrid-lite baseline
  - `train_distill_hybrid_lite_from_latefusion_teacher_v2.py`：warm-start KD 蒸馏 student
- 评估/基准：
  - `benchmark_ptb_efficiency_v4.py`：统一输出 ppl/latency/tok/s/memory
  - `run_ptb_final_cuda_supplement_suite.sh`：CUDA 补充验证 + 最终三候选 benchmark
  - `run_ptb_final_batch_latency_suite.sh`：bs=1/8/32 延迟对比
- 配置与通用：
  - `configs.py`（通用预设，当前 PTB 主流程更多依赖脚本参数）
- 产物与日志：
  - `outputs/hybrid_experiments/*/summary.json`
  - `outputs/distill_experiments/*/summary.json`
  - `outputs/benchmark_reports/*.json`
  - `logs/*.log`

## 4. 训练/评估主链路
1. 训练 teacher（late-fusion last-k，当前最优是 last1 joint）。
2. 训练 hybrid-lite baseline（从零训练，确定可蒸馏初始化）。
3. 对 baseline 做 warm-start KD（teacher 指导；重点扫 `distill_alpha` 与规模）。
4. 用 `benchmark_ptb_efficiency_v4.py` 跑统一离线 PTB 评估。
5. 用 final suites 固化三候选在 bs=1/8/32 的延迟与内存曲线。
6. CUDA 路线先 smoke（扩展构建、内核测试、微基准）再进入正式结论。

## 5. 当前核心假设
- 假设 A：`warm-start + 小 KD 权重（~0.05）` 能显著优于 CE-only 与较大 KD 权重。
- 假设 B：蒸馏可让 31M 级 student 在 PPL 接近 teacher 的同时降低延迟。
- 假设 C：更小 student（23.6M）可在可接受 PPL 退化下明显提高吞吐/降低延迟，适合作为工程 tradeoff。

## 6. 已验证有效 / 无效方法
### 已验证有效
- Warm-start KD 明显有效：
  - `ptb_hybrid_lite_warmstart_kd005`：`test_ppl=51.84`，显著优于同结构 baseline `58.53`。
- Seed 复现稳定：
  - 224x56 l6 kd0.05 两个 seed 在 `test_ppl≈51.77~51.88`。
- Tradeoff 方案有效：
  - 192x48 l4 kd0.05：`test_ppl=53.87`，但 bs32 延迟约 `34.96ms`，快于 teacher（`55.98ms`）。

### 已验证无效
- CE-only warm-start 基本无收益：
  - `ptb_hybrid_lite_warmstart_ce_only`：`test_ppl=59.08`（劣于 kd0.05）。
- KD 权重过大退化：
  - kd0.1/0.15/0.2 对应 `test_ppl≈53.13/53.06/53.65`，均差于 kd0.05。
- 早期 v1 蒸馏配置（如 mild/balanced/hot）PPL 明显偏高（90+ 到 170+），不可用。

## 7. 当前阻塞点
- CUDA 校验链路未打通：
  - `test_cuda_kernels.py` 存在 state_dict key mismatch；
  - `benchmark_cuda.py` 出现 `src` 循环导入报错。
- 基准日志持续出现 tokenizer 长序列 warning（`46038 > 1024`），虽当前流程可跑通，但需确认是否影响可比性叙述。
- 本地仅有 `python3` 命令，执行脚本需注意环境一致性。

## 8. 下一步优先实验
1. 先做 CUDA 路线 smoke 修复（state_dict key 对齐 + 导入路径修复），再复跑补充 suite。
2. 在 warm-start 框架下细扫更小 KD：`alpha ∈ {0.03, 0.05, 0.07}`，每点至少 2 seeds。
3. 围绕 23.6M tradeoff 学生做小步调参（优先提升 PPL，不显著回退延迟）。
4. 固化最终对外结论表：teacher / best / tradeoff 三模型在 bs=1/8/32 的统一报告。

## 9. 2026-09-14 CAC 论文证据审计与重构

- 目标稿：`论文投稿/cac/conference_101719.tex`；编译产物为同目录 `conference_101719.pdf`。
- 新主线：研究异构 Grassmann--Transformer teacher 的 logit KD 在不同初始化与数据域下何时有效；定位为经验研究，不声称新蒸馏目标。
- 已校正的关键事实：
  - `grassmann_v4.py` 使用特征依赖向量门，不是标量门。
  - 训练代码实现 `KL(p_T || p_S)`，且 `batchmean` 只按 batch 归一化，`lambda` 不可脱离实现直接比较。
  - hidden-fusion 记录为 63.93 PPL / 35.37M 参数，并且 checkpoint 不匹配，不能作为融合位置消融。
  - PTB 105.86 PPL 的 random-init 运行是 `lambda=0.3` 且模型配置不同，已从 matched table 删除。
  - 同域 code 使用 5k 文件，PTB→code 使用 40k 文件，跨域结果只保留为受混杂的 stress test。
- 可对外保留的多域描述：warm-start KD 在 PTB、WikiText-2、code 的最佳正系数点相对 warm-start CE 分别改善 13.2%、15.0%、17.2%，但 TinyStories 退化 9.3%。
- 图、表和参考文献均已重新生成并可复现；未启动新训练，也未改动 `datasets/` 或 `checkpoints/`。
- 后续最高优先级：多 seed、token-normalized KL、checkpoint-matched fusion、matched-data cross-domain 四组补充实验。

## 10. 2026-09-15 CAC 英文表达润色

对 `论文投稿/cac/conference_101719.tex` 完成逐段学术语言润色，覆盖摘要、引言、相关工作、方法、实验、图表说明、讨论与结论。统一使用正式、简洁的表达，消除名词所有格和口语化措辞，将修订过程叙述改为可独立阅读的实验说明。逐项比对确认引用与交叉引用、公式环境、表格内容、图件路径、标题层级、导言区及各 LaTeX 命令数量保持一致；未新增强调格式或列表。保留单 seed、数据预算与融合对照的证据边界。本轮仅编辑论文并编译 PDF，未运行训练。

随后按用户新的自然化要求，将贡献列表及行内枚举改为普通段落，删除正文中的加粗和斜体强调，压缩重复总结与模板化过渡。保留已经自然的技术说明及表格最优值标记，引用、公式、表格数据和图件均未改变；本轮明确移除了原有 itemize、item、正文 textbf 和 emph 命令。

## 11. 2026-09-15 架构图与投稿版本说明

新增独立架构图 `论文投稿/cac/figures/fig_architecture_v2.{svg,pdf,png}`，源文件为 `gen_architecture_v2.py`。上半部分展示冻结教师、可训练双分支学生、初始化选择及 KL/CE 两条监督路径，下半部分展示 causal Grassmann mixing 与隐藏状态旁路。初步使用内置图像生成进行构图，最终以原生矢量绘制，保留英文文本和可复现源代码。用户说明昨天修改前的版本已投稿，原标题为 `Bridging the Gap: Warm-start Hybrid-lite Distillation for Grassmann Flow Sequence Models`。新图独立交付；本轮未改主稿引用或标题。标题建议须区分已投稿 PTB 主线与后续多域经验研究主线。

## 12. 2026-09-15 CAC 二次证据审计与 image1_1 替换

- 已将 `论文投稿/cac/figures/image1_1.png` 作为双栏架构图纳入主稿，图注明确了 teacher/student 共享拓扑、多尺度平均和被省略的投影偏置/残差/归一化等细节。
- 发现原稿将 random-init KD 与 CE-from-scratch 误称为 matched：学习率不同，且 WikiText-2/TinyStories 训练长度不同。已将这些数据降级为探索性证据，删除初始化的因果主张。
- 论文主线收紧为可受控验证的四数据集 warm-start KD 系数扫描，标题更新为 `Warm-Start Knowledge Distillation for Grassmann--Transformer Hybrids: An Empirical Study Across Text Domains`。
- WikiText-2 源分支与 teacher 的数据处理记录不一致，因此 RQ1 改为 teacher 与同数据集 CE hybrid student 的预测质量对比，不再把源分支 PPL 当作融合消融。
- 方法公式已补全 Pl\"ucker L2 归一化、有效窗口平均、投影维度与无有效偏移时的零特征处理，并统一了顶层融合门符号 `alpha`。
- 编译产物仍为 IEEE letter 双栏 6 页，无 overfull box、undefined citation/reference 或 duplicate label；未运行新训练。

## 13. 2026-09-16 拒稿后研究重置与仓库整理

- CAC 已拒稿，项目从论文修订切回实验与方法研究。
- 新增 `tools/build_experiment_registry.py`，非破坏性索引 186 个运行目录：159 个完整、25 个 config-only、2 个进一步不完整；结果写入 `docs/generated/experiment_registry.csv` 与 `experiment_inventory.md`。
- 仓库约 18 GiB，其中 178 个输出 checkpoint 约 16.84 GiB，日志约 895 MiB；本轮未移动或删除任何历史产物。
- 发现 25 行有效数据域与原始 `dataset_name` 不一致，主要是 CodeParrot 运行继承了 PTB/WikiText-2 默认标签。后续必须按 dataset path 与 token counts 审计。
- 发现现有 KD 的 `batchmean` KL 对序列位置求和，导致 alpha 与序列长度、域相关 loss scale 耦合。下一实验优先实现 legacy-compatible 的 token-normalized KL，而不是继续旧式系数扫描。
- 新增 `docs/project_map.md`、`docs/experiment_catalog.md`、`docs/research_roadmap.md`、根目录研究状态文件及 H0/H1 预注册协议。
- 远程 wrapper 检查失败：`10.42.0.197` 上 `/songxin` 未挂载，SSH 用户没有免密 sudo。正式 smoke 与训练需在挂载恢复后进行，本地未启动训练。
- 同日后续复检确认挂载已经恢复。`grassmann_lab` 中项目路径为 `/workspace/grassmannflows/grassmann-flows`，可见 159 个 summary、184 个 config 和 4 张 RTX 3090；当时无训练进程。后续实验恢复为“本机 NFS 编辑、197 容器执行”的标准方式。

## 14. 2026-09-16 H0 token-normalized KD smoke

- 在 `src/kd_losses.py` 新增共享 KD loss：`legacy_batchmean` 精确保留历史公式，`token_mean` 按有效预测 token 归一化并使用 `CE + lambda_kd * KL_token_mean`。
- 当前 KD 入口新增 `--kd-loss-mode`、`--kd-lambda` 和 `--kd-chunk-tokens`，并记录 raw/token KL、有效 token 数、有限梯度均值、裁剪比例、非有限梯度比例和 AMP overflow 比例；旧参数默认行为不变。
- PTB 一轮 smoke 的 test PPL：CE-only 59.04、legacy `alpha=0.02` 53.26、token `lambda=2.5/5/10` 分别为 54.08/53.29/53.13。
- `lambda=10` 的裁剪比例为 54.7%，不满足稳定性门槛。`lambda=5` 与 legacy 的理论等效值 5.204 接近，验证/测试 PPL 差均小于 0.05；诊断复跑的有限梯度均值为 0.7872/0.7874，AMP overflow 比例均为 1.4%。
- 正式 H0 系数固定为 `lambda=5`，下一步以 seeds 42/123/456 对 PTB 与 TinyStories 做 CE-only、legacy 和 token-normalized matched confirmation，不在 TinyStories 上重新调参。
- 首次 legacy 任务因 GPU 0 已被外部进程占用而 OOM；随后避开 GPU 0 完成。失败目录和日志均保留，未修改数据集或既有 checkpoint。

## 15. 2026-09-16 拒稿后分阶段实验与创新方案

- 使用 `academic-research-suite` 将后续工作拆为证据修复、机制诊断、方法原型和确认性评测四个闸门；正式训练需在作者确认后执行。
- H0 的关键结论收紧为：在当前固定 256-token、无 padding 的数据上，legacy `alpha=0.02` 与 token-normalized `lambda≈5.204` 仅差全局目标尺度。因此 normalization 是可解释性修复，不能解释 TinyStories 负迁移，也不构成主创新。
- 主方法候选为 Consensus-Routed Branch Distillation：低分歧 token 蒸馏融合分布，高分歧 token 转为 Transformer-to-Transformer 与 Grassmann-to-Grassmann 分支对齐；CE 始终保留，学生推理结构不变。
- 单纯熵加权、置信度加权、分歧抑制和 branch-only KD 均降为强基线或消融，避免与现有 adaptive/multi-teacher/heterogeneous KD 工作发生过宽的新颖性主张。
- 推荐先执行 PTB 三 seed 的 CE/KD matched confirmation、TinyStories seed42 的 CE/KD confirmation，以及无需训练的 branch-JSD/梯度冲突诊断，预计约 25 GPU-hours；机制闸门通过后再实现 CRBD。
- 完整方案、命令矩阵、统计规则、停止条件和算力预算记录于 `research/experiments/post_rejection_program/experiment_plan.md`。本轮未启动训练、未修改数据集或既有 checkpoint。

## 16. 2026-09-16 A1/A2 执行状态与路径校正

- TinyStories seed42 的 CE-only 与 token `lambda=5` 两臂已在 GPU 1/2 运行，保持 14 小时上限；不得因 PTB 调度中断。
- PTB 单轮约需 4.1 分钟，完整 10 epochs 预计 41--43 分钟，作者已批准将硬上限由 30 分钟延长至 50 分钟。
- seed42 PTB 的实际 warm-start 目录是 `outputs/hybrid_experiments/20260328_142654_ptb_hybrid_lite_baseline_mild_e20`，不是预计划中不存在的 `20260402_135900...`。首次 50 分钟配对启动因此在训练前失败；GPU 0/3 随后仍为空闲。
- B 诊断测试的根目录导入已修正，但 `src/__init__.py` 对 `src.data` 的导入仍阻断测试收集。该问题与训练入口无关，需单独修复并复测。

后续经作者确认，PTB seed42 的 CE/KD 已使用正确初始化在 GPU 0/3 重启并进入训练，50 分钟上限生效。`src/__init__.py` 的无效 `src.data` eager import 已移除，endpoint harm 测试样例的标签方向已校正；远端 KD 与 B 诊断测试共 10 项全部通过。B 的正式诊断应等待对应 CE/KD endpoint 完成后执行。

## 17. 2026-09-17 A1 部分结果与 PTB 机制诊断

- PTB matched CE 的 seed42/123/456 test PPL 为 59.1175/59.6566/59.0819。已完成的 token-KD seed42/123 为 51.2771/51.3745，对应改善 7.8405/8.2821 PPL；seed456 KD 仍在运行。
- B 的数值 smoke 发现闭式 CE 范数在饱和概率处发生抵消。现使用显式 CE 梯度向量，并将理论有界的 JSD 与 cosine 限制到合法区间；新增测试后远端 11/11 通过。旧 smoke 保留，新 smoke 的固定索引完全一致且不含原始文本或 logits。
- PTB 全 validation 诊断覆盖 350 chunks、89,250 tokens。JSD 与更负 gradient cosine 的条件关联方向明确，bootstrap 95% CI 不跨零；但对 entropy/NLL 基线的 held-out AUROC 增量为负，delta-NLL R2 增量仅 0.00053。因此暂不能把 disagreement routing 作为已验证创新，必须等待 TinyStories 的相同诊断。

## 18. 2026-09-17 A1/A2 完成后的核心结论

- PTB token-normalized KD 的三 seed test PPL 为 51.2771/51.3745/51.2061，对应 matched CE 的 59.1175/59.6566/59.0819；平均改善 7.9994 PPL（13.49%），方向在三个 seed 上完全一致。
- TinyStories seed42 的 KD test PPL 为 5.4127，显著差于 CE 的 4.8611，恶化 11.35%。CE 同时优于 4.9691 的 teacher，而 KD 甚至差于 5.0654 的 warm-start checkpoint。
- 这组结果确认固定强度 KD 具有明确的 domain dependence。PTB teacher 强于 student，KD 有效；TinyStories teacher 被 CE student 超越，继续强制模仿产生负迁移。当前更合理的下一创新方向是 teacher-quality/confidence-aware KD 或受控 branch-aligned KD，而不是直接声称 JSD routing 已成立。
- PTB 的 JSD--gradient-conflict 关联稳定，但额外 held-out 预测增益接近零且分箱非单调。TinyStories branch diagnostic 尚缺，因此 CRBD 暂不通过机制闸门。

## 19. 2026-09-17 TinyStories 分支诊断与 Stage-B 决策

- TinyStories 2,048-chunk 正式诊断覆盖 522,240 tokens；平均 endpoint delta NLL 为 +0.1069，55.05% token 被 KD 损害，45.35% token 存在负 CE--KD gradient cosine。
- JSD 对 gradient cosine 的条件系数为 -0.0148，95% CI `[-0.0170,-0.0127]`；对 delta NLL 的系数为 +0.00615，CI `[0.00386,0.00836]`。加入 JSD 后，held-out AUROC/R2 分别增加 0.01091/0.00306。
- TinyStories 的 JSD decile 呈清晰风险梯度：harmful endpoint fraction 从最低分位的 37.97% 升至最高分位的 62.26%。该机制证据明显强于 PTB。
- Stage-B 按预注册文字标准通过，但属于弱通过：PTB 只有极小 R2 增量且 AUROC 略降。下一步若作者批准，只运行受限 Stage-C prototype，并必须保留 teacher-quality/confidence gating 与 branch-aligned KD 对照；不得直接宣称 CRBD 已得到验证。

## 20. 2026-09-20 Phase 2A teacher–student transfer landscape audit

- 在新增分析代码前冻结当前脏工作树至 `research/snapshots/20260920/`，记录 HEAD、status、tracked diff、未跟踪研究文件、活跃文件 SHA256、本地/197 环境与四个数据集 manifest；未 reset、clean 或 push。
- 以 NLL 为主量重建四域 transfer table。PTB 三种子均值的 `Delta_teacher/Delta_KD` 为 `+0.168082/+0.144937`；TinyStories seed42 为 `-0.021960/-0.107479`；WT2 与 common-5k Code 的 legacy 单种子值分别为 `+0.068047/+0.162579` 和 `+0.243051/+0.188309`。
- 在固定 validation token 上离线评估 Transformer、Grassmann、learned fusion 与 `alpha=0.0...1.0`。learned fusion 相对最佳分支的 NLL 改善为 PTB `0.146271`、WT2 `0.146483`、TinyStories `0.184168`、Code `0.304930`。
- canonical S0 mean CE–KD cosine 不能解释 KD 符号：WT2 为负但 legacy KD 改善，TinyStories 为正但 matched KD 退化。现有证据更支持 teacher residual advantage 作为待验证假设，而不是 gradient cosine 或 branch JSD 的简单规律。
- PTB 可构造 T_good/T_near 但无 T_bad；TinyStories 可构造 T_near/T_bad 但无 T_good。没有虚构缺失条件。
- 决策：domain dependence 仅 PARTIAL；teacher residual advantage 为 SUPPORTED AS HYPOTHESIS；融合预测价值为 YES、梯度证据仅 partial；disagreement mechanism 为 WEAK。下一项仅推荐、未启动的训练是 controlled teacher-quality alpha experiment。
- 全部产物位于 `research/experiments/phase2_teacher_transfer_audit/`；`AI_RESEARCH_HANDOFF.md` 已追加第 17 节。未修改论文、datasets、checkpoints 或历史输出。

## 21. 2026-09-22 Phase 2B controlled teacher intervention

- validation-only preflight 在同一 WikiText-2 frozen branch pair 内得到完整 triplet：alpha 0.5 的 T_good `Delta_teacher=+0.084900`，alpha 0.3 的 T_near `+0.000330`，alpha 0.0 的 T_bad `-0.266510`。
- C0/C1/C2/C3 使用完全相同的 seed-42 S0，checkpoint SHA256 为 `9e1b4f18f2873c9203b7cf8e7bc5750c272a399ca43d64e3f77adfb8868debef`。test NLL 分别为 4.270742/4.108241/4.139916/4.223972，因而 `Delta_KD` 为 0/+0.162501/+0.130826/+0.046771。
- KD gain 严格满足 T_good > T_near > T_bad，good-to-bad 差为 0.115731 NLL；但 T_bad 仍为正迁移，没有越过正负边界，故按预注册判为 PARTIAL。
- C1/C2/C3 epoch-1 clipping 为 0.5476/0.3946/0.5850，overflow/non-finite 均为 0.0136；初始 KD logit-gradient norm 仅相差约 3%。正式运行不满足 `LOSS_SCALE_CONFOUNDER` 条件。
- 决策：允许的下一步仅为 C0/C1/C3 在 seeds 123/456 的最小三种子确认，但本轮未启动。完整产物位于 `research/experiments/phase2b_controlled_teacher/`；未修改论文、datasets 或 checkpoints。

## 22. 2026-09-22 Phase 2C multiseed teacher utility

- 独立训练了 WikiText-2 Hybrid-lite S0 seeds 123/456，并与原 seed 42 组成三种子控制实验；三个 S0 checkpoint SHA256 均不同，结构、数据、优化器、预算与验证选择规则一致。
- C0/C1(T_a05)/C3(T_a00) 的 test NLL 分别为：seed 42 `4.270742/4.108241/4.223972`，seed 123 `4.277520/4.130183/4.243922`，seed 456 `4.280830/4.123673/4.238226`。
- 配对差 `D=NLL_C3-NLL_C1` 为 `0.115731/0.113739/0.114553`，均值 `0.114674`、样本标准差 `0.001001`，三种子方向一致。T_a00 的 KD gain 也全部为正，但明显弱于 T_a05。
- seed-matched residual advantage 在三种子上保持 T_a05 为正、T_a00 为负。全局 teacher 质量能解释收益强弱排序，但不能解释 T_a00 仍产生正迁移。
- T_a00 的全局 token utility 为负，但在每个 seed 的最高 student-loss quintile 上均为正，positive-utility fraction 均超过 0.54；全体 student-error token 的均值仍为负且 top-1 rescue 很低，因此只记为 conditional utility/complementarity signal，不声称机制。
- repository 中存在 fixed-composition 候选：alpha-only 与 joint-trained teacher 在固定 alpha 0.5、相同架构与相同 source branches 下形成明显质量差。最有信息量的下一项是该固定组成干预；Phase 2D 未启动。
- 完整报告与结果位于 `research/experiments/phase2c_multiseed_teacher_utility/`。本阶段未修改论文、datasets 或 checkpoints。

## 23. 2026-09-29 Phase 2D fixed-composition gate outcome

- 固定组成干预使用 joint Teacher J 与 alpha-only Teacher A，二者均在 effective alpha 0.5 下评估。完整性检查全部通过，179/191 个状态张量不同，差异来自 joint continuation 更新 branch parameters。
- 固定 validation 子集上 J/A fused NLL 为 4.300644/6.376220；joint training 改善两个 branch。Teacher A 虽全局 utility 显著为负，但三 seed 的 hardest-quintile utility 仍为正。
- 梯度预检的 `R_param` 为 1.103/1.149/1.093，无相对 KD-scale mismatch，不创建 scale-control。
- 2,000-line Teacher-A smoke 的 clip fraction 1.00、AMP overflow/non-finite 0.16，违反冻结门槛；正式 D2 未启动，不能报告 `Gain_A` 或 `Q`。
- 当前唯一允许的下一动作是负责人审查是否修订 smoke protocol。不得把 smoke test NLL 当正式 endpoint，不得据此回答 Teacher A 的迁移正负或 Phase 2D 七项最终决策。

## 24. 2026-09-29 Phase 2D fixed-composition completion

- 用户显式批准匹配的 full-data one-epoch J/A stability amendment。J/A clipping 为 0.234694/0.275510，overflow/non-finite 均为 0.013605；门控通过后才启动正式 D2。
- Teacher-A seeds 42/123/456 test NLL 为 4.293136/4.312882/4.306036，`Gain_A` 为 -0.022394/-0.035362/-0.025206；Teacher J 的复用 `Gain_J` 为 +0.162501/+0.147337/+0.157157。
- `Q=NLL_A-NLL_J` 为 +0.184895/+0.182699/+0.182364，均值 0.183319、样本标准差 0.001375，三 seed 完全一致支持 fixed-composition teacher training-state/quality modulation。
- preflight `R_param` 在 1.093--1.149，formal epoch-1 clipping 在 0.503--0.667，overflow/non-finite 均 0.013605；不是 material optimization-scale confound。
- joint training 改善两个 branch NLL 约 2.0，但 fusion gain 略降；branch complementarity 指标混合。Teacher A hardest-quintile utility 仍为正但 full endpoint transfer 为负。
- global teacher NLL 不是跨条件充分解释，因为 Phase 2C 的全球较弱 alpha-0.0 teacher 仍三 seed 正迁移。下一项仅选择 Transformer-only vs Grassmann-only vs fused-teacher KD，尚未启动。

## 25. 2026-09-29 Phase 2E teacher-branch gate outcome

- frozen joint teacher 的 F/T/G validation NLL 为 4.300646/4.445647/4.652064；S0-matched residual advantage 均值为 +0.074515/-0.070486/-0.276903。
- T/F 参数 KD 梯度比 1.232--1.316，预检不属于 scale mismatch。最难 student-loss quintile 上 F utility 高于 T 与 G，但仅是 validation observation。
- reduced T smoke 因 clip=1.00、overflow/non-finite=0.16 失败。matched full-data F/T gate 中 F 通过；T 的 overflow/non-finite 为 0.0136，但 clip=1.00，仍不满足预注册 clip<0.95。
- 三个 formal T arm 未启动，Phase 2E 不得视为完成，不能生成 `Gain_T`、`C_FT` 或七项最终决策。下一步需要负责人明确选择：保持冻结协议并终止，或在任何正式 test endpoint 产生前授权并记录新的 protocol amendment。

## 26. 2026-09-30 Phase 2E teacher-branch completion

- 用户在 formal endpoint 前显式授权 unchanged clipping amendment；必须永久保留 `CLIP_SATURATION_WARNING`。formal source state 为 `a94132b8...`，只运行 seed 42/123/456 的 alpha-1.0 T arms，未重跑 C0/F/G。
- T test NLL 为 4.127957/4.152294/4.144237。`Gain_F/Gain_T/Gain_G` 均值为 0.155665/0.134868/0.040991，三 seed 排序完全一致。
- `C_FT` 为 0.019716/0.022110/0.020564，mean 0.020797、sample SD 0.001214、3/3 positive，勉强越过 0.02 实质阈值；`C_FG` mean 0.114674，`C_TG` mean 0.093877。
- T epoch-1 clipping 三 seed 均为 1.0，但后续最低降至 0.0034；max overflow/non-finite 始终 0.013605，无 NaN/失败。结果不判 optimization-confounded，但 `C_FT` 解释必须保留早期 clipping 约束。
- 可支持 bounded claim：固定 joint teacher 下，加入 Grassmann branch 相对 Transformer-only 带来小而稳定的额外 KD gain。不能声称 Grassmann geometry 因果成立。下一项仅选择 homogeneous T+T ensemble teacher vs T+G teacher control，未启动。

## 27. 2026-09-30 Phase 2F homogeneous ensemble control

- T1 沿用历史 WT2 Transformer；T2 以 seed 123、相同 architecture/data/tokenizer/optimizer/batch/20 epochs 独立训练并按 validation NLL 选 epoch 12。T2 validation/test NLL 为 5.230674/5.288460。
- TT joint teacher 按授权协议训练，在统一强制 alpha=0.5 的 full validation 上得到 NLL 4.293589，TG 为 4.303697。TT 参数量 35,340,801，TG 为 37,749,633；二者不严格 parameter-matched，且 TT 少 6.38% 参数。
- TG 的 branch JSD/fusion gain 为 0.182087/0.147983，高于 TT 的 0.099793/0.112570，但下游结果相反，说明这些互补性指标不足以推出更好的 transferable KD signal。
- 新 TT KD seeds 42/123/456 test NLL 为 4.094289/4.118854/4.111574；复用 TG 为 4.108241/4.130183/4.123673。`Gain_TT` mean 0.168125，`Gain_TG` mean 0.155665；`H=Gain_TG-Gain_TT` mean -0.012460、sample SD 0.001348、3/3 negative。
- TT 早期 clipping 较高但不饱和，overflow/non-finite 与 TG 同量级且无 NaN。teacher validation NLL 差异方向一致，TT/TG alpha LR 又为 `1e-2`/`5e-3`，teacher quality/training 是主要混杂；参数量与 clipping 不提供 TT 获胜的明显正向解释。
- 决策：Phase 2F 否定 TG 优于 homogeneous TT 的假设。Grassmann-specific KD mechanism claim 必须删除，Phase 2E 的 fused gain 降为 generic ensemble supervision/teacher-quality 证据。Phase 2G 对保留该主张没有必要，未启动。

## 28. 2026-10-03 Phase 2G final boundary

- TinyStories warm-start token-mean KD 在冻结的 `lambda=5`、`T=2` 协议下跨 seeds 42/123/456 稳定负迁移。`Delta_KD` 为 -0.107479/-0.107439/-0.106740，mean -0.107219、sample SD 0.000415、3/3 negative，因此预注册决策为 `CONFIRMED`。
- teacher 比三个 matched CE endpoint 分别差 0.021960/0.021422/0.019714 NLL。该结果是 domain- and protocol-specific boundary，不是普适 KD 结论，也不是 Grassmann-specific mechanism。
- 所有 endpoint 均有限。KD clipping 明显高于 CE 但未饱和，max overflow/non-finite 为 0.000611。无效的不完整 GPU-3 seed-456 尝试被排除，仅使用授权后在 GPU 1 从头完成的 retry。
- 项目状态为 `experiments_frozen_manuscript_rewrite_next`。不得启动 Phase 2H；下一任务是基于已验证证据重构论文。

## 29. 2026-10-03 final evidence package

- 实验继续冻结，没有启动 Phase 2H 或任何训练。`research/final_evidence/` 成为论文重构的 authoritative evidence layer。
- 三域 confirmatory gain 为 PTB `+0.144937±0.003938`、WT2 `+0.155665±0.007692`、TinyStories `-0.107219±0.000415` NLL，每个数据集均 3/3 同号。CodeParrot 被单独标为 single-seed legacy observation。
- 最终四项主张围绕 transfer boundary、fixed-composition teacher state、global likelihood insufficiency 与 ensemble control。Phase 2F 正式删除 Grassmann-specific KD、Plücker causality 和 heterogeneity superiority。
- 旧 CAC tex 未修改；逐段审计、主张集、六组表格、十节蓝图与 GPT handoff 已生成。项目状态更新为 `manuscript_reconstruction`，等待明确授权后再写完整论文。

## 30. 2026-10-03 Phase 3A offline safe-distillation diagnostic

- 用户显式授权一次 post-freeze offline diagnostic extension；未运行 KD training、lambda sweep training，未写 checkpoint、读取 test split、改 endpoint、改论文或改 `research/final_evidence/`。
- protocol commit `80904d5` 在任何模型诊断前推送。21 个 seed-level 条件覆盖 PTB fused、WT2 J/A/T/G/TT 与 TinyStories fused；每项使用三个固定 validation calibration subsets，全参数 FP32 HVP，无 AMP 或 last-layer 近似。
- blind table 在 endpoint merge 前固化，SHA256 为 `594ac6d441fbe19776280225a0d652eb38baa7402dbf7a7295e065f7530fd9ad`；blinded commit 为 `ea86510`。
- 二阶 quadratic score 与一阶 validation dot 在 63/63 subset rows 上符号完全相同，seed-level sign accuracy/balanced accuracy/MCC/AUROC 为 0.571/0.650/0.279/0.644。WT2 J、G、TT 三个已确认正迁移 family 被误判为 harmful。
- curvature correction 的绝对值中位数仅 `2.73e-7`，为 first-order term 绝对值的中位数 0.52%，没有改变任何符号。仅 22/63 行存在 finite positive `lambda_safe`，范围 56.44--6429.64；41/63 为 no positive local safe interval。
- calibration sign 仅 11/21 seed conditions 稳定。nominal-LR virtual AdamW 识别全部 6 个 harmful seed rows，但误判 9/15 positive rows；literal formal first step 因 scheduler 初始化 lr=0 为严格 no-op。
- 最终决策为 `STOP_NEW_METHOD_DIAGNOSTIC_FAILED`，不是 `PROCEED_TO_PHASE3B` 或 `THEORY_NEEDS_REVISION`。不得启动 Phase 3B、Safe-KD、lambda tuning、Qwen 或自动 manuscript rewrite。

## 31. 2026-10-03 Phase 4A modern-family pilot authorization

- 用户授权独立的 external-validity pilot，限定 base SmolLM2-135M/360M、TinyStories 后 FineWeb-Edu、seed 42、lambda 1/5、T=2、每 run 10M predicted target tokens。旧实验和 final_evidence 继续冻结，论文不修改。
- Stage 0 数值单测 4/4 通过，两个官方模型和完全一致的 49,152-entry tokenizer 已核验哈希。微批 teacher CE/KD 峰值约 7.47/4.54 GiB；真实 TinyStories accumulated global-batch smoke 通过，峰值 5.06 GiB。
- Transformers 4.57.6 与 NVIDIA PyTorch 内部接口不兼容；仅在 outputs/phase4a_modern/python_deps 安装 4.46.3 和 tokenizers 0.20.3，原环境和框架源码均未改动。
- 正式训练必须在 protocol commit 后启动。seed123/456 只能建议、不自动运行；完成 pilot 后 STOP。

## 32. 2026-10-03 Phase 4A complete / replication review only

- TinyStories 与 FineWeb-Edu 各完成 teacher CE adaptation、独立 student S0 preparation、同一 S0 的 CE/KD1/KD5 continuation；10 个正式 run 各精确 10M predicted targets，总计 100M。验证集选 checkpoint 后才评估 test，所有完整性检查 PASS。
- TinyStories test Gain(lambda1/lambda5) 为 +0.004948632001/-0.004797389422；FineWeb-Edu 为 -0.015718598299/-0.046403663516。全部 validation/test 同号，只有 FineWeb lambda5 超过预注册 abs(gain)>=0.02 门槛；未发现达到门槛的现代正迁移。
- 决策 MODERN_REPLICATION_WORTHWHILE 仅由 FineWeb lambda5 的负迁移 pilot 支持，不是授权继续训练。只有一个 student preparation seed，不报 sample SD 或 3/3 robustness。现代 TinyStories 未清楚复现旧强负迁移边界。
- 两个 lambda5 arm clip_fraction 均为 1.0，永久保留 CLIP_SATURATION_WARNING。所有 run finite、observed overflow/nonfinite=0。不能忽略 clipping，也不能把 clipping 因果认定为负迁移原因。
- teacher validation residual advantage 为 TinyStories +0.173856356061、FineWeb +0.219626011745。FineWeb S0 选中 step256，但 preparation 仍跑满预算；CE endpoint 也略差于 S0。模型预训练包含 FineWeb-Edu，确切重叠未知，135M/360M 预训练预算不同。
- 原 HF 镜像分页跳转失败后，以 pinned official sample-10BT 首个 Parquet shard 的 byte-range streaming 完成同样 document-order prefix；20M/1M/1M token hashes 和 doc-ID/text-hash disjointness 已核验。只提交 compact artifacts，不提交 token、模型、数据正文或 wheels。
- 状态 modern_pilot_complete_replication_review_next。报告/交接/CSV/两组 PDF+PNG 位于 research/experiments/phase4a_modern_generalization_pilot/；final_evidence、旧论文、datasets/、checkpoints/ 未改。推荐负责人审阅固定 teacher 下独立 S0 seeds123/456 的 FineWeb replication；当前 STOP，未启动。

## 33. 2026-10-03 Phase 4B authorization and smoke

- 用户明确授权 FineWeb-only seeds123/456 的独立 S0 preparation 和 CE/KD1/KD5，共8个10M-target run；seed42/teacher/data直接复用，不改 clip/lambda/协议，不扩到 TinyStories。
- source-equivalence 测试证明新增 train.py 只替换 seeds/输出路径/CLI/共享模块引用，旧 train/core 未改。S0 preparation permutation 为对应 seed，continuation 固定4242，保持同 seed 三 arm matched。
- 7项单测通过，teacher/data/frozen-evidence/论文 hashes 已锁定；完整 batch32 的 disposable BF16 KD smoke 有限，峰值5.062GiB，无 checkpoint 保存/test access。
- 次级 best_including_S0 只用 validation 比较原 selected S0 与 trained selected endpoint，tie选step0，绝不替代主比较。协议须在正式训练前 commit/push。完成后 STOP。

## 34. 2026-10-04 Phase 4B confirmed bounded modern negative transfer / STOP

- Protocol commit 1a525c379d7b72e809847dfed6ef1bf71fd0fe50 在 formal 前推送。两个独立 S0 preparation 和六个 continuation 均完整完成，每 run10M targets，共80M；seed42、同一360M teacher、FineWeb token splits直接复用，无重训/retokenization。
- Gain1 seeds42/123/456 为 -0.015718598299/-0.015470071017/-0.009462713488，mean -0.013550460935、sample SD0.003542273400；Gain5 为 -0.046403663516/-0.046078763438/-0.039289819652，mean -0.043924082202、sample SD0.004016675497。两种强度均3/3 negative、全部validation/test同向。决策 FINEWEB_L1_L5_NEGATIVE_REPLICATED。
- Teacher residual validation/test mean为 +0.219866198612/+0.218703697246，但 CE 本身相对S0也3/3退化，test change mean +0.007918178008、SD0.004338019006。best_including_S0 的9/9 arm全部选原 adapted S0，诊断 Gain全为0，不替代主比较。必须明确这是进一步continuation本身不必要/有害的 regime。
- 三个lambda5 clipping均1.0，永久 CLIP_SATURATION_WARNING。lambda1 clipping为0.850123/0.854218/0.798526，虽非全饱和仍频繁受限。所有 observed nonfinite/overflow为0，不能归因clipping causality。
- CE456在step711出现一次finite norm86.482155，clip生效；selected CE是更早step256，因而saved weights不直接受该事件影响，但latertrajectory/selection影响未知。CE456_GRADIENT_SPIKE_WARNING保留，无改gate/exclusion/retry/新控制。
- S0 prep selected256/256/1221，CE selected1221/1221/256，所有KD selected1221。Final audit PASS：独立S0/order hashes、matched continuation order、预算、选择、telemetry、teacher/data/论文/final_evidence哈希均通过。旧Phase4A源码和锁定Phase4B train/adapter源码未改。
- 产物位于 research/experiments/phase4b_fineweb_multiseed_replication/，含完整报告/交接、主比较/step0/optimization/statistics CSV、raw compact telemetry及两组vectorPDF/300-dpiPNG图。状态 experiments_frozen_phase4b_complete；STOP，不运行任何clipping control或后续modern实验，不改论文。
