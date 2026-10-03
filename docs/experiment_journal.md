# 实验日志（experiment_journal）

说明：本文件为追加式记录。仅记录当前仓库实验，不混入其他项目。

---

## [2026-03-29] PTB warm-start 蒸馏首轮扫描（CE-only / KD0.1 / KD0.2）
- 日期：2026-03-29
- 实验名：`ptb_hybrid_lite_warmstart_{ce_only,kd01,kd02}`
- 改动点：
  - 在 `ptb_hybrid_lite_baseline_mild_e20` 初始化上做 warm-start；
  - 对比 `distill_alpha=0.0/0.1/0.2`。
- 执行命令：
  - `bash run_ptb_hybrid_lite_warmstart_distill_suite.sh`
- 关键日志：
  - `logs/ptb_hybrid_lite_warmstart_ce_only.log` -> `test_ppl=59.0784`
  - `logs/ptb_hybrid_lite_warmstart_kd01.log` -> `test_ppl=53.1334`
  - `logs/ptb_hybrid_lite_warmstart_kd02.log` -> `test_ppl=53.6545`
- 结果：
  - KD 显著优于 CE-only；
  - 但 kd0.1/0.2 仍未达到后续最佳水平。
- 结论：
  - warm-start KD 是正确方向，但蒸馏权重需要更细粒度下探。
- 下一步：
  - 扫更小 KD 权重（重点 0.05 附近）。

---

## [2026-04-02] PTB next-stage warm-start 与规模权衡
- 日期：2026-04-02
- 实验名：`ptb_hybrid_lite_warmstart_kd005` + 结构缩小变体
- 改动点：
  - 新增 kd0.05 / kd0.15；
  - 新增更小 student（224x56 l4、192x48 l4）并做 warm-start KD；
  - 增补 seed 复现（seed123/456）。
- 执行命令：
  - `bash run_ptb_next_stage_warmstart_suite.sh`
- 关键日志：
  - `logs/ptb_hybrid_lite_warmstart_kd005.log` -> `best_val_ppl=59.7016`, `test_ppl=51.8431`
  - `logs/ptb_hybrid_lite_warmstart_kd015.log` -> `test_ppl=53.0575`
  - `logs/ptb_hybrid_lite_warmstart_192x48_l4_kd005.log` -> `test_ppl=53.8719`
  - `logs/ptb_hybrid_lite_warmstart_224x56_l6_seed456_kd005.log` -> `test_ppl=51.7745`
- 结果：
  - kd0.05 成为当前最优 KD 点；
  - 更小模型（23.565M）PPL 有退化，但为延迟/吞吐换来更好 tradeoff；
  - seed 复现结果稳定。
- 结论：
  - “小 KD + warm-start” 已被验证；
  - 质量优先候选：31.434M kd0.05；效率优先候选：23.565M kd0.05。
- 下一步：
  - 在 0.05 附近继续精扫（0.03/0.07）并保持多 seed。

---

## [2026-04-02] PTB final CUDA supplement suite
- 日期：2026-04-02
- 实验名：`final_cuda_suite_20260402_145853`
- 改动点：
  - 构建/安装 CUDA 扩展；
  - 跑 `test_cuda_kernels.py` 和 `benchmark_cuda.py`；
  - 对最终三候选做端到端 benchmark（bs32）。
- 执行命令：
  - `bash run_ptb_final_cuda_supplement_suite.sh`
- 关键日志：
  - `logs/final_cuda_suite_20260402_145853.log`
  - kernel test 出现 state_dict key mismatch；
  - cuda micro-benchmark 的 Plucker 子项显示 `~8.37x`，但 grassmann layer benchmark 因循环导入失败；
  - 最终三候选（bs32）：
    - teacher `test_ppl=50.11`, `lat=54.82ms`
    - student-best `test_ppl=51.84`, `lat=48.57ms`
    - student-tradeoff `test_ppl=53.87`, `lat=34.90ms`
- 结果：
  - 端到端三候选指标完整；
  - CUDA 校验链路仍有阻塞（correctness/test 导入问题）。
- 结论：
  - 可先发布模型侧结果，但 CUDA 正确性结论需在修复后补完。
- 下一步：
  - 修复 `test_cuda_kernels.py` 的参数键不一致与 `benchmark_cuda.py` 导入路径问题。

---

## [2026-04-02] PTB final batch-size latency suite（bs=1/8/32）
- 日期：2026-04-02
- 实验名：`final_batch_latency_suite_20260402_150722`
- 改动点：
  - 对 teacher / student-best / student-tradeoff 在 `bs=1,8,32` 全量评测。
- 执行命令：
  - `bash run_ptb_final_batch_latency_suite.sh`
- 关键日志：
  - `logs/final_batch_latency_suite_20260402_150722.log`
  - bs1 延迟（ms）：teacher `15.48`，best `16.09`，tradeoff `11.50`
  - bs8 延迟（ms）：teacher `19.15`，best `19.33`，tradeoff `16.56`
  - bs32 延迟（ms）：teacher `55.98`，best `48.46`，tradeoff `34.96`
- 结果：
  - best student 在中大 batch 上快于 teacher 且 PPL 仅小幅退化；
  - tradeoff student 在所有 batch 下延迟优势最明显。
- 结论：
  - 最终可保留双候选策略：
    - 质量优先：224x56 l6 kd0.05
    - 速度优先：192x48 l4 kd0.05
- 下一步：
  - 输出统一结论表与选型建议（按在线/离线场景选择）。

---

## [2026-09-14] CAC manuscript reconstruction and evidence audit

- 类型：论文重构与已有产物审计；未运行训练实验。
- 输入版本：
  - `paper_cac_draft/conference_101719.tex`
  - `论文投稿/CICAI2026 AuthorKit/LaTex Template/samplepaper.tex`
  - `论文投稿/cac/conference_101719.tex`
- 主要操作：
  - 对照 `summary.json`、benchmark report、训练配置与模型/损失代码逐项核验论文数值和方法公式。
  - 删除无法溯源的 architecture ablation、配置不匹配的 PTB random-init 对照和不存在的 code warm-start `lambda=0.05` 点。
  - 将 5k/40k code 数据预算差异标为跨域混杂，不纳入主结论。
  - 重写目标稿并生成三张可复现图、独立参考文献库和 `revision_summary.md`。
- 验证：
  - `latexmk -pdf -interaction=nonstopmode -halt-on-error conference_101719.tex`
  - 产物：IEEE letter 双栏 6 页；17/17 引用、13 个交叉引用解析；无 overfull/undefined warning；所有字体嵌入。
- 训练与数据：未启动本地或远程训练；未修改 `datasets/`、`checkpoints/`。
- 下一步：执行多 seed、KL normalization、matched fusion 和 matched cross-domain data-budget 四项补充实验。

---

## [2026-09-15] CAC language editing

按用户要求深度润色 `论文投稿/cac/conference_101719.tex` 的学术英语，重点修正句法、冠词、指代、所有格和指标描述；将稿件修订历史改写为正式的实验范围说明。实验表格及数值、公式、引用、现有排版命令均保留，未添加列表或强调格式。通过润色前后结构比对及 `latexmk -pdf -interaction=nonstopmode -halt-on-error conference_101719.tex` 检查，并渲染检查 PDF。未新增实验或修改数据与检查点。

## [2026-09-15] CAC natural academic prose revision

按后续要求，将主稿全部 item 内容改写为连续段落，并移除正文强调格式和行内编号。仅修改明显重复、机械或模板化表达，保留大部分技术段落。比对确认引用与交叉引用、公式环境、表格内容及图件路径完全一致；正文无列表、加粗或斜体强调，无口语缩写或插入式长破折号。编译并渲染检查 6 页 PDF；未运行训练实验。

## [2026-09-15] Architecture figure v2

对照 `src/models/grassmann_v4.py`、蒸馏训练代码与当前方法章节设计新架构图，使用原生 SVG 路径和文本制作白底淡色双面板图，并导出独立 PDF/PNG。核对教师冻结、双分支 late-logit fusion、CE 接未温度化 logits、forward KL、两种初始化及逐特征门的隐藏状态输入。检查渲染后的标签和箭头；无新训练。新图未替换已投稿版本或主稿中的现有图件引用。

---

## [2026-09-15] CAC evidence re-audit and image1_1 integration

- 类型：论文/图件审计与主稿修订；未运行训练。
- 核对内容：`image1_1.png`、`conference_101719.tex`、`grassmann_v4.py`、teacher/student/KD 配置、summary JSON 和 batch-latency reports。
- 主要发现：random-init KD 和 CE-from-scratch 在学习率及部分数据集的 epoch 数上不匹配；WikiText-2 teacher/source-branch 运行的数据处理统计不一致。
- 论文修订：将核心主线改为四数据集 warm-start KD 系数扫描；将 random-init 数据降为探索性观察；替换 RQ1 对照；补全方法公式；加入两篇经 ACL Anthology/PMLR 核验的相关文献。
- 图件：将 `论文投稿/cac/figures/image1_1.png` 以 `0.94\textwidth` 替换主架构图，修订图注以弥补多窗口聚合与残差子层的视觉省略。
- 验证：`latexmk -pdf -interaction=nonstopmode -halt-on-error conference_101719.tex`；产物为 IEEE letter 双栏 6 页，无 overfull/undefined/duplicate-label 问题，所有 PDF 字体嵌入。
- 后续最高优先级：四数据集主结果的多 seed 重复，以及完全匹配的 random-init CE/KD 对照。

---

## [2026-09-16] Post-rejection repository audit and experiment reset

- 类型：只读结果审计、研究状态初始化与非破坏性目录整理；未运行训练。
- 检查范围：核心模型、teacher/baseline/KD/benchmark 入口、37 个 shell suite、186 个输出运行目录、benchmark reports、日志体量和远程执行 wrapper。
- 代码检查：51 个 Python 文件通过 AST parse；37 个顶层 shell 脚本通过 `bash -n`。
- 产物登记：生成 188 行标准化结果记录，覆盖 159 个完整运行、25 个 config-only 运行和 2 个无完整 JSON 的运行。
- 关键发现：CodeParrot 元数据继承错误默认标签；legacy KD 的 KL 使用按 token 求和的 `batchmean`，使跨域 alpha 不可直接比较；random-init 对照与 fusion 变体仍存在配置混杂。
- 研究决策：先执行 H0 token-normalized KD 与完全匹配 controls，再以 TinyStories 负迁移为边界测试 H1 branch-disagreement-aware KD。
- 远程状态：通过 `/home/xin/fuwuqi/agent-tools/exec_grassmann.sh` 检查时，`/songxin` NFS 未挂载且无法自动 sudo 挂载，未启动 smoke 或正式任务。
- 数据与 checkpoint：未修改 `datasets/`、`checkpoints/` 或任何既有输出。

## [2026-09-16] Remote workspace handoff verification

- 阅读工作区级交接文件 `/home/xin/fuwuqi/CODEX_WORKSPACE_HANDOFF.md`，确认 Grassmann 的标准拓扑为：本机 NFS 路径编辑，197 上 `grassmann_lab` 容器执行，共享项目路径 `/workspace/grassmannflows/grassmann-flows`。
- 通过 `/home/xin/fuwuqi/agent-tools/exec_grassmann.sh` 完成只读验证。
- 环境：Python 3.12.3，PyTorch 2.6.0a0+cu126，4 张 RTX 3090；GPU 1--3 基本空闲，检查时无训练进程。
- 共享一致性：远端可见 159 个 `summary.json`、184 个 `config.json`，并可见本轮新增的 `docs/project_map.md`。
- 结论：此前 NFS 阻塞已解除，可以开始 H0 实现与 remote smoke；本轮验证未启动训练。

## [2026-09-16] H0 token-normalized KD PTB smoke

- 类型：损失实现、单元测试、PTB 一轮 smoke 与梯度诊断。
- 实现：新增 `src/kd_losses.py`，保留 legacy batchmean 目标并加入 valid-token mean 目标；训练入口新增显式 loss provenance 与双尺度 KL 日志。
- 单元测试：legacy 公式一致性、序列缩放关系、ignore mask 和相同 logits 零 KL 共 4 项，197 容器内全部通过。
- 固定控制：同一 teacher、同一 224x56 六层 warm-start、seed 42、seq 256、batch 32、lr `1e-4`、temperature 2、一轮。
- 结果：CE-only / legacy-a0.02 / token-l2.5 / token-l5 / token-l10 的 val PPL 分别为 69.31 / 61.69 / 62.88 / 61.73 / 61.35；test PPL 分别为 59.04 / 53.26 / 54.08 / 53.29 / 53.13。
- 稳定性：token-l10 的 clip fraction 为 0.547，未通过门槛。legacy 与 token-l5 诊断复跑的 finite grad mean 为 0.7874/0.7872，nonfinite 及 AMP overflow fraction 均为 0.014。
- 决策：选择 `lambda_kd=5`。在 255 个预测位置下，legacy `alpha=0.02` 的等效 token coefficient 为 5.204，实测结果支持实现一致性。
- 运行异常：首次 legacy arm 在已有 3.7 GiB 占用的 GPU 0 上 OOM；脚本改为仅使用 GPU 1--3 并自动跳过已完成 arm，重试成功。未删除失败产物。
- 关键产物：`research/experiments/h0_kd_normalization/analysis.md`、`run_h0_kd_normalization_smoke.sh`、`run_h0_kd_gradient_diagnostic.sh`。
- 下一步：PTB/TinyStories 三种目标、三个 matched seeds 的确认实验；先核对 TinyStories teacher/init/data provenance，再启动正式任务。

## [2026-09-16] ARS post-rejection staged experiment plan

- 类型：研究缺陷诊断、创新边界检索与预注册实验设计；未运行训练。
- 使用流程：`academic-research-suite` 的 academic-pipeline 与 experiment-agent plan 模式。
- 研究判断：固定长度下的 token normalization 不能消除 TinyStories 负迁移；下一步需要先验证 teacher 分支 JSD 是否预测 CE--KD 梯度冲突。
- 方法候选：Consensus-Routed Branch Distillation，在低分歧 token 使用 fused KD，在高分歧 token 使用架构匹配的双分支 KD；CE 始终保留，推理结构不变。
- 新颖性边界：将 entropy/confidence gating、disagreement suppression、branch-only KD、shuffled routing 和 swapped pairing 纳入基线或机制消融，不把它们单独声明为创新。
- 计划闸门：A1 PTB matched 3-seed；A2 TinyStories seed42 matched；B branch-disagreement diagnostic；C bounded prototype；D confirmatory multi-seed。推荐首批预算约 25 GPU-hours。
- 产物：`research/experiments/post_rejection_program/experiment_plan.md`。
- 状态：等待作者确认 A1+A2+B；没有执行计划中的命令，也没有修改 `datasets/`、`checkpoints/` 或历史输出。

## [2026-09-16] A1/A2 launch and B diagnostic implementation checkpoint

- 作者已确认执行 A1+A2+B。
- 远端预检：所有冻结 teacher、student initialization 和数据路径存在；GPU 1--3 空闲，GPU 0 仍有外部进程占用约 3.7 GiB。
- 已启动 A2 TinyStories seed42 的 CE-only 与 token `lambda=5` 两臂，分别使用 GPU 1/2；两臂进入 epoch 1，预计每臂约 11--12 小时。
- 已在 GPU 3 启动 A1 PTB seed42 CE-only。前两轮分别耗时 243.5/254.7 秒，显著超过计划估计，10 epochs 的预计运行时间约 41--43 分钟，因此存在 30 分钟硬超时风险。
- B 实现新增 branch-logit 可选返回、compact token diagnostic、chunk-grouped CV/chunk bootstrap 分析与测试。默认模型 forward 返回保持不变，尚未运行正式诊断。
- 首次 B 单元测试在 pytest collection 阶段失败：远端分片环境未将项目根目录加入 `sys.path`，无法导入 `src` 与顶层分析模块。按实验规范未自动修补或重试，等待作者决定。
- 当前未修改 `datasets/`、`checkpoints/` 或历史输出；A1/A2 运行目录保留在 `outputs/distill_experiments/`。

## [2026-09-16] A1 timeout extension and pre-training path failure

- 作者确认 GPU 0 外部任务已结束，并同意延长 PTB 运行上限。远端复检显示 GPU 0/3 均为 10 MiB、0% 利用率；TinyStories CE/KD 继续在 GPU 1/2 满载运行，未作改动。
- 使用 50 分钟硬上限并行启动 PTB seed42 CE 与 token `lambda=5`，但两臂均在 student initialization 加载前退出，退出码为 1，未进入任何训练 step。原因是计划中的 `20260402_135900_ptb_hybrid_lite_baseline_224x56_l6_seed42_e20` 目录不存在。
- 通过历史 CE 日志核对，seed42 的实际 matched initialization 为 `outputs/hybrid_experiments/20260328_142654_ptb_hybrid_lite_baseline_mild_e20`；seed123/456 的初始化目录仍分别为 `20260402_140126...` 与 `20260402_140326...`。未自动第三次启动，等待作者确认路径修正。
- B 测试仅加入项目根目录解析后复测。顶层分析模块已可导入，但 `src.branch_diagnostics` 会先执行 `src/__init__.py`，随后因其中的 `from . import data` 产生 unavailable/partially initialized module 错误。测试仍在 collection 阶段失败；按实验协议未继续自动修复或重试。
- 本检查点未修改 `datasets/`、`checkpoints/`，也未删除不完整或失败产物。

## [2026-09-16] A1 seed42 verified relaunch and B test pass

- 作者确认使用核验后的初始化路径重启实验并继续修复 B。配置比较确认 seed42/123/456 的模型维度、层数、窗口、dropout、late-k、20-epoch warm start、学习率、数据路径与 split seed 一致，仅训练随机种子不同。
- PTB seed42 CE 与 token `lambda=5` 分别在 GPU 0/3 进入 epoch 1，运行名为 `h0_ptb_confirm_token_l0_seed42_retry50m_v2` 与 `h0_ptb_confirm_token_l5_seed42_retry50m`，硬上限均为 50 分钟。
- `src/__init__.py` 原先 eager import 仓库中不存在的 `src.data`，使所有 `src.*` 测试在 collection 阶段失败；现已移除该无效导入，不改变模型或训练入口。
- endpoint 测试样例原先让 KD endpoint 优于 CE endpoint，却断言 `NLL(KD)-NLL(CE)>0`。保持预注册符号定义不变，仅修正样例标签。远端完整测试最终为 `10 passed in 4.61s`。
- A2 TinyStories CE/KD 继续在 GPU 1/2 运行；本轮未修改 `datasets/`、`checkpoints/` 或既有实验产物。

## [2026-09-17] A1 partial completion and PTB branch diagnostic

- A1 的三个 CE endpoint 已全部完成。seed42/123/456 的 best-val epoch 均为 1，test PPL 分别为 59.1175、59.6566、59.0819；梯度裁剪、非有限梯度与 AMP overflow 比例均为 0。
- token `lambda=5` 的 seed42/123 已完成，best-val epoch 均为 10，test PPL 分别为 51.2771、51.3745。相对 matched CE 分别改善 7.8405 与 8.2821 PPL。两臂最大裁剪比例约 0.19--0.20，最大非有限梯度与 AMP overflow 比例均为 0.0144，与 H0 smoke 的首批次现象一致。seed456 KD 仍在 GPU 3 运行。
- B 首次 PTB 32-chunk smoke 完整结束，但半精度下的闭式 CE 范数发生抵消，使 gradient cosine 超出理论区间，且 JSD 有轻微负舍入误差。保留该产物后，改为显式 CE logit-gradient 向量计算范数，并对 JSD/余弦施加理论边界；新增饱和 logits 测试，远端 11 项测试全部通过。
- 修正后的 32-chunk smoke 使用相同抽样索引及 SHA-256，保存 8,160 个紧凑 token 统计，不含原始文本或完整 logits。JSD 位于 `[0,0.604]`，gradient cosine 位于 `[-0.996,0.999]`。
- PTB 正式 B 诊断覆盖全部 350 个 validation chunks、89,250 tokens。KD endpoint 的平均 `delta_nll` 为 -0.1555，但 40.34% token 的 `delta_nll>0`；负梯度余弦比例为 36.91%。
- 2000 次 chunk bootstrap 中，控制 teacher entropy/NLL 后的 JSD--gradient-cosine 系数为 -0.0212，95% CI `[-0.0274,-0.0147]`；JSD--delta-NLL 系数为 0.0067，95% CI `[-0.00005,0.0127]`。held-out conflict AUROC 增量为 -0.00032，delta-NLL R2 增量为 0.00053。PTB 支持冲突关联，但独立预测增益很弱；H1 决策等待 TinyStories。
- A2 TinyStories CE/KD 继续在 GPU 1/2 运行；未修改 `datasets/`、`checkpoints/`，所有失败与 smoke 产物均保留。

## [2026-09-17] A1/A2 final results audit

- 远端四张 GPU 均已空闲，A1 六个 PTB summary 与 A2 两个 TinyStories summary 全部存在；所有正式训练均按 validation PPL 选择 best checkpoint。
- PTB 三 seed CE test PPL 为 59.1175/59.6566/59.0819，token `lambda=5` 为 51.2771/51.3745/51.2061。CE 与 KD 的均值分别为 59.2853±0.3220 和 51.2859±0.0846；平均 paired delta 为 -7.9994±0.2454，三个 seed 方向一致，按均值计算改善 13.49%。
- TinyStories seed42 CE/KD test PPL 为 4.8611/5.4127，KD 恶化 0.5516（11.35%）。warm-start 与 teacher 分别为 5.0654/4.9691，CE 最终超过两者，而 KD 低于两者。
- PTB teacher 明显优于 CE student，KD 将 student 拉近 teacher；TinyStories CE student 最终优于 teacher，而固定 `lambda=5` 阻碍 student 超越 teacher。跨域符号更直接地受 teacher 相对质量与 KD 强度解释，不能归因于 normalization 或 branch disagreement 单一因素。
- TinyStories KD 的首轮 clip fraction 为 0.588，之后降至 0.074；非有限/overflow 比例小于 0.0007，且 CE 也有更小比例，因此运行有效但 KD 强度偏激进。
- B 仅完成 PTB 正式诊断；TinyStories branch diagnostic 尚未运行。当前证据不通过 CRBD 机制闸门，不应直接启动正式 CRBD 训练。
- 结果表与结论分别固化为 `research/experiments/post_rejection_program/results.csv` 和 `analysis.md`。

## [2026-09-17] TinyStories Stage-B completion

- 按预注册协议先运行 32-chunk smoke。8,160 个 token 的统计均有限，JSD 与 gradient cosine 位于理论范围；固定 selection seed 可重建完全相同的 indices 与 SHA-256。manifest 确认不保存原始文本或完整 logits。
- 正式诊断使用 2,048 个固定 validation chunks、522,240 个有效 token。mean `delta_nll` 为 +0.10693，55.05% token 的 KD endpoint NLL 高于 CE，45.35% token 的 CE--KD gradient cosine 为负。
- 控制 teacher entropy/NLL 后，JSD--gradient-cosine 系数为 -0.01481，95% chunk-bootstrap CI `[-0.01696,-0.01269]`；JSD--delta-NLL 系数为 +0.00615，CI `[0.00386,0.00836]`，均符合预注册方向。
- 加入 JSD 后，held-out conflict AUROC 从 0.54707 提升到 0.55798（+0.01091），endpoint delta-NLL R2 从 0.04218 提升到 0.04524（+0.00306）。最高 JSD decile 的 harmful endpoint fraction 为 62.26%，最低 decile 为 37.97%。
- 跨域 Stage-B 字面闸门通过：两域系数方向一致，且每域至少一个 held-out metric 改善。但 PTB 增量信号极弱，因此仅授权进入 bounded Stage-C prototype 的科学判断，不构成 CRBD 有效性结论。Stage C 尚未启动。
- 新产物：`research/experiments/h1_branch_disagreement/tinystories_full_seed42_2048/` 与 `cross_domain_summary.csv`。未修改 datasets/checkpoints。

## [2026-09-17] Historical-output audit and Stage-C launch

- 对 `outputs/` 进行了只读再审计：203 个运行目录对应 205 条标准化记录，其中 174 complete、27 config-only、2 incomplete。历史结果没有 branch-disagreement gating、branch-aligned KD、CRBD、shuffled routing 或 swapped pairing，因此 Stage C 不重复旧实验。
- 可直接复用的主证据仍是 PTB 三种子质量/效率结果和 batch-size 延迟。历史 WikiText-2 与 common-5k Code 系列仅有 seed 42；旧 Code full-data/PTB-teacher 系列与 5k Code-teacher 系列不能混合。历史 subspace 输出含 hook-bug 全零版本，且验证脚本对 rank 方向有冲突假设，暂不作为机制证据。
- 新增 `src/crbd_losses.py` 与六项 CRBD loss 测试；训练入口新增显式 `distill_strategy`、branch coefficient、routing temperature/seed 及相应 provenance/metrics。默认 `fixed_fused` 路径保持不变。远端 KD、branch diagnostic、CRBD 共 15 项测试全部通过。
- PTB 2k-line 一轮 smoke 完成：C1 fixed、C5 CRBD、C6 shuffled、C7 swapped 的 validation PPL 分别为 69.2466、69.0009、68.9979、70.5518；所有 checkpoint/summary 完整。C5 与 C6 在极小 smoke 上不可区分，swapped 明显较差。
- TinyStories 20k-line、2-epoch 原型已启动。完成的 C0 CE validation/test PPL 为 5.0183/5.0195，C1 fixed 为 5.3041/5.3018，因此 50% repair 门槛为 C5 validation PPL 不高于 5.1612。C2/C4/C5/C6 仍在运行。
- GPU 3 在四卡并发负载下持续被软件功耗限制至约 210 MHz。该卡上的 C3 两次尝试均被保留为不完整基础设施记录，未用于结果；C3/C7 将迁移到正常 GPU 后以完全相同配置重跑。
- 新产物：`docs/historical_output_audit_20260917.md`、`research/experiments/h2_crbd/protocol.md`、`run_stage_c_ptb_smoke.sh`、`run_stage_c_tinystories_prototype.sh`、`run_stage_c_ts_gpu3_retry.sh`。
- 未修改 `datasets/`、`checkpoints/` 或任何历史输出。

## [2026-09-18] Stage-C TinyStories prototype audit

- 完成 C0/C1/C2/C4/C5/C6/C7；C3 两次均因 GPU 3 锁频被外部终止，只有 config，不计入方法结果。七个完整 arm 的 teacher、warm-start、seed、数据统计、batch、epoch 与学习率完全匹配。
- validation/test PPL：C0 CE 5.0183/5.0195；C1 fixed 5.3041/5.3018；C2 entropy 5.3027/5.3008；C4 branch-only 5.6135/5.6100；C5 CRBD 5.5104/5.5067；C6 shuffled 5.5045/5.5010；C7 swapped 6.3668/6.3616。
- fixed KD 相对 CE 的 validation excess 为 0.2858，CRBD excess 为 0.4921，repair fraction 为 -0.7220，未达到至少 50% 修复要求。C5 比 C6 差 0.0059，但比 C7 好 0.8563。PTB smoke 的 no-worse 门槛通过，TinyStories 修复与 shuffled-control 门槛失败，因此 Stage C 总闸门失败。
- 完整运行的 non-finite/overflow 比例低于 0.23%，但所有 KD arm 两轮的 clip fraction 均为 1.0，CE 平均仅 0.0135。C5 epoch 2 的 weighted fused/branch contributions 约为 2.77/3.53，均叠加在 CE 1.73 上；分别 mean-one normalization 使 routing 不是 fixed-budget mixture，构成明确的 loss-scale confound。
- 决策：不启动 Stage D。若继续，先补 C3，再将 fused/branch routing 改为共享总 KD budget，并以 C1 初始 loss 或梯度范数匹配；不继续当前 separate-normalization 下的 lambda grid。
- 固化产物：`research/experiments/h2_crbd/results.csv` 与 `research/experiments/h2_crbd/analysis.md`。

## [2026-09-20] Phase 2A checkpoint-only transfer audit

- 执行前建立 `research/snapshots/20260920/`。源状态为 `main@67efbc158ad823f7196f0696415f6e32b5e2e2fa` 加大量未提交研究文件；本轮未清理该状态。
- 使用现有 teacher、S0 warm-start 与 S1 WS+CE checkpoints，在 197 四张 RTX 3090 上仅运行 inference。PTB 覆盖完整 validation 350 chunks/89,250 targets；WT2、TinyStories、Code 各用 seed 20260920 固定抽取 512 chunks/130,560 targets。
- 验证融合代码为 `alpha * Transformer logits + (1-alpha) * Grassmann logits`。四个 teacher 均为 last-1；mixed-precision 离线重构最大绝对误差为 0.015625--0.03125。
- 四域 learned fusion 均优于最佳单分支，并均降低相对 S0 的 KL；但相对最佳分支的 S0 gradient cosine 在 Code 上下降，不能宣称普遍改善优化兼容性。
- P1--P7 与现有 `Delta_KD` 仅作 n=4 描述。P1 residual advantage 在四点上符号一致；P3 S0 cosine 未分离正负迁移；P6/P7 与 Stage-B 一致，不支持把 disagreement 当作已验证机制。
- 代表性 token aggregate：fusion 在 Grassmann gold probability 更高的条件下修正 Transformer errors 的计数为 PTB 3,760、WT2 4,621、TinyStories 7,080、Code 7,033；相反 harm 计数为 1,667、3,196、4,260、3,670。该结果是 branch complementarity signal，不是几何因果证明。
- 生成 `REPORT.md`、四张主 CSV、candidate/provenance 文档、附加比较表、raw JSON、运行日志及 PDF/300-dpi PNG 图。没有启动推荐的下一训练实验。

## [2026-09-22] Phase 2B controlled teacher intervention

- 按 Phase 2A 的唯一建议完成 validation-only preflight。WikiText-2 在同一 frozen teacher branch pair 内覆盖 T_good/T_near/T_bad，选择 alpha 0.5/0.3/0.0；所有选择在正式训练前冻结，未使用 test 指标。
- 四臂使用 seed 42、同一 S0 SHA256、相同数据顺序策略与优化配置。C0 CE 的 val/test NLL 为 4.383154/4.270742；C1 good 为 4.216686/4.108241；C2 near 为 4.250850/4.139916；C3 bad 为 4.340190/4.223972。
- `Delta_KD` 严格按 `+0.162501 > +0.130826 > +0.046771` 排列。预注册结论为 PARTIAL：教师 residual advantage 降低时 KD gain 下降，但 bad teacher 没有造成负迁移。
- 正式 epoch 1 中三个 KD arm 均无 clip saturation，overflow/non-finite 均为 1.36%，initial KD logit-gradient norm 近似一致；未标记 `LOSS_SCALE_CONFOUNDER`。GPU3 上的首次 C3 因锁频中止，随后不改配置迁移到 GPU0 完成，旧 incomplete 目录和日志保留。
- 生成 `results.csv`、`REPORT.md`、provenance、四臂 raw artifact copy 与两组 PDF/300-dpi PNG 图。结论支持只复现 C0/C1/C3 的 seeds 123/456，但遵循停止指令，未启动下一阶段。

## [2026-09-22] Phase 2C multiseed teacher utility completion

- canonical seed 42 之外，使用完全相同的 WikiText-2 Hybrid-lite CE 配方独立训练 S0 seeds 123/456。validation-best checkpoint 分别来自 epoch 13/14，test NLL 为 4.261903/4.260188，SHA256 为 `a44b64ba...1f13` 与 `3fbe985d...1757`。
- seeds 123/456 各完成 C0 WS+CE、C1 T_a05、C3 T_a00。三 seed 的 `Delta_KD_a05` 为 0.162501/0.147337/0.157157，`Delta_KD_a00` 为 0.046771/0.033598/0.042604，`D` 为 0.115731/0.113739/0.114553；全部方向一致。
- 新四个 KD arm 的 epoch-1 clipping 为 0.367--0.497，overflow/non-finite 均为 0.0136，initial KD logit-gradient norm 与 seed 42 同量级；无优化失配标记。
- 离线 utility audit 使用固定 512 validation chunks。T_a00 全局 utility 与 student-error partition 均为负，但最高 student-loss quintile 在三 seed 上的 mean utility 均为正，positive fraction 均超过 0.54。top-1 rescue 很低，因此判为 PARTIAL conditional-utility evidence。
- fixed-composition audit 找到 alpha-only 与 joint-trained 两个自然 checkpoint，二者可在固定 alpha 0.5、相同架构与 source branches 下形成未来质量干预。未启动 Phase 2D。
- collector 验证九个 arm 的配置、S0 哈希、有效 alpha 与固定 selection hash 后生成 `results_multiseed.csv`、两张 utility CSV、raw summaries 与图。总报告为 `research/experiments/phase2c_multiseed_teacher_utility/REPORT.md`。

## [2026-09-29] Phase 2D fixed-composition preflight and smoke stop

- 按冻结方案比较 joint Teacher J 与 alpha-only Teacher A，二者均强制 effective alpha 0.5。完整性门槛确认 teacher/source 哈希、架构 schema、state keys/shapes 与融合语义匹配，并复现 179/191 个张量不同。
- 同一 512-chunk WikiText-2 validation 子集上，J/A fused NLL 为 4.300644/6.376220。joint training 同时改善 Transformer 与 Grassmann branch NLL；J 的 branch JSD 略高、top-1 agreement 略低，但 fusion gain margin 略小。
- 三个 S0 的 full-parameter KD-gradient ratio `G_A/G_J` 为 1.102792/1.148522/1.093056，未触发 `KD_SCALE_MISMATCH`，因此未创建 D2-scale。
- Teacher A 全局 utility 在三个 seed 上均约为 -2，但最高 student-loss quintile utility 均为正，positive fraction 为 0.597--0.622；该结果仅为 validation observation。
- seed-42、2,000-line、1-epoch Teacher-A smoke 的 CE/raw KL/weighted KD 为 3.835580/1.114552/5.572758。checkpoint 可重载且 alpha/S0 hash 正确，但 clip fraction=1.00，overflow/non-finite=0.16，未通过预注册 `<0.95` 与 `<0.05` 门槛。
- Phase 2B 的同类 2,000-line Teacher-J smoke 也曾得到 clip=1.00、overflow/non-finite=0.16，而后续 full-data formal epoch 1 为 0.5476/0.0136；因此当前失败可能是 25-step AMP calibration window 的系统性现象，但不能据此绕过新冻结门槛。
- 遵循停止规则，未启动 seeds 42/123/456 正式 D2，未生成 endpoint 图，未修改论文、datasets、历史 checkpoints、lambda 或 AMP 设置。下一步需要研究负责人决定是否显式修订 smoke 协议。

## [2026-09-29] Phase 2D authorized amendment and formal completion

- 研究负责人显式批准将稳定性观察窗修订为同一 seed-42 S0、同一正式配置的 full-data one-epoch J/A 配对门控；原 2,000-line smoke 失败记录保留，不被覆盖。
- 修订门控中 J/A clipping fraction 为 0.234694/0.275510，AMP overflow 与 non-finite fraction 均为 0.013605；teacher、alpha=0.5、S0 hash、目标函数和 checkpoint 重载检查全部通过。
- 不重跑 Phase 2C C0/C1，仅新增 Teacher-A D2。seeds 42/123/456 的 Teacher-A test NLL 为 4.293136/4.312882/4.306036；相对 C0 的 `Gain_A` 为 -0.022394/-0.035362/-0.025206，三 seed 均为负迁移。
- 对应 Teacher-J `Gain_J` 为 +0.162501/+0.147337/+0.157157；主对比 `Q=NLL_A-NLL_J` 为 +0.184895/+0.182699/+0.182364，均值 0.183319、样本标准差 0.001375，三 seed 方向一致。
- formal Teacher-A epoch-1 clipping 为 0.666667/0.503401/0.513605，overflow/non-finite 均为 0.013605；结合 preflight `R_param=1.103/1.149/1.093`，结论不受明显 KD gradient-scale mismatch 混杂。
- joint training 同时显著改善 Transformer 与 Grassmann branch NLL；JSD 和 agreement 暗示分歧略增，但 fusion gain 从 0.164420 降到 0.145003，故主要变化是 branch quality，而不是 fusion-gain margin 增强。
- 决策：fixed-composition teacher-state effect 为 YES，Teacher-A positive transfer 为 NO，cross-seed robustness 为 YES，material scale confound 为 NO，global NLL sufficiency 为 NO。下一项仅选择 option A（Transformer-only vs Grassmann-only vs fused-teacher KD），未启动。
- 生成严格 collector、三组 compact raw formal artifacts、`results_multiseed.csv`、三张 PDF/300-dpi PNG 配对图、完整报告与 GPT handoff；未修改论文、datasets 或历史 checkpoints。

## [2026-09-29] Phase 2E teacher-branch preflight and stability-gate stop

- 固定 joint teacher checkpoint，通过 effective alpha 0.5/1.0/0.0 构造 F/T/G。相同 512-chunk validation 子集上的 teacher NLL 为 4.300646/4.445647/4.652064；三个 S0 上的平均 residual advantage 为 +0.074515/-0.070486/-0.276903。
- T/F full-parameter KD-gradient ratio 为 1.3164/1.2788/1.2324，均在预注册 `[0.5, 2.0]` 范围内，不触发 `KD_SCALE_MISMATCH`。
- token 条件分析显示 T/G 分别在 51.56%/48.44% token 上给 gold token 更高概率；最难 student-loss quintile 的 T/G/F utility 为 0.6356/0.2133/0.7396。这是观测性互补信号，不构成几何因果证据。
- reduced T smoke 复现短窗口伪影：clip=1.00、overflow/non-finite=0.16。预授权的 full-data one-epoch F/T 门控中，F 为 0.2347/0.0136/0.0136 并通过；T 的 overflow/non-finite 降至 0.0136，但 clip 仍为 1.00，违反冻结的 `<0.95` 条件。
- 按“门控通过后才运行 formal”与“不得事后另造规则”的要求，未启动三个正式 T endpoint。Phase 2E 当前为 gate-blocked，等待负责人决定严格终止或显式批准新的前瞻性 protocol amendment；smoke NLL 不得作为正式结果。

## [2026-09-30] Phase 2E authorized completion

- 负责人在任何 formal endpoint 产生前明确授权 clipping amendment：保留 alpha=1.0、lambda=5、T=2、AMP、clip value=1.0、预算、数据、S0 与评估不变，并永久标记 `CLIP_SATURATION_WARNING`。修订先以 Git 状态 `a94132b8...` 固化，再启动三个 formal T arm。
- seed 42/123/456 的 T test NLL 为 4.127957/4.152294/4.144237；均在 epoch 10 得到 validation-best checkpoint。全程无 NaN 或训练失败，max overflow/non-finite 均为 0.013605。
- `Gain_F` 均值 0.155665、`Gain_T` 0.134868、`Gain_G` 0.040991，三个 seed 均满足 F>T>G>0。
- 主对比 `C_FT` 为 0.019716/0.022110/0.020564，均值 0.020797、sample SD 0.001214、3/3 positive，刚超过预注册 0.02 阈值。`C_FG` 均值 0.114674，`C_TG` 均值 0.093877，均为 3/3 positive。
- T 在 gate 与 formal epoch 1 均 clip=1.00，随后降至最低 0.0034；mean epoch clipping 为 0.4990/0.4605/0.4398。终点数值有效，但早期优化约束可能影响较小的 `C_FT`，因此 warning 不得删除。
- 结论：fused 相对 T 与 G 均稳定更优，T/G 均提供正迁移；加入 Grassmann branch 在该协议下带来小而稳定的额外 gain，但不能归因于 Grassmann geometry。唯一选择且未启动的下一实验是 homogeneous Transformer+Transformer ensemble teacher control。

## [2026-09-30] Phase 2F homogeneous ensemble control completion

- 从历史缓存恢复与 T1 完全一致的 WikiText-2 raw 数据，token/chunk 统计逐项复现；未修改项目 `datasets/` 或 `checkpoints/`。独立 seed-123 T2 在相同 20-epoch 协议下选中 epoch 12，validation/test NLL 为 5.230674/5.288460。
- 以最小兼容改动增加显式 `teacher_type=tg|tt`；旧 TG 默认行为与 state keys 保持不变。Phase 2F 加既有 Phase 2B 共 7 项远端测试通过，随后 teacher smoke、KD reduced smoke 与 full-data gate 均完成。
- TT teacher 在强制 alpha=0.5 的 full validation NLL 为 4.293589，优于 TG 的 4.303697。TT/TG 参数量为 35,340,801/37,749,633，非 parameter-matched；TT 少 6.38% 参数。TG 的 JSD 与 fusion gain 更高，但没有转化为下游优势。
- TT KD seeds 42/123/456 test NLL 为 4.094289/4.118854/4.111574，均优于复用 TG 的 4.108241/4.130183/4.123673。`H=NLL_TT-NLL_TG` 为 -0.013952/-0.011329/-0.012099，mean -0.012460、sample SD 0.001348、3/3 negative。
- 三个 TT endpoint 均无 NaN，max overflow/non-finite 为 0.013605。TT 早期 clipping 高于 TG，但未饱和且最低降至 0.0034；更强 clipping 下仍获胜，不构成其优势的明显解释。
- teacher quality 差异与 endpoint 方向一致，且 TT/TG teacher alpha LR 为 `1e-2`/`5e-3`，必须作为混杂保留。结论按预注册规则执行：删除 Grassmann-specific KD mechanism claim；Phase 2E 仅支持 generic ensemble supervision。Phase 2G 不启动。

## [2026-10-03] Phase 2G TinyStories negative-transfer confirmation

- 最终 TinyStories 确认实验在 197 服务器完成。seed 42 复用历史 matched pair；seed 123/456 分别独立训练 S0 后运行 WS+CE 与固定 token-mean WS+KD。
- `Delta_KD` 为 -0.107479/-0.107439/-0.106740，mean -0.107219、sample SD 0.000415、3/3 negative。预注册结论为 `CONFIRMED`。
- 所有 endpoint 均有限。KD gradient norm 与 clipping 高于 CE，但 max overflow/non-finite 仅 0.000611，无数值失败。GPU-3 上受 `SW Power Cap` 影响的不完整 seed-456 尝试被排除；经授权在 GPU 1 从头重跑的 endpoint 为唯一有效结果。
- compact evidence 位于 `research/experiments/phase2g_tinystories_negative_transfer/`。实验现已冻结，下一阶段仅为 manuscript reconstruction，不启动 Phase 2H。

## [2026-10-03] Final evidence ledger and manuscript preparation

- 本阶段未运行实验，只读取冻结的 Phase 2A--2G 产物、PTB benchmark JSON 与旧 CAC tex。
- 建立 `research/final_evidence/FINAL_EVIDENCE_LEDGER.md`，逐项记录 claim、experiment、dataset、seed 数、effect size、status、confound 与 main-paper permission。
- 生成保守的四项 `PAPER_CLAIMS.md`、六组 CSV/Markdown 论文表、`OLD_MANUSCRIPT_AUDIT.md`、十节 `MANUSCRIPT_BLUEPRINT.md` 和 `GPT_MANUSCRIPT_HANDOFF.md`。
- 旧稿保持不变。仓库入口已更新到 final evidence，项目状态为 `manuscript_reconstruction`；完整论文尚未撰写，Phase 2H 仍被禁止。

## [2026-10-03] Phase 3A offline safe-distillation diagnostic completion

- 先锁定并推送 protocol commit `80904d5`，远端 4 项 quadratic/HVP/leakage tests 全部通过；WT2-J real-model smoke 验证 31.43M student 全参数 FP32 HVP 可在 RTX 3090 完成且 checkpoint hash 不变。
- 197 四卡并行完成 21/21 正式 condition JSON。train probe 为前 32 个 train chunks，FP32 microbatch=2；calibration seeds 为 314159/271828/161803，每个 subset 两个 validation chunks。无 test dataset construction、AMP、damping、finite-difference model HVP 或 last-layer fallback。
- `diagnostics_blinded.csv` 先生成并以 SHA256 `594ac6d4...d9ad` 固化、提交、推送；随后 endpoint merger 验证 hash 后才生成 `diagnostics_with_endpoints.csv`。
- first-order 与 quadratic signs 在全部 63 行完全重合，且 seed-level Spearman/AUROC/sign accuracy 均相同。二者正确判断 family-mean PTB、TinyStories、WT2 A、WT2 T，但误判正迁移的 WT2 J、G、TT。
- first/second order 仅 11/21 seed rows 跨 calibration subsets 同号。nominal virtual step 虽无 harmful false positive，却将 9/15 positive seed rows 判为 harm；literal scheduler step 的 lr=0 曲线全部严格平坦。
- 完成四组 PDF/300-dpi PNG 图、condition/family summaries、predictor metrics、REPORT、GPT handoff 与 provenance。终止码 `STOP_NEW_METHOD_DIAGNOSTIC_FAILED`；未启动 Phase 3B 或任何训练。

## [2026-10-03] Phase 4A feasibility and protocol freeze

- 新的 bounded external-validity 授权，沿用 remote-only 197 约束。官方 base SmolLM2-135M revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2、360M revision f8027fd0eaeea54caa13c31d31b9fdc459c38b49 下载并核验官方 LFS SHA256。
- 4 项 CE/KL 数值和梯度、token mask/budget、scheduler 测试通过；Stage 0 模型加载及 KD smoke 通过；真实 train chunks 的 global batch32 / micro4 累积 smoke 通过，无 NaN/Inf。
- TinyStories 新 token subset 固定为 train20M / val1M / test1M，原 seed42 holdout 映射保留。新索引缓存仅写 outputs，新数据与旧 GPT2 tokenization 不做绝对 NLL 混比。
- 采用隔离的 Transformers4.46.3/tokenizers0.20.3，无全局环境替换。正式协议为 lr5e-5、AdamW betas0.9/0.95、wd0.01、5%warmup+cosine、clip1、seq256、global32、每 run10M target tokens、validation-only selection。

## [2026-10-03] Phase 4A modern pilot completion and STOP

- 协议 commit 58f4cd7897858d84d577c3c048cc3fd83294d50c 在任何正式 endpoint 之前推送。TinyStories clean completion 后才准备 FineWeb，并在其正式训练前以 55ca7801f146b2ea05cf95a0664e048cff3b29a7 锁定输入 hashes/streaming transport。正式训练代码和优化设置保持不变。
- 197 服务器完成两域全部 10 个 seed42 正式 run，每 run 1221 updates、精确 10M predicted targets。全部验证选中状态之后再评估 test。FineWeb S0 最优在 step256；其它最优均 step1221。
- TinyStories CE/KD1/KD5 test NLL 为 1.579429492790/1.574480860789/1.584226882212，Gain 为 +0.004948632001/-0.004797389422。FineWeb 为 2.949900313167/2.965618911467/2.996303976684，Gain 为 -0.015718598299/-0.046403663516。
- 两域 lambda5 均全程 clipping (1.0)，永久 CLIP_SATURATION_WARNING；KD1 clipping 分别 0.479934/0.850123。所有 loss/grad/参数有限，observed overflow/nonfinite=0；没有改 lambda 或 clip 来修复结果。
- final read-only audit 检查预算、finite telemetry、选择步骤、source-state/order hashes、split doc-ID/text disjointness 和 Gain，状态 PASS，无 model forward。frozen final_evidence aggregate SHA256 仍为 3d172e634db611edad0d566e74ea0f826ed8cdae69922120baa94ebf9e6e2b96。
- 生成 results_seed42.csv、optimization_audit.csv、selected_state_manifest.csv、decision.json、REPORT、GPT_HANDOFF、raw summaries/压缩 telemetry/doc manifests 与两组 PDF/300-dpi PNG 图，并进行图形检查。
- MODERN_REPLICATION_WORTHWHILE 的唯一 qualifying effect 是 FineWeb lambda5 负迁移；TinyStories 两个小效应未达到 0.02。证据只有单 seed 且受 clipping/pretraining overlap 等限制。只推荐负责人审阅独立 student replication，不自动运行。Phase 4A COMPLETE / STOP，论文与 final_evidence 未改。
