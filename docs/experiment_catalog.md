# Experiment Catalog and Evidence Status

更新时间：2026-09-16

## 实验谱系

仓库中的实验可以分为六代。第一代在 WikiText-2 上复现 Grassmann 与 Transformer，并扫描 window size、reduced dimension 和模型宽度。第二代将相同的单分支比较扩展到 PTB 与 TinyStories。第三代从各自最优的单分支 checkpoint 构建 scalar-alpha、layerwise 和 late-logit fusion teacher。第四代训练 hybrid-lite baseline，并探索 random-init KD、warm-start KD、两阶段 KD 与不同 student 尺度。第五代在 PTB、WikiText-2、TinyStories 和 CodeParrot 上扫描 warm-start KD 系数。第六代进行 batch-size latency、CUDA 和子空间分析。

所有运行的标准化元数据位于 `docs/generated/experiment_registry.csv`。`replicate_key` 只在数据、模型、训练超参数、初始化方式、结果分支和蒸馏方法一致时相同；`control_key` 进一步忽略蒸馏系数，用于寻找候选对照。由于旧配置存在错误默认值，这两个 key 只能帮助筛选，不能替代人工审计。

## 可直接保留的证据

PTB teacher、质量优先 student 和效率优先 student 构成当前最完整的结果链。Teacher 为 36.264M 参数，test PPL 为 50.11。224x56、6 层 student 为 31.434M 参数，warm-start legacy KD 后 test PPL 为 51.84；对应三次运行约为 51.77--51.88。192x48、4 层 student 为 23.565M 参数，test PPL 为 53.87；对应三次运行约为 53.87--53.99。

在 batch size 32 下，teacher、质量优先 student 和效率优先 student 的平均 batch latency 分别为 55.98、48.46 和 34.96 ms。质量优先 student 仅减少约 13.3% 参数，因此不能包装为大幅模型压缩；效率优先 student 减少约 35.0% 参数，才形成更清楚的部署权衡。

PTB 的 CE baseline、单分支基础模型和部分 student 尺度结果有多 seed 支撑。它们适合用于稳定性说明，但运行之间仍需核对 dataset token counts、checkpoint 来源和训练预算。

## 只能作为探索性证据的结果

四数据集 warm-start 系数扫描目前绝大多数只有 seed 42。相对各自 warm-start CE，legacy KD 在 PTB、WikiText-2 和 code 上找到正向点，但在 TinyStories 上所有已测正系数都退化。这一模式值得作为新研究问题，但在完成 token-normalized loss 与多 seed 重复前，不能声称为普遍的域依赖规律。

Random-init CE 与 random-init KD 运行没有统一学习率、epoch、数据预算和部分模型配置，因此不能用于初始化方式的因果比较。早期 fusion 变体也混入不同 checkpoint 或数据处理，不能作为严格的融合位置消融。CodeParrot 的若干 config 将 dataset 错记为 PTB 或 WikiText-2，且 5k 与 40k 文件预算并存，只能在严格按路径和 token count 对齐后比较。

子空间分析已有产物，但尚未建立从 representation statistic 到 PPL 改善的多 seed 因果链。CUDA 微基准包含局部加速信号，但 correctness test 和导入链路曾失败，不能写成完整系统贡献。

## 结果数据的关键异常

当前运行目录共 186 个，159 个完整，25 个 config-only，2 个更不完整。部分同名或近同名运行是失败重试，不应按时间自动取最新结果。Code 元数据存在 25 行 canonical dataset 与原始标签不一致。旧 `collect_results.py` 只扫描 `outputs/experiments`，无法覆盖 hybrid、distill 和失败运行，因此不再适合作为总索引。

PTB legacy KD 的日志显示，`alpha=0.02` 时首轮 CE 约 3.58、KL 约 70.58；到第十轮 CE 约 3.59、KL 约 33.91。TinyStories `alpha=0.01` 的首轮 KL 约 154.23，code 的首轮 KL 约 239.30。这些数量级差异主要来自按 token 求和的 `batchmean`，说明现有 alpha 扫描同时混入了数据域与 loss scale 变化。

## 证据分级规则

可写入主结果的证据至少需要完整 config/summary/metrics、明确的 source checkpoint、有效数据统计、同预算 baseline、至少三个预先指定 seed，以及以 validation 选 checkpoint 后只报告一次 test。只有单 seed 但配置完整的结果可作为探索性表格。配置不匹配、数据预算混杂或只有日志片段的结果只用于形成假设，不用于支持核心主张。

