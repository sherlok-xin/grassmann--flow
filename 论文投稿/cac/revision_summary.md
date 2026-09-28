# CAC 论文重构说明

日期：2026-09-15

## 1. 版本演化与原稿诊断

- `paper_cac_draft/conference_101719.tex` 以 PTB 为主，主线是“warm-start 使 Hybrid-lite 蒸馏有效”，结构完整并包含 Related Work，但把一次未匹配的 hidden-fusion 运行解释成融合位置消融。
- `论文投稿/CICAI2026 AuthorKit/LaTex Template/samplepaper.tex` 扩展到 PTB、WikiText-2、TinyStories 与 Python code，并加入系数扫描和跨域实验；证据范围更广，但删除了 Related Work，并把数据量不一致的 PTB→code 结果与同域结果直接比较。
- 原 `论文投稿/cac/conference_101719.tex` 继承多域叙事，又加入无法在仓库产物中定位的 architecture-ablation 数值；方法公式还把实际的特征依赖向量门写成了标量门。

最可能导致拒稿的风险如下：

1. hidden-fusion 的 72.34 PPL 与仓库记录 63.93 不一致，并且所用 Grassmann checkpoint 与 late fusion 不匹配，不能支持融合位置的因果结论。
2. 63.82/71.15/68.41/74.30/66.92 等 architecture-ablation 数值无可追溯产物。
3. PTB random-init KD 的 105.86 来自不同学生宽度、深度、秩、学习率和 `lambda=0.3`，不能放入 matched comparison。
4. Python `lambda=0.05` warm-start 值与 random-init 值重复，但不存在对应 warm-start 产物。
5. 论文中的 Grassmann 标量融合公式与 `src/models/grassmann_v4.py` 的逐特征门控不符。
6. KD 实现是 `KL(p_T || p_S)`，而且 `batchmean` 仅除以 batch size；若不披露，系数跨长度或实现不可比。
7. 同域 code 训练使用 5,000 个文件，PTB→code 探索使用 40,000 个文件；4.17 PPL 不能归因于跨域初始化。
8. code teacher 与 student 均为 31.43M 参数，因此属于知识迁移而不是模型压缩。
9. 多域系数扫描大多只有单 seed，无法支撑普适机制结论。
10. fused teacher 在分支预训练后又联合训练，性能提升不能单独归因于融合结构。
11. 随机初始化 KD 与 CE-from-scratch 并非受控对照：学习率均不同，WikiText-2 和 TinyStories 的训练 epoch 数也不同，因此不能把 PPL 差异归因于初始化或 KD。
12. WikiText-2 教师与源分支的运行元数据记录了不同的数据处理统计，原先将两者 PPL 直接视为融合改进的做法不够严谨。

## 2. 新的核心主线

论文改为回答一个可由当前对照实验直接支撑的问题：**在固定 CE warm-start 之后，来自 Grassmann--Transformer teacher 的 logit KD 在不同文本数据集上是否一致有效？** 证据沿三个研究问题组织：

- RQ1：teacher 是否比同数据集上的 CE student baseline 更强？
- RQ2：在共用同一 warm-start checkpoint 的受控条件下，KD 对四个文本数据集的影响是否一致？
- RQ3：在 PTB 上能否得到经验证的质量—延迟折衷？

贡献边界被收紧为经验性贡献：warm-start KD 在四个数据集上呈现不一致结果；当前实现的 KL reduction 会改变 `lambda` 的尺度含义；PTB 上存在两个可复现部署点。论文不再声称当前数据证明了初始化效应，也不把观察到的数据集差异解释为已证明的几何机制。

## 3. 关键改动

- 标题改为 `Warm-Start Knowledge Distillation for Grassmann--Transformer Hybrids: An Empirical Study Across Text Domains`，避免原标题中含义空泛的 `Bridging the Gap` 和未充分定义的 `Hybrid-lite Distillation`。
- 重写摘要、引言、方法、实验、讨论、局限性和结论，将不匹配的 random-init 结果降级为探索性观察。
- 方法部分按代码恢复逐特征门控、Pl\"ucker 归一化和多尺度平均，并把蒸馏方向明确写为 `KL(p_tea || p_stu)`；同时解释 `batchmean` 与 token-mean CE 的尺度差异。
- 主表只保留仓库 JSON/benchmark 可追溯结果，warm-start 扫描统一为 `lambda in {0,.01,.02,.03,.04}`。
- 删除了将 random-init 称为 matched comparison 的表格；保留其数值时同时披露学习率和训练时长混杂。
- RQ1 不再直接比较数据处理记录不一致的源分支 PPL，改为比较 teacher 与同数据集的 CE hybrid student baseline。
- 将 PTB→code 放入 stress-test 讨论并显式标注 5k/40k 数据预算混杂，不纳入主结论。
- 使用用户新绘制的 `figures/image1_1.png` 替换研究设计图，并重写图注以说明 teacher/student 共享拓扑、多尺度平均与图中省略操作；保留四域系数扫描和 PTB batch latency 矢量图。
- 补充了两篇已从 ACL Anthology 和 PMLR 核验的最相关工作，用于界定已有初始化研究和同容量蒸馏。
- 新建独立 `refs.bib`，引用仅保留可验证条目；正文恢复 Related Work，并将方法定位到 autoregressive KD、heterogeneous model transfer 与结构化序列模型文献中。

## 4. 仍需补做的实验

正文保留了四项 LaTeX `TODO: additional experiment recommended` 注释：

1. 所有域和主要 `lambda` 至少运行 3 seeds，报告置信区间。
2. 对比 batch-mean 与 token-mean KL，并分别重新调参。
3. 从相同分支 checkpoint、相同参数量和相同步数重跑 late/hidden fusion。
4. 在完全相同的 5k 或 40k code 数据预算下重跑 PTB→code 各基线。
5. 使用相同学习率、epoch 数、student architecture 和数据分割重跑 random-init CE/KD，再判断 warm-start 是否必要。
6. 增加受控的 Transformer-only 或 Grassmann-only student KD 对照，以确定观察到的域敏感性是否与 hybrid architecture 有关。

在这些实验完成前，论文应保持“描述性证据”定位，不宜声称普适机制。

## 5. 交付与验证

- 主稿：`conference_101719.tex`
- 参考文献：`refs.bib`
- 图源：`figures/gen_paper_figures.py`
- 图文件：`figures/image1_1.png`、`figures/fig_lambda_sweep.*`、`figures/fig_batch_latency.*`
- 编译稿：`conference_101719.pdf`
- 构建命令：`latexmk -pdf -interaction=nonstopmode -halt-on-error conference_101719.tex`

最终 PDF 为 IEEE letter 双栏 6 页。19 篇参考文献与全部交叉引用均已解析，没有 overfull box、未定义引用或重复 label；PDF 字体均嵌入。仅保留关键词断行和图页产生的非致命 underfull 提示。
