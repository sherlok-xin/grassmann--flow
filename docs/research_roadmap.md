# Post-Rejection Research Roadmap

更新时间：2026-09-16

## 总体判断

当前工作不缺额外的 alpha 点，而缺少一个经得起审稿的因果主线。最有价值的现象是：同一种 legacy warm-start KD 在 PTB、WikiText-2 和 code 上改善 student，却在 TinyStories 上稳定方向相反。现有实现的 token-summed KL 又使这一现象可能只是 loss scale 与 teacher reliability 的混合结果。新的研究应从这个失败边界出发，而不是继续把“hybrid + KD”本身包装成创新。

两句话的研究叙事可以写成：异构 teacher 的统一 logit KD 在不同文本域上并不稳定，而且常见的 batchmean 实现使蒸馏强度依赖序列长度。我们先建立 token-normalized、严格配对的评测，再利用 teacher 两个分支的置信度与分歧自适应地选择蒸馏信号，以减少负迁移。

## 阶段 0：先修实验有效性

第一项实验是 H0 loss-normalization audit。KD 入口应新增显式的 `legacy_batchmean` 与 `token_mean` 模式，旧模式保持默认兼容；新模式使用有效 token 数归一化，并将目标写为 `CE + lambda_kd * KL_token`，避免把两个不同量纲的损失强行解释成凸组合。所有 config、metrics 和 summary 必须记录 reduction、有效 token 数、CE、raw KL、normalized KL 和两项梯度范数。

远程恢复后先在 PTB 上运行一个 1-epoch smoke：同一 teacher、同一 warm-start checkpoint、同一 batch、seed 和数据，比较 CE-only、legacy `alpha=0.02`、token-normalized `lambda_kd` 为 2.5、5 和 10。筛选标准不是最终 PPL，而是无 NaN、checkpoint 可加载、loss scale 合理、首 200 step 梯度稳定，并确认 legacy 模式复现旧数值。只有 smoke 通过后才运行 10 epochs。

第二项是完全匹配的 random-init 对照。CE 与 KD 必须共享初始化权重、优化器、学习率、scheduler、epoch、batch 和数据顺序。该实验只回答 warm-start 是否必要，不与历史 random-init 运行拼接比较。

## 阶段 1：补齐审稿最容易质疑的证据

在 H0 确定的 token-normalized 权重下，PTB 和 TinyStories 各运行三个固定 seed。PTB 是正迁移域，TinyStories 是负迁移边界，二者先提供机制信号；WikiText-2 和 code 只在机制得到支持后扩展。每个域至少包含 CE-only、最佳固定 KD 和后续自适应 KD，所有方法共享初始化 checkpoint。

随后增加参数和训练预算匹配的 student controls：Transformer-only、Grassmann-only 和 hybrid-lite。三者应报告参数、FLOPs 或测得延迟、CE baseline、KD 结果和 seed 方差。这样才能区分“KD 有效”与“双分支 student 本身更强”。Teacher 侧只保留 checkpoint-matched late fusion；其他 fusion 方案若不能从同一对 source checkpoints 训练，就不进入主消融。

## 阶段 2：创新候选

优先级最高的是 branch-disagreement-aware KD。异构 teacher 的 Transformer 与 Grassmann 分支可分别输出分布。对每个 token 计算两个分支的 Jensen--Shannon divergence 或 top-k restricted divergence，并用低分歧 token 提供更强的 fused-logit KD，高分歧 token 回退到 CE。该方法直接针对 TinyStories 负迁移，额外参数为零，并且可以通过 teacher 分支分歧与 token-level KD gain 的相关性验证机制。

第二候选是 branch-aligned dual distillation。Hybrid-lite student 本身也有两个分支，因此除 fused-logit KD 外，可以让 student Transformer 分支匹配 teacher Transformer 分支，让 student Grassmann 分支匹配 teacher Grassmann 分支。核心消融为 fused-only、branch-only、fused+branch，以及交换 teacher branch 的错误配对。该方向比普通 logit KD 更能体现异构架构，但显存和计算成本较高。

第三候选是 curriculum KD。训练早期按 teacher confidence 或 student-teacher gap 使用较弱权重，随后逐步增加；若分支分歧长期较高，则保持 CE 主导。该方法实现简单，但单独作为创新偏弱，更适合作为前两个方法的稳定训练组件。

Geometry-aware intermediate distillation 暂列高风险候选。它可以匹配 Grassmann 分支的 Plücker direction、窗口关系矩阵或 subspace spectrum，但必须先证明这些统计与预测改进相关，否则容易成为复杂但缺少机制证据的附加 loss。

## 预注册的判定标准

主指标为 test PPL，但模型选择只看 validation PPL。机制实验同时报告 token-level teacher disagreement、teacher entropy、student NLL 和 KD gain。确认性实验使用三个固定 seed，报告均值、标准差和配对差值。若自适应方法在 PTB 不劣于最佳固定 KD，并在 TinyStories 显著缩小负迁移，才进入另外两个域；否则停止扩展，回到 failure analysis。

参数量与 batch 32 延迟必须共同报告。目标不是追求任意 PPL 提升，而是在不增加 student 参数的情况下改善固定质量--效率 Pareto frontier。

## 执行顺序

1. 恢复远程 NFS 挂载并完成环境只读检查。
2. 实现 H0 loss API、单元测试和 legacy parity test。
3. 跑 PTB 1-epoch smoke，再跑 PTB/TinyStories 三 seed 最小矩阵。
4. 根据结果决定是否实现 branch-disagreement-aware KD。
5. 机制成立后再补 WikiText-2、code、单分支 controls 和最终效率评测。

每个实验在运行前写入 `research/experiments/<hypothesis>/protocol.md`，结果回来后追加 analysis，不以事后最佳点替换预注册主点。
