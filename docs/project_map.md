# Grassmann Flows Project Map

更新时间：2026-09-16

## 当前研究主线

仓库最初用于复现 Grassmann Flow 语言模型，随后扩展为异构 Grassmann--Transformer late-fusion teacher 与 hybrid-lite student 的知识蒸馏研究。当前最可信的主问题不是“Grassmann 是否替代注意力”，而是：异构 teacher 的信息在什么条件下能稳定迁移到更小的 hybrid student，以及这种迁移能否形成可验证的质量、参数量与延迟权衡。

## 核心执行链

```text
TextDataset + GPT-2 tokenizer
          |
          +--> GrassmannGPTv4 ---------+
          |                             |
          +--> SmallTransformer --------+--> late-logit fusion teacher
                                                   |
                         CE-trained hybrid-lite ---+--> warm-start KD
                                                            |
                                             PPL + latency + memory benchmark
```

当前应优先阅读和维护的文件如下。

| 层级 | 文件 | 作用 | 当前地位 |
|---|---|---|---|
| 模型 | `src/models/grassmann_v4.py` | Plücker 编码、causal Grassmann mixing、Grassmann block 与 LM | 当前 Grassmann 主实现 |
| 数据与单分支 | `train_exp4_ddp.py` | `TextDataset`、`SmallTransformer`、单分支训练与 DDP 公共逻辑 | 多个后续脚本的实际依赖 |
| Teacher | `train_hybrid_latefusion_alpha_ddp_v1.py` | 从已训练的两个分支构建 late-logit fusion teacher | 当前 teacher 主入口 |
| Student baseline | `train_hybrid_lite_latefusion_baseline_v2.py` | 从零训练同拓扑 hybrid-lite student | warm-start 初始化来源 |
| KD | `train_distill_hybrid_lite_from_latefusion_teacher_v2.py` | 冻结 teacher，对 student 执行 CE+KL 训练 | 当前 KD 主入口 |
| Benchmark | `benchmark_ptb_efficiency_v4.py` | 统一加载 teacher/student 并测 PPL、延迟、吞吐和显存 | 当前效率评估入口 |
| 结果索引 | `tools/build_experiment_registry.py` | 汇总异构 JSON schema，识别不完整运行与元数据错配 | 新增的非破坏性整理工具 |
| 论文 | `论文投稿/cac/conference_101719.tex` | CAC 投稿版本及后续证据审计版本 | 已拒稿，作为研究记录保留 |

`src/models/grassmann.py`、`grassmann_v2.py`、`grassmann_v3.py` 以及 `train_exp*.py`、`train_distill*.py` 的早期版本仍有历史复现实验依赖，因此暂不移动或删除。新实验不应再从这些旧入口派生。

## 模型与损失的实际行为

`GrassmannGPTv4` 先将隐藏状态投影到低维空间，对每个因果偏移计算并 L2 归一化 Plücker 坐标，再投影回模型维度。多个有效偏移的几何特征按位置平均，随后使用由 `[h; g]` 生成的逐特征门进行混合。该混合输出内部包含 LayerNorm 和 dropout，外层 `GrassmannBlock` 又采用 pre-norm 残差结构，最后接 FFN 残差。

Teacher 与 hybrid-lite student 都保留独立的 Grassmann 和 Transformer 分支。它们在最后 `k` 层分别生成 logits，以可学习标量向量 `alpha` 逐层融合，再对所选层求平均。该结构并不在分支之间交换隐藏状态。

KD 入口现保留两种显式模式。`legacy_batchmean` 用于复现历史目标：

```text
(1 - alpha) * CE + alpha * KL(teacher || student)
```

其中 KL 使用 `reduction="batchmean"`。对于形状为 `[batch, sequence, vocabulary]` 的 logits，这会对序列位置求和、只除以 batch size。因此日志中的 KL 通常为 25--240，而 CE 约为 1.4--3.7。新 `token_mean` 模式按有效 shifted labels 归一化，使用 `CE + lambda_kd * KL_token_mean`，并同时记录两种 KL、有效 token 数、梯度裁剪和 AMP 溢出。PTB smoke 已确认 `lambda_kd=5` 与 legacy `alpha=0.02` 基本等效；跨数据集结论仍需匹配种子的 PTB/TinyStories 正式实验。

## 数据流与复现边界

`TextDataset` 读取 Hugging Face dataset 或 `load_from_disk` 产物，过滤空行，将文本按字符预算拼接后使用本地 GPT-2 tokenizer 编码，最后切成无重叠的固定长度块并丢弃尾部余数。训练中没有 padding，标签与输入相同，通过模型内部的 one-token shift 计算 next-token CE。

已有 code 实验复用了默认 `dataset_name`，因此部分 config 将 CodeParrot 错记为 `ptb` 或 `wikitext2`。实验登记工具以实际 `dataset_path` 和实验名恢复 canonical dataset，同时保留原始标签。当前共检测到 25 行 canonical label 与记录标签不一致，这些运行不能仅凭 `dataset_name` 分组。

## 目录与存储

仓库约 18 GiB，其中 `outputs` 约 17 GiB；`logs` 约 895 MiB。代码、论文和文档本身只占很小部分。2026-09-16 H0 smoke 后共有 194 个运行目录，其中 166 个有完整 config 与 summary，26 个仅有 config，2 个没有完整元数据；185 个 checkpoint 中约 16.17 GiB 被完整运行引用。新增的一个 config-only 目录来自 GPU 0 被外部进程占用后的 OOM 失败，按非破坏性审计规则保留。

本轮不移动或删除任何历史产物。后续若需要释放空间，应先生成明确的 keep/archive/delete manifest，至少保留论文引用结果、每个可复现实验组的最佳 checkpoint、所有 config/summary/metrics，以及失败运行的诊断日志。`datasets/` 与 `checkpoints/` 不在本地工作树中，且按项目约束不得改动。

## 远程执行方式

正式训练只能在 `10.42.0.197` 上运行，优先入口为：

```bash
/home/xin/fuwuqi/agent-tools/exec_grassmann.sh '<command>'
```

该脚本会挂载 NFS、启动 `grassmann_lab` 容器、激活 `/workspace/.venvs/grassmannflows`，再在 `/workspace/grassmannflows` 中执行命令。现有 suite 多数自行进入 `/workspace/grassmannflows/grassmann-flows`。2026-09-16 重新检查确认挂载已经恢复：容器可见同一份项目代码与 159 个 summary，PyTorch 2.6.0a0+cu126 可使用 4 张 RTX 3090，检查时没有正在运行的训练任务。

## 当前代码风险

Teacher、baseline 和 KD 脚本各自复制了一份 hybrid 模型定义，修改其中一个文件不会自动同步其他入口。路径迁移兼容代码也在多个脚本中重复。下一阶段应先抽取共享模型与 KD loss，但必须保持旧 checkpoint 的 state-dict key 兼容。

训练 config 未记录 git commit、代码文件 hash 或命令行原文；随机性设置也没有启用 deterministic 模式。旧 shell suite 还硬编码了 GPU 编号、容器路径和输出目录。新实验协议应补充 provenance、effective dataset metadata、loss reduction、有效 token 数和命令行快照。

## 导航入口

- 完整实验登记表：`docs/generated/experiment_registry.csv`
- Checkpoint 保留/复核/候选归档清单：`docs/generated/checkpoint_manifest.csv`
- 存储与不完整运行清单：`docs/generated/experiment_inventory.md`
- Legacy KD loss-scale 审计：`docs/generated/kd_loss_scale_audit.md`
- 证据分级：`docs/experiment_catalog.md`
- 下一阶段路线：`docs/research_roadmap.md`
- 长期项目记忆：`docs/project_memory.md`
- 追加式实验日志：`docs/experiment_journal.md`
