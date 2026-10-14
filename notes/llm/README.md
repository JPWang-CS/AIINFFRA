# 大模型板块（LLM / 大模型系统）

> 定位：把大模型相关学习内容独立成板块，按模型结构、推理、训练、面试和论文整理。
> 约束：这里只是内容组织，不另起学习线。当前学什么仍由 [NOW.md](../../NOW.md) 指定，进度仍由 [PATH.md](../../PATH.md) 维护。

---

## 板块结构

最新外部资料快照：[2026-08-20 更新](updates/2026-08-20.md)。它只记录经过公开论文/官方入口核对的增量信息，不替代 `PATH.md` 的进度状态。

| 子板块 | 内容 | 入口 | 状态 |
|--------|------|------|:--:|
| 模型结构 | Transformer、Attention 与序列模型、MoE、多模态模型结构 | [architectures.md](architectures.md) | 🚧 草稿 |
| 推理系统 | Prefill/Decode、KV cache、vLLM、调度、量化、投机解码、PD 分离 | [inference-systems.md](inference-systems.md) | 🚧 草稿 |
| 训练系统 | 显存账本、混合精度、优化器、DDP/FSDP/ZeRO、TP/PP/EP、checkpoint | [training-systems.md](training-systems.md) | 🚧 草稿 |
| 面试 | 高频题、系统设计、Ascend→GPU 叙事 | [interview.md](interview.md) | 🚧 草稿 |
| 论文 | 八主题阅读器：从模型原理到算法、训练、推理、压缩、多模态与系统 | [论文与算法](../algorithms/README.md) · [论文索引](papers.md) | 🚧 草稿 |
| 算子构建 | RoPE/RMSNorm/GQA/MLA/MoE/FlashAttention/PagedAttention/量化/投机解码 | [operator-building.md](operator-building.md) | 🚧 草稿 |

---

## 学习过程映射（融入现有 PATH 学习计划）

本目录只是**内容聚合**，不维护进度，也不另建学习时间线。当前学什么仍看 [NOW.md](../../NOW.md)，进度仍看 [PATH.md](../../PATH.md)，详细任务和验收看 [PATH 执行参考](../../roadmap/ai-infra-curriculum.md)。

| 子板块 | 学什么 | 对应 PATH 阶段 | 在哪个执行阶段学 |
|--------|--------|----------------|------------------|
| [模型结构](architectures.md) | Transformer 基础、Attention/SSM、MoE 与多模态结构 | 论文线与模型GPU分析参考 | 按论文线当前主题阅读 |
| [推理系统](inference-systems.md) | Prefill/Decode、PagedAttention、batching、量化、投机解码 | 实践落地参考 | 第五/第六篇 |
| [训练系统](training-systems.md) | 显存账本、混合精度、ZeRO/FSDP、TP/PP/EP | 分布式扩展参考 | 第六篇 |
| [面试](interview.md) | 高频题、系统设计、Ascend→GPU 叙事 | 能力复盘 | 全线贯穿 |
| [论文](papers.md) | 八主题阅读入口、论文索引与阅读深度 | 独立论文线 | 不与实践互设前置 |
| [算子构建](operator-building.md) | 模型组件到GPU算子的构建地图 | 实践线内容索引 | 第二到第五篇 |

> 学习顺序不是板块内部单独排序，而是由 PATH 执行参考的阶段推进。

论文线在[统一阅读器](../../roadmap/curriculum/gpu/course-site/index.html?chapter=41)按八类连续阅读：模型基础与完整结构、注意力/序列建模/长上下文、MoE 与条件计算、训练优化与后训练、生成推理与检索、量化与模型压缩、多模态与生成模型、高效计算与系统论文。精读围绕推导、张量维度、代码、例子、质量和适用边界展开；只有相关主张需要实验支撑时才记录对应 benchmark。

---

## 当前覆盖

- 模型结构：语言模型与完整 Transformer、Attention/SSM、MoE、长上下文和多模态结构；模型家族按主题讨论，不作为单独的覆盖清单。
- 推理系统：PagedAttention、continuous batching、prefix cache、chunked prefill、quantization、speculative decoding、PD 分离。
- 训练系统：显存账本、混合精度、ZeRO/FSDP、TP/PP/EP、activation checkpointing。
- 算子构建：RoPE、RMSNorm、GQA/MLA、MoE、FlashAttention、PagedAttention、量化、投机解码。

---

## 怎么用

1. 每个子板块先读“核心概念”，再按“学习任务”动手。
2. 实践主题按问题记录需要的张量形状、缓存容量、通信量或 benchmark；论文结构阅读以机制推导、代码路径、质量条件与限制为主，不要求每篇都补吞吐数字。
3. 能讲清后才更新 [PATH.md](../../PATH.md) 状态；草稿不算掌握。
