# 大模型板块（LLM / 大模型系统）

> 定位：把大模型相关学习内容独立成板块，按结构、推理、训练、面试、论文整理。
> 约束：这里只是内容组织，不另起学习线。当前学什么仍由 [NOW.md](../../NOW.md) 指定，进度仍由 [PATH.md](../../PATH.md) 维护。

---

## 板块结构

| 子板块 | 内容 | 入口 | 状态 |
|--------|------|------|:--:|
| 模型结构 | Transformer、Attention 变体、MoE、SSM、模型家族、HF config | [architectures.md](architectures.md) | 🚧 草稿 |
| 推理系统 | Prefill/Decode、KV cache、vLLM、调度、量化、投机解码、PD 分离 | [inference-systems.md](inference-systems.md) | 🚧 草稿 |
| 训练系统 | 显存账本、混合精度、优化器、DDP/FSDP/ZeRO、TP/PP/EP、checkpoint | [training-systems.md](training-systems.md) | 🚧 草稿 |
| 面试 | 高频题、系统设计、Ascend→GPU 叙事 | [interview.md](interview.md) | 🚧 草稿 |
| 论文 | 精读清单和阅读顺序 | [papers.md](papers.md) | 🚧 草稿 |
| 算子构建 | RoPE/RMSNorm/GQA/MLA/MoE/FlashAttention/PagedAttention/量化/投机解码 | [operator-building.md](operator-building.md) | 🚧 草稿 |

---

## 学习过程映射（融入现有 PATH 学习计划）

本目录只是**内容聚合**，不维护进度，也不另建学习时间线。当前学什么仍看 [NOW.md](../../NOW.md)，进度仍看 [PATH.md](../../PATH.md)，详细任务和验收看 [PATH 执行参考](../../roadmap/ai-infra-curriculum.md)。

| 子板块 | 学什么 | 对应 PATH 阶段 | 在哪个执行阶段学 |
|--------|--------|----------------|------------------|
| [模型结构](architectures.md) | Transformer、GQA/MoE/SSM、HF config、KV cache | 理论线·模型架构 | PATH 执行参考 M1.5 |
| [推理系统](inference-systems.md) | Prefill/Decode、PagedAttention、batching、量化、投机解码 | 算子线 C + 理论线·推理技术 | PATH 执行参考 M3 |
| [训练系统](training-systems.md) | 显存账本、混合精度、ZeRO/FSDP、TP/PP/EP | 算子线 D + 理论线·训练/并行 | PATH 执行参考 M4 |
| [面试](interview.md) | 高频题、系统设计、Ascend→GPU 叙事 | 求职冲刺 | PATH 执行参考 M5 |
| [论文](papers.md) | 精读顺序和清单 | 随各阶段滚动 | PATH 执行参考 M1.5/M3/M4 |
| [算子构建](operator-building.md) | 从读模型升级到构建最新组件/算子 | 理论线 + 算子线 | PATH 执行参考 M2/M2.5/M3 |

> 学习顺序不是板块内部单独排序，而是由 PATH 执行参考的阶段推进。

---

## 当前覆盖

- 模型结构：LLaMA、Qwen、DeepSeek、Mistral、GPT/o、Claude、Gemini、Mamba/Jamba。
- 推理系统：PagedAttention、continuous batching、prefix cache、chunked prefill、quantization、speculative decoding、PD 分离。
- 训练系统：显存账本、混合精度、ZeRO/FSDP、TP/PP/EP、activation checkpointing。
- 算子构建：RoPE、RMSNorm、GQA/MLA、MoE、FlashAttention、PagedAttention、量化、投机解码。

---

## 怎么用

1. 每个子板块先读“核心概念”，再按“学习任务”动手。
2. 模型结构、推理、训练都要留下数字：KV cache 大小、显存账本、通信量、benchmark。
3. 能讲清后才更新 [PATH.md](../../PATH.md) 状态；草稿不算掌握。
