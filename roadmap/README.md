# Roadmap — 学习计划总览

> 这里是学习计划的入口。
> 日常进度看 [PATH.md](../PATH.md)，当前焦点看 [NOW.md](../NOW.md)，详细任务看 [ai-infra-curriculum.md](ai-infra-curriculum.md)。

---

## 当前主线

```text
PATH B：Triton 实现阶段
vec add -> matmul -> fused softmax -> flash attention -> GQA/fused MLP
```

A4/A5 作为背景收尾，不抢主线。

并行强化：最新模型与算子构建能力（GQA/MLA/MoE/FlashAttention/PagedAttention 等）。

## 学习计划清单

| 文件 | 对应阶段 | 内容 |
|------|---------|------|
| [ai-infra-curriculum.md](ai-infra-curriculum.md) | PATH 全路线 | 全阶段执行计划、任务、验收、关键数字（含 M2.5 算子构建） |
| [vllm.md](vllm.md) | C 推理系统 | vLLM 源码深挖：PagedAttention、Scheduler、量化 |
| [distributed.md](distributed.md) | D 分布式训练 | 显存账本、DDP/FSDP/ZeRO、TP/PP/EP |
| [agents.md](agents.md) | E Agent | Tool Use、ReAct、RAG、MCP demo |
| [interviews.md](interviews.md) | 求职冲刺 | 高频题、系统设计、面试叙事 |
| [leetgpu-ladder.md](leetgpu-ladder.md) | ⭐ 可选深钻 | 超出 B 级的 CUDA 优化菜单 |

## 全景

```text
算子线：A CUDA -> B Triton -> C 推理 -> D 分布式 -> E Agent
理论线：GPU优化 · 量化 · 注意力 · 模型架构 · 推理技术 · 训练/并行
```

## 使用方式

1. 打开 [ai-infra-curriculum.md](ai-infra-curriculum.md) 看当前模块。
2. 进入对应专项计划完成源码/demo。
3. 正确性 + 性能数字都满足后更新 PATH。
4. 写一篇 weekly 回顾。