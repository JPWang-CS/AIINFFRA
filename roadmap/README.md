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
| [execution-system.md](execution-system.md) | 所有阶段 | 知识→自写→LeetGPU/reference→服务器→profiler→归档的统一流程 |
| [gpu-foundations.md](gpu-foundations.md) | A–D 挂载 | GPU 硬件→优化动作映射、P0–P8 极致性能阶梯，以及整机/SM/存储/指令/runtime/集群课程；[知识入口](../notes/cuda/README.md) |
| [multi-node-multi-gpu.md](multi-node-multi-gpu.md) | D 多机多卡 | topology、NCCL、RDMA、DeviceMesh、混合并行、EP 与排障 |
| [ai-infra-curriculum.md](ai-infra-curriculum.md) | PATH 全路线 | 全阶段执行计划、任务、验收、关键数字（含 M2.5 算子构建） |
| [vllm.md](vllm.md) | C 推理系统 | vLLM 源码深挖：PagedAttention、Scheduler、量化 |
| [distributed.md](distributed.md) | D 分布式训练 | 显存账本、DDP/FSDP/ZeRO、TP/PP/EP |
| [agents.md](agents.md) | E Agent | Tool Use、ReAct、RAG、MCP demo |
| [interviews.md](interviews.md) | 求职冲刺 | 高频题、系统设计、面试叙事 |
| [leetgpu-ladder.md](leetgpu-ladder.md) | ⭐ 可选深钻 | 超出 B 级的 CUDA 优化菜单 |

## 全景

```text
算子/系统线：A CUDA -> B Triton -> C 推理 -> D 分布式 -> E Agent
GPU 底层挂载：执行/SM/存储 -> Tensor Core/指令 -> pipeline/profiling -> runtime/互联
理论线：GPU优化 · 量化 · 注意力 · 模型架构 · 推理技术 · 训练/并行
```

## 使用方式

1. 打开 [ai-infra-curriculum.md](ai-infra-curriculum.md) 看当前模块。
2. 按 [统一执行系统](execution-system.md) 完成知识、平台验收、服务器与 profiler 闭环。
3. 进入对应专项计划完成源码/demo。
4. 正确性 + 性能数字 + 面试口径都满足后更新 PATH。
5. 写一篇 weekly 回顾。
