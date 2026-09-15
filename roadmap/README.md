# Roadmap — 学习计划总览

> 这里是学习计划的入口。
> 日常进度看[PATH.md](../PATH.md)，当前焦点看[NOW.md](../NOW.md)，正式课程从[curriculum/README.md](curriculum/README.md)进入；`ai-infra-curriculum.md`只保留兼容路由。

---

## 当前主线

新计划以 GPU 高性能 LLM 算子为主干：NPU→GPU 架构迁移、完整算子、极致优化、量化、Prefill/Decode GPU 分析；Mini Transformer贯穿验证，vLLM负责真实系统落地。论文线单列。

## 学习计划清单

| 文件 | 对应阶段 | 内容 |
|------|---------|------|
| [execution-system.md](execution-system.md) | 所有阶段 | 知识→自写→LeetGPU/reference→服务器→profiler→归档的统一流程 |
| [curriculum/README.md](curriculum/README.md) | 正式课程 | 按篇/章/节模块化路由 |
| [gpu-foundations.md](gpu-foundations.md) | 第一篇兼容入口 | GPU硬件、执行、内存、数值、资源、流水、编译、profiling、代际 |
| [multi-node-multi-gpu.md](multi-node-multi-gpu.md) | 第六篇扩展 | topology、NCCL、RDMA、DeviceMesh、混合并行、EP与排障 |
| [ai-infra-curriculum.md](ai-infra-curriculum.md) | 兼容入口 | 指向模块化正式课程 |
| [vllm.md](vllm.md) | 第六篇加分项 | vLLM源码深挖：PagedAttention、Scheduler、量化 |
| [distributed.md](distributed.md) | 第六篇扩展 | 显存账本、DDP/FSDP/ZeRO、TP/PP/EP |
| [agents.md](agents.md) | 非当前路线 | Tool Use、ReAct、RAG、MCP demo |
| [interviews.md](interviews.md) | 求职冲刺 | 高频题、系统设计、面试叙事 |
| [leetgpu-ladder.md](leetgpu-ladder.md) | ⭐ 可选深钻 | 超出 B 级的 CUDA 优化菜单 |

## 全景

```text
实践：GPU架构 → 完整算子 → 极致优化 → 量化 → Prefill/Decode → Mini Transformer → vLLM → 多GPU/MoE
论文：经典理论 → 公式 → 作者代码 → 系统论文 → 最新论文（独立推进）
```

## 使用方式

1. 打开[curriculum/README.md](curriculum/README.md)看正式课程路由。
2. 按 [统一执行系统](execution-system.md) 完成知识、平台验收、服务器与 profiler 闭环。
3. 进入对应专项计划完成源码/demo。
4. 正确性 + 性能数字 + 面试口径都满足后更新 PATH。
5. 写一篇 weekly 回顾。
