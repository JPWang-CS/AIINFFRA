# Distributed / 多机多卡知识入口

> 这里只放可跨实验复用的知识，不维护进度。执行任务见 [多机多卡专项路线](../../roadmap/multi-node-multi-gpu.md)。

## 索引

| 文档 | 内容 |
|------|------|
| [multi-node-gpu.md](multi-node-gpu.md) | 拓扑、NCCL、RDMA、collective、并行映射、排障 |
| [训练系统](../llm/training-systems.md) | DDP/FSDP/TP/PP/EP 与显存账本 |
| [理论速览](../algorithms/remaining-theory-primer.md) | NCCL、ZeRO、并行概念速查 |

## 边界

- 本目录：通信和多机多卡机制。
- `notes/llm/training-systems.md`：训练系统如何使用这些机制。
- `roadmap/`：学习顺序、实验和验收。
- `solutions/distributed/`：亲手跑通的代码与结果。
- `papers/`：单篇论文精读。
