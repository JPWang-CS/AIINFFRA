# Distributed 实验产物

> 当前仅建立验收结构，不代表已完成。实际状态只看 [PATH.md](../../PATH.md)。
> 执行顺序见 [多机多卡专项路线](../../roadmap/multi-node-multi-gpu.md)。

## 建议目录

```text
solutions/distributed/
├── collectives/       4-rank 语义实验
├── nccl/              nccl-tests 命令与结果
├── ddp/               单卡/单机/多机对照
├── fsdp2/             显存与吞吐实验
├── tensor_parallel/   DeviceMesh + DTensor
├── expert_parallel/   AllToAll / DeepEP 读码实验
└── reports/           拓扑、扩展效率、故障定位报告
```

## 入库门槛

- 单卡或单进程 reference 正确；
- rank 数、节点数、GPU/NIC/topology 明确；
- 命令可复现；
- 有正确性和性能数字；
- 至少一条 profiler/NCCL 证据；
- 使用 [分布式记录模板](../../templates/distributed-record.md)。

系统/通信实验没有对应 LeetGPU 题时，使用 `nccl-tests`、PyTorch reference 和端到端一致性作为平台门；不要伪造 LeetGPU 验收。
