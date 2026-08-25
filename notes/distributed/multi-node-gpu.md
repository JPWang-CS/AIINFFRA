# 多机多卡知识地图

## 1. 五层模型

```text
模型并行层：DP / FSDP / TP / PP / CP / EP
通信语义层：P2P / AllReduce / AllGather / ReduceScatter / AllToAll
通信库层：torch.distributed / NCCL / NVSHMEM / DeepEP
节点互连层：NVLink / NVSwitch / PCIe
跨节点网络：NIC / GPUDirect RDMA / InfiniBand / RoCE / switch fabric
```

定位性能问题时必须逐层问，不能把所有慢都归因于 NCCL。

## 2. 基本实体

| 实体 | 含义 |
|------|------|
| process/rank | SPMD 程序的一个执行实例 |
| local rank | 当前节点内的 rank 编号，通常绑定一张 GPU |
| world size | 总进程数 |
| ProcessGroup | 一组 ranks 与对应通信上下文 |
| DeviceMesh | N 维 rank 网格，把并行维映射到拓扑 |
| DTensor Placement | `Shard`、`Replicate`、`Partial` |

现代 PyTorch 的重要演进是：从手工维护大量 process groups，走向 `DeviceMesh + DTensor` 描述多维并行布局；FSDP2 与 TP 可建立在这一抽象上。

## 3. 链路不是一种带宽

| 路径 | 典型场景 | 主要限制 |
|------|----------|----------|
| HBM↔SM | kernel load/store | HBM、L2、访问模式 |
| GPU↔GPU NVLink/NVSwitch | 节点内 collective/P2P | link 数、拓扑、路由 |
| GPU↔GPU PCIe | 无 NVLink 的 P2P | PCIe generation、switch、ACS/IOMMU |
| GPU↔NIC | 跨节点 | PCIe affinity、GPUDirect RDMA |
| NIC↔fabric↔NIC | 跨节点 | IB/RoCE、rail、拥塞、路由、QP |

GPUDirect RDMA 提供 NIC 与 GPU memory 的直接数据路径，避免数据必须先复制到 host buffer；控制面、队列建立和同步仍然存在。

## 4. InfiniBand 与 RoCE

| 项 | InfiniBand | RoCE |
|----|------------|------|
| 链路层 | IB 原生 fabric | Ethernet 上承载 RDMA |
| 常见要求 | Subnet Manager、PKey/SL/VL | PFC/ECN、无损/拥塞配置 |
| 排障重点 | port state/rate、LID/GID、routing | VLAN/GID、PFC/ECN、MTU、拥塞 |

不能只看到“400 Gbit/s”就推导 GPU collective 带宽；还要考虑编码、协议、单/双端口、multi-rail、PCIe、消息分片和并发。

## 5. Collective 的数据语义

| 原语 | 输入/输出 | 常见使用 |
|------|-----------|----------|
| Broadcast | root → all | 初始化/控制 |
| AllReduce | 每卡输入，所有卡完整归约结果 | DDP gradient |
| ReduceScatter | 归约后每卡一份 shard | FSDP gradient |
| AllGather | 每卡 shard，所有卡拼完整 | FSDP parameter、TP |
| AllToAll | 每卡向每卡发送不同 shard | MoE dispatch/combine |
| P2P send/recv | 指定 rank | PP、ring attention |

算法选择依赖消息大小和拓扑：ring 通常带宽友好，tree 小消息 latency 可能更优，hierarchical collective 会分别利用节点内和节点间路径。

## 6. `algbw`、`busbw` 与 logical bandwidth

- algorithm bandwidth：从用户消息大小与完成时间算出的有效吞吐。
- bus bandwidth：按 collective 算法的归一化通信量修正，用于更接近物理链路利用。
- logical bandwidth：某些 EP 系统把 local-rank traffic 也计入，不能直接等于 RDMA wire bandwidth。

看论文或 README 数字时先问指标定义。

## 7. 通信与计算重叠

真正 overlap 需要：

- 数据依赖允许；
- 通信被切成合适粒度；
- 使用不同 stream/engine 或低 SM 通信路径；
- GEMM 与通信没有在 SM、HBM、PCIe 上互相抢到得不偿失；
- 时间线显示并发，而不是 API 异步但最终串行等待。

DDP bucket、FSDP prefetch、PP micro-batch、MoE dispatch/combine pipeline 都是这一原则的不同表现。

## 8. Topology-aware parallelism

```text
高频、延迟敏感：尽量放高速域（TP、部分 CP/EP）
消息较少或可隐藏：可跨节点（PP、部分 DP/FSDP）
```

这只是起点。最终要结合模型 shape、batch、并行度和实际 collective profile。

## 9. 故障与 slow rank

常见层次：

1. rank/collective 调用顺序错误；
2. rendezvous、端口、防火墙或 DNS；
3. GPU/NIC affinity 不佳；
4. NVLink/P2P 未启用或拓扑异常；
5. IB/RoCE link、GID、MTU、拥塞配置；
6. 某 rank GPU 降频、ECC、数据加载或 kernel 变慢；
7. timeout 只是结果，不是根因。

排障证据：NCCL debug 日志、topology dump、`nccl-tests`、低层网络测试、跨 rank 时间线和 per-rank heartbeat。

## 10. 资料

- [NCCL docs](https://docs.nvidia.com/deeplearning/nccl/)
- [GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed)
- [DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html)
- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [nccl-tests](https://github.com/NVIDIA/nccl-tests)
