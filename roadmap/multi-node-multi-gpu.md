# 多机多卡专项路线：从拓扑到大模型并行

> 对应 PATH D / M4。当前 Triton 主线完成阶段性目标后再进入；现在只建立索引，不并行开课。
> 知识入口：[多机多卡知识地图](../notes/distributed/multi-node-gpu.md)。实验统一使用 [分布式记录模板](../templates/distributed-record.md)。

---

## 1. 能力出口

完成后能够：

- 从 `rank/process/device/node` 解释 PyTorch/NCCL 程序如何运行；
- 读懂单机 PCIe/NVLink/NVSwitch 与跨机 InfiniBand/RoCE 拓扑；
- 手算 collective 通信量和理论时间下界；
- 跑通单卡、单机多卡、两机多卡三层 baseline；
- 用 `nccl-tests`、NCCL debug、Nsight Systems 定位带宽低、slow rank、hang；
- 设计 DP/FSDP2/TP/PP/CP/EP 的 `DeviceMesh`，解释为什么某个并行维放在节点内或跨节点；
- 把通信优化转化为端到端 step time、MFU、tokens/s 或 TTFT/TPOT 改善。

---

## 2. 六级学习阶梯

| 级别 | 主题 | 环境 | 最小产出 |
|------|------|------|----------|
| D0 | 进程、rank、ProcessGroup、collective | CPU/Gloo 或单 GPU | 4-rank collective 模拟 |
| D1 | 单机多卡拓扑与 NCCL | 2–8 GPU 单节点 | `nccl-tests` 四原语基线 |
| D2 | DDP、FSDP2、DTensor/TP | 单机 2–8 GPU | loss 一致、显存与吞吐表 |
| D3 | 多节点启动与 RDMA | 2 nodes × 2+ GPU | 两机 DDP + NCCL 网络基线 |
| D4 | 混合并行 DP×TP×PP×CP | 2+ nodes | DeviceMesh/拓扑映射图 |
| D5 | MoE EP 与通信-计算重叠 | 8+ GPU，最好多节点 | AllToAll/dispatch-combine 分析 |

---

## 3. D0：分布式语义，不先背框架

必须掌握：

- world、rank、local rank、node rank；
- one process per GPU；
- rendezvous、store、ProcessGroup；
- point-to-point 与 collective；
- SPMD、barrier、同步与异步 work handle。

实验：用 Gloo/CPU 写 4-rank `all_reduce`、`all_gather`、`reduce_scatter_tensor`、`all_to_all_single`，检查每个 rank 的输入输出。

验收：

- [ ] 能手算 4 ranks 每个原语的结果；
- [ ] 能解释 collective 顺序不一致为什么会 hang；
- [ ] 能区分 global rank 与 local rank。

---

## 4. D1：单机多卡拓扑与 NCCL

先观察，不先跑训练：

```text
nvidia-smi topo -m
nvidia-smi topo -p2p r
```

再用 `nccl-tests` 建立 baseline：

- AllReduce：DDP 梯度同步；
- AllGather：FSDP 参数、TP 拼接；
- ReduceScatter：FSDP/ZeRO 梯度分片；
- AllToAll：MoE token 交换。

记录 `algbw` 与 `busbw`，不要只看一个“GB/s”。对不同 message size 画 latency/bandwidth 曲线，识别小消息 latency-bound 与大消息 bandwidth-bound。

验收：

- [ ] 一张 GPU↔NVLink/PCIe↔NIC 拓扑图；
- [ ] 四个 collective 的 size sweep；
- [ ] 能解释 ring/tree 与 channel 数为什么随消息大小和拓扑变化；
- [ ] 能说明 P2P 可达不等于链路一定走 NVLink。

---

## 5. D2：单机多卡并行抽象

### 5.1 DDP

- 每卡完整模型；
- backward 时 bucketed gradient AllReduce；
- 学习 bucket、overlap、`no_sync`、gradient accumulation。

### 5.2 FSDP2

- 用 `fully_shard` 理解参数、梯度、优化器状态分片；
- forward 前 AllGather，backward 后 ReduceScatter；
- 记录 peak memory 与通信增加。

### 5.3 DTensor + DeviceMesh + TP

```text
DeviceMesh：物理/逻辑 rank 网格
Placement：Shard / Replicate / Partial
DTensor：带全局 shape 和布局语义的分布式张量
```

从 `ColumnwiseParallel`、`RowwiseParallel` 开始，观察算子前后 placement 与 collective。

验收：

- [ ] 单卡、DDP、FSDP2 三组 loss 对齐；
- [ ] 显存、step time、吞吐表；
- [ ] 能画一层 MLP 的 TP 通信；
- [ ] 能解释 DDP/FSDP/TP 解决的是容量还是吞吐/延迟问题。

---

## 6. D3：真正的多节点

### 6.1 网络路径

```text
GPU HBM
 -> PCIe/NVLink
 -> NIC
 -> InfiniBand 或 RoCE fabric
 -> remote NIC
 -> remote GPU
```

GPUDirect RDMA 的重点是 NIC 与 GPU memory 的直接数据路径，减少经过 host bounce buffer；它不等于“完全没有 CPU 控制参与”。

### 6.2 启动

两节点示意：

```bash
# node 0
torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 \
  --master-addr=<node0-ip> --master-port=29500 train.py

# node 1
torchrun --nnodes=2 --nproc-per-node=8 --node-rank=1 \
  --master-addr=<node0-ip> --master-port=29500 train.py
```

### 6.3 排障顺序

```text
host reachability
-> NIC link/layer/rate
-> GPU-NIC affinity and topology
-> low-level RDMA test
-> nccl-tests
-> minimal torch.distributed
-> training job
```

不要一上来调模型。先让网络和 NCCL baseline 正常。

验收：

- [ ] 两机 `nccl-tests` AllReduce/AllToAll size sweep；
- [ ] 两机 DDP loss 与单机一致；
- [ ] 记录跨机/单机 bandwidth 比值；
- [ ] 人为制造错误 NIC 或 rank mismatch，留下一次排障记录。

---

## 7. D4：拓扑感知混合并行

原则不是死记“TP 节点内、PP 跨节点”，而是比较每种并行的通信频率、消息大小和同步敏感性。

| 并行维 | 常见通信 | 频率/敏感性 | 常见放置 |
|--------|----------|-------------|----------|
| TP | AllReduce/AllGather/ReduceScatter | 每层，高频、延迟敏感 | NVLink/NVSwitch 域内 |
| FSDP/DP | AllGather/ReduceScatter/AllReduce | 每 step，可 bucket/overlap | 节点内外均可 |
| PP | P2P activation | stage 边界，消息较少 | 可跨节点 |
| CP/SP | Ring/P2P/AllGather | attention 内高频 | 优先高速域 |
| EP | AllToAll dispatch/combine | 每个 MoE layer | 分层 NVLink + RDMA |

学习 `DeviceMesh`：例如 2 nodes × 8 GPUs 构造 `(dp=2, tp=8)`，将 TP 放节点内、DP 放节点间；再比较 `(fsdp=2, tp=8)` 的通信与显存。

验收：

- [ ] 给 16/64/256 GPU 画 rank mesh；
- [ ] 手算每层或每 step 的主要通信量；
- [ ] 用理论链路带宽估算通信下界；
- [ ] 解释 topology-aware rank placement。

---

## 8. D5：MoE、GPU 发起通信与重叠

现代 EP 不只是调用普通 `all_to_all`：

- dispatch/combine 的 token layout 与 expert load balance；
- 节点内 NVLink、节点间 RDMA 的 hierarchical forwarding；
- high-throughput prefill/training 与 low-latency decode 使用不同策略；
- 通信 kernel 占用 SM 会与 GEMM 竞争；
- GPU-initiated communication、copy engine、low-SM/zero-SM 路径用于释放计算资源；
- micro-batch/pipeline 将通信与计算重叠。

读码顺序：PyTorch AllToAll → DeepEP API/benchmark → profile-data 时间线 → NCCL EP 论文/实现。

验收：

- [ ] 画 dispatch → expert GEMM → combine 数据流；
- [ ] 区分 logical bandwidth 与 physical link bandwidth；
- [ ] 解释 prefill HT 与 decode LL 的目标不同；
- [ ] 用 Nsight 时间线判断是否真正 overlap。

---

## 9. 通信量与扩展效率手算

### Ring AllReduce

每 rank 通信量：

```text
2 * (N - 1) / N * message_size
```

### 理论通信时间下界

```text
T_comm >= bytes_on_bottleneck_link / effective_bandwidth
```

还需加 launch、协议、同步、拥塞、跨 rail 与 slow rank 开销。

### Strong scaling

固定全局问题规模：

```text
efficiency = T_1 / (N * T_N)
```

### Weak scaling

每卡工作量固定：比较 `T_1 / T_N` 或吞吐随 GPU 数增长的接近线性程度。必须注明 global/local batch 是否变化。

---

## 10. 精选资料

### 官方/项目

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [NCCL performance and tuning](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/performance_and_tuning.html)
- [NCCL networking troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/networking_troubleshooting.html)
- [CUDA GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [PyTorch Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
- [PyTorch DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html)
- [PyTorch Tensor Parallel](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [nccl-tests](https://github.com/NVIDIA/nccl-tests)
- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [DeepSeek profile-data](https://github.com/deepseek-ai/profile-data)

### 论文路由

- ZeRO → 分片基础；
- Megatron-LM / Efficient Large-Scale Training → TP/PP/DP；
- MegaScale → 大规模网络、重叠、可观测性、故障；
- ByteScale → 长上下文动态 mesh 与负载均衡；
- NCCL EP / DeepEP → 现代 MoE 通信；
- 2026 Collective Communication taxonomy → planning/runtime/coordination 全景。

最新入口统一见 [papers/watchlist-2026.md](../papers/watchlist-2026.md)。

---

## 11. 最终项目

做一份 2 nodes × 8 GPUs（无条件时先 2×2）报告：

1. 拓扑与软件环境；
2. 四个 collective 的 size sweep；
3. DDP/FSDP2/TP 中至少两种端到端实验；
4. 理论通信下界与实测差距；
5. 一次 hang 或 slow-rank 排障；
6. 一张 compute-communication timeline；
7. 一套 64 GPU 混合并行设计与取舍。

完成标准：正确性、性能数字、可复现命令、故障记录、面试口径齐全，才回写 `PATH.md`。
