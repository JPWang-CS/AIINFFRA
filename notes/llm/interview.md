# 大模型板块：面试

> 子板块目标：把大模型相关面试题按模块整理，每个题都要能给出“结论 + 取舍 + 证据/数字”。
> 状态：🚧 草稿。

---

## 1. 面试准备原则

- 不要背答案，要能画图、能推公式、能算数。
- 每个题准备一个 1 分钟版本和 3 分钟版本。
- 用仓库里的 `solutions/`、`weekly/`、`notes/` 做项目证据。

---

## 2. 模型结构高频题

| 题 | 要能回答 |
|----|---------|
| MHA/GQA/MQA/MLA 区别 | KV cache 大小、Decode 带宽、代价 |
| MoE 为什么难 | 权重显存、AllToAll、负载均衡 |
| RoPE 为什么外推困难 | 相对位置旋转、插值/NTK/YaRN |
| RMSNorm vs LayerNorm | reduce 差异、为什么主流用 RMSNorm |
| Mamba 能替代 Attention 吗 | 线性复杂度、状态容量、混合架构 |
| Flash Attention 为什么快 | tiling、online softmax、HBM vs SRAM |
| Decode 为什么 memory-bound | 每 token 读全部权重和 KV |

---

## 3. 推理系统高频题

| 题 | 要能回答 |
|----|---------|
| Prefill vs Decode | compute-bound vs memory-bound |
| PagedAttention | block table、碎片化、copy-on-write |
| Continuous Batching | iteration 调度、GPU 利用率 |
| Chunked Prefill | 消除 prefill/decode 干扰 |
| Prefix Cache / RadixAttention | 前缀复用、树管理 |
| Quantization | GPTQ/AWQ/SmoothQuant/KV 量化 |
| Speculative Decoding | draft-verify、接受率 |
| PD 分离 | 资源隔离、KV 传输 |
| 长上下文 | KV cache、Flash Attention、Ring Attention |

---

## 4. 训练系统高频题

| 题 | 要能回答 |
|----|---------|
| 7B 模型训练显存 | 参数、梯度、master weight、Adam 状态 |
| DDP vs FSDP | 通信原语、显存、适用场景 |
| ZeRO-1/2/3 | 分片对象和通信量 |
| TP vs PP vs EP | 切什么、通信模式、适用场景 |
| 3D parallel 设计 | TP×PP×DP 拓扑、NVLink/跨节点 |
| 混合精度 | FP16/BF16/FP8、loss scaling、master weight |
| Activation checkpointing | 显存换重算 |
| MoE 并行 | AllToAll、负载均衡 |

---

## 5. 系统设计题

准备一两个端到端设计：

### 5.1 设计 LLM 推理服务

```text
QPS、并发、延迟、显存
-> KV cache 估算
-> PagedAttention + continuous batching
-> 量化
-> 多卡 TP/PP
-> 调度和容量规划
```

### 5.2 设计大模型训练集群

```text
模型大小、数据规模、卡数
-> 显存账本
-> TP/PP/DP 拓扑
-> 通信瓶颈
-> 故障恢复/checkpoint
```

---

## 6. Ascend → GPU 叙事

核心一句话：

> 我理解的不是某个 vendor 的 API，而是异构计算中“计算、访存、并行、数据搬运”的本质。

可展开：

- Ascend Cube/Vector/Scalar 与 CUDA SM/warp 的对应。
- Ascend L1/UB tiling 与 CUDA shared memory tiling。
- Ascend pipe/double buffer 与 CUDA 手动双缓冲。
- Ascend profiling 与 Nsight。
- 从算子优化到模型推理/训练系统。

---

## 7. 需要准备的项目

- 从仓库整理 3-5 个可讲项目：
  1. GEMM naive → tiled → 性能数字。
  2. Softmax 3-pass → online → benchmark。
  3. Flash Attention 读码/复现。
  4. vLLM 推理链路分析。
  5. FSDP/ZeRO 显存账本和最小 demo。

每个项目准备：

```text
背景 -> 目标 -> 方案 -> 关键数字 -> 取舍 -> 还可以怎么优化
```

---

## 8. 关联材料

- [面试题库](../../roadmap/interviews.md)
- [最新模型与结构](../algorithms/latest-model-architectures.md)
- [训练系统](training-systems.md)
- [推理系统](inference-systems.md)
