# 剩余理论主题速览：先懂再深钻

> 目标：把 PATH 里还标着“待写”的主题按类别补齐入口，用“一句话 + 核心机制 + 最小验证 + 面试口径”讲清楚。
> 定位：这是入门速览，不是替代每篇精读笔记。每个主题后续还要按 [notes 规则](README.md) 写成独立笔记。
> 状态：`🚧 速览` 表示 Agent 已生成、用户还没消化，不计入“已学”。

---

## 目录

1. [A. GPU 优化算法](#a-gpu-优化算法)
2. [B. 注意力与序列建模](#b-注意力与序列建模)
3. [C. 量化](#c-量化)
4. [D. 推理系统](#d-推理系统)
5. [E. 分布式训练](#e-分布式训练)
6. [F. 模型结构补充](#f-模型结构补充)
7. [怎么使用这份速览](#怎么使用这份速览)

---

## A. GPU 优化算法

### A1. Norm 的 reduce 模式（LayerNorm / RMSNorm）

**一句话**：Norm 是对每一行做一次规约，再按规约结果做 elementwise 缩放，所以它和 Softmax 一样是 memory-bound 算子。

**为什么重要**：几乎所有 LLM 都有 Norm；训练/推理框架里它是稳定存在的基础算子，也是面试手写 kernel 的常见题。

**核心机制**：

```text
LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
RMSNorm:   y = x / sqrt(mean(x^2) + eps) * gamma
```

- LayerNorm 需要 mean 和 variance 两次 reduce；RMSNorm 只算 mean(x²)。
- 常见实现可以两遍读：先 reduce，再 normalize；也可以 fused，尽量少读 HBM。
- 和 Softmax 的优化套路一致：shared memory tree reduce、warp shuffle、多 block 并行。

**对 Infra 影响**：

- Norm 在 decode 阶段也是 HBM 密集。
- 融合 residual + norm + 下一个 GEMM 可以减少 kernel launch 和中间读写。
- Triton 写 Norm 比 CUDA 更简单，但要验证编译器是否真的少读了一遍。

**最小验证**：用 [reference/cuda/layernorm/layernorm.cu](../../reference/cuda/layernorm/layernorm.cu) 对照，写一个 RMSNorm CUDA 或 Triton kernel，对比 PyTorch 输出并记录带宽。

**面试口径**：“Norm 是 row-wise reduce，瓶颈是 HBM；优化方向是减少 pass、warp reduce、尽量保持 coalesced access。”

---

### A2. Work partitioning（FlashAttention-2 的思路）

**一句话**：FlashAttention-2 调整了 thread block 内的工作分配，减少非矩阵乘操作，让 tensor core 更忙。

**为什么重要**：FlashAttention V1 已经用 tiling + online softmax 解决显存；V2 主要解决“算力没打满”的问题。

**核心机制**：

- V1 每个 thread block 负责一个 Q tile，遍历所有 KV tile。
- V2 把 Q tile 拆得更细，每个 warp 负责一部分 Q rows，减少寄存器压力和同步。
- 减少 `rescale` 和 `softmax` 等非 GEMM 操作在关键路径上的占比。
- V3 更偏向硬件调度、TMA 和 warp specialization，方向仍是“让算力更忙、访存更高效”。

**对 Infra 影响**：

- 读 kernel 时，除了看 tiling，还要看每个 warp 负责多少行、同步点在哪。
- 性能对比不能只看 kernel 数量，要看 tensor core 利用率和非 GEMM 开销。

**最小验证**：读 [flash-attention-2.md](../../papers/attention/flash-attention-2.md)，在笔记里画 V1/V2 的 tile 和 warp 分工图。

**面试口径**：“FlashAttention V1 解决 IO，V2 解决并行度和非 matmul 开销，核心是 tile 划分和工作分配。”

---

## B. 注意力与序列建模

### B1. MHA -> MQA -> GQA -> MLA

**一句话**：Attention 的 KV 共享程度越来越高，KV cache 越来越小，是“省显存换少量计算/精度”的结构演进。

**核心机制**：

| 变体 | K/V head 数 | KV cache 趋势 |
|------|------------|--------------|
| MHA | 每个 query head 一组 K/V | 最大 |
| MQA | 所有 query head 共享一组 K/V | 最小 |
| GQA | 分成 G 组共享 | 折中 |
| MLA | 把 K/V 压成低秩 latent | 最省 |

**最小验证**：手算三种结构的 KV cache 大小，例如 4096 seq、32 layers、head_dim=128、fp16。

```text
KV = 2 * num_kv_heads * head_dim * seq_len * layers * 2 bytes
```

**面试口径**：“从 MHA 到 GQA 是减少 KV head 数，MLA 是低秩压缩；都是为了减少 Decode 必须读的 KV 字节。”

详细见 [最新模型与结构](latest-model-architectures.md)。

---

### B2. 线性注意力 / Ring Attention

**一句话**：线性注意力把 QK^T 的显式矩阵改成先合并 K/V，让复杂度从 O(seq²) 降到 O(seq)；Ring Attention 则用分块 + 通信支持超长序列。

**核心机制**：

```text
标准: out = softmax(QK^T) @ V
线性: out = Q @ (K^T @ V)   // 先合并 K/V
```

- 代价是表达能力受限，很多方案用局部 attention 或 SSM 混合补偿。
- Ring Attention 把序列切成块，在设备间环形传递 KV，避免一次放完整矩阵。

**对 Infra 影响**：

- 长上下文 benchmark 更依赖这类实现。
- 通信和计算可以 overlap，但调度复杂度高。

**最小验证**：手算 seq=4096 时标准 attention 和线性 attention 的 FLOPs/KV 对比；读一篇 linear attention 或 Ring Attention 论文摘要即可。

**面试口径**：“标准 attention 是 O(seq²) 显存/计算；线性 attention 通过改变结合顺序降复杂度，但精度和实现都更复杂。”

---

### B3. FlashAttention 1/2/3 对比

**一句话**：FA1 解决显存和 IO，FA2 解决并行度和非 matmul 开销，FA3 更多利用 Hopper 硬件特性。

| 版本 | 主要贡献 | 关注点 |
|------|---------|--------|
| FA1 | tiling + online softmax，O(N) 显存 | HBM 读写 |
| FA2 | warp/tile 分工优化 | tensor core 利用率 |
| FA3 | TMA / warp specialization / Hopper | 硬件调度 |

**面试口径**：“FlashAttention 的演进从 IO 优化走向硬件适配。”

---

## C. 量化

### C1. 量化基础：INT8 / FP8

**一句话**：用更低精度表示权重/激活/KV，省显存和带宽；不同方法解决不同精度问题。

**核心机制**：

```text
quantize(x) = round(x / scale) + zero_point
dequantize(q) = (q - zero_point) * scale
```

- 对称量化通常只算 scale，非对称量化还带 zero_point。
- FP8 有 E4M3/E5M2 两种格式，E4M3 精度更高、范围小，E5M2 范围大、精度低。

**最小验证**：读 [quantization-int8-fp8.md](quantization-int8-fp8.md)，手算一个 tensor 的 scale/zero_point。

**面试口径**：“量化把数值域映射到低精度，重点是找对 scale 和量化粒度。”

---

### C2. GPTQ

**一句话**：训练后逐层量化权重，最小化权重量化误差，是 LLM 权重量化常用的 baseline。

**核心机制**：

- 按 layer 收集输入激活，作为校准数据。
- 用近似二阶信息（Hessian）逐列更新权重。
- 一次量化一层，减少误差累积。

**对 Infra 影响**：

- 通常只量化权重，激活仍是高精度。
- 推理时加载 INT4/INT8 权重，能明显降显存和带宽。
- 需要校准集，离线完成，不影响在线服务。

**最小验证**：在 vLLM 里用 AWQ/GPTQ 模型跑一次 benchmark，对比 FP16 的显存和 TPOT。

**面试口径**：“GPTQ 按 layer 逐层校准，用二阶信息做权重量化，目标是让量化后的权重尽量保持原输出。”

---

### C3. AWQ

**一句话**：不只看权重，还看激活的重要性，保护少数重要 channel。

**核心机制**：

- 统计每个 channel 的激活幅度。
- 对重要 channel 用更小的 scale，降低量化误差。
- 比 GPTQ 更少依赖校准数据，速度通常更快。

**面试口径**：“AWQ 认为权重重要性由激活决定，所以按激活统计保护重要 channel。”

---

### C4. SmoothQuant

**一句话**：把激活的难量化部分“迁移”到权重，解决激活离群值。

**核心机制**：

- 激活离群值导致激活量化困难。
- 通过数学变换把激活的 scale 迁移到权重，让两边都更容易量化。
- 通常用于 W8A8 全量化。

**面试口径**：“SmoothQuant 把激活的量化难度转移到权重，使 W8A8 更可行。”

---

### C5. KV cache 量化

**一句话**：把 K/V 降到 INT8/FP8，省显存但需要校准和测试精度。

**核心机制**：

- 按 token 或按 channel 选择 scale。
- 精度敏感度因层而异，可以只量化部分层。
- 和 GQA/MLA 可以叠加使用。

**最小验证**：在 vLLM 里开 KV cache 量化，跑不同 seq_len 对比显存和困惑度。

**面试口径**：“KV cache 量化省的是 Decode 带宽和显存，但要验证长上下文和低 bit 下的精度。”

---

### C6. 量化方法对比表

| 方法 | 量化对象 | 是否需要校准 | 主要代价 |
|------|---------|:--:|---------|
| GPTQ | 权重 | 是 | 离线时间长 |
| AWQ | 权重 | 少 | 对激活分布敏感 |
| SmoothQuant | 权重+激活 | 是 | 需要重写 kernel |
| KV cache 量化 | KV | 是 | 精度风险 |

---

## D. 推理系统

### D1. Prefill vs Decode

**一句话**：Prefill 一次处理完整 prompt，compute-bound；Decode 逐 token 生成，memory-bound。

| 维度 | Prefill | Decode |
|------|---------|--------|
| 计算量 | 大，可并行 | 小 |
| 瓶颈 | 算力 | HBM 带宽 |
| 优化 | Flash Attention、chunked prefill | 权重量化、KV 优化、speculative decoding |

**面试口径**：“Prefill 是 compute-bound，Decode 是 memory-bound，所以两个阶段要分开优化。”

---

### D2. Continuous Batching / Chunked Prefill

**一句话**：Continuous batching 让请求不按 batch 整体等待，iteration 级调度随时加入/退出；chunked prefill 把长 prefill 切块，避免长时间霸占 GPU。

**核心机制**：

- 传统静态 batching 要等整批生成完，GPU 空闲多。
- Continuous batching 每个 iteration 重新组合请求，完成即退出，新请求即加入。
- Prefill 计算密集、Decode 带宽密集；chunked prefill 把 prefill 拆成小块和 decode 混跑。

**最小验证**：读 vLLM scheduler 源码，画出“一个 iteration 里 prefill/decode 请求如何混合”。

**面试口径**：“连续批处理用 iteration-level scheduling 提高 GPU 利用率；chunked prefill 解决 prefill 和 decode 互相干扰。”

---

### D3. Prefix Cache / RadixAttention

**一句话**：多轮对话和 RAG 请求会重复相同前缀，前缀缓存复用这些 KV，RadixAttention 用树结构管理共享前缀。

**核心机制**：

- vLLM 的 prefix cache 常用 hash 精确匹配整段前缀。
- SGLang 的 RadixAttention 用 radix tree，允许部分前缀共享和 copy-on-write。
- 代价是 KV 管理更复杂，需要 eviction、refcount、cache hit 调度。

**最小验证**：读 [SGLang 文档](https://docs.sglang.ai/) 或源码，画一个多轮对话的 radix tree 变化。

**面试口径**：“前缀缓存复用重复 KV，RadixAttention 用树做最长前缀匹配，能省 prefill 但增加显存管理复杂度。”

---

### D4. Speculative Decoding

**一句话**：小模型先 draft，大模型再 verify，让多个 token 在一次 forward 中验证。

**核心机制**：

- 小模型快速生成候选 token。
- 大模型一次验证多个候选，接受一致部分，回退不一致部分。
- 收益取决于 draft 接受率和额外显存。

**面试口径**：“投机解码用 draft-verify 提高 token/s，但收益依赖 draft 模型质量和额外显存。”

详细见 [投机解码](speculative-decoding.md)。

---

### D5. PD 分离

**一句话**：把 Prefill 和 Decode 放到不同 GPU/实例，避免两种 workload 互相干扰。

**核心机制**：

- Prefill 实例负责长 prompt 计算。
- Decode 实例负责低延迟生成。
- 之间要传 KV cache 或中间状态，传输开销是主要挑战。

**面试口径**：“PD 分离用硬件隔离解决 prefill/decode 的资源争抢，但要处理 KV 传输。”

详细见 [PD 分离](pd-disaggregation.md)。

---

## E. 分布式训练

### E1. NCCL / 集合通信

**一句话**：NCCL 是 GPU 间集合通信库，AllReduce/ReduceScatter/AllGather/All-to-All 是分布式训练的基础。

**核心机制**：

- AllReduce：所有卡得到完整结果。
- ReduceScatter：每卡得到部分结果。
- AllGather：每卡把部分结果广播成完整结果。
- All-to-All：每卡和所有卡交换不同数据。

**最小验证**：画 Ring AllReduce 的通信量，理解为什么通信量随卡数增长但每卡负载可控。

**面试口径**：“集合通信按数据分布和归约方式分，通信量和带宽是分布式训练的第一账本。”

---

### E2. DDP / FSDP

**一句话**：DDP 复制模型、同步梯度；FSDP 分片参数，训练更大模型。

| 方案 | 参数 | 梯度 | 优化器状态 | 通信 |
|------|------|------|-----------|------|
| DDP | 每卡全量 | 每卡全量 | 每卡全量 | AllReduce 梯度 |
| FSDP | 分片 | 分片 | 分片 | AllGather 参数 + ReduceScatter 梯度 |

**面试口径**：“DDP 是数据并行 + 梯度同步；FSDP 是参数/梯度/优化器状态分片，省显存换通信。”

---

### E3. ZeRO / FSDP

**一句话**：ZeRO 把优化器状态、梯度、参数分片到多卡，让单卡显存不再必须装下完整模型；FSDP 是 PyTorch 对 ZeRO-3 风格参数分片的实现。

**核心机制**：

- ZeRO-1：分片优化器状态。
- ZeRO-2：分片优化器状态 + 梯度。
- ZeRO-3：再分片参数，前向/反向时 all-gather 临时取回。
- FSDP 使用类似思路，把模型切成 flat parameter shards。

**最小验证**：手算 7B 模型 fp16 + AdamW 的显存账本：参数 14GB、梯度 14GB、fp32 master 28GB、Adam m/v 各 28GB，合计约 112GB；再算 ZeRO-3 分片后每卡多少。

**面试口径**：“ZeRO 按参数/梯度/优化器状态依次分片，FSDP 是 PyTorch 的 ZeRO-3 风格实现；分片越多显存越省，通信越贵。”

---

### E4. TP / PP / EP 通信

**一句话**：张量并行把单个矩阵切成多卡算，流水线并行把层切到多卡，专家并行把 expert 分布到多卡；它们解决不同瓶颈，通信模式不同。

**核心机制**：

- TP：权重分片，forward 后需要 all-reduce；通信频率高，通常需要 NVLink。
- PP：层分片，卡间传 activation；通信次数少但延迟叠加，有 bubble。
- EP：MoE expert 分片，token 需要 AllToAll 到对应 expert 所在卡。

**最小验证**：画一个 2 卡 TP、4 卡 PP、MoE 8 expert 的通信图，标出每步传输的数据量。

**面试口径**：“TP 切权重、PP 切层、EP 切 expert；TP 通信最频繁，PP 有 bubble，EP 有 AllToAll 负载均衡问题。”

---

### E5. 混合精度 / Activation Checkpointing

**一句话**：混合精度用 FP16/BF16/FP8 降低显存和加速；activation checkpointing 不存全部激活，反向时重算。

**核心机制**：

- FP16 范围小容易溢出，BF16 和 FP32 范围接近但精度低。
- 梯度更新时通常保留 FP32 master copy。
- activation checkpointing 用“时间换显存”，重算代价约一个 forward。

**面试口径**：“混合精度省显存，激活 checkpointing 用重算换显存，适合大模型训练。”

---

## F. 模型结构补充

### F1. Mamba / SSM

**一句话**：SSM 用固定大小的 hidden state 做序列建模，推理时逐 token 更新状态，避免 Attention 的二次复杂度。

**核心机制**：

```text
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

- Mamba 的关键是 selective scan：根据输入动态选择保留/遗忘信息。
- 混合架构（如 Jamba）把 Attention 和 SSM 层交替，兼顾长距离和表达力。
- GPU 上实现 SSM 需要专门 kernel，不是简单 PyTorch 循环。

**面试口径**：“SSM 是线性时间状态更新，长序列省显存；但状态容量有限，常见做法是和 Attention 混合。”

### F2. MoD / MTP

- MoD（Mixture-of-Depths）：不是每个 token 都经过每一层，减少激活计算。
- MTP（Multi-Token Prediction）：一次预测多个未来 token，训练和投机解码相关。

**面试口径**：“MoD 是稀疏计算，MTP 是训练目标/投机解码相关，两者都在改变‘每 token 必须算满每一层’的假设。”

---

## 怎么使用这份速览

1. 每次只挑一个主题，从“一句话”开始，能不看原文讲出来。
2. 做“最小验证”：能算账就手算，能跑就写一个小 kernel/脚本。
3. 把“面试口径”改成自己的话，不要背模板。
4. 一个主题能讲 3 分钟后，再把它升级成独立精读笔记，并更新 [PATH.md](../../PATH.md) 状态。

---

## 关联入口

- [最新模型与结构](latest-model-architectures.md) — 模型家族和组件全景
- [模型追踪表](model-tracker.md) — 最新模型、结构、学习状态
- [PATH.md](../../PATH.md) — 理论线全貌
- [roadmap/ai-infra-curriculum.md](../../roadmap/ai-infra-curriculum.md) — 怎么把这些主题排进学习节奏