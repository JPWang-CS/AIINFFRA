# ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

**Authors**: Rajbhandari, Rasley, Ruwase, He (Microsoft DeepSpeed)  
**Venue**: SC 2020 (Best Paper) | **arxiv**: [1910.02054](https://arxiv.org/abs/1910.02054)  
**实现**: DeepSpeed / PyTorch FSDP | **优先级**: P0 | **状态**: ✅ 精读 | **日期**: 2026-05-28

> **一句话**: 把模型状态（参数、梯度、优化器状态）分片到所有 GPU，消除数据并行的冗余，显存节省 Nd 倍（N 是 GPU 数），通信量只增加 50%。

---

## 为什么这篇论文重要

这是 **大模型训练的基石**：
- GPT-3 (175B) / Llama-7B+ 能训出来，ZeRO 是关键
- **PyTorch FSDP** 就是 ZeRO-3 的官方实现（Meta 抄的 DeepSpeed）
- 后续所有大模型训练框架（Megatron-DeepSpeed, Colossal-AI）都基于 ZeRO

**面试必考**："大模型训练怎么省显存？" → ZeRO + Tensor Parallelism。  
**算子线 D**（本仓库）的理论基础——虽然你不会手写 ZeRO，但要能讲清原理 + 画数据流图。

---

## 解决了什么问题

### 数据并行（DP）的显存冗余

**标准 DP**（PyTorch `DataParallel` / `DistributedDataParallel`）：
```
GPU 0: 完整模型权重 + 完整梯度 + 完整优化器状态
GPU 1: 完整模型权重 + 完整梯度 + 完整优化器状态
...
GPU N: 完整模型权重 + 完整梯度 + 完整优化器状态
```

**每个 GPU 都存一份完整的模型状态** → N 个 GPU 就有 N 份冗余。

### 模型状态的显存占用

以 Adam 优化器 + FP16 混合精度训练为例：
```
参数（FP16）:        2 bytes/param
梯度（FP16）:        2 bytes/param
优化器状态（FP32）:
  - momentum（FP32）: 4 bytes/param
  - variance（FP32）: 4 bytes/param
  - master copy（FP32）: 4 bytes/param

总计: 2 + 2 + 4 + 4 + 4 = 16 bytes/param
```

**1B 参数 = 16 GB 显存**（仅模型状态！还没算激活）。  
7B 模型 = 112 GB，A100 40GB 单卡根本放不下。

### 数据对比（问题严重性）

| 模型 | 参数量 | 模型状态显存 (Adam FP16) | 单卡 A100 (40GB) |
|:---:|:---:|:---:|:---:|
| BERT-Large | 0.3B | 5 GB | ✅ 能放 |
| GPT-2 | 1.5B | 24 GB | ❌ 放不下 |
| Llama-7B | 7B | 112 GB | ❌ 需要 3 卡 |
| GPT-3 | 175B | 2.8 TB | ❌ 需要 70 卡 |

标准 DP：8 卡训练 7B → **每卡都要 112 GB**（虽然只是训练，不是 8 个独立模型！）。

---

## 核心思想：Zero Redundancy（零冗余）

> **不需要的数据就不要存。** 每个 GPU 只存它负责更新的那部分参数 + 对应的梯度 + 优化器状态。

### 三个阶段（递进式优化）

| Stage | 分片内容 | 显存节省 (per GPU) | 通信量 vs DP | 何时用 |
|:---:|---|:---:|:---:|---|
| **ZeRO-1** | 优化器状态 | **4×** | = DP | 小模型（< 1B），单机多卡 |
| **ZeRO-2** | 优化器状态 + 梯度 | **8×** | = DP | 中模型（1-10B） |
| **ZeRO-3** | 优化器状态 + 梯度 + 参数 | **Nd×** | +50% vs DP | 大模型（10B+），必须用 |

**Nd** = GPU 数量。8 卡 ZeRO-3 → 每卡显存节省 **8 倍**。

---

## ZeRO-1: 分片优化器状态

### 标准 DP 的问题

```python
# 每个 GPU 都存完整的 optimizer state
optimizer = Adam(model.parameters())
# momentum: [全部参数] (FP32)
# variance: [全部参数] (FP32)
```

**冗余**：8 卡 DP，momentum/variance 存了 8 份，但它们计算出来完全一样（因为梯度 allreduce 后一致）。

### ZeRO-1 的做法

```python
# 每个 GPU 只存 1/N 的 optimizer state
rank = dist.get_rank()
world_size = dist.get_world_size()

# 参数分片（逻辑上）
params_per_rank = len(model.parameters()) // world_size
start = rank * params_per_rank
end = (rank + 1) * params_per_rank

# 只存自己负责的那部分
optimizer = Adam(model.parameters()[start:end])
```

**Forward/Backward**：正常跑（参数还是完整的）。  
**Optimizer Step**：
1. AllReduce 梯度（和标准 DP 一样）
2. 每个 rank 只更新自己负责的参数分片
3. AllGather 更新后的参数 → 所有 GPU 重新同步完整参数

**显存节省**：
- Optimizer state: 12 bytes/param → 12 / N bytes/param
- 参数 + 梯度：还是完整存储（2 + 2 = 4 bytes/param）
- **总计**: 16 → 4 + 12/N ≈ **4×** 节省（N=8 时）

**通信量**：
- 标准 DP: AllReduce 梯度（2ψ，ψ 是参数量）
- ZeRO-1: AllReduce 梯度 + AllGather 参数（2ψ + 2ψ = 4ψ）
- **增加**: 2× vs 标准 DP ❌

**优化**（论文 Section 3.2）：
- 用 Reduce-Scatter 替代 AllReduce（每个 rank 只收到自己负责的梯度）
- 通信量降回 2ψ，**和标准 DP 一样** ✅

---

## ZeRO-2: 分片梯度

### 问题

ZeRO-1 还是每个 GPU 存完整梯度（2 bytes/param）。能不能也分片？

### ZeRO-2 的做法

```python
# Backward 时，每层的梯度算完后：
# 1. Reduce-Scatter 到负责这层参数的 rank
# 2. 本地梯度立即释放（不全局存）

for layer in model.layers():
    loss.backward()  # 算出 layer.grad
    
    # Reduce-Scatter: 每个 rank 只收到自己负责的那部分梯度
    reduced_grad = reduce_scatter(layer.grad, group=world_group)
    
    # 释放本地完整梯度
    layer.grad = None
    
    # 存储自己负责的梯度分片
    layer.grad_shard = reduced_grad
```

**显存占用**：
- 参数: 2 bytes/param（完整）
- 梯度: 2 / N bytes/param（分片）
- Optimizer state: 12 / N bytes/param（分片）
- **总计**: 2 + 2/N + 12/N ≈ **2.x bytes/param**（N=8 时）

**vs 标准 DP (16 bytes/param)** → **8×** 节省 ✅

**通信量**：
- Reduce-Scatter 梯度: ψ
- AllGather 参数: 2ψ
- **总计**: 3ψ ≈ 标准 DP (2ψ) 的 1.5× ❌

但省了 8× 显存，通信多 50% 可以接受（显存是瓶颈）。

---

## ZeRO-3: 分片参数（核心）

### 终极目标

**连参数也分片** → 每个 GPU 只存 1/N 的参数 + 1/N 的梯度 + 1/N 的优化器状态。

### 挑战

参数分片了，Forward/Backward 怎么算？

**答案**：按需 AllGather，用完就扔。

### ZeRO-3 的完整流程

```python
# 初始化：每个 rank 只存自己负责的参数分片
rank = dist.get_rank()
world_size = dist.get_world_size()

for param in model.parameters():
    shard_size = param.numel() // world_size
    param.data = param.data[rank * shard_size : (rank + 1) * shard_size]
    # 现在 param.data 只有 1/N

# Forward pass
for layer in model.layers():
    # 1. AllGather 这一层的完整参数（临时）
    full_param = all_gather(layer.param.data, group=world_group)
    
    # 2. 用完整参数做 forward
    output = layer.forward(input, full_param)
    
    # 3. 立即丢弃完整参数（释放显存）
    del full_param
    
    # 现在显存里只有这一层的激活（下一层要用）

# Backward pass
for layer in reversed(model.layers()):
    # 1. AllGather 这一层的完整参数（临时）
    full_param = all_gather(layer.param.data, group=world_group)
    
    # 2. 算梯度
    grad = layer.backward(grad_output, full_param)
    
    # 3. Reduce-Scatter 梯度（每个 rank 只收自己负责的分片）
    grad_shard = reduce_scatter(grad, group=world_group)
    layer.grad_shard = grad_shard
    
    # 4. 丢弃完整参数 + 完整梯度
    del full_param, grad

# Optimizer step
for layer in model.layers():
    # 每个 rank 只更新自己负责的参数分片
    optimizer.step(layer.param.data, layer.grad_shard)
```

**关键洞察**：
- **Forward/Backward 时**：临时 AllGather 参数 → 用完立即释放
- **显存峰值**：某个时刻只有"当前层的完整参数 + 激活"，不是所有层一起存
- **通信**：每层 Forward 一次 AllGather（2ψ/L，L 是层数）+ Backward 一次 AllGather + Reduce-Scatter（4ψ/L）

### 显存占用（ZeRO-3）

```
参数分片: 2 / N bytes/param
梯度分片: 2 / N bytes/param
Optimizer state 分片: 12 / N bytes/param

总计: 16 / N bytes/param
```

**7B 模型 (112 GB)，8 卡 ZeRO-3**:  
每卡只需 **14 GB**（模型状态）+ 激活 → A100 40GB 能放下 ✅

**vs 标准 DP (112 GB/卡)** → **8×** 节省 ✅

### 通信量（ZeRO-3）

```
Forward: 每层 AllGather 参数 (2ψ/L)
Backward: 每层 AllGather 参数 (2ψ/L) + Reduce-Scatter 梯度 (2ψ/L)
总计: 6ψ/L × L = 6ψ
```

**vs 标准 DP (2ψ AllReduce)** → **3×** 通信量 ❌

但能训 10× 更大的模型（显存是瓶颈），通信多 3× 可接受。

---

## 数据流图（ZeRO-3 Forward）

```
GPU 0 持有: Param[0:N/8] (分片)
GPU 1 持有: Param[N/8:N/4]
...
GPU 7 持有: Param[7N/8:N]

Layer 1 Forward:
  Step 1: AllGather Param_Layer1
    GPU 0 → broadcast Param[0:N/8]
    GPU 1 → broadcast Param[N/8:N/4]
    ...
    所有 GPU 现在有完整 Param_Layer1
  
  Step 2: 每个 GPU 独立算 forward（输入不同，data parallel）
    GPU 0: output_0 = Layer1(input_0, Param_Layer1)
    GPU 1: output_1 = Layer1(input_1, Param_Layer1)
  
  Step 3: 释放 Param_Layer1 的 AllGather 副本
    所有 GPU 只保留自己负责的分片
```

---

## 与 Tensor Parallelism (TP) 的区别

| 方法 | 分片对象 | 每个 GPU 算什么 | 通信 | 适用 |
|---|---|---|---|---|
| **Data Parallel (DP)** | 数据（每 GPU 不同 batch） | 完整前向/反向 | AllReduce 梯度 | 小模型 |
| **Tensor Parallel (TP)** | 参数（权重矩阵按列/行切） | 部分计算（矩阵乘的一部分） | AllReduce 激活 | 大模型（单层放不下） |
| **ZeRO (FSDP)** | 参数 + 梯度 + 优化器状态 | 完整前向/反向（临时 gather 参数） | AllGather 参数 + RS 梯度 | 大模型（显存不够） |

**组合使用**（Megatron-DeepSpeed）：
```
8 机 64 卡训练 GPT-3:
  - TP=8 (单机内，层内并行，低延迟)
  - ZeRO-3 (跨机，层间参数分片)
  - Pipeline Parallel (层分到不同机器)
```

---

## 性能数据（论文 Table 3-5）

### vs 标准 DP (PyTorch DDP)

| 模型 | 配置 | DDP 显存/卡 | ZeRO-3 显存/卡 | 加速比 |
|:---:|---|:---:|:---:|:---:|
| 1.5B | 64 V100 (32GB) | OOM | 18 GB | ∞ |
| 10B | 128 V100 | OOM | 28 GB | ∞ |
| 100B | 400 V100 | OOM | 31 GB | ∞ |

**结论**：DDP 根本跑不了，ZeRO-3 能跑。

### vs Megatron (TP only)

| 模型 | Megatron (TP=4) | DeepSpeed (ZeRO-3) | 吞吐比 |
|:---:|:---:|:---:|:---:|
| 20B (64 V100) | 15 samples/s | **18 samples/s** | 1.2× |

ZeRO-3 吞吐略高，因为通信更高效（TP 的 AllReduce 激活更频繁）。

### 端到端训练时间（GPT-2 1.5B）

| 方法 | 16 V100 训练时间 (1 epoch) |
|---|:---:|
| PyTorch DDP | OOM |
| Megatron TP=4 | 12.5 小时 |
| **DeepSpeed ZeRO-2** | **10.8 小时** |

---

## 实现细节（PyTorch FSDP）

### 基本用法

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = MyModel()

# Wrap 成 FSDP（自动做 ZeRO-3）
model = FSDP(
    model,
    sharding_strategy="FULL_SHARD",  # ZeRO-3
    # sharding_strategy="SHARD_GRAD_OP",  # ZeRO-2
    cpu_offload=None,  # 不 offload 到 CPU
)

# 训练照常
optimizer = Adam(model.parameters())
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

FSDP 会自动：
- Forward 前 AllGather 参数
- Backward 后 Reduce-Scatter 梯度
- 释放临时的完整参数

### 关键参数

```python
FSDP(
    model,
    sharding_strategy="FULL_SHARD",  # ZeRO-3: 分片参数+梯度+优化器
    cpu_offload=CPUOffload(offload_params=True),  # ZeRO-Offload: 参数卸到 CPU
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Overlap 通信和计算
    mixed_precision=MixedPrecision(...),  # FP16 训练
)
```

---

## 在 Ascend 的对应

| DeepSpeed (NVIDIA) | Ascend (HCCL) | 备注 |
|---|---|---|
| NCCL AllGather | HCCL AllGather | 集合通信原语一致 |
| NCCL Reduce-Scatter | HCCL ReduceScatter | |
| GPU Global Memory | NPU HBM | |
| ZeRO-3 逻辑 | 通用（跨平台） | 只是通信库换成 HCCL |

**可移植性**：ZeRO 是纯软件逻辑（参数分片 + 通信），不依赖硬件特性。昇腾训 Llama 用的也是 ZeRO-3（通过 MindSpore 或 Megatron-DeepSpeed）。

---

## 与我何干（学习路径）

### D1 — DP/FSDP/TP/PP 概念（算子线 D）
理解四种并行方式：
- **DP**（数据并行）：标准 PyTorch DDP
- **FSDP**（ZeRO-3）：本篇论文
- **TP**（Tensor 并行）：Megatron，权重矩阵切片
- **PP**（Pipeline 并行）：层切到不同 GPU，流水线

能画出每种的数据流图 + 通信模式。

### D2 — 通信原语（AllReduce / AllGather / RS）
- AllReduce = Reduce + Broadcast
- AllGather = 每个 rank broadcast 自己的数据
- Reduce-Scatter = 每个 rank 只收自己负责的那部分 reduce 结果

能手画这三个的通信拓扑（ring / tree）。

### 面试必考题

**Q1: ZeRO 是什么？**  
A: 把模型状态（参数 + 梯度 + 优化器状态）分片到所有 GPU，消除数据并行的冗余。ZeRO-3 能节省 Nd 倍显存（N 是 GPU 数）。

**Q2: ZeRO 三个阶段的区别？**  
A: ZeRO-1 只分片优化器状态（4× 节省）；ZeRO-2 分片优化器 + 梯度（8×）；ZeRO-3 分片所有（Nd×），但通信量增加 50%。

**Q3: ZeRO-3 Forward 时参数怎么办？**  
A: 临时 AllGather 当前层的完整参数 → 算 forward → 立即释放。显存峰值只有"当前层参数 + 激活"，不是所有层一起存。

**Q4: ZeRO vs Tensor Parallelism？**  
A: ZeRO 是数据并行的优化（分片模型状态，减少冗余），TP 是模型并行（权重矩阵切片，单层放不下时用）。可以组合：TP 单机内（低延迟），ZeRO 跨机（省显存）。

**Q5: PyTorch FSDP 是什么？**  
A: PyTorch 官方的 ZeRO-3 实现。API 类似 DDP，但自动做参数分片 + AllGather/RS。

**Q6: 通信量增加多少？**  
A: ZeRO-3 是标准 DP 的 1.5×（AllGather 参数 + RS 梯度 vs 只 AllReduce 梯度）。但能训 10× 更大的模型，trade-off 值得。

---

## 代码对照

### DeepSpeed 官方
- [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)
- ZeRO 实现在 `deepspeed/runtime/zero/`

### PyTorch FSDP
- [torch.distributed.fsdp](https://pytorch.org/docs/stable/fsdp.html)
- Tutorial: [Getting Started with FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

### 本仓库参考
D 线分布式（算子线）走到 D1 时，我会给你一个 FSDP 的 minimal example（训一个小模型，对比 DDP/FSDP 的显存占用）。

---

## 扩展：ZeRO-Offload / ZeRO-Infinity

### ZeRO-Offload (2020)
- 优化器状态 offload 到 CPU 内存
- 显存再省 4×，但训练慢 10-20%（CPU-GPU 数据传输）
- 适合单机多卡 + 显存紧张

### ZeRO-Infinity (2021)
- 参数 + 优化器状态 offload 到 NVMe SSD
- 能训 **万亿参数模型**（单机！）
- 慢 2-3×，但能跑就是胜利

---

## 参考资料

- **论文**: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- **DeepSpeed 官方**: [www.deepspeed.ai](https://www.deepspeed.ai/)
- **PyTorch FSDP 文档**: [FSDP](https://pytorch.org/docs/stable/fsdp.html)
- **配套课程**: D1 分布式训练（算子线 D），[roadmap/distributed.md](../../roadmap/distributed.md)
- **视频**: [Microsoft Research Talk](https://www.microsoft.com/en-us/research/video/zero-memory-optimizations-toward-training-trillion-parameter-models/) (1 小时，作者讲解)

---

*D 线分布式的理论基础。能讲清 ZeRO-3 的数据流 + 通信模式，面试大模型训练岗位的硬通货。*
