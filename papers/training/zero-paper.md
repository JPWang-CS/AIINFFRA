# ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

**Authors**: Rajbhandari, Rasley, Ruwase, He (Microsoft DeepSpeed)  
**Venue**: SC 2020 (Best Paper) | **arxiv**: [1910.02054](https://arxiv.org/abs/1910.02054)  
**实现**: DeepSpeed / PyTorch FSDP | **优先级**: P0 | **状态**: ✅ 精读 | **日期**: 2026-05-28

> **一句话**: ZeRO 按阶段消除数据并行中重复保存的优化器状态、梯度和参数；每个阶段的峰值显存与 collective 通信都不同，不能用一个固定倍数概括。

---

## 为什么这篇论文重要

论文的核心价值在于把“每个 rank 都保存一份模型状态”的冗余拆开，并给出状态分片、按需聚合和通信量之间的取舍。PyTorch FSDP 的 full-shard 路径与 ZeRO-3 有相近语义，但 API、参数重分片时机和实现细节应按版本文档核对。

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

以一种 Adam 混合精度配置为例：
```
参数（FP16）:        2 bytes/param
梯度（FP16）:        2 bytes/param
优化器状态（FP32）:
  - momentum（FP32）: 4 bytes/param
  - variance（FP32）: 4 bytes/param
  - master copy（FP32）: 4 bytes/param

总计: 2 + 2 + 4 + 4 + 4 = 16 bytes/param
```

在这组精度假设下，模型状态为 16 bytes/parameter；它不包含激活、临时 buffer、通信 bucket、碎片和框架保留。换用 BF16/FP32 梯度、低精度状态或卸载时，字节数需要重新列式。

### 数据对比（问题严重性）

标准 DP 在每个 rank 复制上述模型状态；能否放下取决于设备容量和其他活跃对象，而不是只由参数量决定。

---

## 核心思想：Zero Redundancy（零冗余）

> **不需要的数据就不要存。** 每个 GPU 只存它负责更新的那部分参数 + 对应的梯度 + 优化器状态。

### 三个阶段（递进式优化）

| Stage | 分片内容 | 显存节省 (per GPU) | 通信量 vs DP | 何时用 |
|:---:|---|:---:|:---:|---|
| **ZeRO-1** | 优化器状态 | 状态项按 rank 分片，参数/梯度仍复制 | 参数可完整驻留且状态是主要压力 |
| **ZeRO-2** | 优化器状态 + 梯度 | 状态与梯度按 rank 分片，参数仍复制 | 反向后可及时释放完整梯度 |
| **ZeRO-3** | 优化器状态 + 梯度 + 参数 | 参数按层按需 all-gather，再重新分片 | 单层或整模容量受限，接受更多通信 |

表中的“按 rank 分片”不等于固定 N 倍峰值节省：参数精度、梯度精度、激活、临时 gather、bucket 和分片边界都会改变实际峰值。

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

**容量核算**：若状态总计 12 bytes/param 且被均匀分片，ZeRO-1 的模型状态近似为完整参数与梯度 `4 + 12/N` bytes/param；真实峰值还要加入临时 buffer、bucket 和不均匀分片。

**通信量**取决于梯度归约与参数同步的 collective 组合。论文讨论了用 Reduce-Scatter 让每个 rank 直接得到自己负责的梯度分片，再在需要时同步参数；报告通信时应说明 payload、rank 数、算法和计量口径，不能把某一种实现的 2ψ 或 4ψ 当作阶段定义。

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

**容量核算**：在参数 2 bytes、梯度 2 bytes、优化器状态 12 bytes 的假设下，ZeRO-2 的静态模型状态近似为 `2 + 14/N` bytes/param，因为参数仍然完整复制。反向中产生的完整梯度可在 Reduce-Scatter 后释放；临时梯度和 bucket 仍会抬高峰值。

**通信量**：Reduce-Scatter 梯度、参数 all-gather/同步和实现中的 bucket 重叠共同决定通信。与标准 DP 的比较必须固定 collective 算法、payload、是否包含参数同步和是否发生重叠；“省多少显存、增加多少通信”不是只由 ZeRO-2 这个名称决定。

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
- **Forward/Backward 时**：临时 AllGather 当前模块参数，计算完成后按配置重新分片；参数是否在反向前保持未分片取决于实现。
- **显存峰值**：当前模块的 unsharded 参数、激活、梯度和通信 buffer 可能同时活跃；不是简单地把所有层参数相加。
- **通信**：每层的 all-gather、reshard 和 reduce-scatter 量取决于参数/梯度 dtype、rank 数、分片边界、bucket 与重叠，应该从实际 trace 或明确的 collective 模型计算。

### 显存占用（ZeRO-3）

```
若沿用前面的精度假设，参数、梯度和 optimizer state 的持久分片合计约为 `16 / N` bytes/param；临时 all-gather 参数、激活、通信 bucket 和 allocator 保留不在这个静态项内。
```

以 7B 参数和上述精度为例，持久模型状态的静态分片项可按 `7B × 16 / N` 估算；是否能放入某张卡还要加入当前层 all-gather、激活、通信 buffer 和设备余量。

### 通信量（ZeRO-3）

```
Forward/backward 的每层 all-gather、reshard 和 reduce-scatter 都对通信有贡献。若以参数量 ψ、参数/梯度字节数和 ring 等算法为输入，可以逐层累加 payload；不能从“ZeRO-3”单独推出固定的 6ψ。
```

与标准 DP 的通信比较应标明是否包含梯度 reduce、参数 all-gather、reshard、bucket 和重叠；模型可训练规模与吞吐也要在相同硬件和 batch 下实测。

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

论文的性能表是其硬件、模型、并行配置和测量边界下的结果。复现实验应同时记录可运行的最大模型、每卡峰值显存、step time、吞吐、通信/计算重叠和收敛状态；“DDP OOM”只说明该配置无法运行，不是所有 DDP 配置的普遍结论。

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

## 求职追问与示范回答

**Q1: ZeRO 是什么？**  
A: 把数据并行中重复的模型状态按阶段分片，在需要计算时临时聚合；显存收益与精度、rank 数、激活和通信 buffer 一起核算。

**Q2: ZeRO 三个阶段的区别？**  
A: ZeRO-1 分片 optimizer state，ZeRO-2 再分片梯度，ZeRO-3 还分片参数并在模块计算前后聚合/重分片；通信量由 collective、dtype、bucket 和重叠决定。

**Q3: ZeRO-3 Forward/Backward 时参数怎么办？**
A: 当前模块计算前临时 all-gather 参数，计算后按配置重新分片；反向计算前通常还要再次 all-gather 当前模块参数。峰值除了 unsharded 参数和激活，还可能包含梯度、prefetch 的下一模块、通信 bucket、重叠 buffer 和 allocator 保留，需由时间线或 profiler 核对。

**Q4: ZeRO vs Tensor Parallelism？**  
A: ZeRO 是数据并行的优化（分片模型状态，减少冗余），TP 是模型并行（权重矩阵切片，单层放不下时用）。可以组合：TP 单机内（低延迟），ZeRO 跨机（省显存）。

**Q5: PyTorch FSDP 是什么？**  
A: PyTorch 的参数分片训练 API；FULL_SHARD 采用参数/梯度/优化器状态分片，并在前向/反向按需 all-gather 与 reduce-scatter。具体重分片、预取和 offload 行为按版本配置确认。

**Q6: 通信量如何回答？**
A: 先定义 payload 和计时边界，再分别列参数 all-gather、梯度 reduce-scatter、reshard、bucket 和重叠；给出目标 rank 数与 dtype 下的推导或 trace，不引用脱离配置的固定倍数。

---

## 代码对照

### DeepSpeed 官方
- [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)
- ZeRO 实现在 `deepspeed/runtime/zero/`

### PyTorch FSDP
- [torch.distributed.fsdp](https://pytorch.org/docs/stable/fsdp.html)
- Tutorial: [Getting Started with FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

### 本仓库参考
可用小模型对比 DDP/FSDP 的显存、step time 和通信 trace；该实验结果应单独记录版本、dtype、wrap policy 和 batch。

---

## 扩展：ZeRO-Offload / ZeRO-Infinity

### ZeRO-Offload (2020)
- 优化器状态 offload 到 CPU 内存
- 省下的显存与训练时间取决于 CPU/GPU 带宽、预取和计算重叠；不能预设固定百分比
- 适合单机多卡 + 显存紧张

### ZeRO-Infinity (2021)
- 参数 + 优化器状态 offload 到 NVMe SSD
- 可以把更多状态放到 NVMe，但可运行规模与吞吐取决于访问模式、设备和预取实现

---

## 参考资料

- **论文**: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- **DeepSpeed 官方**: [www.deepspeed.ai](https://www.deepspeed.ai/)
- **PyTorch FSDP 文档**: [FSDP](https://pytorch.org/docs/stable/fsdp.html)
- **配套课程**: D1 分布式训练（算子线 D），[roadmap/distributed.md](../../roadmap/distributed.md)
- **官方论文页**: [Microsoft Research](https://www.microsoft.com/en-us/research/publication/zero-memory-optimizations-toward-training-trillion-parameter-models/)

---

*D 线分布式的理论基础。能讲清 ZeRO-3 的数据流 + 通信模式，面试大模型训练岗位的硬通货。*
