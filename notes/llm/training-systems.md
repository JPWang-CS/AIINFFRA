# 大模型板块：训练系统

> 子板块目标：能讲清大模型训练的内存、通信、并行、精度和工程链路，能回答“为什么 FSDP/ZeRO/TP/PP 这样设计”。
> 状态：🚧 草稿，需要配合小模型实验和显存账本手算消化。

---

## 1. 训练全链路

```text
数据加载 -> 前向 -> 损失 -> 反向 -> 梯度同步 -> 优化器更新 -> checkpoint
```

大模型训练系统和推理系统不同，主要矛盾是：

- 显存放不下完整参数/梯度/优化器状态。
- 单卡算力不够，需要多卡并行。
- 多卡并行带来通信。
- 数值精度和训练稳定性冲突。

---

## 2. 训练显存账本

训练时每份参数可能同时存在多种副本：

| 数据 | 大小（7B，FP32） | 说明 |
|------|-----------------|------|
| 模型参数 | 28 GB | 真正可用的 FP32 权重 |
| 梯度 | 28 GB | 反向累积的梯度 |
| Adam m | 28 GB | 一阶动量 |
| Adam v | 28 GB | 二阶动量 |
| 合计 | 112 GB | 单卡基线 |

实际常用混合精度：

- FP16/BF16 模型权重：14 GB。
- FP32 master weight：28 GB。
- 梯度 FP16/BF16：14 GB。
- Adam m/v：各 14 GB 或 28 GB 视实现而定。

> 面试不要只背数字，要能说明为什么需要 master weight、为什么 Adam 状态最大。

---

## 3. 混合精度

| 格式 | 精度 | 范围 | 适用 |
|------|------|------|------|
| FP32 | 高 | 高 | master weight、loss scaling |
| FP16 | 中 | 窄，易溢出 | 前向/反向加速 |
| BF16 | 中 | 和 FP32 接近 | 前向/反向，训练更稳 |
| FP8 | 低 | 看 E4M3/E5M2 | 训练和推理都在探索 |

常见策略：

- 用 FP16/BF16 计算，保留 FP32 master weight。
- loss scaling 防止 FP16 梯度下溢。
- 梯度累积时用 FP32 累加。

---

## 4. 优化器

Adam/AdamW 是主流，显存成本高：

```text
每个参数额外存 2 个状态（m、v）
```

降低优化器显存的方向：

- 8-bit Adam：把 m/v 压到 8-bit。
- Adafactor：用 factorized state。
- LAMB/LARS：大 batch 训练时用。

---

## 5. 数据并行：DDP

DDP 是“每卡一份完整模型 + 梯度 AllReduce”：

- 每卡保存完整参数、梯度、优化器状态。
- 反向后 AllReduce 梯度。
- 通信量约等于 `2 * 参数量`。

适用：模型单卡能放下，想增加吞吐。

---

## 6. ZeRO / FSDP

ZeRO 按阶段分片：

| 阶段 | 分片对象 | 省什么 |
|------|---------|--------|
| ZeRO-1 | 优化器状态 | 最大的一项 |
| ZeRO-2 | 优化器状态 + 梯度 | 再加梯度 |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 | 单卡不再放完整模型 |

FSDP 是 PyTorch 的 ZeRO-3 风格实现：

- 参数分片。
- forward 前 AllGather 取回参数。
- backward 后 ReduceScatter 梯度。
- 通信和计算可以 overlap。

核心取舍：

```text
显存越省，通信越多
```

---

## 7. 张量并行 TP

把单个 Linear/Attention 权重切到多卡：

- Column Parallel：输入完整，输出按列切。
- Row Parallel：输入按行切，输出 AllReduce。
- 每层 forward 都有通信。

适合：单层太大放不下，或需要低延迟推理。

---

## 8. 流水线并行 PP

把层切到多卡：

- 卡 0 算前几层，传给卡 1 算后几层。
- 主要通信是 activation P2P。
- 有 pipeline bubble。

常用调度：

- GPipe：一个 batch 切 micro-batch。
- 1F1B：前向/反向交错，减少 bubble。
- Interleaved：把模型再切成多个 stage，进一步降低 bubble。

---

## 9. 专家并行 EP / 上下文并行 CP

| 并行 | 切什么 | 通信 | 适用 |
|------|--------|------|------|
| EP | Expert 权重 | AllToAll | MoE |
| CP | 序列 | Ring Attention / AllGather | 超长序列 |

MoE 的负载均衡：

- router 可能把太多 token 送到一个 expert。
- 训练时用 auxiliary loss。
- 推理时用调度/重路由。

---

## 10. 3D/混合并行

实际大模型训练通常同时用：

```text
DP × TP × PP（× EP / CP）
```

例子：

```text
64 卡
TP=4：单层权重切 4 份
PP=4：模型层切 4 段
DP=4：数据复制 4 份
=> 4 × 4 × 4 = 64
```

设计原则：

- TP 放同一节点，NVLink。
- PP 放跨节点但通信少。
- DP 负责扩展吞吐。

---

## 11. 其他训练工程

| 技术 | 作用 | 代价 |
|------|------|------|
| Gradient accumulation | 用小 batch 模拟大 batch | 训练时间变长 |
| Activation checkpointing | 反向重算激活 | 约多一次 forward |
| Flash Attention | 省显存、加速 attention | 对反向也要实现 |
| Checkpoint/load 优化 | 保存恢复模型 | 存储和 I/O |
| 数据并行 shuffle | 避免每个 rank 同分布 | 数据管线复杂度 |

---

## 12. 训练系统学习任务

1. 手算 7B/70B 模型在 DDP、FSDP、ZeRO-3 下的显存和通信量。
2. 用 PyTorch 跑一个最小 DDP 或 FSDP 脚本。
3. 画 TP=2、PP=4、DP=4 的卡拓扑和通信图。
4. 跑一个小模型 forward，记录 activation memory 和 peak memory。
5. 对 MoE 模型画 expert 路由和 AllToAll。

验收：

- 能口算 7B 混合精度训练的显存。
- 能说清 ZeRO-1/2/3 各分片什么。
- 能解释 FSDP 为什么 AllGather/ReduceScatter。
- 能设计一个 TP×PP×DP 拓扑。

---

## 13. 关联材料

- [分布式训练 Demo](../../roadmap/distributed.md)
- [多机多卡专项路线](../../roadmap/multi-node-multi-gpu.md)
- [多机多卡知识地图](../distributed/multi-node-gpu.md)
- [剩余理论主题速览](../algorithms/remaining-theory-primer.md)
- [ZeRO 论文](../../papers/training/zero-paper.md)
- [PATH 执行参考](../../roadmap/ai-infra-curriculum.md)
