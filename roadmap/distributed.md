# 分布式训练计划（算子线 D：训练系统）

> 对应 PATH 算子线 D，以及大模型板块 [训练系统](../notes/llm/training-systems.md)。
> 目标：能算显存、能画通信图、能跑最小 demo，能讲清 DDP/FSDP/ZeRO/TP/PP/EP。
> 本页负责并行策略基础；拓扑、NCCL、InfiniBand/RoCE、GPUDirect RDMA、多节点启动、EP 通信和排障见 [多机多卡专项路线](multi-node-multi-gpu.md)。

---

## 1. 前置

- [ ] 理解训练 forward/backward/optimizer 链路
- [ ] 理解显存三部分：参数、梯度、优化器状态
- [ ] 理解 FP16/BF16/FP32 的区别
- [ ] 有至少一张 GPU 或多卡模拟环境

## 2. 显存账本

以 7B 模型、混合精度 + AdamW 为例：

```text
FP16/BF16 参数：14 GB
FP32 master weight：28 GB
梯度：14 GB
Adam m：28 GB
Adam v：28 GB
合计：112 GB
```

训练前先会算这个账，再看并行策略。

## 3. 集合通信

| 原语 | 行为 | 典型用途 |
|------|------|---------|
| AllReduce | 所有卡得到完整结果 | DDP 梯度同步 |
| ReduceScatter | 每卡得到部分归约结果 | ZeRO-2/FSDP 梯度分片 |
| AllGather | 每卡把分片拼成完整结果 | FSDP 参数取回 |
| AllToAll | 每卡和所有卡交换不同数据 | MoE expert 路由 |

完成定义：
- [ ] 能画 Ring AllReduce 两阶段
- [ ] 能说明通信量为什么约 2 倍模型大小

## 4. DDP

理解：
- 每卡完整模型
- 反向后 AllReduce 梯度
- bucket 减少通信次数

最小 demo：

```bash
torchrun --nproc_per_node=2 ddp_demo.py
```

完成定义：
- [ ] 跑通单卡 baseline
- [ ] 跑通 DDP
- [ ] 能画梯度 AllReduce 时序

## 5. ZeRO / FSDP

| 阶段 | 分片对象 | 省什么 |
|------|---------|--------|
| ZeRO-1 | 优化器状态 | 最大项 |
| ZeRO-2 | 优化器状态 + 梯度 | 再加梯度 |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 | 单卡不再放完整模型 |

FSDP 是 PyTorch 的 ZeRO-3 风格实现：

```text
forward 前：AllGather 参数
forward 后：丢弃非本卡参数
backward 中：ReduceScatter 梯度
backward 后：更新本卡分片的优化器状态
```

完成定义：
- [ ] 能口算 ZeRO-1/2/3 分别省多少
- [ ] 能说清 FSDP 为什么通信增加
- [ ] 跑 FSDP demo

## 6. Tensor Parallel（TP）

| 类型 | 切分方式 | 通信 |
|------|---------|------|
| Column Parallel | 权重按输出列切 | 输入广播，输出拼接 |
| Row Parallel | 权重按输入行切 | 输出 AllReduce |

TP 适合：
- 单层权重放不进单卡
- 单节点 NVLink
- 低延迟推理

完成定义：
- [ ] 用虚拟设备模拟 TP
- [ ] 能画 Column/Row parallel 通信图

## 7. Pipeline Parallel（PP）

PP 把层切到多卡：

```text
卡0: Layer 0..7
卡1: Layer 8..15
卡2: Layer 16..23
卡3: Layer 24..31
```

调度：
- GPipe：先完整 forward，再完整 backward
- 1F1B：交错 forward/backward
- Interleaved：多个 stage 降低 bubble

完成定义：
- [ ] 能画出 micro-batch 时间线
- [ ] 能解释 pipeline bubble

## 8. Expert Parallel（EP）

MoE 场景：
- expert 分布到不同卡
- token 根据 router 结果发送到对应卡
- 使用 AllToAll

完成定义：
- [ ] 能画 8 expert 分布在 4 卡的映射
- [ ] 能说负载不均衡的影响

## 9. 3D / 混合并行

例子：64 卡

```text
TP=4（单层切 4）
PP=4（层切 4 段）
DP=4（数据复制 4 份）
=> 4 * 4 * 4 = 64
```

设计原则：
- TP 放同节点，NVLink
- PP 跨节点，通信少
- DP 负责扩展吞吐
- EP 只在 MoE 加

## 10. 常见坑

| 坑 | 解法 |
|----|------|
| 只背术语不会算显存 | 先手算 7B 账本 |
| 通信量只看模型大小 | 还要看梯度和优化器状态 |
| FSDP 和 DDP 混淆 | 看参数/梯度/优化器是否分片 |
| TP 跨节点 | TP 应尽量放 NVLink 内 |
| 只看文档不跑 demo | 至少跑单机多卡 demo |

## 11. 输出物

- [ ] `solutions/distributed/ddp_demo.py`
- [ ] `solutions/distributed/fsdp_demo.py`
- [ ] `solutions/distributed/tp_linear.py`
- [ ] `notes/llm/training-systems.md`