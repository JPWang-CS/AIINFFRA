# 分布式训练 Demo (Phase 4: Week 31-40)

理解主流并行策略的原理、通信模式和实现。

## Demo 计划

### ddp-demo (Data Parallel)

PyTorch DDP 的最小可运行示例：
- `single_gpu.py` — 单卡 baseline
- `ddp_demo.py` — 多卡 DDP + gradient allreduce 时序
- 关键概念：bucket、通信与计算 overlap、allreduce 的通信量

```bash
torchrun --nproc_per_node=4 ddp_demo.py
```

### fsdp-demo (Fully Sharded Data Parallel)

FSDP 的最小示例，展示参数分片：
- `fsdp_demo.py` — FSDP wrap + mixed precision
- 关键概念：ZeRO Stage 1/2/3、参数 gather/scatter、offloading

### tp-demo (Tensor Parallel)

Megatron-style Tensor Parallel 的原理演示（单卡模拟）：
- `tp_linear.py` — 把 Linear 层切分到 2/4 个"虚拟设备"
- 关键概念：Column-parallel、Row-parallel、allreduce/gather 的通信量

## 学习目标

- 能画出 FSDP / TP / PP 的通信拓扑图
- 能计算每种策略的通信量和显存占用
- 理解 NCCL ring allreduce 的带宽模型

## 并行策略速查

| 策略 | 切什么 | 通信原语 | 通信量 | 适用场景 |
|------|--------|---------|--------|---------|
| DP | 数据 | AllReduce (grad) | 2(N-1)/N × params | 小模型多卡 |
| FSDP | 参数+优化器 | AllGather + ReduceScatter | ~ params | 大模型 |
| TP | 权重矩阵 | AllReduce (forward/backward) | 每层都通信 | 超大单层 |
| PP | 层 | P2P Send/Recv | 很小（仅激活） | 层数多的模型 |
| EP | Expert | AllToAll | 取决于 MoE 配置 | MoE 模型 |
