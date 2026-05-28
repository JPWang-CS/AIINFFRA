# ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

**Authors**: Rajbhandari, Rasley, Ruwase, He (Microsoft DeepSpeed)
**Venue**: SC 2020 | **优先级**: P0 | **状态**: ✅ | **日期**: 2026-05-28

---

## 解决了什么问题

数据并行 (DP) 中每个 GPU 都存一份完整的模型状态（参数 + 梯度 + 优化器状态），导致显存冗余。以 Adam + FP16 训练为例，每 1B 参数需要 16GB 仅存模型状态，大模型单卡根本放不下。

## 怎么解决的

**ZeRO（Zero Redundancy Optimizer）**: 把模型状态分片到所有 GPU，消除 DP 中的冗余。

| Stage | 分片内容 | 显存节省 (per GPU) | 通信量 |
|-------|---------|:---:|------|
| ZeRO-1 | 优化器状态 (momentum, variance) | **4×** | = DP |
| ZeRO-2 | 优化器状态 + 梯度 | **8×** | = DP |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 | **Nd×** | +50% vs DP |

**ZeRO-3 的核心流程**:
```
Forward:  AllGather 参数碎片 → 计算 → 丢弃参数（不存本地）
Backward: AllGather 参数碎片 → 计算梯度 → ReduceScatter 梯度 → 丢弃参数
Update:   每个 rank 只更新自己那一片参数
```

1. **ZeRO-1**: 优化器状态分片，计算完梯度后 allreduce → 每个 rank 只更新自己的参数分片
2. **ZeRO-2**: 梯度也分片，reduce-scatter 替代 allreduce
3. **ZeRO-3**: 参数也分片，forward/backward 时按需 allgather

## 关键数据

| 配置 | 1B 模型 | 10B 模型 | 100B 模型 |
|------|:---:|:---:|:---:|
| 无 ZeRO (DP only) | 16 GB/GPU | OOM | OOM |
| ZeRO-1 (64 GPUs) | 4 GB/GPU | 40 GB/GPU | OOM |
| ZeRO-3 (64 GPUs) | 0.25 GB/GPU | 2.5 GB/GPU | 25 GB/GPU |

PyTorch FSDP 就是 ZeRO-3 的 PyTorch 官方实现，核心机制完全一致。

## 与我何干

> "不需要的数据就不要存"。梯度算完立即 reduce、参数用完立即释放，HBM 永远只放当前计算需要的东西。这和 Flash Attention 的核心思想（不存中间矩阵）是同一个哲学：**compute 比 memory access 便宜**。ZeRO 的三个阶段是面试必考题，要能画图说明每个阶段的数据流动。
