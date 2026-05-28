# Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Authors**: Dao, Fu, Ermon, Rudra, Ré (Stanford)
**Venue**: NeurIPS 2022 | **优先级**: P0 | **状态**: ✅ | **日期**: 2026-05-28

---

## 解决了什么问题

标准 self-attention 的 O(N²) 显存和带宽瓶颈 —— 每次 forward 都要把 N×N 的 attention 矩阵写回 HBM 再读出来做 softmax，HBM 带宽成为限制因素。

## 怎么解决的

通过 **tiling + recomputation** 避免 materialize 完整的 N×N attention 矩阵到 HBM：

1. **Tiling**: Q 按行分块 (Br 行)，K/V 按行分块 (Bc 行)，每次只在 SRAM 内处理 Br×Bc 的 tile
2. **Online Softmax**: 维护 running max (m) 和 running exp sum (l)，增量更新，不需要知道全局 max
3. **Recomputation**: 反向传播时不存储 attention 矩阵，而是从 Q/K/V 重新计算（用存储的 m, l 做数值稳定）
4. **Causal Masking**: 自动利用下三角结构，跳过不需要计算的 tile

## 关键数据

| 指标 | 标准 Attention | Flash Attention |
|------|:---:|:---:|
| HBM 读写量 | O(N²) | O(N² · d / SRAM_size) |
| 显存峰值 | O(N²) | O(N) |
| 训练加速 (GPT-2, seq=1K) | 1× | **3-5×** |
| 训练加速 (GPT-2, seq=4K) | 1× | **2-4×** |
| 数值误差 | - | < 0.1% vs 标准实现 |

## 与我何干

> 不是所有中间结果都值得写回 HBM —— 有时 recompute 的代价（SRAM 内重算）远小于 memory access（HBM 读写）。昇腾 Da Vinci 的 L1/UB 有同样的优化空间，tiling 思路可直接迁移。

## Ascend NPU 对应

Da Vinci AiCore 的 L1 Buffer (1MB) 和 Unified Buffer 就是 Flash Attention 里的 "SRAM"。CUDA 的 shared memory per SM 更大（A100 164KB），所以 tile size 可以设得更大。但优化思路完全一致。
