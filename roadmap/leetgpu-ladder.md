# 算子优化路线

> 5 个算子 × 多层优化阶梯。LeetGPU 题号仅用作 baseline/正确性验证。

## GEMM (Week 1-3)

| 版本 | 技术要点 | 验证 |
|------|---------|------|
| v0 naive | 每线程直接读 global memory，算一个输出 | LeetGPU `2_matrix_multiplication` |
| v1 tiled | shared memory TILE×TILE，bank conflict 处理 | |
| v2 vec4 | `float4` 128-bit vectorized load | |
| v3 double buffer | 双 shared memory buffer，copy-compute overlap | |
| v4 tensor core | `mma.sync`，FP16→FP32，warp-level matrix fragment | LeetGPU `22_gemm` (FP32) / `57_fp16_batched_matmul` |

**目标**: tensor core GEMM 达到硬件峰值 70%+

## Softmax (Week 4-5)

| 版本 | 技术要点 | 验证 |
|------|---------|------|
| v0 3-pass | max → exp sum → normalize，每个 pass 扫一遍 | LeetGPU `5_softmax` |
| v1 online | 单 pass，running max + running sum | |
| v2 warp reduce | `__shfl_down_sync` 做 warp-level max/sum | |
| v3 fused | softmax 结果直接喂给下一步（如 attention 中），不写回 HBM | |

**目标**: online softmax 的 bandwidth utilization > 80%

## RMSNorm (Week 6)

| 版本 | 技术要点 | 验证 |
|------|---------|------|
| v0 naive | 两 pass：mean → normalize | LeetGPU `50_rms_normalization` |
| v1 warp reduce | warp shuffle + shared memory cross-warp reduce | |
| v2 fused residual | RMSNorm(x) + x 一次完成，避免额外 memory round-trip | |

**目标**: 融合版本比 PyTorch 原生快 2×+

## Flash Attention (Week 7-9)

| 版本 | 技术要点 | 验证 |
|------|---------|------|
| v0 naive attn | QK^T → softmax → ×V，完整 O(N²) | LeetGPU `6_softmax_attention` |
| v1 tiled | Q/K/V 分块，online softmax 增量更新 | |
| v2 causal | causal mask 跳过无效 tile | LeetGPU `53_casual_attention` |
| v3 multi-head | B×H×N×d，batch + head 维并行 | LeetGPU `12_multi_head_attention` |

**目标**: 对 seq_len=4096，比 naive attention 快 5×+，显存降至 O(N)

## GQA Attention (Week 10)

| 版本 | 技术要点 | 验证 |
|------|---------|------|
| v1 GQA | 从 MHA 扩展，K/V head 数 < Q head 数 | LeetGPU `80_grouped_query_attention` |

**目标**: 完成全部 5 个算子的优化线，整理 portfolio

---

## 进度追踪

| # | 算子 | v0 | v1 | v2 | v3 | v4 | 最终 GFLOPS / BW% |
|---|------|:--:|:--:|:--:|:--:|:--:|------|
| 1 | GEMM | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | |
| 2 | Softmax | ⏳ | ⏳ | ⏳ | ⏳ | - | |
| 3 | RMSNorm | ⏳ | ⏳ | ⏳ | - | - | |
| 4 | Flash Attn | ⏳ | ⏳ | ⏳ | ⏳ | - | |
| 5 | GQA Attn | ⏳ | - | - | - | - | |

⏳ 待做 | 🚧 进行中 | ✅ 完成
