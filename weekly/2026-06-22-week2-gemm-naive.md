# Week 2 — GEMM Naive

> 2026-06-16 ~ 2026-06-22 · 算子线 A2/A3 · 平台：LeetGPU `2_matrix_multiplication` + `22_gemm`

## 做了什么

### GEMM float naive（A2）
- 写出了 `gemm_naive`，2D grid (16×16)，LeetGPU `2_matrix_multiplication` 跑通 ✅
- 每个 thread 计算 C 的一个元素（dot product over K）
- 理解了 naive GEMM 瓶颈：算术强度 ~0.25 FLOP/byte，memory-bound
- 画出访问模式：A 行连续（coalescing OK），关键是数据复用率低（A 读 K 次，B 读 M 次）
- ⚠️ 笔记曾误判 B 访问 uncoalesced，实际是连续合并访问——真正瓶颈在数据复用率

### GEMM fp16 naive（A2+）
- LeetGPU `22_gemm` fp16 跑通（2026-06-22）
- 实现了 BLAS 标准公式：`C = alpha * A × B + beta * C`
- half 精度 + float alpha/beta 混合运算
- 写回用 `__float2half_rn(sum)` 显式 round，而非隐式转

### GEMM fp16 tiled（A3+）
- LeetGPU `22_gemm` fp16 跑通（2026-06-22）
- TILE=32，shared memory 分块
- 双 `__syncthreads` 正确放置：
  - 第一个：确保所有 thread 加载完 tile 再开始计算
  - 第二个：确保所有 thread 计算完再覆写下一轮 tile
- ⚠️ LeetGPU K=16 太小，naive vs tiled 加速比几乎无差异——待 4090 大 K 验证

### Code Review（fp16 naive）
- `alpha` 乘在循环内（正确但浪费 K 次乘法）
- `beta == 0.0f` 分支后还做了一次无用乘法
- 整体逻辑正确，性能 OK

## 关键数据

| 算子 | 平台 | 精度 | 状态 |
|------|------|:----:|:----:|
| GEMM float naive | LeetGPU `2_matrix_multiplication` | FP32 | ✅ |
| GEMM fp16 naive | LeetGPU `22_gemm` | FP16 | ✅ |
| GEMM fp16 tiled | LeetGPU `22_gemm` | FP16 | ✅ |
| Tiled 加速比 | — | — | ⚠️ K=16 待 4090 验证 |
| Shared memory 用量 | As + Bs = 2×(32×32)×2B = 4KB | — | — |

## 卡点 / 怎么解决的

1. **Naive GEMM 的 B 矩阵误解**：先以为 B 访问是 uncoalesced（strided），实际上 `threadIdx.x→K` 映射下 B 的访问是连续的。真正瓶颈是 HBM 带宽浪费而非 coalescing——每个 thread 单独读 A 和 B，A 被重复读取 K 次。**解法**：下个阶段 shared memory tiling。

2. **`C[idx] = sum` vs `C[idx] += sum`**：给 LeetGPU 写时误用了 `+=`，平台 C 没有被清零（依赖未初始化内存）。**正确做法**：`=` 直写，每个 thread 独占一个输出。**解法已纠正**。

3. **LeetGPU GEMM tiled 速度**：K=16 时 naive 就已经可以复用大部分数据（K 太小，memory-bound 不明显），tiled 没有测量到显著加速。**待做**：本地 4090 跑 K=1024+ 验证 5-10× 提速。

## 面试可用点

- GEMM 的 arithmetic intensity = O(K)：K 小 memory-bound，K 大 compute-bound
- Global memory coalescing 条件：同一 warp 32 线程读 128B 对齐连续地址
- Tiled GEMM：shared memory 搬到片上复用，10× 提速的经典案例
- `__syncthreads` 两个位置的 race condition 分析

## 产出物

- [x] solutions/cuda/gemm/gemm_naive.cu
- [x] solutions/cuda/gemm/gemm_fp16_naive.cu
- [x] solutions/cuda/gemm/gemm_fp16_tiled.cu
- [x] notes/cuda/code-review-gemm-fp16-naive.md
- [x] LeetGPU `2_matrix_multiplication` 跑通
- [x] LeetGPU `22_gemm` fp16 两版跑通
