# Lesson 03 — GEMM Tiled (Shared Memory)

> 主题：用 shared memory tiling 加速 GEMM，理解 `__syncthreads` 和 bank conflict
> 前置：完成 [Lesson 02](02-gemm-naive.md)，理解 naive GEMM 的瓶颈（memory-bound）
> 平台：LeetGPU `2_matrix_multiplication` / 本地 FP32
> 状态：⏳ 待做（见 [PATH.md](../PATH.md)）

📚 **本课涉及的底层知识**：
- [内存层级详解](../notes/cuda/memory-model.md) — shared memory tiling、bank conflict
- [Warp 与同步](../notes/cuda/warp-and-sync.md) — `__syncthreads`
- [CUDA API 速查](../notes/cuda/cuda-cheatsheet.md) — shared memory 语法

🎯 **这对理解 Triton 有什么用**：
- shared memory tiling → **这是 Triton 的 `tl.load` + `tl.dot` 在底层做的事**
- `__syncthreads` → Triton 自动插入，但知道为什么需要帮你 debug
- 详细对照见 → [Triton 底层 CUDA 对照](../notes/cuda/triton-under-the-hood.md)

---

## Part 1：原理回顾

Naive GEMM 的问题：每个 thread 从 global memory 读 `2×K` 个元素。K=1024 时，2K 次 global memory read per output element → 严重 memory-bound。

Tiled GEMM 的思路：**把 A 和 B 切成 TILE×TILE 的小块，先搬到 shared memory，在片上高速计算。**

```
把 K 维度切成 TILE 大小的段：

C[block_y : block_y+TILE][block_x : block_x+TILE] +=
    A[block_y : block_y+TILE][t : t+TILE] × B[t : t+TILE][block_x : block_x+TILE]

每个 block 做 TILE×TILE 的输出，循环 K/TILE 次：
  1. 加载 A_tile（所有 thread 合作搬）
  2. 加载 B_tile
  3. __syncthreads()
  4. 在 shared memory 上计算部分积
  5. __syncthreads()（在覆写 tile 之前）
```

> **Ascend 对照**：这和 Ascend 的 L1 Buffer tiling 完全一样——把大矩阵切开，小块搬入片上内存（L1 Buffer/Shared Memory），在片上计算。区别是 Ascend 用 pipe 机制做搬运和计算的流水线重叠，CUDA 需要你手动控制 `__syncthreads()`。

---

## Part 2：你要写的代码

```cpp
#define TILE 32  // 为什么选 32？思考和实验

__global__ void gemm_tiled(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    // 1. 声明 shared memory 的 A_tile 和 B_tile
    //    A_tile: TILE×TILE float
    //    B_tile: TILE×TILE float
    // TODO

    // 2. 计算这个 thread 对应的 row 和 col
    //    (blockIdx 配合 TILE 做 offset)
    // TODO

    float sum = 0.0f;

    // 3. 循环遍历 K 维度的 tile
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // 3a. 从 global memory 加载 A_tile
        //     每个 thread 加载一个元素到 shared memory
        //     注意边界（K 可能不被 TILE 整除）
        // TODO

        // 3b. 加载 B_tile（同理）
        // TODO

        __syncthreads();  // 确保所有 thread 加载完毕

        // 3c. 在 shared memory 上计算部分点积
        //     循环 k=0..TILE，累加 A_tile[row][k] * B_tile[k][col]
        //     （注意：A_tile 用 threadIdx.y 索引行，B_tile 用 threadIdx.x 索引列）
        // TODO

        __syncthreads();  // 确保所有 thread 计算完毕，然后才覆写 tile
    }

    // 4. 写入结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

---

## Part 3：TILE 大小的选择

```
影响因素：
1. Shared memory 容量：TILE×TILE×2×4B = TILE² × 8B
   TILE=32 → 8 KB（任何卡都够）
   TILE=64 → 32 KB（T4 需要调整 shared mem 配置）

2. Block 大小：TILE×TILE 个 thread
   TILE=32 → 1024 threads → 刚好上限

3. Occupancy：每个 block 用多少 shared memory
   Shared mem 用多了 → 同时驻留 SM 的 block 数少 → occupancy 低

建议：先从 TILE=32 开始，跑通后再试 TILE=16 和 TILE=64 对比
```

---

## Part 4：Bank Conflict（会了就是加分项）

TILE=32 时，A_tile 和 B_tile 都是 32×32 float。同一 warp 的 thread 访问同一列的不同行 → 32-way bank conflict？

```cpp
// 加载 A_tile 时，threadIdx.x（列）相同的线程访问连续的行
// A_tile[threadIdx.y][threadIdx.x]
// warp 内 threadIdx.x 相同 → 全部访问同一个 bank → 32-way conflict！

// 解法：padding
__shared__ float As[TILE][TILE + 1];  // 多一列，打破 stride=32 的对齐
```

> 具体原理见 → [memory-model.md](../notes/cuda/memory-model.md) §3.3

---

## Part 5：对比和验证

跑通后对比 naive vs tiled 的 GFLOPS：
- T4 FP32 预期：naive ~20 GFLOPS，tiled ~200 GFLOPS（~10× 提升）
- 如果不到 10×：检查 shared memory 是否正确加载、`__syncthreads` 位置是否正确

---

## ✅ 本课检验清单

- [x] ✅ `gemm_fp16_tiled` LeetGPU 跑通（2026-06-22，TILE=32）→ [solutions/cuda/gemm_fp16_tiled.cu](../solutions/cuda/gemm_fp16_tiled.cu)
- [x] ✅ 能解释 `__syncthreads` 在两个位置各起什么作用
- [x] ✅ 知道 TILE=32 的 shared memory 用量（As+Bs = 4KB），以及为什么 TILE=64 可能有 bank conflict
- [ ] 对比 naive vs tiled GFLOPS，提升 ≥ 5×（⏳ LeetGPU K=16 太小无差异，待 4090 大 K 验证）

---

## 知识库索引

| 想深入理解 | 去看 |
|-----------|------|
| Shared memory tiling 完整原理 | [memory-model.md](../notes/cuda/memory-model.md) |
| Bank conflict 详解和解决 | [memory-model.md](../notes/cuda/memory-model.md) §3.3 |
| `__syncthreads` 的陷阱 | [warp-and-sync.md](../notes/cuda/warp-and-sync.md) §3.2 |
| Triton GEMM 的底层实现 | [triton-under-the-hood.md](../notes/cuda/triton-under-the-hood.md) §4 |
| Tiled GEMM 参考实现（先自己写！） | [reference/cuda/gemm/gemm.cu](../reference/cuda/gemm/gemm.cu) |

---

*Lesson 03 · GEMM Tiled · 源自原 week-03（前半）*
