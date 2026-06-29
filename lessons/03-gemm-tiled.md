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

## Part 0：tiled 和 naive 在模型里一样，区别是规模

Tiled GEMM 和 naive GEMM 算的是同一个东西——矩阵乘法。在模型里的位置完全一样（QKV projection、FFN gate/up/down、attention output projection）。

**什么时候 tiling 赢**：K 维度足够大，且 L2 cache 不够。例如：
- LLaMA-7B 的 FFN：$d_{\text{model}}=4096 \to d_{\text{ff}}=11008$，这个 K 维度巨大，A+B 数据量远超 L2
- 老 GPU（Kepler/Maxwell）L2 小，tiling 始终有效
- 4090 L2=72MB，小规模(K<8K) naive 已够快

**面试关键点**："Tiling 什么时候必须？" → 当数据量超出 L2 cache 容量时，shared memory tiling 是唯一避免反复读 HBM 的手段。

## Part 1：原理回顾

Naive GEMM 的问题：每个 thread 从 global memory 读 `2×K` 个元素。K=1024 时，2K 次 global memory read per output element → 严重 memory-bound。

Tiled GEMM 的思路：**把 A 和 B 切成 TILE×TILE 的小块，先搬到 shared memory，在片上高速计算。**

把 K 维度切成 TILE 大小的段：

$$
C[block\_y : block\_y + TILE][block\_x : block\_x + TILE] \mathrel{+}= A[block\_y : block\_y + TILE][t : t + TILE] \times B[t : t + TILE][block\_x : block\_x + TILE]
$$

每个 block 做 $TILE \times TILE$ 的输出，循环 $K/TILE$ 次：
  1. 加载 A_tile（所有 thread 合作搬）
  2. 加载 B_tile
  3. $\text{\_\_syncthreads()}$
  4. 在 shared memory 上计算部分积
  5. $\text{\_\_syncthreads()}$（在覆写 tile 之前）

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
1. Shared memory 容量：$TILE \times TILE \times 2 \times 4\text{B} = TILE^{2} \times 8\text{B}$
   $TILE = 32 \to 8$ KB（任何卡都够）
   $TILE = 64 \to 32$ KB（T4 需要调整 shared mem 配置）

2. Block 大小：$TILE \times TILE$ 个 thread
   $TILE = 32 \to 1024$ threads → 刚好上限

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

- [x] ✅ `gemm_fp16_tiled` LeetGPU 跑通（2026-06-22，TILE=32）→ [solutions/cuda/gemm/tiled_fp16.cu](../solutions/cuda/gemm/tiled_fp16.cu)
- [x] ✅ 能解释 `__syncthreads` 在两个位置各起什么作用
- [x] ✅ 知道 TILE=32 的 shared memory 用量（As+Bs = 4KB），以及为什么 TILE=64 可能有 bank conflict
- [x] ✅ 4090 实测: K=2048 naive 5033 GFLOPS vs tiled 3118 GFLOPS (0.6x); K=8192 类似（tiled 未加速，见下）

---

## Part 6：我的实现 — `gemm_fp16_tiled`（✅ 已通过）

> 2026-06-25，LeetGPU `22_gemm` fp16，TILE=32，shared memory tiling
> 代码归档在 [solutions/cuda/gemm/tiled_fp16.cu](../solutions/cuda/gemm/tiled_fp16.cu)

```cpp
constexpr int tileLen = 32;

__global__ void kernel(const half* A, const half* B, half* C,
                       int M, int N, int K, float alpha, float beta) {
    int m = blockIdx.x * tileLen + threadIdx.x;
    int n = blockIdx.y * tileLen + threadIdx.y;
    float sum = 0.0f;

    // 每个线程先搬运暂存
    __shared__ half As[tileLen][tileLen];
    __shared__ half Bs[tileLen][tileLen];

    // 遍历 K
    for (int t = 0; t < (K + tileLen - 1) / tileLen; ++t) {
        // A 矩阵按列步进
        int aK = t * tileLen + threadIdx.y;
        As[threadIdx.x][threadIdx.y] = (m < M && aK < K)
            ? A[m * K + aK] : __float2half_rn(0.0f);

        // B 矩阵按行步进
        int bK = t * tileLen + threadIdx.x;
        Bs[threadIdx.x][threadIdx.y] = (n < N && bK < K)
            ? B[bK * N + n] : __float2half_rn(0.0f);

        __syncthreads();

        for (int k = 0; k < tileLen; k++) {
            sum += __half2float(As[threadIdx.x][k]) *
                   __half2float(Bs[k][threadIdx.y]);
        }

        __syncthreads();
    }

    if (m < M && n < N) {
        C[m * N + n] = __float2half_rn(
            alpha * sum + beta * __half2float(C[m * N + n]));
    }
}
```

| 关键点 | 说明 |
|--------|------|
| Grid 铺法 | blockIdx.x→M，blockIdx.y→N，和 naive 版（x→K, y→M）不同 |
| A tile 加载 | threadIdx.y 走 K 维度(列)，threadIdx.x 走 M 维度(行) |
| B tile 加载 | threadIdx.x 走 K 维度(行)，threadIdx.y 走 N 维度(列) |
| Syncthreads #1 | 确保所有线程加载完 tile 再开始计算 |
| Syncthreads #2 | 确保所有线程计算完再覆写下一轮 tile（防race） |
| Shared mem 用量 | As + Bs = 2×(32×32)×2B = **4KB**（任何 GPU 都够） |
| alpha/beta | 支持 BLAS 标准公式：`C = α·A·B + β·C` |
| ⚠️ 4090 实测 | K=2048/8192 tiled 0.6x naive — 不是 bug，是 4090 L2(72MB) 缓存大 + occupancy 低(1024 threads/block vs 256)。**Tiling 在 4090 上不总是赢的** — 这是真实的学习记录。详见 [benchmark.cu](../solutions/cuda/gemm/benchmark.cu) |

---

### 为什么 4090 上 tiled 没加速？（面试加分点）

1. **RTX 4090 的 L2 太大** (72MB)——A+B=16MB(K=2048) 或 64MB(K=8192) 大部分在 L2 cache 里缓存了。Naive 的重复读取实际从 L2 命中了，没有真正从 HBM 重读。

2. **Tiled 的同步开销 > 实际节省**——K=8192 时 (8192/32=256) × 2 syncthreads = **512 次同步** vs naive 0 次。

3. **Occupancy 差**——TILE=32→1024 threads/block，4090 SM 最多 1536 threads → 只能跑 1 block/SM（浪费 512 线程槽）。Naive 256 threads → 6 blocks/SM → 更好的延迟隐藏。

4. **真需要 tiling 的场景**：老 GPU（Kepler/Maxwell 小 L2）、超大矩阵（K>16K 超出 L2 容量）、或组合 Tensor Core（Tiled fp16→Tensor Core 比 cuda core 快 10x）。

> `[面试]` 这道题可以讲："Tiling 不是万能药——在 Ada Lovelace 的大 L2 上，naive 的 cache 优化已足够。真正的加速来自 Tensor Core 而非手写 tiling。"

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
