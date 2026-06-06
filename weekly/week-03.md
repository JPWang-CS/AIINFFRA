# Week 3 — GEMM Tiled + Softmax Naive

> 目标：用 shared memory tiling 加速 GEMM，写出 Softmax，理解 warp reduce
> 时间：5-7 小时
> 前置：完成 Week 2，理解 naive GEMM 的瓶颈（memory-bound）

📚 **本周涉及的底层知识**：
- [内存层级详解](../cuda-kernels/notes/memory-model.md) — shared memory tiling、bank conflict
- [Warp 与同步](../cuda-kernels/notes/warp-and-sync.md) — warp shuffle、`__syncthreads`
- [CUDA API 速查](../cuda-kernels/notes/cuda-cheatsheet.md) — shared memory 语法

🎯 **这对理解 Triton 有什么用**：
- shared memory tiling → **这是 Triton 的 `tl.load` + `tl.dot` 在底层做的事**
- `__syncthreads` → Triton 自动插入，但知道为什么需要帮你 debug
- warp shuffle → Triton 的 `tl.max`/`tl.sum` 底层用的就是这个
- 详细对照见 → [Triton 底层 CUDA 对照](../cuda-kernels/notes/triton-under-the-hood.md)

---

## Day 1-2：Tiled GEMM（3h 动手）

### 1.1 原理回顾

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

### 1.2 你要写的代码

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

### 1.3 TILE 大小的选择

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

### 1.4 Bank Conflict（会了就是加分项）

TILE=32 时，A_tile 和 B_tile 都是 32×32 float。同一 warp 的 thread 访问同一列的不同行 → 32-way bank conflict？

```cpp
// 加载 A_tile 时，threadIdx.x（列）相同的线程访问连续的行
// A_tile[threadIdx.y][threadIdx.x]
// warp 内 threadIdx.x 相同 → 全部访问同一个 bank → 32-way conflict！

// 解法：padding
__shared__ float As[TILE][TILE + 1];  // 多一列，打破 stride=32 的对齐
```

> 具体原理见 → [memory-model.md](../cuda-kernels/notes/memory-model.md) 第 3.3 节

### 1.5 对比和验证

跑通后对比 naive vs tiled 的 GFLOPS：
- T4 FP32 预期：naive ~20 GFLOPS，tiled ~200 GFLOPS（~10× 提升）
- 如果不到 10×：检查 shared memory 是否正确加载、`__syncthreads` 位置是否正确

---

## Day 3：Softmax Naive（2h 动手）

### 2.1 定义

```
softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))

"max trick" 是必须的——不做减法，exp(大数) 会溢出到 inf。
```

### 2.2 朴素实现（3-pass）

```cpp
// 每个 block 处理一行（或一组行）
// 每个 thread 处理该行的部分元素
__global__ void softmax_naive(const float* input, float* output,
                               int B, int D) {
    int row = blockIdx.x;
    if (row >= B) return;
    
    // Pass 1: 找 max
    float max_val = -FLT_MAX;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float val = input[row * D + j];
        if (val > max_val) max_val = val;
    }
    // 需要 block 内 reduce max → 见 2.3
    
    // Pass 2: 算指数和
    float sum = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        sum += expf(input[row * D + j] - max_val);
    }
    // 需要 block 内 reduce sum
    
    // Pass 3: normalize + 写回
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        output[row * D + j] = expf(input[row * D + j] - max_val) / sum;
    }
}
```

先完成 3 个 pass 的框架，不用 reduce。让 thread 0 串行做 reduce（效率低但能跑对）：

```cpp
// 临时方案：thread 0 负责 reduce
// 更好的方案见 Day 3 的 warp reduce
__shared__ float shared_max[256];  // 假设 blockDim=256
__shared__ float shared_sum[256];

shared_max[threadIdx.x] = max_val;
shared_sum[threadIdx.x] = sum;
__syncthreads();

if (threadIdx.x == 0) {
    float global_max = shared_max[0];
    float global_sum = shared_sum[0];
    for (int i = 1; i < blockDim.x; i++) {
        if (shared_max[i] > global_max) global_max = shared_max[i];
        global_sum += shared_sum[i];
    }
    shared_max[0] = global_max;  // 写入 shared mem 让其他 thread 读取
    shared_sum[0] = global_sum;
}
__syncthreads();

max_val = shared_max[0];
sum = shared_sum[0];
```

### 2.3 Warp Reduce（优化版）

thread 0 串行 reduce 太慢。用 warp shuffle 做 log(N) 级 reduce：

```cpp
// Warp-level reduce max
__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        if (other > val) val = other;
    }
    return val;  // warp 内所有 lane 拿到同一个 max
}

// 用法：每个 warp 先内部 reduce
float w_max = warp_reduce_max(my_local_max);

// 然后每个 warp 的 lane 0 把结果写到 shared memory
// thread 0 对 warp_results 做最后一次 reduce
// 完整代码见 → [warp-and-sync.md](../cuda-kernels/notes/warp-and-sync.md)
```

> **Ascend 对照**：Ascend 的 Vector Unit 有硬件 reduce 指令。CUDA 用 warp shuffle 模拟同样的效果。都是把"一组计算单元内的值快速聚合"。

### 2.4 LeetGPU 上的 Softmax

LeetGPU `5_softmax` 题：

```cpp
extern "C" void solve(const float* input, float* output, int N) {
    // 注意：这题是 1D softmax（没有 batch 维度）
    // N ≤ 500,000
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;  // 1D softmax 只需要查一个值
    // 或者用多个 block，但需要 inter-block reduce
}
```

---

## ✅ Week 3 检验清单

- [ ] `gemm_tiled` 跑通，结果与 CPU 一致
- [ ] 对比 naive vs tiled GFLOPS，提升 ≥ 5×
- [ ] 能解释 `__syncthreads` 在两个位置各起什么作用
- [ ] 知道 TILE=32 的 shared memory 用量（8 KB），以及为什么 TILE=64 可能有 bank conflict
- [ ] `softmax_naive` 跑通，结果正确
- [ ] 理解 warp shuffle reduce 的原理和 `__shfl_down_sync` 的语义
- [ ] 知道 Triton 的 `tl.max`/`tl.sum` 底层就是用 warp shuffle 实现的

## 知识库索引

| 想深入理解 | 去看 |
|-----------|------|
| Shared memory tiling 完整原理 | [memory-model.md](../cuda-kernels/notes/memory-model.md) |
| Bank conflict 详解和解决 | [memory-model.md](../cuda-kernels/notes/memory-model.md) §3.3 |
| Warp shuffle 完整说明 | [warp-and-sync.md](../cuda-kernels/notes/warp-and-sync.md) §4 |
| `__syncthreads` 的陷阱 | [warp-and-sync.md](../cuda-kernels/notes/warp-and-sync.md) §3.2 |
| Triton GEMM 的底层实现 | [triton-under-the-hood.md](../cuda-kernels/notes/triton-under-the-hood.md) §4 |
| LeetGPU Softmax 题 | [leetgpu-challenges.md](../cuda-kernels/notes/leetgpu-challenges.md) → `5_softmax` |

---

*Week 3 · GEMM Tiled + Softmax Naive*
