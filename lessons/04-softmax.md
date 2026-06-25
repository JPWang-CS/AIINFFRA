# Lesson 04 — Softmax (Naive + Warp Reduce)

> 主题：写出 Softmax，理解 max trick、3-pass 结构、warp shuffle reduce
> 前置：完成 [Lesson 03](03-gemm-tiled.md)，理解 shared memory 和 `__syncthreads`
> 平台：LeetGPU `5_softmax`
> 状态：⏳ 待做（见 [PATH.md](../PATH.md)）

📚 **本课涉及的底层知识**：
- [Warp 与同步](../notes/cuda/warp-and-sync.md) — warp shuffle、`__shfl_down_sync`
- [CUDA API 速查](../notes/cuda/cuda-cheatsheet.md) — shared memory、atomics
- [内存层级详解](../notes/cuda/memory-model.md) — bandwidth 分析

🎯 **这对理解 Triton 有什么用**：
- warp shuffle → Triton 的 `tl.max`/`tl.sum` 底层用的就是这个
- online softmax（v1）→ **Flash Attention 的核心**，[Lesson 05](05-flash-attn-reading.md) 会用到
- 详细对照见 → [Triton 底层 CUDA 对照](../notes/cuda/triton-under-the-hood.md)

---

## Part 1：定义

```
softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))

"max trick" 是必须的——不做减法，exp(大数) 会溢出到 inf。
```

---

## Part 2：朴素实现（3-pass）

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
    // 需要 block 内 reduce max → 见 Part 3

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
// 更好的方案见 Part 3 的 warp reduce
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

---

## Part 3：Warp Reduce（优化版）

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
// 完整代码见 → [warp-and-sync.md](../notes/cuda/warp-and-sync.md)
```

> **Ascend 对照**：Ascend 的 Vector Unit 有硬件 reduce 指令。CUDA 用 warp shuffle 模拟同样的效果。都是把"一组计算单元内的值快速聚合"。

---

## Part 4：LeetGPU 上的 Softmax

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

## ✅ 本课检验清单

- [ ] `softmax_naive` 跑通，结果正确
- [ ] 理解 max trick 为什么必须（防止 exp 溢出）
- [ ] 理解 warp shuffle reduce 的原理和 `__shfl_down_sync` 的语义
- [ ] 知道 Triton 的 `tl.max`/`tl.sum` 底层就是用 warp shuffle 实现的
- [ ] 了解 online softmax（单 pass）的思路 → 通向 Flash Attention

---

## 知识库索引

| 想深入理解 | 去看 |
|-----------|------|
| Warp shuffle 完整说明 | [warp-and-sync.md](../notes/cuda/warp-and-sync.md) §4 |
| `__syncthreads` 的陷阱 | [warp-and-sync.md](../notes/cuda/warp-and-sync.md) §3.2 |
| LeetGPU Softmax 题 | [leetgpu-challenges.md](../notes/cuda/leetgpu-challenges.md) → `5_softmax` |
| Softmax 参考实现（含 online 版） | [reference/cuda/softmax/softmax.cu](../reference/cuda/softmax/softmax.cu) |
| online softmax 在 attention 里的用法 | [Lesson 05](05-flash-attn-reading.md) |

---

*Lesson 04 · Softmax · 源自原 week-03（后半）*
