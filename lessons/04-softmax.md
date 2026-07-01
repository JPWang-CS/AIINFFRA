# Lesson 04 — Softmax (Naive + Warp Reduce)

> 主题：写出 Softmax，理解 max trick、3-pass 结构、warp shuffle reduce
> 前置：完成 [Lesson 03](03-gemm-tiled.md)，理解 shared memory 和 `__syncthreads`
> 平台：LeetGPU `5_softmax`
> 状态：🚧 优化中 — 3-pass naive 已跑通（2026-07-01）。见 [PATH.md](../PATH.md) / [NOW.md](../NOW.md)

📚 **本课涉及的底层知识**：
- [Warp 与同步](../notes/cuda/warp-and-sync.md) — warp shuffle、`__shfl_down_sync`
- [CUDA API 速查](../notes/cuda/cuda-cheatsheet.md) — shared memory、atomics
- [内存层级详解](../notes/cuda/memory-model.md) — bandwidth 分析

🎯 **这对理解 Triton 有什么用**：
- warp shuffle → Triton 的 `tl.max`/`tl.sum` 底层用的就是这个
- online softmax（v1）→ **Flash Attention 的核心**，[Lesson 05](05-flash-attn-reading.md) 会用到
- 详细对照见 → [Triton 底层 CUDA 对照](../notes/cuda/triton-under-the-hood.md)

---

## Part 0：这个算子在模型里干嘛？

Softmax 是 **Attention 的归一化层**——把 raw attention scores 变成概率分布：

```
Attention 计算流程（以 LLaMA 为例）：

x (输入) → Q = x @ W_Q    ┐
          → K = x @ W_K    ├─ 三个 GEMM
          → V = x @ W_V    ┘
              ↓
S = Q @ K^T / √d_k          ← QK^T 点积，得到 raw scores (N×N 矩阵)
              ↓
P = softmax(S)              ← ⬅ Softmax！把每行变成概率（Σ=1）
              ↓
O = P @ V                   ← 加权求和，得到 attention output
```

**Softmax 的作用**：
1. **归一化**：让每行（每个 query token 对所有 key token 的关注度）变成概率分布——权重和为 1
2. **非线性**：通过指数函数拉开差距（高分更高，低分更低）
3. **可微**：反向传播梯度流畅

**什么模型用**：所有使用 Attention 的模型——LLaMA/GPT/BERT/DeepSeek/Mistral/Qwen/Claude 系列。没有 softmax 就没有 self-attention。

**注意**：FFN 里的激活函数（SiLU/GELU/ReLU）看起来像但不同——它们也是 element-wise 归一化，但不需要 sum=1。"softmax 和 GELU 的区别"是面试高频题。

> `[面试]` 必问："softmax 在 Transformer 的哪里？" → 在 Attention 里，把 scaled dot-product scores 变成概率权重。一个 N×N 矩阵，每行做一次 softmax。

## Part 1：数学定义

$$
\text{softmax}(x_i) = \frac{\exp(x_i - \max x)}{\sum_j \exp(x_j - \max x)}
$$

"max trick" 是必须的——不做减法，$\exp(\text{大数})$ 会溢出到 inf。

---

## Part 2：朴素实现（3-pass）

```cpp
// 每个 block 处理一行（或一组行）
// 每个 thread 处理该行的部分元素
__global__ void softmax_naive(const float* input, float* output,
                               int B, int D) {
    int row = blockIdx.x;
    if (row >= B) return;

    // TODO Pass 1: 找 max
    //   每个 thread 扫自己负责的列，记局部 max
    //   然后用 reduce 合并成全局 max（见下面的"临时方案"）

    // TODO Pass 2: 算指数和
    //   用全局 max 算 exp(x_i - max)，累加

    // TODO Pass 3: normalize + 写回
    //   exp(x_i - max) / 全局和
}
```

**这个 kernel 缺什么**：3 个 pass 都要做 block 内 reduce（把各线程的局部 max/sum 合并成全局值）。先不写 warp shuffle，用 thread 0 串行 reduce 让结果跑对：

```cpp
// 临时方案：thread 0 负责 reduce，其余线程等待
// 1. 每个线程把局部值写到 shared memory
// 2. __syncthreads()
// 3. thread 0 串行遍历 shared memory，合并成全局值
// 4. 全局值写回 shared memory
// 5. __syncthreads()
// 6. 所有线程读取全局值，继续计算
// 提示：max 用 fmaxf，sum 用 +=
// 更好的方案见 Part 3 的 warp reduce
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

## Part 5：我的实现（已跑通 · 3-pass 跨 block）

> 我自己写的，LeetGPU `5_softmax` 跑通（2026-07-01）。完整文件 → [solutions/cuda/softmax/softmax_naive.cu](../solutions/cuda/softmax/softmax_naive.cu)

思路和 Part 2 的"每 block 一行"不同：这题是 **1D softmax over N（N ≤ 500,000，跨多个 block）**，所以做**两级归约**——每个 block 用 shared memory tree reduce 出局部值，再把各 block 的 partial 拷回 host 做最后一次归约。三个 kernel 串行：

```cpp
// Kernel 1：每个 block 求局部 max → partial_max[blockIdx]
__global__ void findMax_kernel(const float* input, float* partial_max, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    __shared__ float inputMax[256];
    inputMax[tid] = (idx < N) ? input[idx] : -INFINITY;      // 越界补 -INF，不污染 max
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {           // shared mem 树归约
        if (tid < s) inputMax[tid] = fmaxf(inputMax[tid], inputMax[tid + s]);
        __syncthreads();
    }
    if (tid == 0) partial_max[blockIdx.x] = inputMax[0];
}

// Kernel 2：用全局 max 求局部 Σexp(x-max) → partial_sum[blockIdx]
__global__ void countSum_kernel(const float* input, float* partial_sum, float global_max, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    __shared__ float inputSum[256];
    inputSum[tid] = (idx < N) ? expf(input[idx] - global_max) : 0.0f;   // 越界补 0
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) inputSum[tid] += inputSum[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial_sum[blockIdx.x] = inputSum[0];
}

// Kernel 3：归一化写回
__global__ void softmax_kernel(const float* input, float* output, float global_max, float global_sum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) output[idx] = expf(input[idx] - global_max) / global_sum;
}

extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *partial_max, *partial_sum;
    cudaMalloc(&partial_max, blocksPerGrid * sizeof(float));
    cudaMalloc(&partial_sum, blocksPerGrid * sizeof(float));
    float* partial_host = (float*)malloc(blocksPerGrid * sizeof(float));

    // Step 1: 各 block 局部 max → 拷回 host 求全局 max
    findMax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, partial_max, N);
    cudaDeviceSynchronize();
    cudaMemcpy(partial_host, partial_max, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    float global_max = -INFINITY;
    for (int i = 0; i < blocksPerGrid; i++) global_max = fmaxf(global_max, partial_host[i]);

    // Step 2: 各 block 局部 sum → 拷回 host 求全局 sum
    countSum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, partial_sum, global_max, N);
    cudaDeviceSynchronize();
    cudaMemcpy(partial_host, partial_sum, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    float global_sum = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++) global_sum += partial_host[i];

    // Step 3: normalize
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, global_max, global_sum, N);
    cudaDeviceSynchronize();

    free(partial_host);
    cudaFree(partial_max);
    cudaFree(partial_sum);
}
```

**正确性要点（都做对了）**：max trick 防 `exp` 溢出、越界线程补 `-INF`/`0` 不污染归约、树归约步步 `__syncthreads`、写回用 `=`（每元素独占输出，无需累加）。

---

## Part 6：优化建议（下一步）

softmax 是 **memory-bound**，优化主线就一条：**少读几遍 HBM**。当前 3-pass 把大数组 `input` 读了 **3 遍**：

| Pass | kernel | 读/写 HBM | 说明 |
|:-:|------|------|------|
| 1 | findMax | 读 input ×1 | 求全局 max |
| 2 | countSum | 读 input ×1 | 求 Σexp（`exp` 第 1 次） |
| 3 | softmax | 读 input ×1 + 写 output ×1 | 归一化（`exp` 第 2 次） |

三笔冗余成本 → 三个优化方向（对应 [NOW.md](../NOW.md) 的 TODO）：

**① fuse max+sum，用 online softmax 一趟出** `[面试]`
3-pass 的隐含依赖是"先知道全局 max 才能算 sum"。online softmax 打破它：
```
m_new = max(m_old, x_i)
s_new = s_old * exp(m_old - m_new) + exp(x_i - m_new)   // max 变了就修正已累加的 sum
```
一趟同时维护 `(m, s)` → **Kernel 1+2 合成一个**，`input` 从读 3 遍降到 2 遍，省 1/3 HBM。
这就是 **Flash Attention 不把 attention matrix 落回 HBM 的同一个 trick**，[Lesson 05](05-flash-attn-reading.md) 会原样再见到。

**② warp shuffle reduce**（骨架见 Part 3）
把 `for (s >>= 1)` 的 shared memory 树归约换成 `__shfl_down_sync`——warp 内（32 线程）寄存器直接交换，不走 shared memory、warp 内不用 `__syncthreads`。对应昇腾 Vector 单元的硬件 reduce 指令，只是 NV 要你用 shuffle 手动模拟。

**③ 跨 block 归约挪回 GPU** `[面试]`
当前用 `cudaMemcpy` D2H + CPU 循环做最后一级归约 → 2 次 round-trip + 把 3 个 kernel 强制串行化。两条路，是 occupancy 和协调开销的权衡：
- **多 block + 两级归约**：partial 用第二个小 kernel（或 atomic）在 GPU 上归约，别回 host。能打满带宽，但要处理 inter-block。
- **单 block + grid-stride loop** 扫完整个 N：完全免掉 inter-block 归约和 host round-trip，但只用 1 个 SM，打不满 HBM 带宽——小 N 划算，大 N 不划算。

**验收**：三版（3-pass / online / warp shuffle）跑通后对比吞吐（GB/s），确认 online 版接近 HBM 带宽上限。

> 建议顺序：① online 融合（收益最大 + 扣题）→ ② warp shuffle → ③ 去 host 归约 → benchmark。

---

## ✅ 本课检验清单

- [x] `softmax_naive` 跑通，结果正确（3-pass 跨 block，2026-07-01）
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
