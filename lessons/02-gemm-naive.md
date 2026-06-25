# Lesson 02 — GEMM v0 Naive

> 主题：理解矩阵乘的 GPU 实现，写出 naive GEMM，定位瓶颈
> 前置：完成 Lesson 01，能写简单 kernel + 计时
> 平台：LeetGPU（`2_matrix_multiplication` 或 `22_gemm`）
> 状态：✅ 已完成（2026-06-16，见 [PATH.md](../PATH.md)；我的解法在 [solutions/cuda/](../solutions/cuda/)）

📚 **本课涉及的底层知识**：
- [内存层级详解](../notes/cuda/memory-model.md) — coalescing、shared memory、arithmetic intensity
- [CUDA API 速查](../notes/cuda/cuda-cheatsheet.md) — 2D grid/block、计时
- [LeetGPU 题库](../notes/cuda/leetgpu-challenges.md) — `22_gemm` 题签名和约束

🎯 **这对理解 Triton 有什么用**：
- naive GEMM 的瓶颈分析 → 理解为什么 Triton 的 `tl.load` + shared memory tiling 是必须的
- GEMM 的访问模式 → 知道 Triton 的 `tl.dot` 相比手写 FMA loop 加速了多少
- arithmetic intensity → 理解为什么 GEMM 从 memory-bound 变成 compute-bound（K 足够大时）
- 详细对照见 → [Triton 底层 CUDA 对照](../notes/cuda/triton-under-the-hood.md)

---

## Part 1：GEMM 是什么 + 为什么它是一切的基础

### 1.1 定义

```
C[M×N] = α × A[M×K] × B[K×N] + β × C[M×N]

简化版（α=1, β=0）：C = A × B
```

每个输出元素是两个向量的点积：

```
C[i][j] = Σ(k=0..K-1) A[i][k] × B[k][j]
```

> **Ascend 对照**：Ascend 上你用 Cube Unit 的 `mmad` 指令直接做矩阵乘，硬件帮你搞定 tiling、数据搬运、累加。CUDA 的 v0 阶段没有这种待遇——你要自己让每个 thread 做 dot product。

### 1.2 计算量

一个 M×K × K×N 的矩阵乘需要：
- **2×M×N×K 次 FLOP**（每次乘加算 2 FLOP）
- M=1024, N=1024, K=1024 → 2.1 GFLOP
- 读 A 和 B 各需 M×K + K×N = 2M 次内存访问（same order）

**GEMM 是 compute-heavy 算子**——计算密度 O(K)，K 越大，计算相对访存的优势越大。

> `[面试]` 这是经典考点：GEMM 的 arithmetic intensity = O(K)。K=64 时 memory-bound，K=1024 时 compute-bound。

### 1.3 数据布局

**Row-major**：`A[i][j] = A[i * K + j]`（C 语言的默认布局）

```
矩阵 A (M×K):         矩阵 B (K×N):
[a₀₀ a₀₁ a₀₂]  K=3   [b₀₀ b₀₁]  N=2
[a₁₀ a₁₁ a₁₂]  M=2   [b₁₀ b₁₁]  K=3
                       [b₂₀ b₂₁]

C[i][j] = aᵢ₀×b₀ⱼ + aᵢ₁×b₁ⱼ + aᵢ₂×b₂ⱼ
```

---

## Part 2：写出 Naive GEMM（动手）

### 2.1 任务

写一个 kernel，每个 thread 计算 C 的一个元素。**所有数据直接从 global memory 读，不做任何优化。**

### 2.2 你要写的代码

```cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

// TODO: 你来写
__global__ void gemm_naive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    // 1. 用 blockIdx 和 threadIdx 计算 row, col
    // 2. 对 row-th 行、col-th 列，做 dot product over K
    // 3. 写入 C[row * N + col]
}

int main() {
    int M = 512, N = 512, K = 512;
    // 1. 分配 host + device 内存
    // 2. 随机初始化 A 和 B（用 rand() / RAND_MAX）
    // 3. 拷到 device
    // 4. 配置 2D block（如 16×16）
    // 5. 启动 kernel + 计时（cudaEvent）
    // 6. 拷回结果
    // 7. 验证正确性（对比 CPU 实现）
    // 8. 输出 GFLOPS
    return 0;
}
```

### 2.3 关键提示

**2D grid/block**：GEMM 天然适合 2D。每一维都独立索引：

```cpp
// block 是 2D 的 (16, 16)
dim3 blockDim(16, 16);
dim3 gridDim((N + 15) / 16, (M + 15) / 16);

// kernel 内用两个维度
int row = blockIdx.y * blockDim.y + threadIdx.y;   // M 维度
int col = blockIdx.x * blockDim.x + threadIdx.x;   // N 维度
```

> 💡 `dim3` 是 CUDA 内置三维向量类型，没有 `dim1`/`dim2`，未指定维度默认=1 → [cuda-cheatsheet §1](../notes/cuda/cuda-cheatsheet.md#1-kernel-声明与启动)

> **Ascend 对照**：这和 Ascend 的 tiling 分块思路一样。你把输出矩阵 C 切成 (M/16)×(N/16) 个小块，每个 block 算一块。但 CUDA v0 阶段，每个 block 内的 thread 还是一个个从 global memory 读 A 和 B，没有用 shared memory。

**计算 GFLOPS**：

```cpp
float gflops = (2.0f * M * N * K) / (ms / 1000.0f) / 1e9f;
// 2*M*N*K = 总的浮点操作数（乘+加算 2 FLOP）
```

### 2.4 我的实现 — `2_matrix_multiplication`（✅ 已通过）

> LeetGPU 题目：A(M×N) × B(N×K) = C(M×K)，row-major，float32
> 完整代码归档在 → [solutions/cuda/](../solutions/cuda/)

```cpp
#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C,
                                             int M, int N, int K) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;  // K 维度索引
    int m = blockDim.y * blockIdx.y + threadIdx.y;  // M 维度索引
    int idx = m * K + k;

    if ((k < K) && (m < M)) {
        float sum = 0;
        for (int n = 0; n < N; n++) {
            sum += A[m * N + n] * B[n * K + k];
        }
        C[idx] = sum;  // 每线程独占一个输出，用 = 不用 +=
    }
}

extern "C" void solve(const float* A, const float* B, float* C,
                      int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + 15) / 16, (M + 15) / 16);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```

| 关键点 | 说明 |
|--------|------|
| Grid 铺法 | blockIdx.x → K, blockIdx.y → M |
| 每线程 | 读 A 的一行 × B 的一列，沿 N 累加 |
| Coalescing | ✅ 其实是连续的：同一 warp 内 `threadIdx.x→k` 相邻，`B[n*K+k]`、`C[m*K+k]` 都是合并访问。（stride=K 是"单线程随 n 走"的轨迹，不是 coalescing 的判据）|
| 正确性 | 应写 `C[idx] = sum`：每线程独占一个输出，无需累加/atomic。`+=` 依赖 C 被清零，是埋雷 |
| 真正瓶颈 | 数据复用率低：A 被读 K 次、B 被读 M 次 → memory-bound（算术强度≈0.25）|
| 后续优化 | shared memory tiling（→ [Lesson 03](03-gemm-tiled.md)） |

### 2.5 LeetGPU 版（`22_gemm` 题，FP16）— 进阶

如果有兴趣挑战 FP16 + tensor core，可以试 `22_gemm`。签名和上面的不同：

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// LeetGPU 的 solve 签名（FP16）：
// A: M×K, B: K×N, C: M×N — all half precision
// alpha, beta: float scale factors
// 计算: C = alpha * (A × B) + beta * C
extern "C" void solve(const half* A, const half* B, half* C,
                      int M, int N, int K, float alpha, float beta) {
    // TODO: 你来写 kernel + launch
    // 注意：所有指针已经是 device pointer，不需要 cudaMalloc/cudaMemcpy
    // 注意：half 和 float 的混合运算需要显式转换
}
```

LeetGPU 完整约束见 → [题库 22_gemm](../notes/cuda/leetgpu-challenges.md)

### 2.6 正确性验证

写一个 CPU 版本的 GEMM 作为 ground truth：

```cpp
void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
```

比较 GPU 和 CPU 结果的最大绝对误差，< 1e-3 即可通过。

---

## Part 3：分析瓶颈

### 3.1 跑起来，看数据

在 GPU 上跑你的 naive GEMM。M=N=K=512 大概需要多长时间？GFLOPS 是多少？

**参考数据**（T4 上的典型结果）：
- T4 理论峰值：~65 TFLOPS (FP16) / ~8 TFLOPS (FP32)
- Naive GEMM FP32：通常 10-50 GFLOPS，只有峰值的 **0.5-2%**

### 3.2 为什么这么慢？

naive GEMM 的瓶颈是 **global memory 延迟**。

```
每个 thread 算一个 C[i][j]：
  → 读 A 的 K 个元素（global memory）
  → 读 B 的 K 个元素（global memory）
  → 算 K 次乘加
  → 写 1 个结果

计算量：2K FLOP
访存量：2K × 4 bytes = 8K bytes
算术强度：2K / 8K = 0.25 FLOP/byte  ← 极低！
```

T4 的 HBM 带宽 ~320 GB/s。0.25 FLOP/byte × 320 GB/s ≈ **80 GFLOPS** 是这个 kernel 能达到的理论上限——远低于 T4 FP32 峰值的 8 TFLOPS。

**所以 naive GEMM 是 memory-bound。大部分时间花在等数据上，计算单元空闲。**

> **Ascend 对照**：Ascend 上你不做 L1 Buffer tiling 也一样——Cube Unit 等数据从 HBM 过来，利用率上不去。优化的核心思路两个平台完全一样：**把数据搬进片上内存，复用多次，减少 HBM 访问**。

### 3.3 画出访问模式

```
C[i][j] 需要 A 的第 i 行和 B 的第 j 列：

A: [aᵢ₀  aᵢ₁  aᵢ₂  ...  aᵢₖ₋₁]  ← 读 K 个连续元素（好，coalesced）
B:  b₀ⱼ                           ← 读 K 个元素，但跨 N 的步长！
    b₁ⱼ
    b₂ⱼ
    ...
    bₖ₋₁ⱼ
```

A 的访问是连续的（coalesced），B 的访问是跨步的（stride = N）。GPU 每次读 B 实际上都在读一个 cache line 但只用一个 float——严重的带宽浪费。

> `[面试]` Global memory coalescing 是 CUDA 面试的高频题。规则：同一 warp 内的 32 个线程访问的地址要在 128B 对齐的连续区间内，才能合并成一次 transaction。

---

## ✅ 本课检验清单

- [x] ✅ 2026-06-16 写出了 `gemm_naive`，在 LeetGPU 上跑通（`2_matrix_multiplication`，float，2D grid 16×16；写回用 `=` 而非 `+=`）
- [ ] 知道自己的 kernel 跑了多少 GFLOPS
- [ ] 能解释为什么 naive GEMM 慢：arithmetic intensity 低，memory-bound
- [ ] 能画出每个 thread 访问 A 和 B 的 pattern，指出 coalescing 问题
- [ ] 能用 grid-stride loop 重写 kernel（处理 M/N > grid 的情况）
- [ ] 知道下一步优化方向：shared memory tiling（→ [Lesson 03](03-gemm-tiled.md)）

---

## 知识库索引

| 想深入理解 | 去看 |
|-----------|------|
| 内存层级、coalescing、bank conflict | [memory-model.md](../notes/cuda/memory-model.md) |
| CUDA API 速查（计时、grid/block） | [cuda-cheatsheet.md](../notes/cuda/cuda-cheatsheet.md) |
| LeetGPU GEMM 题完整规格 | [leetgpu-challenges.md](../notes/cuda/leetgpu-challenges.md) → `22_gemm` |
| Triton GEMM 的实现（之后对照） | [triton-cheatsheet.md](../notes/triton/triton-cheatsheet.md) |
| ⭐ 可选深钻（tensor core 等） | [roadmap/leetgpu-ladder.md](../roadmap/leetgpu-ladder.md) |
| GEMM 参考实现（先自己写！） | [reference/cuda/gemm/gemm.cu](../reference/cuda/gemm/gemm.cu) |

---

*Lesson 02 · GEMM v0 Naive · 源自原 week-02*
