# Week 1 — CUDA 基础 + 第一个 Kernel

> 目标：理解 CUDA 编程模型，写出 Vector Addition 并在 GPU 上跑通
> 时间：4-6 小时（分 2-3 天）
> 前置：有 Ascend C 经验，不需要从头学异构计算
> 平台：LeetGPU（浏览器直接写），不需要真机

📚 **本周涉及的底层知识**（跳到对应文档看完整原理）：
- [CUDA API 速查](../cuda-kernels/notes/cuda-cheatsheet.md) — 需要查 API 时随时看
- [内存层级详解](../cuda-kernels/notes/memory-model.md) — Day 1 内存部分
- [Warp 与同步](../cuda-kernels/notes/warp-and-sync.md) — Day 1 warp 部分
- [GPU 架构对比](../cuda-kernels/notes/gpu-architecture.md) — Ascend ↔ NVIDIA

🎯 **这对理解 Triton 有什么用**：
- `blockIdx` / `threadIdx` → 对应 Triton 的 `tl.program_id` 和 `tl.arange`
- `<<<grid, block>>>` → 对应 Triton 的 `grid` lambda
- grid-stride loop → Triton 自动处理，但你写的 kernel 没有它就不安全
- 详细对照见 → [Triton 底层 CUDA 对照](../cuda-kernels/notes/triton-under-the-hood.md)

---

## Day 1：CUDA 编程模型（1.5h 阅读 + 笔记）

### 1.1 核心概念

CUDA 的并行模型是 SIMT（Single Instruction, Multiple Threads）。和 Ascend 最大的不同：**Ascend 是你告诉硬件"搬运数据→计算→写回"，CUDA 是你告诉每个线程"你算哪个元素"。**

```
一个 kernel 启动时的层级：

Grid
├── Block (0,0)           Block (0,1)           Block (0,2)
│   ├── Thread 0          ├── Thread 0           ├── ...
│   ├── Thread 1          ├── Thread 1
│   ├── ...               ├── ...
│   └── Thread 255        └── Thread 255
├── Block (1,0)           ...
└── ...
```

**三个关键内置变量**：

```cpp
threadIdx.x   // 我在这个 block 里是第几个 thread
blockIdx.x    // 我这个 block 在 grid 里是第几个
blockDim.x    // 每个 block 有多少 thread

// 全局索引 = blockIdx.x * blockDim.x + threadIdx.x
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

> **Ascend 对照**：Ascend 没有 thread 概念。你用的是 tiling——把数据切成块，每块交给一个 AiCore。CUDA 的 grid/block 类似 tiling 的"分块→分配到计算单元"，但 CUDA 的 thread 粒度更细（每个 thread 算几个元素）。

### 1.2 Kernel 是什么

```cpp
// __global__ 表示这是一个 kernel，CPU 调用，GPU 执行
__global__ void my_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f;  // 每个 thread 处理一个元素
    }
}

// CPU 端调用：
my_kernel<<<grid_size, block_size>>>(d_data, N);
//         ^^^^^^^^^^^^^^^^^^^^^^^^  这是 CUDA 特有的 launch syntax
//         grid: 多少个 block
//         block: 每个 block 多少个 thread
```

> **Ascend 对照**：`<<<grid, block>>>` 就是 Ascend 的"把 tiling 任务分发到多少个 AiCore 上"。grid 决定分多少块，block 决定每块用多少计算资源。

### 1.3 内存层级（重要）

```
GPU 内存层级（从快到慢）：
Register    — 每个 thread 私有，最快，但最少（~255 个 32-bit reg/thread）
Shared Mem  — 每个 block 共享，快，程序员显式管理（~48-164 KB/block）
L2 Cache    — 所有 SM 共享，硬件自动管理
Global Mem  — 整个 GPU 共享（HBM），最慢但最大，程序员管理
```

> **Ascend 对照**：
> - Shared Memory ≈ **L1 Buffer / Unified Buffer**——都是片上、程序员显式管理
> - Global Memory ≈ **HBM**——一样的东西
> - Register ≈ Ascend 没有直接对应，但类似 Cube Unit 内部的寄存器文件
>
> 关键差异：CUDA 没有 Ascend 的 **L0A/L0B/L0C** 专用缓存层。Shared Memory 承担了这个角色。

### 1.4 Warp（CUDA 独有的概念，重点理解）

**一个 warp = 32 个 thread 的组。同一 warp 内的 32 个线程在执行同一条指令。**

```cpp
// 如果 warp 内线程走不同分支：
if (threadIdx.x % 2 == 0) {
    // 偶数线程执行这条
} else {
    // 奇数线程执行这条 → 但硬件只能串行！先执行 if，再执行 else
    // 这就是 warp divergence
}
```

> **Ascend 对照**：升腾的 Vector Unit 也有 SIMD 宽度（比如 256-bit，一次处理 8 个 FP32）。但 CUDA 的 warp divergence 是它独有的性能杀手——同一 warp 内分支会导致串行化。Ascend 没有这个因为它是数据流驱动的。

`[面试]` Warp divergence 是 CUDA 面试必考题。

---

## Day 2：Vector Addition — 你的第一个 Kernel（2h 动手）

### 任务

写一个 GPU 程序，做两个向量的逐元素加法：`C[i] = A[i] + B[i]`。

**自己写，不要复制仓库里的 `vector_add.cu`。写完可以对照参考。**

### 2.1 你要写的代码框架

```cpp
#include <cuda_runtime.h>
#include <cstdio>

// TODO: 你来写这个 kernel
__global__ void vector_add_kernel(const float* A, const float* B, float* C, int N) {
    // 1. 计算全局索引
    // 2. 边界判断
    // 3. 做加法
}

int main() {
    int N = 1 << 20;  // 1M 个元素
    size_t bytes = N * sizeof(float);

    // 1. 在 CPU 上分配内存并初始化
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }

    // 2. 在 GPU 上分配内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 3. 把数据从 CPU 拷到 GPU
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 4. 启动 kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 5. 等 GPU 跑完，把结果拷回 CPU
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 6. 验证结果
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            errors++;
        }
    }
    printf("Errors: %d / %d\n", errors, N);

    // 7. 清理
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return errors == 0 ? 0 : 1;
}
```

> **Ascend 对照**：
> - `cudaMalloc` ≈ Ascend 的 `hi_malloc` / `aclrtMalloc`——在设备端分配内存
> - `cudaMemcpy` ≈ Ascend 的 `aclrtMemcpy`——Host↔Device 数据传输
> - `<<<grid, block>>>` ≈ Ascend 的 kernel launch（但没有显式 task 分发）

### 2.2 在 LeetGPU 上跑（推荐，零环境配置）

1. 打开 [leetgpu.com](https://leetgpu.com)，注册登录
2. 搜索 "Vector Addition"
3. 选择 **CUDA** 标签
4. 你只需要写 kernel 函数体，平台提供 `solve` 入口：

```cpp
// LeetGPU 的 starter 模板：
#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    // TODO: 你来写
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```

> **注意**：LeetGPU 的 `solve` 参数已经是 device pointer，你不需要 `cudaMalloc`/`cudaMemcpy`。写完 kernel 就行。

### 2.3 思考题（做完 kernel 后回答）

1. `threadsPerBlock = 256` 为什么选 256 而不是 128 或 512？
2. 如果 N = 25,000,000，`blocksPerGrid` 是多少？
3. 数据从 CPU 拷到 GPU 要多久？这个时间在总耗时中占比多少？
4. 这个 kernel 是 compute-bound 还是 memory-bound？为什么？

---

## Day 3：搞懂执行模型 + 错误检查（1.5h）

### 3.1 加 CUDA Error Check

裸写 `cudaMalloc` 很危险——出错了你不知道。把这几个宏记熟：

```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// 用法：所有 CUDA API 调用都包一层
CUDA_CHECK(cudaMalloc(&d_A, bytes));
CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
CUDA_CHECK(cudaDeviceSynchronize());  // kernel launch 后用这个等 GPU 跑完
```

### 3.2 Grid-Stride Loop（处理 N > grid 容量的情况）

上面的 kernel 有一个隐患：如果 N 比 `blocksPerGrid × threadsPerBlock` 大怎么办？用 **grid-stride loop**：

```cpp
__global__ void vector_add_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;   // 所有线程总数
    
    for (int i = idx; i < N; i += stride) {  // 跳着处理
        C[i] = A[i] + B[i];
    }
}
```

> 这样不管 N 多大，所有元素都会被覆盖。grid-stride loop 是 CUDA 编程的基础惯用法。

### 3.3 用 cudaEvent 计时

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
vector_add_kernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel time: %.3f ms\n", ms);

// 算 bandwidth
float gb_per_sec = (3.0f * N * sizeof(float)) / (ms / 1000.0f) / 1e9f;
printf("Bandwidth: %.2f GB/s\n", gb_per_sec);
```

> Vector Add 是典型的 memory-bound kernel：3N 个 float 的读写 ÷ kernel 耗时 = bandwidth
> 对比 GPU 的理论 HBM 带宽（T4: ~320 GB/s，A100: ~2 TB/s），看你的 kernel 利用率。

---

## ✅ Week 1 检验清单

完成以下所有项再进入 Week 2：

- [ ] 能用自己的话说清 thread/block/grid/warp 四个概念
- [ ] 写出了 `vector_add_kernel`，在 LeetGPU 或本地 GPU 上跑通，结果全对
- [ ] 理解了 grid-stride loop 的写法
- [ ] 能给 kernel 加 CUDA error check + cudaEvent 计时
- [ ] 能回答：为什么 Vector Add 是 memory-bound？
- [ ] 能说出 Ascend 和 CUDA 在编程模型上的 3 个关键差异

---

## 知识库索引

> 这些文档覆盖完整原理，本文只做梗概。遇到不懂的跳过去读。

| 想深入理解 | 去看 |
|-----------|------|
| CUDA 所有 API 的快速查阅 | [cuda-cheatsheet.md](../cuda-kernels/notes/cuda-cheatsheet.md) |
| 内存层级、coalescing、bank conflict | [memory-model.md](../cuda-kernels/notes/memory-model.md) |
| Warp 调度、divergence、shuffle、同步 | [warp-and-sync.md](../cuda-kernels/notes/warp-and-sync.md) |
| NVIDIA vs Da Vinci 架构对比 | [gpu-architecture.md](../cuda-kernels/notes/gpu-architecture.md) |
| Triton 生成的 CUDA 代码长什么样 | [triton-under-the-hood.md](../cuda-kernels/notes/triton-under-the-hood.md) |
| LeetGPU 全部题目 | [leetgpu-challenges.md](../cuda-kernels/notes/leetgpu-challenges.md) |
| 完整 Vector Add 参考（先自己写！） | [../cuda-kernels/vector_add.cu](../cuda-kernels/vector_add.cu) |
| 工具函数 | [../cuda-kernels/include/cuda_utils.h](../cuda-kernels/include/cuda_utils.h) |

---

*Week 1 · CUDA 基础*
