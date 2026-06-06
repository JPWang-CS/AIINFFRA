# CUDA API 速查

> 日常写 CUDA 时最常用的语法、API、惯用法。每个条目给出最小可用代码。
> 深入原理见 → [memory-model.md](./memory-model.md) | [warp-and-sync.md](./warp-and-sync.md)

---

## 1. Kernel 声明与启动

```cpp
// 声明 — 三个前缀之一
__global__ void kernel(float* data) { }  // CPU 调用，GPU 执行
__device__ float helper(float x) { }     // GPU 调用，GPU 执行
__host__   void cpu_func() { }           // CPU 调用，CPU 执行

// 启动
kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(args);

// 典型值
dim3 block(256);        // 1D block，256 threads
dim3 grid( (N+255)/256 );  // 1D grid，足够覆盖 N 个元素

dim3 block2d(16, 16);   // 2D block，256 threads
dim3 grid2d( (W+15)/16, (H+15)/16 );  // 2D grid
```

### 内置变量（kernel 内可用）

```cpp
threadIdx.x, threadIdx.y, threadIdx.z   // block 内索引 (0 ~ blockDim-1)
blockIdx.x,  blockIdx.y,  blockIdx.z    // grid 内索引 (0 ~ gridDim-1)
blockDim.x,  blockDim.y,  blockDim.z    // 每个 block 的线程数
gridDim.x,   gridDim.y,   gridDim.z     // grid 中的 block 数

// 全局 ID（1D）
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 全局 ID（2D）
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

### 限制

| 项目 | 上限 |
|------|------|
| Threads per block | 1024 |
| Block dim (x/y) | 1024 |
| Block dim (z) | 64 |
| Grid dim (x) | 2³¹-1 |
| Shared memory per block | 48 KB (默认) / 164 KB (配置后) |
| Registers per thread | 255 |

---

## 2. 内存管理

```cpp
// 分配/释放
cudaMalloc(&d_ptr, size);     // 在 GPU 上分配
cudaFree(d_ptr);              // 释放

// 数据传输
cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);    // CPU → GPU
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);    // GPU → CPU
cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);  // GPU → GPU
cudaMemcpy(dst, src, size, cudaMemcpyDefault);         // 自动推断

// 零拷贝 / 统一内存（简化版，性能差）
cudaMallocManaged(&ptr, size);  // CPU/GPU 都能访问，驱动自动迁移
cudaFree(ptr);
```

### 命名惯例

```cpp
float* h_A;   // host 指针
float* d_A;   // device 指针
```

---

## 3. 错误检查

```cpp
// 宏（每个 CUDA API 调用后使用）
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// kernel launch 后用这两个检查
CUDA_CHECK(cudaGetLastError());       // 检查 launch 是否成功
CUDA_CHECK(cudaDeviceSynchronize());  // 等 GPU 跑完 + 检查 runtime 错误
```

---

## 4. 计时

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>(...);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

---

## 5. Grid-Stride Loop

处理 N 大于总线程数的情况：

```cpp
__global__ void kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;   // 所有线程总数
    
    for (int i = idx; i < N; i += stride) {
        data[i] = data[i] * 2.0f;
    }
}
// 优点：不管启动多少个 block，保证覆盖所有元素
// 且减少 grid 尺寸切换的开销
```

---

## 6. Shared Memory

```cpp
// 方式 1：静态大小
__global__ void kernel() {
    __shared__ float tile[32][32];   // 编译期确定大小
}

// 方式 2：动态大小（启动时指定）
__global__ void kernel() {
    extern __shared__ float smem[];  // 大小由 launch 时的第三个参数决定
}

// 启动
kernel<<<grid, block, shared_mem_bytes>>>();

// 同步（barrier）
__syncthreads();  // block 内所有线程在此等待，确保 shared memory 写入可见
```

> 深入理解 → [memory-model.md](./memory-model.md)

---

## 7. Warp 操作

```cpp
// Warp shuffle（warp 内 thread 之间通信，不经过 shared memory）
float val = __shfl_down_sync(0xffffffff, val, offset);  // 从 lane+offset 拿值

// Warp vote
int mask = __ballot_sync(0xffffffff, condition);  // 哪些 lane 满足条件

// Active mask（全 warp 参与）
#define FULL_MASK 0xffffffff
```

> 深入理解 → [warp-and-sync.md](./warp-and-sync.md)

---

## 8. 原子操作

```cpp
atomicAdd(&addr, val);     // 浮点加法
atomicMax(&addr, val);     // 最大值
atomicExch(&addr, val);    // 交换
atomicCAS(&addr, old, new); // compare-and-swap

// 注意：原子操作串行化，大量冲突时性能很差
```

---

## 9. 设备信息

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

printf("Name: %s\n", prop.name);
printf("SM count: %d\n", prop.multiProcessorCount);
printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
printf("Shared mem per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
printf("Warp size: %d\n", prop.warpSize);
printf("Clock rate: %d MHz\n", prop.clockRate / 1000);
printf("HBM bandwidth: %.1f GB/s\n",
       2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
```

---

## 10. 编译命令

```bash
# 基本编译
nvcc -O3 -arch=sm_75 kernel.cu -o kernel

# 常见 arch
# sm_75 = Turing (T4, RTX 2080)
# sm_80 = Ampere (A100)
# sm_86 = Ampere (RTX 3090, A40)
# sm_89 = Ada Lovelace (RTX 4090)
# sm_90 = Hopper (H100)
```

---

## 11. 常见模式速查

| 场景 | 模式 | 说明 |
|------|------|------|
| element-wise op | 1D grid + grid-stride loop | 最简单 |
| 2D matrix op | 2D block (16×16) + 2D grid | GEMM/Conv |
| reduction | 1D block + shared mem + warp shuffle | max/sum |
| tiled GEMM | 2D block + shared mem tile + `__syncthreads` | 见 week-03 |
| online softmax | block per row + warp reduce | 见 week-04 |

---

## 相关文档

| 文档 | 内容 |
|------|------|
| [memory-model.md](./memory-model.md) | 内存层级、coalescing、bank conflict |
| [warp-and-sync.md](./warp-and-sync.md) | Warp 调度、divergence、shuffle |
| [gpu-architecture.md](./gpu-architecture.md) | NVIDIA vs Da Vinci 架构对比 |
| [triton-under-the-hood.md](./triton-under-the-hood.md) | Triton 生成的 CUDA 代码长什么样 |
