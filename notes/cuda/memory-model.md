# CUDA 内存层级详解

> 理解 GPU 内存体系是算子优化的核心。本文覆盖内存层级、访问模式、常见问题和优化方法。
> 速查 API → [cuda-cheatsheet.md](./cuda-cheatsheet.md)

---

## 1. 内存层级全景

```
快 ┌─────────────────────────────────────────────┐
   │ Register (255 × 32-bit / thread)            │  ~0 cycle，每个 thread 私有
   │  — 编译器自动分配，程序员不能直接控制          │
   │  — 太多变量 → spill 到 local memory (慢)     │
   ├─────────────────────────────────────────────┤
   │ L1 / Shared Memory (192 KB / SM on A100)    │  ~30 cycles，block 内共享
   │  — 程序员显式管理 (__shared__)                │
   │  — L1 cache 和 shared memory 共享同一片 SRAM │
   │  — 可配置比例（如 64KB shared + 128KB L1）    │
   ├─────────────────────────────────────────────┤
   │ L2 Cache (40 MB on A100)                    │  ~200 cycles，所有 SM 共享
   │  — 硬件自动管理                                │
   │  — 全局内存的缓存                               │
   ├─────────────────────────────────────────────┤
   │ Global Memory / HBM (80 GB on A100)         │  ~600 cycles，所有 SM 共享
   │  — 容量最大，延迟最高                           │
   │  — 程序员显式管理 (cudaMalloc)                  │
慢 └─────────────────────────────────────────────┘
       ↑ 还有 Constant Memory / Texture Memory（已过时，不展开）
```

> **Ascend 对照**：
> - Shared Memory ≈ L1 Buffer / Unified Buffer — 都是片上、程序员显式管理
> - L2 Cache ≈ L2 Cache — 完全一致
> - Global Memory ≈ HBM — 完全一致
> - Register ≈ Ascend 的 L0 Buffer — 最快的存储层
> - CUDA 缺少 Ascend 的 L0A/L0B/L0C 专用缓存层

### 各存储层的大小和延迟参考 (A100)

| 层级 | 容量 | 延迟 | 带宽 | 作用域 |
|------|------|------|------|--------|
| Register | 256 KB/SM | ~0 | ~8 TB/s | 1 thread |
| Shared Memory | 164 KB/SM | ~30 cycles | ~1.5 TB/s | 1 block |
| L1 Cache | 64 KB/SM | ~30 cycles | ~1.5 TB/s | 1 SM |
| L2 Cache | 40 MB | ~200 cycles | ~4 TB/s | 全 GPU |
| HBM | 40/80 GB | ~600 cycles | ~1.6/2.0 TB/s | 全 GPU |

---

## 2. Global Memory — 你的数据默认住在这里

### 2.1 带宽和延迟

Global memory 是 GPU 上最大的存储，也是**最慢的**。一个 warp 发起一次 global memory 访问，要等 ~600 个时钟周期。在这段时间里，计算单元是空闲的——这就是为什么 naive kernel 往往受限于 memory。

```
A100 理论 HBM 带宽：~2.0 TB/s = 2000 GB/s
实际能达到的使用率：70-85%（优化好的 kernel）

你的 kernel 如果只用了 50 GB/s 的 bandwidth，
意味着你在用 GPU 的 2.5% 的带宽能力。
```

### 2.2 Memory Coalescing（合并访问）— `[面试]`

**这是 CUDA 最重要的内存优化概念之一。**

规则：**同一 warp 内的 32 个线程访问的地址，如果能落在 128B 对齐的连续区间内，GPU 会合并成一次 transaction。**

```cpp
// ❌ 非合并访问（stride = N）
// 每个线程读一个元素，但地址间隔 N×4B，远大于 128B
for (int k = 0; k < K; k++) {
    sum += A[row * K + k] * B[k * N + col];
    // B 的访问：thread 间 stride = N，非连续！
}

// ✅ 合并访问
// thread 0→A[0], thread 1→A[1], thread 2→A[2] ... 连续！
float val = A[idx];  // idx 在每个 thread 中递增 1
```

**图示**

```
Warp 内 32 个 thread 访问 global memory：

✅ 合并访问（一次 128B transaction 覆盖）：
Thread:  0     1     2     3     ... 31
Addr:   0x00  0x04  0x08  0x0C  ... 0x7C   ← 128B 连续

❌ 跨步访问（32 次 transaction，浪费带宽）：
Thread:  0     1     2     3     ... 31
Addr:   0x00  0x400 0x800 0xC00 ... 0xF400  ← 间隔 1KB
```

### 2.3 对齐要求

| 类型 | 大小 | 对齐 |
|------|------|------|
| float, int | 4B | 4B |
| float2, int2 | 8B | 8B |
| float4, int4 | 16B | 16B |

128B 对齐的起始地址 + 连续访问 = 最优。使用 `float4` 可以一次读 16B，减少 transaction 次数。

> `[面试]` 高频考题：给一个 kernel 的访问模式，判断是否 coalesced。

---

## 3. Shared Memory — 你的手动缓存

### 3.1 为什么需要 Shared Memory

Global memory latency 太高。如果你的数据需要重复读（比如 GEMM 中每个 block 多次读 A 的同一行），**先把数据搬到 shared memory，在片上高速读**。

```
GEMM 中不加 tiling：
  每个 C[i][j] 要从 global memory 读 K 次 A + K 次 B
  → K=1024 时，每个输出元素要忍受 2048 次 global memory 延迟

GEMM 加 tiling（TILE=32）：
  先搬 32×32 的 A_tile 到 shared memory（所有 thread 合作搬）
  在 shared memory 上计算 32×32 的部分积
  → 每个元素从 global memory 搬运一次，计算复用 32 次
```

> **Ascend 对照**：这和 Ascend 的 L1 Buffer tiling 完全一样——把大矩阵切开，小块搬进片上内存再计算。区别是 Ascend 有 pipe 机制做搬运和计算的流水，CUDA 需要手动调用 `__syncthreads()` 控制同步点。

### 3.2 声明和使用

```cpp
// 静态大小
__global__ void kernel() {
    __shared__ float tile[32][32];    // 32×32 = 1024 floats = 4 KB
    
    // 加载数据
    tile[threadIdx.y][threadIdx.x] = A[row * K + col];
    __syncthreads();  // ⚠️ 必须！确保所有 thread 都写完了
    
    // 使用数据
    float val = tile[threadIdx.y][threadIdx.x];
    __syncthreads();  // ⚠️ 要再次写入 tile 之前，确保所有 thread 用完
}
```

**为什么 `__syncthreads()` 必须？**
- 同一 block 内的 thread 物理上不是同时执行——它们在不同的 warp 上
- 你 thread(0,0) 写完 tile 时，thread(0,1) 可能还没开始写
- `__syncthreads()` 是一个 barrier：所有 thread 在这里等，直到 block 内所有 thread 都到达
- 相当于 Ascend 的 `block_sync()` / `pipe_barrier`

> `[面试]` 经典错误：在条件分支里用 `__syncthreads()`。如果 block 内有些 thread 不经过这个分支，barrier 就会死等 → 需要确保所有 thread（或没有 thread）到达同一个 `__syncthreads`。

### 3.3 Shared Memory Bank Conflict

Shared memory 由 32 个 bank 组成，每个 bank 4B 宽。每个时钟周期，一个 bank 只能服务一个 thread。如果同一 warp 内的多个 thread 访问同一个 bank 的不同地址 → 串行化 → bank conflict。

```cpp
// 一个 bank 的宽度是 4B（一个 float）
// 地址映射：bank = (addr / 4) % 32

// ✅ 无 bank conflict：连续访问不同 bank
tile[0][0], tile[0][1], tile[0][2], ... tile[0][31]
// bank:   0,        1,        2,    ...       31

// ❌ 2-way bank conflict：stride = 16 导致 2 threads/bank
tile[0][0], tile[1][0], tile[2][0], ... tile[31][0]
// 每 16 列卷回同一个 bank（32 × 4B = 128B → 32 banks × 4B，重复）

// ✅ 常见解法：padding
__shared__ float tile[32][33];  // 多一列，打破 stride=32 的周期
// 此时 tile[i][0] 和 tile[i+1][0] 的 bank 不同
```

> `[面试]` 高频题：给一个 shared memory 的访问 pattern，判断是几 way bank conflict，怎么解决。

### 3.4 容量限制和 Occupancy 权衡

| GPU | Shared Memory per SM | 最大 block 数/SM | 限制因素 |
|-----|---------------------|:---:|------|
| T4 | 64 KB (default) | 16 | block 的 shared mem 用得多 → 并发 block 数减少 |
| A100 | 164 KB (max) | 32 | |

**Occupancy = active warps / max warps per SM**。shared memory 用太多 → occupancy 下降 → 隐藏延迟的能力下降。需要在 tile size 和 occupancy 之间权衡。

> 这是 CUDA 特有的 trade-off，Ascend 没有 occupancy 这个概念。

---

## 4. Register 和 Occupancy

### 4.1 Register 分配

每个 thread 的 local variable 默认放 register（编译器决定）。一个 SM 上的 register 总数是固定的（A100: 65536 × 32-bit）。

```cpp
// 这个 thread 用了多少 register？
// 可以用 nvcc --ptxas-options=-v 或 ncu 查看
__global__ void kernel() {
    float a, b, c, d;  // 4 registers
    float arr[128];    // 128 registers（一般会 spill 到 local memory）
}
```

### 4.2 Register Spilling

当 thread 需要的 register 超过硬件上限时，编译器会把溢出的变量放到 **local memory**（没错，local memory 其实在 HBM 上，非常慢）。

```bash
# 编译时查看 register 用量
nvcc -Xptxas -v kernel.cu
# 输出：Used 64 registers, 0 bytes spill stores
# 如果 spill stores > 0，说明有变量被挤出 HBM，要优化
```

### 4.3 Occupancy 计算

$$
\text{Occupancy} = \frac{\text{active\_warps\_per\_SM}}{\text{max\_warps\_per\_SM}}
$$

限制因素（取最紧的那个）：
1. Register per thread：$\text{active\_warps} = \lfloor 65536 \;/\; (\text{reg\_per\_thread} \times 32) \rfloor$
2. Shared memory per block：$\text{active\_blocks} = \lfloor \text{shmem\_per\_SM} \;/\; \text{shmem\_per\_block} \rfloor$
3. Max blocks per SM：hardware limit (A100 = 32)

A100 例子：
- reg/thread = 64，shared/block = 32 KB，block_size = 256 (=8 warps)
- Register 限制：$\lfloor 65536 / (64 \times 32) \rfloor = 32$ warps = 4 blocks
- Shared mem 限制：$\lfloor 164\text{KB} / 32\text{KB} \rfloor = 5$ blocks
- Block 限制：max 32
- 最终：$\min(4, 5, 32) = 4$ blocks = 32 warps
- $\text{Occupancy} = 32/64 = 50\%$

> **Ascend 对照**：Ascend 没有 occupancy 概念，因为它的任务调度模型不同。但调 register/shared memory 用量来最大化并行度的思路是通用的。

---

## 5. 判断 Memory Bound vs Compute Bound

### 5.1 用 Nsight Compute 判断

```
关键指标：
- Memory Throughput / Peak Memory Bandwidth → 接近 100% = memory bound
- SM Utilization / Peak Compute Throughput → 接近 100% = compute bound
```

### 5.2 手指算法（Roofline Model）

$$
\text{Arithmetic Intensity (AI)} = \frac{\text{Total FLOPs}}{\text{Total Bytes Moved}}
$$

GEMM 例子：
- $M=N=K=1024$，FLOPs $= 2.1\text{G}$，Bytes $= \sim\!12\text{ MB}$
- $\text{AI} = 2.1\text{G} \;/\; 12\text{M} \approx 175 \text{ FLOP/byte}$
- A100: $312 \text{ TFLOPS} \;/\; 2000 \text{ GB/s} \approx 156 \text{ FLOP/byte}$ ← 转折点
- $\text{AI} > 156 \to$ compute bound；$\text{AI} < 156 \to$ memory bound
- $1024 \times 1024 \times 1024$ GEMM：$\text{AI} = 175 > 156 \to$ compute bound ✓

Vector Add 例子：
- $N=1\text{M}$，FLOPs $= 1\text{M}$，Bytes $= 12\text{ MB}$
- $\text{AI} = 1\text{M} \;/\; 12\text{M} = 0.083 \text{ FLOP/byte}$
- 远远 $< 156 \to$ 严重 memory bound ✗
```

> FLOPs 计算：乘加算 2 FLOP
> Bytes 计算：A(读) + B(读) + C(写) = 3N × 4B = 12N bytes

---

## 6. 常见内存优化策略速查

| 问题 | 症状 | 解法 |
|------|------|------|
| 非 coalesced 访问 | 低 bandwidth 利用率 | 调整 thread→数据映射，让同一 warp 访问连续地址 |
| 重复读 global memory | 高 latency，低 compute 利用率 | shared memory tiling |
| Bank conflict | Nsight 显示 shared memory 吞吐低 | padding（加一列）或重排访问模式 |
| Register spill | local memory 访问多 | 减少局部变量，拆分复杂表达式 |
| 低 occupancy | SM 上 active warp 太少 | 减小 shared memory 或 register 用量 |

---

## 相关文档

| 文档 | 内容 |
|------|------|
| [cuda-cheatsheet.md](./cuda-cheatsheet.md) | API 速查、代码模板 |
| [warp-and-sync.md](./warp-and-sync.md) | Warp 调度、divergence、shuffle |
| [gpu-architecture.md](./gpu-architecture.md) | NVIDIA vs Da Vinci 架构对比 |
