# Triton → CUDA 底层对照

> 学 CUDA 不是为了手写极致优化，而是为了**看懂 Triton 在底层做了什么**。
> 本文建立 Triton 概念和 CUDA 实现的对照关系。
> Triton API 速查 → [../../triton-kernels/notes/triton-cheatsheet.md](../../triton-kernels/notes/triton-cheatsheet.md)

---

## 1. 核心对照表

| Triton 概念 | 对应的 CUDA 实现 | 说明 |
|---|---|---|
| `tl.program_id(0)` | `blockIdx.x` | 当前 block 在哪个位置 |
| `tl.program_id(1)` | `blockIdx.y` | 2D block indexing |
| `tl.num_programs(0)` | `gridDim.x` | grid 中有多少个 block |
| `tl.load(ptr)` | shared mem load + `__syncthreads` (自动) | Triton 编译器自动分配 shared mem |
| `tl.store(ptr, val)` | shared mem write + copy to global (自动) | |
| `@triton.jit` | `__global__` kernel function | Triton 自动生成 launch code |
| `tl.arange(0, BLOCK)` | `threadIdx.x` 的范围 | 向量化表达，不需手动分 thread |
| block-level program | 1 block = 1 program instance | Triton 不是 thread-level，是 block-level |
| `tl.zeros(...)` | shared memory allocation + init | 编译器自动分配 |
| `tl.dot(a, b)` | `mma.sync` (Tensor Core) 或手动 FMA loop | 取决于硬件和 dtype |

---

## 2. 最核心差异：Block-Level vs Thread-Level

**这是理解 Triton 的关键。**

```python
# Triton — block-level programming
# 你描述一个 block 在做什么，不关心 block 内 thread 怎么分配
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)   # 这是一个向量！
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)             # 一条语句加载 BLOCK_SIZE 个元素
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y                                       # 向量运算
    tl.store(output_ptr + offsets, output, mask=mask)
```

```cpp
// CUDA 等价 — thread-level programming
// 你需要显式管理每个 thread
__global__ void add_kernel(float* x, float* y, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = x[idx] + y[idx];
    }
}
// 还要手动计算 grid/block 尺寸，分配 shared memory 等
```

**Triton 编译器自动做的事**（你不需手写 CUDA）：
1. 决定 block 内的 thread 数量和分布
2. 分配 shared memory（`tl.load` 自动用 shared memory 缓冲）
3. 插入 `__syncthreads()` barrier
4. 处理 bank conflict（尽可能避免）
5. 向量化 load/store（`float4` 等）
6. 选择 Tensor Core 路径（如果可用且有益）

---

## 3. Triton 的 Memory Hierarchy vs CUDA

```
Triton 编程模型          CUDA 实现
─────────────────────────────────────────────
Program (每个 program    Block
  instance 处理一个 tile)
                       
tl.load (从 global)     shared memory buffer (多 thread 合作 load)
                        → __syncthreads
                        → 从 shared memory 读取

tl.store (写到 global)  写入 shared memory buffer
                        → __syncthreads
                        → 多 thread 合作 write 到 global memory

block 内的数据           Register / Shared Memory (编译器自动决定)
                        → 小 tensor 放 register
                        → 大 tensor 放 shared memory
```

**关键洞察**：Triton 让你不用管 shared memory 的分配细节。它自动分析你的 kernel，决定哪些数据放 register、哪些放 shared memory、什么时候 `__syncthreads`。这就是为什么 Triton 的生产力高。

---

## 4. 常见 Triton Kernel 生成的 CUDA 结构

### Vector Add — 简单的 element-wise

```python
# Triton
@triton.jit
def add(x, y, out, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x_val = tl.load(x + offsets, mask=mask)
    y_val = tl.load(y + offsets, mask=mask)
    tl.store(out + offsets, x_val + y_val, mask=mask)
```

生成的 CUDA 大致逻辑：
```cpp
// 编译器生成（简化）
__global__ void add_kernel(float* x, float* y, float* out, int N, int BLOCK) {
    int pid = blockIdx.x;
    int base = pid * BLOCK + threadIdx.x;  // tl.arange → threadIdx.x
    
    // tl.load → coalesced global load
    if (base < N) {
        out[base] = x[base] + y[base];
    }
}
// element-wise 的 Triton ≈ 直接翻译成 CUDA，几乎无 overhead
```

### Tiled GEMM — 编译器自动插入 shared memory

```python
# Triton
@triton.jit
def gemm(A, B, C, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 编译器分析这些 access pattern，决定用 shared memory tiling
    a_ptrs = A + ...  # block of A
    b_ptrs = B + ...  # block of B
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)   # 编译器自动：多 thread 合作 load A_tile 到 shared mem
        b = tl.load(b_ptrs)   # 编译器自动：多 thread 合作 load B_tile 到 shared mem
        acc += tl.dot(a, b)   # 编译器自动：Tensor Core MMA 或 manual FMA loop
                              # 自动插入 __syncthreads
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    
    tl.store(c_ptrs, acc)
```

编译器在背后做的事：
1. **自动分块**：分析 `BLOCK_M × BLOCK_K` 的 A tile 和 `BLOCK_K × BLOCK_N` 的 B tile
2. **自动 shared memory allocation**：两个 tile 各放 shared memory
3. **自动 `__syncthreads()`**：load 完后 barrier，再计算
4. **自动 double buffering**：如果性能有利，编译器会分配两个 buffer 做 ping-pong
5. **自动选择 Tensor Core 路径**：如果 dtype 和尺寸匹配（FP16/BF16/INT8），用 `mma.sync`

你写的是：
```python
a = tl.load(a_ptrs)
b = tl.load(b_ptrs)
acc += tl.dot(a, b)
```

CUDA 等价物你如果手写需要 ~50 行：shared memory 分配、多 thread 合作加载、`__syncthreads`、FMA loop / mma 调用、bank conflict padding。

---

## 5. `tl.dot` 的底层实现

`tl.dot` 是 Triton GEMM 的核心。它的底层实现取决于硬件：

| 硬件 | dtype | `tl.dot` 实现 |
|------|-------|--------------|
| Ampere+ (A100/H100) | FP16/BF16/INT8 | `mma.sync` (Tensor Core) |
| Ampere+ (A100/H100) | FP32 | 手写 FMA loop（无 Tensor Core 加速） |
| Turing (T4) | FP16 | `mma.sync` (Turing Tensor Core) |
| Turing (T4) | FP32 | FMA loop |

**Tensor Core 的 `mma.sync` 长这样**（了解一下，不需要手写）：
```cpp
// 这是编译器调用的底层指令，不是你写的
// wmma API（较简单）：
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
wmma::load_matrix_sync(a_frag, a_ptr, lda);
wmma::load_matrix_sync(b_frag, b_ptr, ldb);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // 一次 16×16×16 的矩阵乘+累加
wmma::store_matrix_sync(c_ptr, c_frag, ldc, wmma::mem_row_major);
```

> 这就是为什么 FP16 GEMM 比 FP32 快很多——Tensor Core 一步算 16×16×16 = 4096 次乘加，而 FP32 的 FMA loop 一步只算一次。

---

## 6. 你学 CUDA 到什么程度就够了

针对你的目标（Triton 为主，CUDA 为辅）：

| 你需要理解 | 为什么 | 深度 |
|-----------|--------|------|
| grid/block/warp | 理解 `program_id` 和并行度 | 能用话说清 |
| shared memory | 知道 Triton 的 `tl.load` 背后在做什么 | 能解释原理 |
| `__syncthreads` | 知道 Triton 插入 barrier 的时机和原因 | 能解释为什么需要 |
| memory coalescing | 知道你写的 Triton 代码的 memory access 是否高效 | 能判断好坏 |
| bank conflict | 知道 Triton 有时需要手动 padding 来避免 | 能识别并解决 |
| occupancy | 知道为什么 tile size 不是越大越好 | 能解释 trade-off |
| Tensor Core 概念 | 知道 `tl.dot` 在什么时候用硬件加速 | 能说出条件 |
| PTX/SASS/MMA | ❌ 不需要 | 编译器的事 |
| 手写 tensor core GEMM | ❌ 不需要 | Triton 帮你做 |

---

## 7. 调试：看 Triton 生成的 CUDA 代码

```python
# 方法 1：打印中间表示
import triton
triton.runtime.driver.set_active_to_cpu()
# 写入 TTGIR 和 LLVM IR

# 方法 2：export PTX
@triton.autotune(...)
@triton.jit
def my_kernel(...): ...

# 用 TRITON_INTERPRET=1 环境变量在 CPU 上跑（调试模式）

# 方法 3：用 triton.compile 拿到编译产物
compiled = triton.compile(my_kernel, signature=..., constants=...)
# 查看 generated PTX
```

---

## 相关文档

| 文档 | 内容 |
|------|------|
| [../../triton-kernels/notes/triton-cheatsheet.md](../../triton-kernels/notes/triton-cheatsheet.md) | Triton 语法速查 |
| [memory-model.md](./memory-model.md) | CUDA 内存层级（理解 Triton 背后在做什么） |
| [warp-and-sync.md](./warp-and-sync.md) | Warp 调度（理解 Triton 的 block 内并行） |
| [gpu-architecture.md](./gpu-architecture.md) | GPU 架构基础 |
