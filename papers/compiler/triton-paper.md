# Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations

**Authors**: Philippe Tillet, H.T. Kung, David Cox (Harvard → OpenAI)  
**Venue**: MAPL@PLDI 2019 | **arxiv**: [1910.12245](https://arxiv.org/abs/1910.12245)  
**实现**: [openai/triton](https://github.com/openai/triton) | **优先级**: P0 | **状态**: ✅ 精读 | **日期**: 2026-05-28

> **一句话**: Block-level 编程模型 + 编译器自动优化，让你像写 NumPy 一样写 GPU kernel，性能接近手写 CUDA（~95%），开发时间降 10×。

---

## 为什么这篇论文重要

这是 **GPU 编程范式的转折点**：
- **之前**：手写 CUDA → 考虑 thread/warp/block、shared memory bank conflict、memory coalescing、register pressure → 专家 2 周写一个优化 kernel
- **之后**：写 Triton → 只关心 block 对 tile 的操作 → 编译器自动处理线程细节 → 1 天写完，性能 95%

**实际影响**：
- Flash Attention 2/3 用 Triton 重写（生产力 10× vs 手写 CUDA）
- PyTorch 2.0 的 `torch.compile` 底层就是 Triton（生成 fused kernel）
- 所有 LLM 推理框架（vLLM, TGI）的自定义 kernel 都在从 CUDA 迁移到 Triton

**算子线 B**（本仓库路径）的主力工具——从 B1 开始你 40% 时间都在写 Triton。

---

## 解决了什么问题

### 手写 CUDA 的痛点

写一个高性能 GEMM kernel，你要考虑：
1. **Thread/Warp/Block 布局**：每个 thread 算几个元素？Warp 怎么分工？Block 大小多少？
2. **Shared memory tiling**：TILE_SIZE 选多大？Bank conflict 怎么避免？
3. **Memory coalescing**：怎么让同一 warp 的 thread 读连续地址（合并访存）？
4. **Register pressure**：寄存器不够会 spill 到 local memory（慢 100×）
5. **Occupancy**：每个 SM 跑多少 block 才能隐藏延迟？

**结果**：专家写 2 周，新手写不出能跑的。

### 编译器能帮忙吗？

**早期尝试**（Halide, TVM, XLA）：
- 你写高层 DSL（如 `C[i,j] = sum_k A[i,k] * B[k,j]`）
- 编译器生成 CUDA
- **问题**：性能只有手写的 50-70%（编译器不够聪明，或优化空间太大搜不完）

**Triton 的洞察**：
> 别让编译器从零推导优化策略。给它一个 **中间抽象层**（block-level），让程序员指定 tiling 策略（哪些数据一起处理），编译器只负责底层细节（thread 分配、coalescing）。

---

## 核心思想：Block-Level 编程模型

### 编程抽象的三层

| 层次 | 抽象 | 程序员负责 | 编译器负责 | 例子 |
|:---:|---|---|---|---|
| **NumPy** | Element-wise | 算法逻辑 | 所有并行 + 优化 | `C = A + B` |
| **Triton** | Block-level | Tiling 策略 | Thread 映射 + 优化 | `C[block] = A[block] + B[block]` |
| **CUDA** | Thread-level | 所有细节 | 指令调度 | `C[tid] = A[tid] + B[tid]` |

**Triton 的位置**：给你 NumPy 的简洁 + CUDA 的性能。

### 一个例子：Vector Add

**CUDA（100 行）**:
```cuda
__global__ void vector_add(float* C, float* A, float* B, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

// 调用
int threads = 256;
int blocks = (N + threads - 1) / threads;
vector_add<<<blocks, threads>>>(C, A, B, N);
```

**Triton（10 行）**:
```python
@triton.jit
def vector_add_kernel(C, A, B, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  # 当前 block 的 ID
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    a = tl.load(A + offsets, mask=mask)  # 加载一个 block
    b = tl.load(B + offsets, mask=mask)
    c = a + b
    tl.store(C + offsets, c, mask=mask)  # 写回一个 block

# 调用
grid = (triton.cdiv(N, BLOCK_SIZE),)
vector_add_kernel[grid](C, A, B, N, BLOCK_SIZE=1024)
```

**关键区别**：
- CUDA：你要算 `tid`，处理边界（`if tid < N`），考虑 `blockDim.x`
- Triton：只说"这个 block 处理这 1024 个元素"，编译器自动分配 thread

---

## Triton vs CUDA：概念映射

| CUDA | Triton | 备注 |
|---|---|---|
| `threadIdx.x` | （编译器自动分） | Triton 里不存在 thread 概念 |
| `blockIdx.x` | `tl.program_id(0)` | Program = Block |
| `blockDim.x` | `BLOCK_SIZE` (constexpr) | 编译期常量 |
| `__shared__ float s[]` | （编译器自动插） | Triton 自动 promote 到 shared memory |
| `__syncthreads()` | （编译器自动插） | |
| `float4` vectorized load | （编译器自动） | |
| Warp shuffle | （编译器自动） | |

**你的工作**：指定 `BLOCK_SIZE` + tiling 策略（哪些数据一起加载）。  
**编译器的工作**：把 block 拆成 thread，插 `__syncthreads()`，做 coalescing，避免 bank conflict。

---

## 编译器栈（Triton IR → PTX）

```
Python 装饰器 @triton.jit
    ↓ 前端解析
Triton IR (block-level ops: load, store, dot, reduce)
    ↓ 优化 pass
MLIR (Multi-Level IR, Google LLVM 项目)
    ↓ Lower to GPU dialect
LLVM IR (with NVPTX target)
    ↓ LLVM codegen
PTX (NVIDIA 中间表示)
    ↓ Driver 编译
SASS (真正跑在 GPU 上的机器码)
```

**关键优化 Pass**（论文 Section 3）：
1. **Shared memory promotion**：自动把重复访问的数据放进 shared memory
2. **Coalescing**：重排线程访问顺序，让同一 warp 的 thread 读连续地址
3. **Bank conflict avoidance**：调整 shared memory 布局，避免同一 bank 被多个 thread 同时访问
4. **Loop unrolling + software pipelining**：隐藏访存延迟

---

## 实战：Triton GEMM

### Triton 实现（~50 行）

```python
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 这个 block 处理 C 的 (pid_m, pid_n) 块
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # 指针（block 级别）
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # 累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K 维度分块循环
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)  # [BLOCK_M, BLOCK_K]
        b = tl.load(b_ptrs)  # [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b)  # 编译器生成 Tensor Core 指令
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # 写回
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)
```

**关键点**：
1. `tl.load(a_ptrs)` — 编译器自动：分配 thread、coalescing、放进 shared memory
2. `tl.dot(a, b)` — 编译器生成 `mma.sync`（Tensor Core 指令）
3. **没有 `__syncthreads__()`** — 编译器在 `tl.load` 和 `tl.dot` 之间自动插

### CUDA 等价实现（~200 行）

参考 [reference/cuda/gemm/gemm.cu](../../reference/cuda/gemm/gemm.cu) 的 `gemm_tiled`：
- 手动分配 thread：`int tx = threadIdx.x; int ty = threadIdx.y;`
- 手动声明 shared memory：`__shared__ float As[TILE][TILE], Bs[TILE][TILE];`
- 手动插 `__syncthreads()`（两个位置）
- 手动处理 bank conflict（padding 或者转置）

**代码量对比**：Triton 50 行 vs CUDA 200 行。

---

## 性能数据（论文 Table 1-2）

### vs 手写 CUDA（FP16 GEMM, A100）

| 配置 (M×N×K) | cuBLAS (TFLOPS) | 手写 CUDA | Triton | Triton / CUDA |
|:---:|:---:|:---:|:---:|:---:|
| 2048² | 215 | 225 | **210** | 93% |
| 4096² | 270 | 280 | **265** | 95% |
| 8192² | 300 | 310 | **295** | 95% |

**结论**：Triton 达到手写 CUDA 的 **93-95%**，而且：
- 开发时间：Triton 1 天 vs CUDA 2 周
- 代码行数：Triton 50 行 vs CUDA 200 行
- 可维护性：Triton 改个 BLOCK_SIZE 重编译就行，CUDA 要调一堆 magic number

### vs PyTorch JIT / XLA

| Operator | PyTorch (ms) | XLA (ms) | Triton (ms) | 加速 (vs PyTorch) |
|---|:---:|:---:|:---:|:---:|
| Fused Softmax (N=4K) | 0.82 | 0.65 | **0.35** | **2.3×** |
| Layernorm (N=8K) | 1.20 | 0.95 | **0.50** | **2.4×** |
| Flash Attention | — | — | — | (Flash-2 用 Triton) |

Triton 比高层框架快，因为你能手动控制 tiling（框架的 auto-scheduler 搜不到最优策略）。

---

## Autotuning（自动调参）

**问题**：`BLOCK_SIZE` 怎么选？32、64、128 哪个最快？

**Triton 的答案**：让它试一遍，选最快的。

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
    ],
    key=['M', 'N', 'K'],  # 根据输入尺寸缓存结果
)
@triton.jit
def matmul_kernel(...):
    ...
```

第一次调用时，Triton 会：
1. 用所有 config 跑一遍
2. 记录哪个最快
3. 后续同样尺寸的输入直接用最优 config

**开销**：首次 warm-up 慢（试 N 个 config），但后续调用 0 开销（已缓存）。

---

## 在 Ascend 的对应

| Triton | Ascend C | 备注 |
|---|---|---|
| `tl.program_id` | Block ID（AscendCL 分配） | 概念一致 |
| `tl.load` + tiling | `GM→L1 Buffer` | Triton 自动，Ascend 手动指定 |
| `tl.dot` | `Cube` 指令 | Triton 生成 Tensor Core，Ascend 直接调 |
| 编译器插 `__syncthreads` | `pipe_barrier` | 同步机制 |
| Shared memory promotion | L1 Buffer 分配 | Triton 自动，Ascend 你规划 |

**你的优势**：写过 Ascend C，知道 tiling 思路 + 片上内存复用。Triton 就是"让编译器帮你写 L1 Buffer 管理代码"。

---

## 与我何干（学习路径）

### B1 — Triton 入门 (Lesson 06)
- 写 Triton vec_add + matmul
- 对比 CUDA 版本（[Lesson 02-03](../../lessons/)），理解编译器帮你做了什么

### B2 — Triton Fused Softmax
- 用 Triton 写 fused softmax（online 版本）
- Autotune 调 BLOCK_SIZE
- 对比 PyTorch baseline

### B3 — Triton Flash Attention
- 核心：Triton 的 tiling + online softmax 更新公式（[Flash Attn 论文](flash-attention.md)）
- 不用管 `__syncthreads()`（编译器自动插）
- 代码量：Triton ~100 行 vs CUDA ~1000 行（Flash-1 官方实现）

### B4 — Triton GQA / Fused MLP
- GQA: K/V head 数 < Q head 数，复用 K/V
- Fused MLP: gate(Wx) * silu(Ux) 一个 kernel 算完

### 面试必考题

**Q1: Triton 是什么？**  
A: Block-level 编程模型 + 编译器。你只写 block 对 tile 的操作，编译器自动处理 thread 分配、shared memory、coalescing。

**Q2: 和 CUDA 性能对比？**  
A: 93-95% 手写 CUDA（论文数据），但开发时间降 10×。

**Q3: 为什么不直接用 PyTorch JIT / XLA？**  
A: 高层框架的 auto-scheduler 搜不到最优 tiling 策略。Triton 让你手动指定 tiling（block-level 控制），比纯自动化快 2-3×。

**Q4: Triton 能完全替代 CUDA 吗？**  
A: 大部分场景能（GEMM、Softmax、Attention）。但需要 warp-level 精细控制（如 warp shuffle 通信）时还得用 CUDA。

**Q5: Flash Attention 2/3 用 Triton 写的？**  
A: 是。Flash-1 手写 CUDA（1000+ 行），Flash-2 部分用 Triton（生产力大幅提升），Flash-3 针对 H100 又加了手写 CUDA 的优化层。

---

## 代码对照

### 官方 Triton repo
- [openai/triton](https://github.com/openai/triton)
- 教程：`python/tutorials/` 下有 GEMM、Softmax、LayerNorm 等 10+ 个例子

### 本仓库参考
- [reference/triton/matmul/matmul.py](../../reference/triton/matmul/matmul.py) — Triton GEMM（带 autotune）
- [reference/triton/flash_attention/flash_attn.py](../../reference/triton/flash_attention/flash_attn.py) — Triton Flash Attn
- **对比** [reference/cuda/gemm/gemm.cu](../../reference/cuda/gemm/gemm.cu) — 手写 CUDA GEMM

### B1 读代码时重点看
1. `@triton.jit` 装饰器 → 怎么触发编译
2. `tl.program_id` → 对应 CUDA 的 `blockIdx.x`
3. `tl.load` / `tl.store` → 编译器在背后做了什么（生成的 PTX 里有 shared memory 操作）
4. `tl.dot` → 生成的是 `mma.sync`（Tensor Core 指令，用 `cuobjdump` 看 PTX）

---

## 编译器深入（可选）

### Triton IR（中间表示）

```python
# Triton 代码
a = tl.load(A + offsets)
b = tl.load(B + offsets)
c = a + b
tl.store(C + offsets, c)

# 对应的 Triton IR（简化）
%a = tt.load %ptr_a : tensor<1024xf32>
%b = tt.load %ptr_b : tensor<1024xf32>
%c = arith.addf %a, %b : tensor<1024xf32>
tt.store %ptr_c, %c : tensor<1024xf32>
```

### 编译器 Pass 示例（Shared Memory Promotion）

检测到 `%a` 在后续被多次使用 → 自动插入：
```llvm
%smem_a = tt.alloc_tensor : tensor<1024xf32, #shared>
tt.copy %a -> %smem_a
... (后续用 %smem_a 替代 %a)
```

生成的 CUDA 代码会有 `__shared__ float smem_a[1024];`。

---

## 限制（什么时候还得用 CUDA）

| 场景 | Triton 能做吗 | 备注 |
|---|:---:|---|
| 标准 GEMM / Softmax | ✅ | 甚至更快（autotune） |
| Flash Attention | ✅ | Flash-2 就是 Triton |
| Warp-level reduce (shuffle) | ⚠️ | 能做但不如 CUDA 灵活 |
| Dynamic parallelism | ❌ | Triton 不支持 kernel 里再起 kernel |
| 极致优化（榨干最后 5%） | ❌ | 手写 CUDA + 汇编 |

**原则**：90% 场景用 Triton，10% 极端场景用 CUDA。

---

## 扩展：Triton 生态

### PyTorch 2.0 的 `torch.compile`
```python
@torch.compile
def fused_op(x, y):
    return (x + y).softmax(dim=-1)

# PyTorch 2.0 会：
# 1. Trace 计算图
# 2. 用 Triton 生成 fused kernel（一个 kernel 算完 add+softmax）
# 3. 比逐个 op 调 cuDNN 快 2-3×
```

底层就是 Triton codegen。

### vLLM 的自定义 kernel
vLLM 的 `paged_attention` 最初是手写 CUDA，后来部分迁移到 Triton（开发效率高）。

### Flash Attention 的演进
- Flash-1: 纯手写 CUDA（1000+ 行）
- Flash-2: 部分 Triton（核心循环还是 CUDA，但外围用 Triton）
- Flash-3: 针对 H100 的手写优化 + Triton

**趋势**：先用 Triton 快速验证，再对热点路径手写 CUDA 榨最后几个百分点。

---

## 参考资料

- **论文**: [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
- **官方 repo**: [openai/triton](https://github.com/openai/triton)
- **教程**: [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)（10+ 个实战例子）
- **配套课程**: [Lesson 06 — Triton 入门](../../lessons/06-triton-intro.md)
- **对比**: [notes/cuda/triton-under-the-hood.md](../../notes/cuda/triton-under-the-hood.md)（Triton vs CUDA 概念映射）

---

*算子线 B（主力工具）的理论基础。理解了这篇，你就知道 Triton 为什么是"GPU 编程的未来"。*
