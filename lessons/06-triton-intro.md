# Lesson 06 — Triton 入门：写第一个 Kernel

> 主题：写出第一个 Triton kernel（Vector Add → MatMul），对比 CUDA
> 前置：建议先完成 [Lesson 05](05-flash-attn-reading.md)，理解 tiling 和 online softmax；如果当前主线直接进入 Triton，也可以先写 vec add 再补 A5
> 状态：🚧 当前主线（详细任务见 [roadmap/ai-infra-curriculum.md](../roadmap/ai-infra-curriculum.md)）

📚 **本课重点知识库**：
- [Triton 语法速查](../notes/triton/triton-cheatsheet.md) — 本课主力参考
- [Triton 底层 CUDA 对照](../notes/cuda/triton-under-the-hood.md) — Triton 代码对应什么 CUDA
- [Triton vs CUDA 对比](../notes/triton/triton-vs-cuda.md) — 编程模型差异

🎯 **这是分水岭**：从这里开始，Triton 成为主力优化工具（见 [PATH.md](../PATH.md) 权重）。

---

## Part 0：Triton 写的算子 = 同样的数学，不同的写法

Triton 写的 Vector Add / MatMul / Softmax / Flash Attention **和 CUDA 版算的数学完全一样**，模型里的位置也一样。

**关键差异**：
- **CUDA**：你写 thread-level 逻辑——每个 thread 读哪个元素、怎么同步、怎么 avoid bank conflict
- **Triton**：你写 block-level 逻辑——声明"这个 tile 怎么算"，编译器自动分配线程、插入同步、优化访存

**为什么 Triton 是主力**：同样功能的 GEMM，CUDA 要 ~100 行（手动 tiling/sync），Triton ~30 行。性能接近手写 CUDA（~93-95%），开发时间降 5-10×。

---

## Part 1：环境

```bash
pip install triton
# 需要 NVIDIA GPU（T4 或以上）
# 如果没 GPU，用 TRITON_INTERPRET=1 在 CPU 上模拟运行（调试）
# LeetGPU 也支持在线跑 Triton
```

---

## Part 2：第一个 Kernel — Vector Add in Triton

```python
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 你的第一个 Triton kernel
    # TODO:
    # 1. 获取 program_id
    # 2. 计算偏移量
    # 3. tl.load x 和 y
    # 4. 做加法
    # 5. tl.store 结果
    pass

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=256)
    return output
```

参考 → [Triton 语法速查](../notes/triton/triton-cheatsheet.md) §1

> **CUDA 对照**：`tl.program_id(0)` ≈ `blockIdx.x`，`tl.arange(0, BLOCK_SIZE)` ≈ 一个 block 内的 `threadIdx.x` 全体。Triton 一次操作一整个 block 的 tile，不写单个 thread。

---

## Part 3：第二个 Kernel — Matrix Multiply in Triton

用 `tl.dot` 做矩阵乘。这是你未来最常用的 Triton 操作。

参考 → [Triton 语法速查](../notes/triton/triton-cheatsheet.md) §7

```python
@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # TODO: 对照 cheatsheet 的 GEMM pattern 自己写
    pass
```

写完后对照参考实现 → [reference/triton/matmul/matmul.py](../reference/triton/matmul/matmul.py)

---

## Part 4：对比 CUDA vs Triton

写完 Triton GEMM 后，对比你在 Lesson 02-03 写的 CUDA GEMM：

| 维度 | CUDA | Triton |
|------|------|--------|
| 代码行数 | ~50 行（tiled） | ~25 行 |
| shared memory | 手动分配 + `__syncthreads` | `tl.load` 自动处理 |
| Tensor Core | 需要手动 `mma.sync` / `wmma` | `tl.dot` 自动选择 |
| bank conflict | 手动 padding | 编译器尽量自动避免 |
| 调试难度 | ncu 逐 kernel 看 | 打印 IR 或 CPU 模拟 |

完整对比 → [triton-vs-cuda.md](../notes/triton/triton-vs-cuda.md)

---


## Part 5：怎么一步步写对

### 5.1 Vector Add 的完整心理模型

```text
N = 1000, BLOCK_SIZE = 256
grid = ceil(1000 / 256) = 4 个 block

block 0: 处理 [0, 256)
block 1: 处理 [256, 512)
block 2: 处理 [512, 768)
block 3: 处理 [768, 1000)，但 1000 不是 256 的倍数
         => 最后 24 个位置越界，必须用 mask
```

写代码时的固定顺序：

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

不要先写优化，先保证这三件事正确：
1. 每个元素只被一个 block 处理。
2. 越界位置不参与计算。
3. launch 的 grid 覆盖全部元素。

### 5.2 MatMul：从输出 tile 反推指针

假设输出是 `[M, N]`，每个 block 算一个 `BLOCK_M x BLOCK_N` 的 tile。

```text
pid_m = tl.program_id(0)   # 这个 block 负责哪一组行
pid_n = tl.program_id(1)   # 这个 block 负责哪一组列

A tile 起点 = pid_m * BLOCK_M * K
B tile 起点 = pid_n * BLOCK_N
C tile 起点 = pid_m * BLOCK_M * N + pid_n * BLOCK_N
```

然后循环 `K`：

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs)   # [BLOCK_M, BLOCK_K]
    b = tl.load(b_ptrs)   # [BLOCK_K, BLOCK_N]
    acc += tl.dot(a, b)
    a_ptrs += BLOCK_K
    b_ptrs += BLOCK_K * N
```

调试顺序：
1. 先固定 `BLOCK_K = K`，不做 K 循环，只验证单个 tile 正确。
2. 再做 K 循环。
3. 最后做 autotune。

### 5.3 Softmax：先想 reduce 方向

一维 softmax：

```python
x = tl.load(ptr + offsets, mask=mask, other=-float('inf'))
x_max = tl.max(x, axis=0)
num = tl.exp(x - x_max)
denom = tl.sum(num, axis=0)
y = num / denom
```

二维 softmax 每行独立：

```python
row = tl.arange(0, BLOCK_M)[:, None]
col = tl.arange(0, BLOCK_N)[None, :]
offsets = row * N + col

x = tl.load(x_ptr + offsets)
x_max = tl.max(x, axis=1)          # 每行一个 max
num = tl.exp(x - x_max[:, None])   # 广播到 [BLOCK_M, BLOCK_N]
denom = tl.sum(num, axis=1)
y = num / denom[:, None]
```

关键：`axis=1` 后形状变成 `[BLOCK_M]`，要恢复成 `[BLOCK_M, 1]` 才能广播。

### 5.4 Flash Attention：保持 running state

Triton 版的核心是维护三个 running state：

```python
m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
l = tl.zeros((BLOCK_M,), dtype=tl.float32)
acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

for kv_start in range(0, N, BLOCK_KV):
    k = tl.load(...)   # [BLOCK_KV, HEAD_DIM]
    v = tl.load(...)   # [BLOCK_KV, HEAD_DIM]
    scores = tl.dot(q, tl.trans(k)) * scale  # [BLOCK_M, BLOCK_KV]
    m_new = tl.maximum(m, tl.max(scores, axis=1))
    correction = tl.exp(m - m_new)
    p = tl.exp(scores - m_new[:, None])
    l = l * correction + tl.sum(p, axis=1)
    acc = acc * correction[:, None] + tl.dot(p, v)
    m = m_new
```

不要一上来就写 causal，先不加 mask 跑通，再加 causal。

### 5.5 GQA：先理解 head 映射

GQA 里 query head 数 > key/value head 数：

```text
Q head 0..3  -> 共用 K/V head 0
Q head 4..7  -> 共用 K/V head 1
```

Triton 实现时，先在 host 端把 Q 按 group 重排，或者把 Q head 索引除以 group size 后再映射到 K/V head。

## ✅ 本课检验清单

### 环境与基础
- [ ] `python -c "import triton, torch"` 能跑
- [ ] Triton Vector Add 跑通，和 PyTorch 对齐
- [ ] Triton MatMul 跑通，记录 GFLOPS

### 理解
- [ ] 能解释 `tl.program_id` / `tl.arange` / `tl.load` / `tl.store` 对应 CUDA 什么
- [ ] 能解释 `tl.dot` 在 GPU 上怎么被编译执行
- [ ] 能说出 Triton 相比手写 CUDA 的 3 个主要简化点
- [ ] 能解释 mask 和 other 的作用

### 进阶
- [ ] Triton Fused Softmax 跑通
- [ ] Triton Flash Attention 跑通并对比 PyTorch ref
- [ ] GQA 或 Fused MLP 跑通
- [ ] 能用 autotune 给出至少一组调参结论

---

## 知识库索引

| 想深入理解 | 去看 |
|-----------|------|
| Triton 所有 API | [triton-cheatsheet.md](../notes/triton/triton-cheatsheet.md) |
| Triton → CUDA 底层实现 | [triton-under-the-hood.md](../notes/cuda/triton-under-the-hood.md) |
| Triton vs CUDA 编程模型 | [triton-vs-cuda.md](../notes/triton/triton-vs-cuda.md) |
| Triton MatMul 参考实现 | [reference/triton/matmul/matmul.py](../reference/triton/matmul/matmul.py) |
| 接下来去哪 | [PATH.md](../PATH.md) — Triton 算子 + 推理系统 |

---

*Lesson 06 · Triton 入门 · 源自原 week-04（后半）*
