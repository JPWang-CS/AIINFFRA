# Lesson 06 — Triton 入门：写第一个 Kernel

> 主题：写出第一个 Triton kernel（Vector Add → MatMul），对比 CUDA
> 前置：完成 [Lesson 05](05-flash-attn-reading.md)，理解 tiling 和 online softmax
> 状态：⏳ 待做（见 [PATH.md](../PATH.md)）

📚 **本课重点知识库**：
- [Triton 语法速查](../notes/triton/triton-cheatsheet.md) — 本课主力参考
- [Triton 底层 CUDA 对照](../notes/cuda/triton-under-the-hood.md) — Triton 代码对应什么 CUDA
- [Triton vs CUDA 对比](../notes/triton/triton-vs-cuda.md) — 编程模型差异

🎯 **这是分水岭**：从这里开始，Triton 成为主力优化工具（见 [PATH.md](../PATH.md) 权重）。

---

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

## ✅ 本课检验清单

- [ ] 写完了 Triton Vector Add，在 GPU（或 CPU 模拟）上跑通
- [ ] 写完了 Triton MatMul，能解释 `tl.dot` 在 A100 上用什么硬件加速
- [ ] 能说出 Triton 相比手写 CUDA 的 3 个主要简化点
- [ ] 理解 `tl.program_id` / `tl.arange` / `tl.load` 对应的 CUDA 概念

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
