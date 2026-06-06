# Week 4 — 读 Flash Attn + Triton 入门

> 目标：能读懂 Flash Attention 的 CUDA 代码，写出第一个 Triton kernel
> 时间：5-7 小时
> 前置：完成 Week 3，理解 shared memory tiling 和 warp reduce

📚 **本周重点知识库**：
- [Triton 语法速查](../triton-kernels/notes/triton-cheatsheet.md) — 本周主力参考
- [Triton 底层 CUDA 对照](../cuda-kernels/notes/triton-under-the-hood.md) — Triton 代码对应什么 CUDA
- [内存层级详解](../cuda-kernels/notes/memory-model.md) — 回顾 tiling 原理

---

## Day 1：Flash Attention 代码阅读（2h 阅读，不动手写）

### 1.1 为什么读 Flash Attention

Flash Attention 是 Triton 生态的标志性算子。它融合了 tiling、online softmax、shared memory、causal mask——是 CUDA 优化技术的集大成者，也是面试的超级高频题。

你不需要手写它（Triton 帮你写），但**读懂了 CUDA 实现，你就理解了 Triton 的 `tl.load` → `tl.dot` → online softmax → `tl.store` 流水线在做什么**。

### 1.2 核心思想（3 句话）

```
Standard Attention:
  S = QK^T / √d           → O(N²) 显存（存整个 S 矩阵）
  P = softmax(S)
  O = P × V

Flash Attention:
  把 Q 按行分块（Q_tile），K/V 按列分块（K_tile/V_tile）
  对每个 Q_tile，循环遍历所有 K_tile/V_tile：
    1. 加载 Q_tile, K_tile, V_tile 到 shared memory
    2. 用 online softmax 增量计算 attention
    3. 不存 S 矩阵 → O(N) 显存
```

### 1.3 对照阅读

打开仓库里的 Flash Attn 参考代码，和下面注释一起看：

→ [../cuda-kernels/flash_attention/flash_attn.cu](../cuda-kernels/flash_attention/flash_attn.cu)

**阅读路线**（按这个顺序理解）：

```
1. 参数理解（28-29 行）:
   Q: N×d, K: N×d, V: N×d, O: N×d
   N = seq_len, d = head_dim
   causal: 是否 causal mask

2. Tiling 策略（23-24 行）:
   BR = 32: Q 每次处理 32 行
   BC = 32: K/V 每次处理 32 行
   → Q 切成 ⌈N/BR⌉ 个块，K/V 切成 ⌈N/BC⌉ 个块

3. 外层循环（79 行）:
   for kv_start in 0..N step BC:  ← 遍历 K/V 块
       每个 Q 块 × 所有 K/V 块

4. 内层（106-131 行）:
   score = Q_row · K_row / √d   ← 计算注意力分数
   m_new = max(m, score)        ← online softmax: 更新 max
   p = exp(score - m_new)       ← 对应当前要加上的项
   acc *= exp(m - m_new)        ← 重新缩放旧累加器
   acc += p * V_row             ← 加上当前项
   l = l * scale + p            ← 重新缩放旧归一化因子
   m = m_new
```

**这个 kernel 的 CUDA 概念清单**（看你能认出几个）：
- shared memory allocation（39-43 行）
- `__syncthreads()`（55、101、132 行）
- thread 合作加载 tile（46-54 行、83-100 行）
- register 累加器（75 行，`acc[128]` 在寄存器里）
- online softmax（120-129 行）

> 如果能认出所有这些并且能解释作用 → CUDA B 级目标达成 ✓

### 1.4 这些在 Triton 里怎么写

读完了 CUDA 版本，对比 Triton 版（伪代码）：

```python
@triton.jit
def flash_attn(Q, K, V, O, N, d, BLOCK_Q, BLOCK_KV):
    pid_q = tl.program_id(0)
    q_start = pid_q * BLOCK_Q
    
    # 加载 Q block → triton 自动分配 shared memory
    q = tl.load(Q + q_start * d + ...)  # BLOCK_Q × d
    
    m = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, d), dtype=tl.float32)
    
    for kv_start in range(0, N, BLOCK_KV):
        k = tl.load(K + kv_start * d + ...)  # BLOCK_KV × d
        v = tl.load(V + kv_start * d + ...)
        
        scores = tl.dot(q, tl.trans(k)) * (1.0 / tl.sqrt(d))
        # ... online softmax（和 CUDA 逻辑完全相同，只是用 traion 向量化）
        # ... 累加
    
    tl.store(O + q_start * d + ..., acc / l)
```

**关键差异**：
- 没有显式 `__shared__` → Triton 自动分配
- 没有显式 `__syncthreads()` → Triton 自动插入
- 没有手动 thread 合作 → `tl.load` 自动多 thread 合作
- `tl.dot` 自动用 Tensor Core（如果硬件和 dtype 支持）

---

## Day 2-3：Triton 入门——写第一个 Kernel（3h 动手）

### 2.1 环境

```bash
pip install triton
# 需要 NVIDIA GPU（T4 或以上）
# 如果没 GPU，用 TRITON_INTERPRET=1 在 CPU 上模拟运行（调试）
```

### 2.2 第一个 Kernel：Vector Add in Triton

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

参考 → [Triton 语法速查](../triton-kernels/notes/triton-cheatsheet.md) §1

### 2.3 第二个 Kernel：Matrix Multiply in Triton

用 `tl.dot` 做矩阵乘。这是你未来最常用的 Triton 操作。

参考 → [Triton 语法速查](../triton-kernels/notes/triton-cheatsheet.md) §7

```python
@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # TODO: 对照 cheatsheet 的 GEMM pattern 自己写
    pass
```

### 2.4 对比 CUDA vs Triton

写完 Triton GEMM 后，对比你在 Week 2-3 写的 CUDA GEMM：

| 维度 | CUDA | Triton |
|------|------|--------|
| 代码行数 | ~50 行（tiled） | ~25 行 |
| shared memory | 手动分配 + `__syncthreads` | `tl.load` 自动处理 |
| Tensor Core | 需要手动 `mma.sync` / `wmma` | `tl.dot` 自动选择 |
| bank conflict | 手动 padding | 编译器尽量自动避免 |
| 调试难度 | ncu 逐 kernel 看 | 打印 IR 或 CPU 模拟 |

---

## ✅ Week 4 检验清单

- [ ] 读完了 Flash Attention CUDA 代码，能标注出每个 `__syncthreads` 的作用
- [ ] 能用自己的话解释 online softmax 的更新公式
- [ ] 理解了 Q 分块（BR）和 K/V 分块（BC）的 tiling 策略
- [ ] 写完了 Triton Vector Add，在 GPU（或 CPU 模拟）上跑通
- [ ] 写完了 Triton MatMul，能解释 `tl.dot` 在 A100 上用什么硬件加速
- [ ] 能说出 Triton 相比手写 CUDA 的 3 个主要简化点

## 知识库索引

| 想深入理解 | 去看 |
|-----------|------|
| Triton 所有 API | [triton-cheatsheet.md](../triton-kernels/notes/triton-cheatsheet.md) |
| Triton → CUDA 底层实现 | [triton-under-the-hood.md](../cuda-kernels/notes/triton-under-the-hood.md) |
| Flash Attention 论文笔记 | [../papers/attention/flash-attention.md](../papers/attention/flash-attention.md) |
| Shared memory tiling 回顾 | [memory-model.md](../cuda-kernels/notes/memory-model.md) |

---

*Week 4 · Flash Attn + Triton 入门*
