# Lesson 05 — 读懂 Flash Attention 的 CUDA 实现

> 主题：读懂 Flash Attention 的 CUDA 代码（不手写），认全里面的 CUDA 概念
> 前置：完成 [Lesson 04](04-softmax.md)，理解 shared memory tiling + online softmax
> 状态：⏳ 待做（见 [PATH.md](../PATH.md)）

📚 **本课重点知识库**：
- [Triton 底层 CUDA 对照](../notes/cuda/triton-under-the-hood.md) — Triton 代码对应什么 CUDA
- [内存层级详解](../notes/cuda/memory-model.md) — 回顾 tiling 原理
- [Flash Attention 论文笔记](../papers/attention/flash-attention.md)

🎯 **这对理解 Triton 有什么用**：
读懂了 CUDA 实现，你就理解了 Triton 的 `tl.load` → `tl.dot` → online softmax → `tl.store` 流水线在做什么。

---

---
	
## Part 0：这个算子在模型里干嘛？

Flash Attention **不是新算子**——它和 naive attention 算的东西**一模一样**，改变的是**怎么算**：

```
同样的数学公式：
  O = softmax(QK^T / √d) × V

不同的执行策略：
  Naive Attention:   算完整 N×N 矩阵 → 存 HBM → 读回来 softmax → 写 HBM → 读回来 × V
                      ↑ O(N²) 显存，大量 HBM 读写
  
  Flash Attention:   把 Q 切成小块 → 每块循环遍历所有 K/V 块
                     → 用 online softmax 增量累加 O
                     → 不存中间 N×N 矩阵
                     ↑ O(N) 显存，HBM 读写大幅减少
```

**最终结果一样**（数值误差 < 1e-5），但显存从 $O(N^{2})$ 降到 $O(N)$，速度提升 2-10×。

**什么模型用**：所有现代 LLM 的默认 attention 实现——LLaMA 2/3、GPT-4、DeepSeek-V2/V3、Mistral、Qwen、Claude。PyTorch 的 `torch.nn.functional.scaled_dot_product_attention` 内部自动走 Flash Attention（如果满足 dtype/causal 条件）。

## Part 1：为什么读 Flash Attention

Flash Attention 是 Triton 生态的标志性算子。它融合了 tiling、online softmax、shared memory、causal mask——是 CUDA 优化技术的集大成者，也是面试的超级高频题。

你不需要手写它（Triton 帮你写），但**读懂了 CUDA 实现，你就理解了 Triton 的 `tl.load` → `tl.dot` → online softmax → `tl.store` 流水线在做什么**。

---

## Part 2：核心思想（3 句话）

```
Standard Attention:
$$
\begin{aligned}
S &= \frac{QK^{T}}{\sqrt{d}} &\quad&\to O(N^{2})\text{ 显存（存整个 S 矩阵）} \\
P &= \text{softmax}(S) \\
O &= P \times V
\end{aligned}
$$

Flash Attention:
  把 Q 按行分块（$Q_{\text{tile}}$），K/V 按列分块（$K_{\text{tile}}/V_{\text{tile}}$）
  对每个 $Q_{\text{tile}}$，循环遍历所有 $K_{\text{tile}}/V_{\text{tile}}$：
    1. 加载 $Q_{\text{tile}}, K_{\text{tile}}, V_{\text{tile}}$ 到 shared memory
    2. 用 online softmax 增量计算 attention
    3. 不存 S 矩阵 → $O(N)$ 显存
```

---

## Part 3：对照阅读

打开仓库里的 Flash Attn 参考代码，和下面注释一起看：

→ [reference/cuda/flash_attention/flash_attn.cu](../reference/cuda/flash_attention/flash_attn.cu)

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
   $\text{score} = Q_{\text{row}} \cdot K_{\text{row}} \;/\; \sqrt{d}$   ← 计算注意力分数
   $m_{\text{new}} = \max(m, \text{score})$        ← online softmax: 更新 max
   $p = \exp(\text{score} - m_{\text{new}})$       ← 对应当前要加上的项
   $\text{acc} \mathrel{*}= \exp(m - m_{\text{new}})$          ← 重新缩放旧累加器
   $\text{acc} \mathrel{+}= p \times V_{\text{row}}$           ← 加上当前项
   $l = l \times \text{scale} + p$            ← 重新缩放旧归一化因子
   $m = m_{\text{new}}$
```

**这个 kernel 的 CUDA 概念清单**（看你能认出几个）：
- shared memory allocation（39-43 行）
- `__syncthreads()`（55、101、132 行）
- thread 合作加载 tile（46-54 行、83-100 行）
- register 累加器（75 行，`acc[128]` 在寄存器里）
- online softmax（120-129 行）

> 如果能认出所有这些并且能解释作用 → CUDA B 级目标达成 ✓

---

## Part 4：这些在 Triton 里怎么写

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
        # ... online softmax（和 CUDA 逻辑完全相同，只是用 triton 向量化）
        # ... 累加

    tl.store(O + q_start * d + ..., acc / l)
```

**关键差异**：
- 没有显式 `__shared__` → Triton 自动分配
- 没有显式 `__syncthreads()` → Triton 自动插入
- 没有手动 thread 合作 → `tl.load` 自动多 thread 合作
- `tl.dot` 自动用 Tensor Core（如果硬件和 dtype 支持）

---

## ✅ 本课检验清单

- [ ] 读完了 Flash Attention CUDA 代码，能标注出每个 `__syncthreads` 的作用
- [ ] 能用自己的话解释 online softmax 的更新公式
- [ ] 理解了 Q 分块（BR）和 K/V 分块（BC）的 tiling 策略
- [ ] 能认出代码里的 shared memory / register 累加器 / online softmax 三个部分

---

## 知识库索引

| 想深入理解 | 去看 |
|-----------|------|
| Flash Attention 论文笔记 | [../papers/attention/flash-attention.md](../papers/attention/flash-attention.md) |
| Triton → CUDA 底层实现 | [triton-under-the-hood.md](../notes/cuda/triton-under-the-hood.md) |
| Shared memory tiling 回顾 | [memory-model.md](../notes/cuda/memory-model.md) |
| Flash Attn 参考实现（CUDA） | [reference/cuda/flash_attention/flash_attn.cu](../reference/cuda/flash_attention/flash_attn.cu) |
| Flash Attn 参考实现（Triton） | [reference/triton/flash_attention/flash_attn.py](../reference/triton/flash_attention/flash_attn.py) |
| 下一步：自己写 Triton kernel | [Lesson 06](06-triton-intro.md) |

---

*Lesson 05 · Flash Attention 阅读 · 源自原 week-04（前半）*
