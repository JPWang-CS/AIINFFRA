# Online Softmax：增量更新，省一趟 HBM 读

> GPU 优化算法类 · Flash Attention 的心脏 · 给 A4/A5 铺路

---

## 解决了什么问题

### 原始公式

$$
\begin{aligned}
\text{Pass 1:}\quad m &= \max(x) \\[4pt]
\text{Pass 2:}\quad S &= \sum_{j} \exp(x_j - m) \\[4pt]
\text{Pass 3:}\quad y_i &= \frac{\exp(x_i - m)}{S}
\end{aligned}
$$

每趟都从 HBM 完整读一遍 x，共 3N 次读 + N 次写。

### 朴素 softmax：3 趟 HBM 读

```
         HBM 读数            HBM 写数
Pass 1:  N 次（读 x 找 max）     0
Pass 2:  N 次（读 x 算 Σexp）    0
Pass 3:  N 次（读 x 算输出）     N 次（写 y）
─────────────────────────────────────
合计:    3N 次读               N 次写
```

每趟都完整读一遍 x，因为 Pass 2 等 Pass 1 的 m、Pass 3 等 Pass 2 的 S。

### Online softmax：2 趟 HBM 读

```
         HBM 读数            HBM 写数
Pass 1:  N 次（读 x，边读边更新 m 和 S）     0
Pass 2:  N 次（重读 x，用 m,S 算 y_i）      N 次（写 y）
────────────────────────────────────────────────
合计:    2N 次读               N 次写
```

**省了什么**：朴素版的 Pass 1 和 Pass 2 是两趟独立的读——第一趟找 max、第二趟求 Σexp。Online 把它们**合并成一趟**——读 x 的同时维护 m 和 S。但输出 y 那趟跑不掉，因为必须先知道最终的 m 和 S。

### Flash Attention：真正的 1 趟

```
         HBM 读数                  HBM 写数
遍历 K/V 块，每块读进来就算 O += P_ij × V_j
O 本身是增量累积的，不需要最后再归一化重读 x
────────────────────────────────────────────────
合计:    1 趟（每块一次进去就出来）
```

Flash Attention 不需要输出独立的 softmax，它直接把 softmax 结果喂给 V 累加到 O——所以 O 的更新不需要等到 m,S 确定。这是比 standalone online softmax 多出来的融合优势。

### 三句话总结

| | 读 x 几趟 | 怎么做到的 |
|---|:---:|---|
| 朴素 3-pass | 3 | 老老实实扫三遍 |
| Online | 2 | Pass1(找max)+Pass2(Σexp) 合并成一趟 |
| Flash Attn | 1 | 连输出那趟也融合掉了——O 增量累积，不需要重读 x |

## 核心思路

维护 **running max ($m$)** 和 **running sum ($S$)**，每读一个新块就增量更新：

**初始状态：** $m_{\text{old}} = -\infty,\; S_{\text{old}} = 0$

**读到新块 $x_{\text{new}}$ 时：**

$$
\begin{aligned}
m_{\text{new}} &= \max(m_{\text{old}}, \max(x_{\text{new}})) \\
\alpha &= \exp(m_{\text{old}} - m_{\text{new}}) &\text{// 修正因子} \\
S_{\text{new}} &= S_{\text{old}} \cdot \alpha + \sum \exp(x_{\text{new}} - m_{\text{new}})
\end{aligned}
$$

**最终输出：**

$$
y_i = \frac{\exp(x_i - m_{\text{final}})}{S_{\text{final}}}
$$

**关键洞察**：当发现更大的 max 时，之前累积的 $S_{\text{old}}$ 要乘一个修正因子 $\exp(m_{\text{old}} - m_{\text{new}})$，因为之前算的是 $\exp(x - m_{\text{old}})$，现在要改成 $\exp(x - m_{\text{new}})$。

---

## 数学推导（为什么等价）

### 标准 Softmax

$$
y_i = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)}, \quad m = \max_j x_j
$$

### 分块计算推导

假设将输入分为两块 A 和 B，设 $m_A = \max A$，$m_B = \max B$。

全局最大值：
$$
m = \max(m_A, m_B)
$$

**分子：** 对于块 A 中的元素 $x_i \in A$

$$
\exp(x_i - m) = \exp(x_i - m_A) \cdot \exp(m_A - m)
$$

**分母：**

$$
\begin{aligned}
\sum_j \exp(x_j - m) &= \sum_{j \in A} \exp(x_j - m) + \sum_{j \in B} \exp(x_j - m) \\
&= \left[\sum_{j \in A} \exp(x_j - m_A)\right] \cdot \exp(m_A - m) + \left[\sum_{j \in B} \exp(x_j - m_B)\right] \cdot \exp(m_B - m) \\
&= S_A \cdot \exp(m_A - m) + S_B \cdot \exp(m_B - m)
\end{aligned}
$$

其中 $S_A = \sum_{j \in A} \exp(x_j - m_A)$，$S_B$ 同理。

**结论**：每个块的贡献都要乘一个修正因子 $\exp(m_{\text{block}} - m)$，这就是 online 更新公式的来源。

> **为什么会有"A、B 两块"这个推导**：就是为了证明——数据分给多个线程各算各的、最后 merge，结果和一次算全部 x 完全等价。merge 公式就是 `m = max(m_A, m_B)`、`S = S_A·e^(m_A-m) + S_B·e^(m_B-m)`。每个线程把自己的那段 x 当成一个独立的小 softmax，算出局部状态，交给 merge。

### 多线程下全局 max 怎么来的：两级归约

每个线程只看到自己那段 x，不知道自己段的 max 是不是全局最大。所以分两步：

**第一步：各算各的局部状态**

```
Thread 0 扫 x[0..255]    → m₀, S₀
Thread 1 扫 x[256..511]  → m₁, S₁
Thread 2 扫 x[512..767]  → m₂, S₂
Thread 3 扫 x[768..1023] → m₃, S₃
```

此时每个线程手里只有自己的 $(m_k, S_k)$，不知道别人的。

**第二步：归约 merge**

所有线程把各自的 $(m, S)$ 交出来，逐级合并：

```
        m₀,S₀    m₁,S₁         m₂,S₂    m₃,S₃
          └── merge ──┘           └── merge ──┘
             ↓                       ↓
        m₀₁, S₀₁                m₂₃, S₂₃
                 └───── merge ─────┘
                        ↓
                  m_global, S_global
```

每次 merge 用的就是分块推导里的公式。最后 `m_global` 广播给所有线程，大家才知道真正的全局 max 是多少。

**归约怎么做**：CUDA 里用 warp shuffle（线程间直接交换值，不经过 shared memory），或 shared memory（线程写到 smem，同步后 thread 0 串行或并行 merge）。规模大到跨 block 时用 atomic 或多 kernel launch。

**注意**：如果线程内部是逐元素 online 更新的，那它的局部 $S_k$ 是**按自己那段的局部 max 归一化的**——这就是为什么 merge 时要乘修正因子 $\exp(m_k - m_\text{global})$。因为 $S_k = \sum \exp(x - m_k)$，要改成 $\sum \exp(x - m_\text{global})$。

---

## 性能对比

| 方法 | HBM 读 | HBM 写 | 省了什么 |
|------|:---:|:---:|------|
| 朴素 3-pass | $3N$ | $N$ | — |
| **Online** | **$2N$** | $N$ | 把找 max 和求 Σexp 合并成一趟 |
| Flash Attn | $N$ | $0$ | Softmax 融合进 O 累加，不单独输出 |

Online 比朴素省 $1N$ 读，实际提速 ~1.5-2×。

---

## 在 Ascend 的对应

和 L1 Buffer 的 pipe 操作一样——你不可能把整个 tensor 搬进 L1，只能分块流式处理。`Pipe` 的 `IterateNext` 就是增量更新，online softmax 是同样的思想在 reduce 操作上的体现。

---

## 与我何干（面试 + 实战）

**A4 Softmax (Lesson 04)**: 你会写 naive 3-pass $\to$ 改成 online 1-pass，亲手验证提速。

**A5 Flash Attention (Lesson 05)**: Flash Attn 的核心就是 online softmax——Q/K/V 分块处理时，每个新的 K tile 来了就增量更新 attention 的 running max/sum，避免存完整的 $N \times N$ attention 矩阵到 HBM。你读 `flash_attn.cu` 时满屏都是 `m_new`、`S_new`、`correction`，就是这套公式在跑。

**[面试]** 高频题：
- "Flash Attention 为什么能省显存？" $\to$ online softmax 不需要存中间矩阵
- "online softmax 的数值稳定性？" $\to$ running max 保证 $\exp(x - m)$ 不会溢出
- "能用 warp shuffle 优化吗？" $\to$ 可以，warp 内先 shuffle 算 local max/sum，再跨 warp 合并（A4 会做）

---

## 参考

- 数学推导：[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)（Milakov 2018，2 页 note，Flash Attn 论文引用的）
- 实战代码：A4 你自己写的 + [reference/cuda/softmax/softmax.cu](../../reference/cuda/softmax/softmax.cu) `softmax_online`
- Flash Attn 论文：[papers/attention/flash-attention.md](../../papers/attention/flash-attention.md)，Algorithm 1 就是 online softmax

---

*下一条建议学：[parallel reduce](parallel-reduce.md)（为 A4 的 warp reduce 铺路）*
