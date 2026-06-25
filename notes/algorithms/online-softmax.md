# Online Softmax（单趟增量 Softmax）

> GPU 优化算法类 · Flash Attention 的心脏 · 给 A4/A5 铺路

---

## 解决了什么问题

朴素 softmax 需要 3 趟遍历：
```
Pass 1: max_val = max(x)           // 数值稳定
Pass 2: sum_exp = Σ exp(x - max)   // 分母
Pass 3: y = exp(x - max) / sum_exp // 归一化
```
每趟都要从 HBM 读一次输入（N 个元素），**3N 次访存**。能不能**一趟算完**？

## 核心思路

维护 **running max (m)** 和 **running sum (l)**，每读一个新块就增量更新：

```
初始: m_old = -∞, l_old = 0

读到新块 x_new:
  m_new = max(m_old, max(x_new))           // 更新全局 max
  correction = exp(m_old - m_new)          // 旧块的 exp 要修正
  l_new = l_old * correction + Σ exp(x_new - m_new)
  
最终: y[i] = exp(x[i] - m_final) / l_final
```

**关键洞察**：当发现更大的 max 时，之前累积的 `l_old` 要乘一个修正因子 `exp(m_old - m_new)`，因为之前算的是 `exp(x - m_old)`，现在要改成 `exp(x - m_new)`。

## 数学推导（为什么等价）

标准 softmax:
```
y[i] = exp(x[i] - m) / Σ exp(x[j] - m)
```

分块计算（块 A, B）:
```
m = max(m_A, m_B)
分子: exp(x[i] - m) = exp(x[i] - m_A) · exp(m_A - m)
分母: Σ_A exp(x - m) + Σ_B exp(x - m)
    = [Σ_A exp(x - m_A)] · exp(m_A - m) + [Σ_B exp(x - m_B)] · exp(m_B - m)
    = l_A · exp(m_A - m) + l_B · exp(m_B - m)
```
每个块的贡献都要乘一个修正因子，这就是 online 更新公式的来源。

## 性能对比

| 方法 | HBM 读次数 | 适用场景 |
|------|:---:|------|
| 3-pass naive | 3N | 数据能一次性放进 shared memory |
| **online** | **N** | 数据分块处理（Flash Attn、长序列） |

**Bandwidth 节省**: 3× → 1×，理论提速 **3倍**（实际 ~2-2.5× 因为计算复杂度略增）。

## 在 Ascend 的对应

和 L1 Buffer 的 pipe 操作一样——你不可能把整个 tensor 搬进 L1，只能分块流式处理。`Pipe` 的 `IterateNext` 就是增量更新，online softmax 是同样的思想在 reduce 操作上的体现。

## 与我何干（面试 + 实战）

**A4 Softmax (Lesson 04)**: 你会写 naive 3-pass → 改成 online 1-pass，亲手验证提速。

**A5 Flash Attention (Lesson 05)**: Flash Attn 的核心就是 online softmax——Q/K/V 分块处理时，每个新的 K tile 来了就增量更新 attention 的 running max/sum，避免存完整的 N×N attention 矩阵到 HBM。你读 `flash_attn.cu` 时满屏都是 `m_new`、`l_new`、`correction`，就是这套公式在跑。

**[面试]** 高频题：
- "Flash Attention 为什么能省显存？" → online softmax 不需要存中间矩阵
- "online softmax 的数值稳定性？" → running max 保证 `exp(x - m)` 不会溢出
- "能用 warp shuffle 优化吗？" → 可以，warp 内先 shuffle 算 local max/sum，再跨 warp 合并（A4 会做）

## 参考

- 数学推导：[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)（Milakov 2018，2 页 note，Flash Attn 论文引用的）
- 实战代码：A4 你自己写的 + [reference/cuda/softmax/softmax.cu](../../reference/cuda/softmax/softmax.cu) `softmax_online`
- Flash Attn 论文：[papers/attention/flash-attention.md](../../papers/attention/flash-attention.md)，Algorithm 1 就是 online softmax

---

*下一条建议学：[parallel reduce](parallel-reduce.md)（为 A4 的 warp reduce 铺路）*
