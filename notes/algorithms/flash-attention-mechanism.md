# Flash Attention 机制详解

> 注意力机制：IO-aware tiling 与 online softmax

---

## 解决了什么问题

标准 self-attention 的两大瓶颈：
1. **显存 $O(N^{2})$**：$N \times N$ 的 attention 矩阵（$QK^{T}$ 和 softmax 后的权重）必须写回 HBM
2. **Bandwidth-bound**：每次 forward 都要从 HBM 读写这个巨大矩阵，HBM 带宽成为限制

序列长度 N=4096 时，FP16 的单头 attention 矩阵约占 32 MiB；N=16384 时约占 512 MiB。矩阵随序列长度平方增长，读写它会形成显著的 HBM 流量；实际耗时还取决于实现、shape、精度和设备，不能由容量示例直接推出固定带宽或加速比。

## 核心思路（3 个技巧组合）

### 1. Tiling（分块）
不要一次性算完整的 $QK^{T}$，而是把 Q/K/V 都切成小块：

$$
\begin{aligned}
Q &: [N, d] \text{ 按行切成 } B_r \times d \text{ 的块 } (B_r \approx 32\text{-}128) \\
K &: [N, d] \text{ 按行切成 } B_c \times d \text{ 的块 } (B_c \approx 32\text{-}128) \\
V &: [N, d] \text{ 同 } K
\end{aligned}
$$

每次只在 SRAM (shared memory / L1) 里处理 **$B_r \times B_c$** 的小 attention tile，算完后立即用它更新输出，**不写回 HBM**。

### 2. Online Softmax（增量更新 O）
因为分块了，softmax 的 max 和 sum 要增量维护（详见 [online-softmax.md](online-softmax.md)）。对一个 Q tile 的每一行，跨 K/V tile 维护运行态 $(m,\ \ell,\ O)$；第 $j$ 个 K/V tile 到来时：

$$
\begin{aligned}
S_j &= Q_{\text{tile}} K_j^{\top} / \sqrt{d} && [B_r, B_c]\ \text{scores} \\
m^{\text{new}} &= \max\bigl(m,\ \mathrm{rowmax}(S_j)\bigr) \\
f &= \exp(m - m^{\text{new}}) && \text{回溯修正因子} \\
P_j &= \exp(S_j - m^{\text{new}}) && \text{未归一化权重} \\
\ell &\leftarrow \ell \cdot f + \mathrm{rowsum}(P_j) \\
O   &\leftarrow O \cdot f + P_j V_j \\
m   &\leftarrow m^{\text{new}}
\end{aligned}
$$

所有块扫完后归一化：$O \leftarrow O / \ell$（只在最后做一次，循环里不归一化）。伪代码（对应上面的公式）：

```
对 Q 的每个块 (Br 行,整块一起向量化算):
  初始: m = full(Br, -inf)              # [Br]    每行 running max
        l = zeros(Br)                   # [Br]    每行 running sum
        O = zeros(Br, d)                # [Br, d] 输出累加器

  遍历 K 的每个块 (Bc 行):
    S = Q_block @ K_block^T / sqrt(d)   # [Br, Bc]  attention scores
    m_new = max(m, max(S, dim=1))       # [Br] ⊕ [Br] → [Br]
    correction = exp(m - m_new)         # [Br]      回溯修正因子
    P = exp(S - m_new[:, None])             # [Br,Bc] - [Br,1]沿Bc广播 → [Br,Bc]  未归一化权重
    l = l * correction + sum(P, dim=1)      # [Br]·[Br] + [Br] → [Br]
    O = O * correction[:, None] + P @ V_block  # [Br,d]·[Br,1]沿d广播 + [Br,Bc]@[Bc,d] → [Br,d]
    m = m_new                           # [Br]

  O /= l[:, None]                        # [Br,d] / [Br,1]，沿特征维广播
```

关键 ①：**O 累积的是未归一化的 `P @ V`**（`P = exp(S - m_new)`），除以 `l` 的归一化**只在最后做一次**——不能在循环里用 `softmax(S)` 提前归一化，否则各块的分母不一致，结果错。
关键 ②：**每来一个 K/V 块就同步修正 m、l、O**（旧的 O 和 l 都乘 correction 拉回同一基准），最终得到与整行一次性算 softmax 完全相同的输出。

### 3. Recomputation（反向时不存 attention）
前向不存 $N \times N$ 的 attention 矩阵（省显存）；需要反向传播时，常见实现从 Q/K/V 和保存的统计量重新计算局部结果。重算是否划算取决于 HBM 流量、片上复用、算术吞吐和反向实现，不能概括为“重算必然更快”。

反向需要保存或重新获得输入、前向输出及归一化统计。统计可以表示为每行的 m、l，也可合并为 log-sum-exp；具体 dtype、布局和其他训练状态取决于实现。下面的容量表仅比较选定张量，不是训练峰值显存。

## 数据对比

| 方法 | 显存示例（N=4096, d=64, FP16） | 延迟 | Seq=16384 |
|------|:---:|:---:|:---:|
| PyTorch naive | 32 MiB (attention) + 1.5 MiB (QKV) | 需按固定环境实测 | 需按固定环境实测 |
| FlashAttention | 1.5 MiB (QKV) + 0.03125 MiB (m,l，统计量按 FP32 估算) | 需按固定环境实测 | 需按固定环境实测 |

**显存复杂度**: 中间 attention 矩阵从 $O(N^{2})$ 降为与 tile 状态和输出相关的线性规模；具体常数取决于实现是否保存额外统计量。

## 伪代码（单头，forward）

```python
# Q, K, V: [N, d]
Br, Bc = 32, 32  # block size
Tr = (N + Br - 1) // Br
Tc = (N + Bc - 1) // Bc

O = torch.zeros_like(Q, dtype=torch.float32)  # 教学参考：FP32 输出与累加

for i in range(Tr):  # 遍历 Q 的块
    Qi = Q[i*Br : (i+1)*Br, :]  # [Br, d]
    Oi = torch.zeros_like(Qi, dtype=torch.float32)
    mi = torch.full((Qi.shape[0],), -float('inf'), device=Q.device)
    li = torch.zeros(Qi.shape[0], device=Q.device)
    
    for j in range(Tc):  # 遍历 K 的块
        Kj = K[j*Bc : (j+1)*Bc, :]  # [Bc, d]
        Vj = V[j*Bc : (j+1)*Bc, :]
        
        S = (Qi.float() @ Kj.float().T) / (d ** 0.5)  # 尾块按实际长度计算
        
        # Online softmax update
        mi_new = torch.maximum(mi, S.max(dim=1).values)
        correction = torch.exp(mi - mi_new)
        li = li * correction + torch.sum(torch.exp(S - mi_new[:, None]), dim=1)
        
        Oi = Oi * correction[:, None] + (torch.exp(S - mi_new[:, None]) @ Vj.float())
        mi = mi_new
    
    O[i*Br : (i+1)*Br, :] = Oi / li[:, None]
```

## Causal Mask（因果注意力优化）

Decoder 的 causal mask 是下三角：`QK^T` 的上三角全是 $-\infty$（未来 token 不能看）。

FlashAttention 可以利用这个结构：第 $i$ 个 Q tile 不必计算确定被 mask 的未来 K tile；但 tile 边界、序列长度和实现的 mask 处理会影响实际节省，不能把计算量或访存一概写成减半。

求职追问可以落到实现边界：显存收益来自不物化完整 score/probability 矩阵和 tile 复用，而不是把 dense attention 的计算复杂度改成线性；重算是否划算要看反向的重算范围与实际内存/计算瓶颈；任意 mask 虽可保持数学定义，但非规则 mask 可能破坏 tile 跳过与访存规律，必须按 shape 和 mask 分布测量。

## 在 Ascend 的对应

它与 Ascend Cube 算子的 tiling 可以在数据分块层面类比：
- **CUDA Flash Attn 的 Br×Bc tile** = Ascend 的 L1 Buffer 分块大小
- **Online 更新** = Ascend `Pipe` 的流式处理（不存完整中间矩阵）
- **Recomputation** = Ascend 也常用（前向省片上内存，反向重算）

区别在于具体算子接口、片上存储层次、同步和矩阵乘指令不同；CUDA 参考实现需要显式表达这些边界，不能只凭 tile 名称推断两种平台的执行路径。

## 与我何干

**A5 Flash Attn 读代码 (Lesson 05)**：你会读 [reference/cuda/flash_attention/flash_attn.cu](../../reference/cuda/flash_attention/flash_attn.cu)，看到满屏的 `m_new`、`l_new`、`correction`、双层 `for` 循环（Q 块、K 块），就是上面伪代码的 CUDA 实现。

**B3 Triton Flash Attn (算子线 B)**：用 Triton 重写，会发现 tiling 和 online softmax 的逻辑简洁很多（Triton 帮你管线程），但核心算法一模一样。

**C2 vLLM PagedAttention**：两者都要处理分块后的 K/V 数据，但 page 是 KV cache 的分配与寻址单位，计算 tile 是 kernel 内部的工作单位；一个 page 不必等于一个 attention tile。实现需要额外处理 page table、非连续物理位置和 tile 内的加载组合。


## 论文 + 代码

- **论文精读**: [FlashAttention 原论文](https://arxiv.org/abs/2205.14135)
- **参考实现**: [reference/cuda/flash_attention/flash_attn.cu](../../reference/cuda/flash_attention/flash_attn.cu) (单头 causal, Br=Bc=32)
- **作者仓库**: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)（多头、backward、Flash-2/3 实现）

## Flash Attention 2 / 3 简述

- **Flash-2**: 改进 work partitioning，减少非矩阵乘开销，并在单头场景跨 thread block 并行；论文报告的速度取决于 GPU、shape、精度和基线，不能简写为无条件的“~2×”。
- **Flash-3**: 针对 H100 的异步 WGMMA + TMA，进一步压榨硬件

核心算法（tiling + online softmax）没变，优化的是 GPU 硬件利用率。

---

*前置：[online-softmax.md](online-softmax.md) · 配套：A5 读代码 [lessons/05-flash-attn-reading.md](../../lessons/05-flash-attn-reading.md)*
