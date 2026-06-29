# Flash Attention 机制详解

> 注意力演进类 · IO-aware tiling + online softmax · A5 读代码前必看

---

## 解决了什么问题

标准 self-attention 的两大瓶颈：
1. **显存 $O(N^{2})$**：$N \times N$ 的 attention 矩阵（$QK^{T}$ 和 softmax 后的权重）必须写回 HBM
2. **Bandwidth-bound**：每次 forward 都要从 HBM 读写这个巨大矩阵，HBM 带宽成为限制

序列长度 N=4096 时，FP16 的 attention 矩阵就要 32MB（单头）；N=16384 时 512MB。A100 的 HBM 带宽只有 1.5TB/s，读写这个矩阵吃掉大量时间。

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
因为分块了，softmax 的 max 和 sum 要增量维护（详见 [online-softmax.md](online-softmax.md)）：

```
对 Q 的每一行（$B_r$ 行）:
  初始: $m = -\infty,\; l = 0,\; O = 0$  (running max, sum, output)
  
  遍历 K 的每个块 ($B_c$ 行):
    $S = Q_{\text{block}} \times K_{\text{block}}^{T}$        ($B_r \times B_c$)
    $m_{\text{new}}, l_{\text{new}} = \text{online\_update}(S, m, l)$
    $\text{correction} = \exp(m - m_{\text{new}})$
    $O = O \times \text{correction} + \text{softmax}(S) \times V_{\text{block}}$
  
  $O \;\mathrel{/}= l_{\text{final}}$
```

关键：**O 是增量累积的**，每来一个 K/V 块就更新一次，最终得到完整输出。
关键：**O 是增量累积的**，每来一个 K/V 块就更新一次，最终得到完整输出。

### 3. Recomputation（反向时不存 attention）
前向不存 $N \times N$ 的 attention 矩阵（省显存），反向传播时从 Q/K/V 重新算一遍。因为**重算比存储+读取更快**（HBM 慢，compute 快）。

只需要额外存：
- `m` 和 `l`（每行 2 个 FP32，总共 2N 个数）
- Q/K/V 本身（本来就要存）

## 数据对比

| 方法 | 显存（N=4096, d=64, FP16） | Seq=4096 延迟 (A100) | Seq=16384 |
|------|:---:|:---:|:---:|
| PyTorch naive | 32 MB (attention) + 1.5 MB (QKV) | 100 ms | OOM |
| Flash Attention | 1.5 MB (QKV) + 0.064 MB (m,l) | **20 ms** | 400 ms |

**提速**: ~5× (短序列) ~ 10×+ (长序列，naive 会 OOM)  
**显存节省**: $O(N^{2}) \to O(N)$

## 伪代码（单头，forward）

```python
# Q, K, V: [N, d]
Br, Bc = 32, 32  # block size
Tr = N // Br
Tc = N // Bc

O = torch.zeros(N, d)      # 输出
m = torch.full((N,), -inf) # running max (每行)
l = torch.zeros(N)         # running sum (每行)

for i in range(Tr):  # 遍历 Q 的块
    Qi = Q[i*Br : (i+1)*Br, :]  # [Br, d]
    Oi = torch.zeros(Br, d)
    mi = torch.full((Br,), -inf)
    li = torch.zeros(Br)
    
    for j in range(Tc):  # 遍历 K 的块
        Kj = K[j*Bc : (j+1)*Bc, :]  # [Bc, d]
        Vj = V[j*Bc : (j+1)*Bc, :]
        
        S = Qi @ Kj.T  # [Br, Bc] attention scores
        
        # Online softmax update
        mi_new = torch.maximum(mi, S.max(dim=1))
        correction = torch.exp(mi - mi_new)
        li = li * correction + torch.sum(torch.exp(S - mi_new[:, None]), dim=1)
        
        Oi = Oi * correction[:, None] + (torch.exp(S - mi_new[:, None]) @ Vj)
        mi = mi_new
    
    O[i*Br : (i+1)*Br, :] = Oi / li[:, None]
```

## Causal Mask（因果注意力优化）

Decoder 的 causal mask 是下三角：`QK^T` 的上三角全是 -inf（未来 token 不能看）。

Flash Attention **自动利用这个结构**：第 i 个 Q 块只需要处理前 i 个 K 块，后面的直接跳过。计算量和访存都省一半。

## 在 Ascend 的对应

和你写 Ascend Cube 算子的 tiling 策略完全一样：
- **CUDA Flash Attn 的 Br×Bc tile** = Ascend 的 L1 Buffer 分块大小
- **Online 更新** = Ascend `Pipe` 的流式处理（不存完整中间矩阵）
- **Recomputation** = Ascend 也常用（前向省片上内存，反向重算）

区别：Ascend 有 `MatMul` 指令直接算 tile，CUDA 要手写循环；但思想完全一致。

## 与我何干

**A5 Flash Attn 读代码 (Lesson 05)**：你会读 [reference/cuda/flash_attention/flash_attn.cu](../../reference/cuda/flash_attention/flash_attn.cu)，看到满屏的 `m_new`、`l_new`、`correction`、双层 `for` 循环（Q 块、K 块），就是上面伪代码的 CUDA 实现。

**B3 Triton Flash Attn (算子线 B)**：用 Triton 重写，会发现 tiling 和 online softmax 的逻辑简洁很多（Triton 帮你管线程），但核心算法一模一样。

**C2 vLLM PagedAttention**：Flash Attn 是 PagedAttention 的基础——PagedAttention 把 KV Cache 切成 block，每个 block 的 attention 计算就是 Flash Attn 的一个 tile。

**[面试]** 必考：
- "Flash Attention 为什么快？" → tiling 避免 HBM 读写 N×N 矩阵 + online softmax
- "为什么 recomputation 反而更快？" → HBM 慢（1.5TB/s），重算在 SRAM 里（19TB/s），带宽差 10×+
- "Flash Attention 显存复杂度？" → O(N)，只存 Q/K/V + m/l
- "能处理任意 mask 吗？" → 可以，但非结构化 mask（如稀疏 mask）加速效果会打折扣，causal 是最优情况

## 论文 + 代码

- **论文精读**: [papers/attention/flash-attention.md](../../papers/attention/flash-attention.md)
- **参考实现**: [reference/cuda/flash_attention/flash_attn.cu](../../reference/cuda/flash_attention/flash_attn.cu) (单头 causal, Br=Bc=32)
- **官方 repo**: [HazyResearch/flash-attention](https://github.com/Dao-AILab/flash-attention)（多头、backward、优化版 Flash-2/3）

## Flash Attention 2 / 3 简述

- **Flash-2**: 改进 work partitioning（每个 block 处理更多 Q，减少跨 block 通信），~2× 提速
- **Flash-3**: 针对 H100 的异步 WGMMA + TMA，进一步压榨硬件

核心算法（tiling + online softmax）没变，优化的是 GPU 硬件利用率。

---

*前置：[online-softmax.md](online-softmax.md) · 配套：A5 读代码 [lessons/05-flash-attn-reading.md](../../lessons/05-flash-attn-reading.md)*
