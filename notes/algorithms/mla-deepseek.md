# MLA（Multi-head Latent Attention）

> 注意力演进类 · DeepSeek-V2/V3 的 KV Cache 压缩技术，比 GQA 更激进

---

## 解决了什么问题

GQA 通过减少 KV heads 数量（G < H）降低 KV cache，但 **d_k（每个 head 的维度）不变**。对于长上下文服务（128K tokens），即使 G=8，KV cache 仍然很大。

**MLA（DeepSeek-V2, 2024）的洞察**：不仅减少 heads，还**压缩 d_k 维度**——将 K/V 投影到低秩的潜在向量（latent vector），存储压缩后的表示，推理时按需解压。

### GQA vs MLA 对比

| 方法 | 存储的 KV | 维度 | KV Cache per token per layer |
|------|----------|------|------------------------------|
| MHA (H=128, d_k=128) | H 个完整 K/V | 128×128 = 16384 | 32 KB |
| GQA (G=8, d_k=128) | G 个完整 K/V | 8×128 = 1024 | 2 KB |
| **MLA (d_c=512)** | **1 个低秩向量** | **512** | **1 KB** |

DeepSeek-V2（H=128，d_k=128，d_c=512）：MLA 比 GQA 再省 **2×**，比 MHA 省 **32×**。

---

## 核心思路

### 传统 KV 投影（MHA/GQA）

```python
# 每个 token: x [d_model] → K, V [H × d_k]
K = x @ W_K  # [d_model, H × d_k]
V = x @ W_V  # [d_model, H × d_k]
# 存储到 KV cache: H × d_k × 2 floats
```

### MLA：低秩压缩投影

```python
# Step 1: 压缩（Down-projection）
c_KV = x @ W_DKV           # [d_model] → [d_c]，d_c ≪ H × d_k
# 存储到 KV cache: d_c floats （远小于 H × d_k）

# Step 2: 推理时解压（Up-projection）
K = c_KV @ W_UK             # [d_c] → [H × d_k]
V = c_KV @ W_UV             # [d_c] → [H × d_k]

# Attention 照常计算
attn = softmax(Q @ K.T / sqrt(d_k)) @ V
```

**核心**：存的不是 K 和 V，而是生成它们的**压缩中间表示** c_KV（维度更小）。推理时 K/V 按需从 c_KV 解压，不用常驻。

### 数学等价性

```
传统: K = x W_K = x W_DKV W_UK
     = (x W_DKV) W_UK = c_KV W_UK

等价于先做低秩分解: W_K ≈ W_DKV × W_UK
                  rank = d_c（远小于 H × d_k）
```

这是标准的低秩矩阵分解（LoRA 的推理版本）。

### Q 也做压缩（对称设计）

```python
# Q 的压缩（减少计算量，不减少存储，因为 Q 不存 KV cache）
c_Q = x @ W_DQ               # [d_model] → [d_c_q]
Q   = c_Q @ W_UQ             # [d_c_q] → [H × d_k]
```

---

## 进一步优化：解耦 RoPE

### 问题

MLA 存储 c_KV 而不是 K，但 Rotary Position Embedding（RoPE）需要在 K 上施加位置相关的旋转，这与"存压缩向量、推理时解压"的流程冲突——因为旋转后的 K 取决于 token 位置，无法从单一的 c_KV 恢复。

### 解法：Decoupled RoPE

将 K 拆分为两部分：
- **带 RoPE 的小 K**（d_k_rope 维）：位置相关，直接存储（小，额外开销可接受）
- **无 RoPE 的大 K**（d_k_nope 维）：通过 c_KV 低秩解压

```python
# KV cache 存储：
cache_c   = c_KV          # [d_c]，低秩向量
cache_k_rope = K_rope     # [d_k_rope]，带 RoPE 的位置相关部分

# 推理时：
K_nope = c_KV @ W_UK_nope  # 从低秩恢复
K = concat(K_nope, K_rope)  # 完整的 K
```

DeepSeek-V2 中 `d_k_rope = 64`（小），`d_k_nope = 128`（大，走低秩），`d_c = 512`。

---

## 关键数据

### DeepSeek-V2（2024）

| 参数 | 值 |
|------|---|
| 总参数 | 236B（MoE，激活 21B） |
| num_heads H | 128 |
| d_k | 128 |
| d_c (KV 压缩维度) | 512 |
| d_k_rope | 64 |
| KV cache per token/layer | 512 + 64 = 576 bytes ≈ **1.1× GQA(G=8)**，但质量接近 MHA |

**KV Cache 节省**（对比 MHA）：128 heads → 512 维 = **32× 节省**。

### DeepSeek-V3（2024）

- 671B 参数（MoE），激活 37B
- 沿用 MLA 设计，d_c=512
- 128K 上下文下 KV cache 只有 ~3.2 GB（vs 等效 MHA 的 ~100 GB）

### 质量对比（DeepSeek-V2 paper，英文推理基准）

| 机制 | MMLU | HumanEval | 相对 MHA |
|------|:----:|:---------:|:--------:|
| MHA（baseline） | 77.8 | 52.4 | 100% |
| MQA | 74.1 | 47.6 | -4% |
| GQA (G=8) | 76.9 | 51.2 | -1% |
| **MLA** | **77.6** | **52.1** | **-0.3%** |

MLA 质量接近 MHA，KV cache 比 GQA 小 2×。

---

## 在 Ascend 的对应

MLA 是权重矩阵分解层面的优化，不依赖特定硬件。  
关键 kernel：**两次矩阵乘**（Down-projection + Up-projection），可以用 Triton/CUDA GEMM 实现，也可以融合为一个 kernel。  
Ascend 上 AscendC 的 `Matmul` 算子同样适用。

---

## 与我何干

**注意力演进理解**：GQA → MLA 是减少 KV cache 的两个层级：GQA 减少 head 数，MLA 减少每 head 的维度（低秩分解）。

**C2/C4 推理系统**：DeepSeek 系列模型的量化推理（AWQ/FP8）需要理解 MLA 的 KV projection 结构。

**理论延伸**：LoRA 训练 = 低秩分解 of weight delta。MLA = 低秩分解 of KV projection。同一数学工具，不同应用场景。

**[面试]**：
- "MLA 和 GQA 的区别？" → GQA 减少 KV heads 数，MLA 减少每个 head 的维度（低秩压缩 d_k），两者可叠加
- "MLA 存什么到 KV cache？" → 低秩压缩向量 c_KV（维度 d_c ≪ H×d_k），推理时按需解压
- "为什么要解耦 RoPE？" → RoPE 是位置相关的，无法从与位置无关的 c_KV 恢复；小的 K_rope 单独存储

## 参考

- DeepSeek-V2: [arxiv 2405.04434](https://arxiv.org/abs/2405.04434)
- DeepSeek-V3: [arxiv 2412.19437](https://arxiv.org/abs/2412.19437)
- LoRA（低秩分解训练）: [arxiv 2106.09685](https://arxiv.org/abs/2106.09685)
- GQA: [papers/attention/gqa.md](../../papers/attention/gqa.md)
