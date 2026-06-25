# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

**Authors**: Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai (Google)  
**Venue**: EMNLP 2023 | **arxiv**: [2305.13245](https://arxiv.org/abs/2305.13245)  
**优先级**: P0 | **状态**: ✅ 精读

> GQA 将注意力头分组共享 KV，在保持接近 MHA 质量的同时将 KV Cache 减至 G/H，并提出从 MHA checkpoint 仅用 5% 训练量就能转换的 uptrain 方法。

---

## 背景：MHA vs MQA vs GQA

### 定义

**MHA (Multi-Head Attention)** — Vaswani et al. 2017：每个 Q head 有独立的 K/V head，**H 个 KV heads**。

**MQA (Multi-Query Attention)** — Shazeer 2019：所有 Q heads 共享**同一组** K/V，**1 个 KV head**。

**GQA (Grouped Query Attention)**：将 H 个 Q heads 分为 G 组，每组共享一个 KV head，**G 个 KV heads**。

```
MHA: Q_i @ K_i.T    (每个 Q head 有独立 KV)
GQA: Q_i @ K_{i//（H/G)}.T  (每组 H/G 个 Q heads 共享一个 KV)
MQA: Q_i @ K_0.T    (所有 Q heads 共享同一个 KV)
```

- G = H：退化为 MHA
- G = 1：退化为 MQA
- 实践中常用 **G = H/8**（如 32 Q heads → 4 KV heads；64 Q heads → 8 KV heads）

### 对比表

| 机制 | KV Heads | KV Cache 相对 MHA | 质量 | 推理速度 |
|------|:--------:|:------------------:|------|---------|
| MHA | H | 100% | 最高 | 最慢 |
| GQA (G=H/8) | H/8 | **12.5%** | 接近 MHA | 接近 MQA |
| MQA | 1 | 1.6% | 下降明显 | 最快 |

---

## 核心思想

**问题**：MQA 质量比 MHA 差，且无法从已有 MHA checkpoint 低成本转换（大量算力浪费）。

**洞察**：
1. KV cache 是推理内存瓶颈，G 个 KV heads 已足够捕捉注意力多样性，H 个存在冗余
2. Q heads 和 FFN 可完全保留，只需调整 KV projection
3. 可以从 MHA checkpoint **uptrain** 而非从头训练

---

## Uptrain 方法（从 MHA checkpoint 转换）

### Step 1：KV Head 压缩（Mean Pooling）

将同组 KV heads 的权重做 mean pooling：

```python
# 原始 MHA: W_K shape [H, d_model, d_k]
# 目标 GQA: W_K shape [G, d_model, d_k]，G = H//8

groups = H // G  # 每组包含多少个原 KV head
for g in range(G):
    W_K_new[g] = W_K[g*groups : (g+1)*groups].mean(dim=0)  # mean pooling
    W_V_new[g] = W_V[g*groups : (g+1)*groups].mean(dim=0)

# W_Q 完全保留，不变
```

论文对比了三种初始化策略，mean pooling 效果最好（保留了每个 head 的信息）。

### Step 2：继续预训练（Uptrain）

在原始预训练数据上继续训练约 **5% 的原始 token 量**（如 T5-Large: 原始50B tokens → uptrain 约 2.5B tokens）：
- 使用相同学习率调度，从当前 checkpoint 继续
- batch size 不变
- 无需修改其他架构，只有 KV projection 层变小

**为什么只需 5%**：Q weights 和 FFN 完全保留，只有 KV projection（占总参数 ~4%）需要适配，fine-tuning 收敛快。

---

## 性能数据

### 质量对比（T5-Large，SuperGLUE）

| 模型 | SuperGLUE |
|------|:---------:|
| MHA (baseline) | 89.6 |
| MQA from scratch | 87.8 (-1.8) |
| MQA uptrained | 88.4 (-1.2) |
| **GQA uptrained (G=H/8)** | **89.1 (-0.5)** |

GQA 质量损失仅 0.5，远优于 MQA。

### KV Cache 内存节省（LLaMA 2 70B 实例）

```
LLaMA 2 70B: H=64, G=8, d_k=128, 80 layers, FP16
每 token KV = 2 × 8 × 128 × 2 × 80 = 320 KB/token

如用 MHA (G=64): 2.56 MB/token
batch=64, seq=4096: 2.56 MB × 64 × 4096 = 671 GB → 不可能
batch=64, seq=4096 with GQA: 320 KB × 64 × 4096 ≈ 84 GB → 可行（2× A100 80GB）
```

---

## 实际使用（哪些模型用了 GQA）

| 模型 | Q Heads | KV Heads (G) | 比例 |
|------|:-------:|:------------:|:----:|
| LLaMA 2 7B | 32 | 32 | MHA |
| **LLaMA 2 70B** | 64 | **8** | 1/8 |
| **LLaMA 3 8B** | 32 | **8** | 1/4 |
| **LLaMA 3 70B** | 64 | **8** | 1/8 |
| **LLaMA 3.1 405B** | 128 | **8** | 1/16 |
| **Mistral 7B** | 32 | **8** | 1/4 |
| **Qwen2 72B** | 64 | **8** | 1/8 |
| DeepSeek-V2 | 128 | — | MLA（不同机制）|

**规律：G=8 是最常见选择，大模型（70B+）几乎全用 GQA。**

---

## KV Cache 和推理系统的影响

### KV Cache 内存公式

```
单 token 单层 KV = 2 × G × d_k × sizeof(dtype) bytes
```

### 与 PagedAttention (vLLM) 的关系

GQA 减少每个 page 的内存占用 → 同等显存容纳更多并发请求 → **提升吞吐量**。

实际效果：GQA + PagedAttention 组合在 A100 80GB 上可支持 **8× 以上** 的并发请求（对比 MHA without PagedAttention）。

### 与 Flash Attention 的交互

FlashAttention-2/3 原生支持 GQA（`num_heads_k < num_heads_q`）。在 CUDA kernel 层，每个 CTA 处理一个 Q head group，共享同一个 K/V head 的 SRAM tile，**避免重复从 HBM 加载 K/V**。

---

## 实现细节（Kernel 层面）

### PyTorch 实现（分组计算）

```python
# Q: [B, H, S, d_k]  K/V: [B, G, S, d_k]
Q_grouped = Q.view(B, G, H//G, S, d_k)  # [B, G, H/G, S, d_k]
K_grouped = K.unsqueeze(2)              # [B, G, 1, S, d_k]
V_grouped = V.unsqueeze(2)

scores = torch.einsum('bghsd,bgjsd->bghsj', Q_grouped, K_grouped)
# K/V 只加载 G 次，不是 H 次
```

### Triton Kernel 实现要点

GQA 对 Triton attention kernel 的**唯一核心改动**：

```python
head_idx = tl.program_id(0)             # Q head index [0, H)
kv_head_idx = head_idx // (H // G)      # 映射到 KV head [0, G)
                                         # 这一行是 GQA 的关键

k_ptr = K_ptr + kv_head_idx * kv_head_stride  # 基于 kv_head_idx
v_ptr = V_ptr + kv_head_idx * kv_head_stride
```

### Flash Attention 原生支持

```python
from flash_attn import flash_attn_func

# q: [B, S, H, d_k],  k/v: [B, S, G, d_k]
# Flash Attention 自动处理 head group broadcast
out = flash_attn_func(q, k, v, causal=True)
```

---

## 与我何干

**理论线**：理解 GQA 是理解为什么 70B 模型可以在 2× A100 上部署的关键。

**B4 Triton GQA**（算子线）：在 Triton attention kernel 里加一行 `kv_head_idx = head_idx // (H // G)`，是 Triton 实战的好题目。

**C2 vLLM 推理系统**：vLLM 的 KV cache memory 估算和 block size 设计都依赖 GQA 的 KV size。

**面试必考**：

**Q: GQA 的 KV cache 节省怎么算？**  
A: 节省比例 = G/H。LLaMA 2 70B: 8/64 = 12.5%，KV cache 只有 MHA 的 1/8。

**Q: 从 MHA checkpoint 如何转换？**  
A: Mean pool 同组 KV heads 权重，uptrain 5% token 量，Q heads 和 FFN 不变。

**Q: 为什么 G=8 是甜点？**  
A: G 太小（MQA）质量差，G 太大（MHA）KV cache 大，G=8 在质量和内存之间取得平衡。

## 参考

- **论文**: [GQA](https://arxiv.org/abs/2305.13245)
- **Flash Attention 2 GQA 支持**: [flash-attention-2.md](flash-attention-2.md) §3.1.1
- **vLLM KV Cache**: [paged-attention.md](../inference/paged-attention.md)
- **后续 MLA (DeepSeek-V2)**: [notes/algorithms/mla-deepseek.md](../../notes/algorithms/mla-deepseek.md)
