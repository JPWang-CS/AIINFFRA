# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

**Authors**: Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai (Google)  
**Venue**: EMNLP 2023 | **arxiv**: [2305.13245](https://arxiv.org/abs/2305.13245)  
> GQA 将注意力头分组共享 KV，在保持接近 MHA 质量的同时把 KV Cache 的元素数按 KV-head 数量从 H 降到 G；论文还提出从 MHA checkpoint 使用约 5% 原预训练计算量进行 uptraining 的方法。

---

## 背景：MHA vs MQA vs GQA

### 定义

**MHA (Multi-Head Attention)** — Vaswani et al. 2017：每个 Q head 有独立的 K/V head，**H 个 KV heads**。

**MQA (Multi-Query Attention)** — Shazeer 2019：所有 Q heads 共享**同一组** K/V，**1 个 KV head**。

**GQA (Grouped Query Attention)**：将 H 个 Q heads 分为 G 组，每组共享一个 KV head，**G 个 KV heads**；通常要求 H 能被 G 整除。

```
MHA: Q_i @ K_i.T    (每个 Q head 有独立 KV)
GQA: Q_i @ K_{i//(H/G)}.T  (每组 H/G 个 Q heads 共享一个 KV)
MQA: Q_i @ K_0.T    (所有 Q heads 共享同一个 KV)
```

- G = H：退化为 MHA
- G = 1：退化为 MQA
- 某些模型采用 **G = H/8**（如 32 Q heads → 4 KV heads；64 Q heads → 8 KV heads），这不是 GQA 的固定取值。

### 对比表

| 机制 | KV Heads | KV Cache 相对 MHA | 质量 | 推理速度 |
|------|:--------:|:------------------:|------|---------|
| MHA | H | 100% | 原模型作为对照，不保证对所有任务最优 | 作为对照 |
| GQA (G=H/8) | H/8 | **12.5%** | 取决于模型与 uptraining | 取决于实现与 shape |
| MQA | 1 | $1/H$（即 $100/H$%） | 取决于模型与训练 | 取决于实现与 shape |

---

## 核心思想

将所有 Query heads 的 K/V 合并为一组，可能损失原模型的表达能力。GQA 在完全独立与全部共享之间增加可调的分组数，研究缓存容量、推理速度与质量如何权衡。

已有 MHA checkpoint 可以保留 Query 与 FFN 权重，将同组 KV 投影取平均作为初始化，再继续训练以适应新结构。是否能维持原模型质量仍需评估，不能把 head 共享理解为无损删除冗余参数。

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

论文对比了 mean pooling、选择单个 head 和随机初始化等策略；在论文实验设置中，mean pooling 的质量表现最好，但该排序属于其模型、数据与训练配置，不是所有 checkpoint 的保证。

### Step 2：继续预训练（Uptrain）

论文的 uptraining 配方在原始预训练数据上继续使用约 **5% 的原始预训练计算量/数据规模**（例如文中 T5 设置使用约 5% token 量）：
- 使用相同学习率调度，从当前 checkpoint 继续
- batch size 不变
- 无需修改其他架构，只有 KV projection 层变小

**为什么可用较小的 uptraining 预算**：Q weights 和 FFN 保留，主要改变 KV projection；但所需预算仍取决于 checkpoint、数据和目标 KV-head 数，不能把 5% 当成普适收敛阈值。

---

## 性能数据

### 质量与计时对比（T5-XXL，生成任务）

原论文 Table 1 比较摘要、翻译与问答任务，不是 SuperGLUE 分类结果。下表摘取同为 T5-XXL 的三项，MQA/GQA 使用论文中 5% 额外预训练配方：

| 模型 | 七项任务的平均得分（论文口径） | 每样本、每 TPUv4 芯片的推理时间 |
|------|:---------:|:---------:|
| MHA-XXL | 47.2 | 1.51 s |
| MQA-XXL | 46.6 | 0.24 s |
| GQA-8-XXL | 47.1 | 0.28 s |

该平均值混合了论文选定任务的指标，只适合同表对照，不能解释成准确率百分比。作者在 8 个 TPU 上测量，并分别调整各模型的并行方案与可容纳 batch；这些结果不是固定 batch 的 GPU kernel benchmark，更不是本仓库实测。来源：[原论文 §3 与 Table 1](https://arxiv.org/html/2305.13245v3#S3)。

### KV Cache 内存节省（LLaMA 2 70B 实例）

```
LLaMA 2 70B: H=64, G=8, d_k=128, 80 layers, FP16
每 token KV = 2 × 8 × 128 × 2 × 80 = 327,680 bytes = 320 KiB/token

如用 MHA (G=64): 2,621,440 bytes = 2.5 MiB/token
batch=64, seq=4096: 2.5 MiB × 64 × 4096 = 640 GiB
batch=64, seq=4096 with GQA: 320 KiB × 64 × 4096 = 80 GiB
```

这是按给定 shape、层数和 FP16 存储计算的容量示例，不包含 allocator 元数据、临时 workspace、权重和其他运行时缓冲区；部署可行性仍需结合设备容量与运行时配置核对。

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

这些模型展示了不同的 KV-head 取值；G 的选择需要在质量、Cache 容量、访存和 kernel 并行度之间权衡，不能从一个模型的配置外推通用“甜点值”。

---

## KV Cache 和推理系统的影响

### KV Cache 内存公式

```
单 token 单层 KV = 2 × G × d_k × sizeof(dtype) bytes
```

### 与 PagedAttention (vLLM) 的关系

GQA 减少每个 page 中 K/V 的元素数量，在 page size、调度和带宽不成为新瓶颈的条件下，可能提高同等显存可容纳的并发量；端到端吞吐仍需实测。

PagedAttention 的 page table、碎片管理和 kernel 调度仍是独立因素，不能由 Cache 元素数直接推出固定并发倍数。

### 与 Flash Attention 的交互

FlashAttention 的相关实现支持 `num_heads_k < num_heads_q` 的 GQA/MQA 配置，并要求 query head 数能被 KV head 数整除。kernel 是否在一个 CTA 内复用 K/V tile、以及能否避免重复 HBM 加载，取决于具体版本、head grouping 和调度策略，不能仅由 API 形状推出。

---

## 实现细节（Kernel 层面）

### PyTorch 实现（分组计算）

```python
# Q: [B, H, S, d_k]  K/V: [B, G, S, d_k]
Q_grouped = Q.view(B, G, H//G, S, d_k)  # [B, G, H/G, S, d_k]
K_grouped = K.unsqueeze(2)              # [B, G, 1, S, d_k]
V_grouped = V.unsqueeze(2)

scores = torch.einsum('bghtd,bgjud->bghtu', Q_grouped, K_grouped)
# 输出 [B, G, H//G, query_len, key_len]；逻辑上每组复用一个 KV head
```

### Triton Kernel 实现要点

GQA 对 Triton attention kernel 的最小头映射改动之一是：

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
# 需要 H % G == 0；具体 head broadcast 与布局由实现处理
out = flash_attn_func(q, k, v, causal=True)
```

实现检查可以围绕三个可计算问题展开：若有 H 个 query heads、G 个 KV heads，单 token 单层 Cache 元素数相对 MHA 为 $G/H$，前提是 head dimension 和 dtype 相同；若把 MHA checkpoint 转为 GQA，需要按组初始化 KV projection 并继续训练，5% 是论文配方而不是正确性条件；若实现 `kv_head_idx = head_idx // (H/G)`，必须先验证 `H % G == 0`，并分别检查 K/V 的 stride、query/key 序列维度和 page layout，否则结果可能形状正确却读取了错误的 head。

---

## 与我何干

**理论线**：理解 GQA 可以把 KV-head 数、Cache 容量和 Decode 访存联系起来；具体部署仍需把权重、激活和运行时缓冲区一起计算。

**B4 Triton GQA**（算子线）：在 Triton attention kernel 里加一行 `kv_head_idx = head_idx // (H // G)`，是 Triton 实战的好题目。

**C2 vLLM 推理系统**：vLLM 的 KV cache memory 估算和 block size 设计都依赖 GQA 的 KV size。


## 参考

- **论文**: [GQA](https://arxiv.org/abs/2305.13245)
- **Flash Attention 2 GQA 支持**: [flash-attention-2.md](../../notes/algorithms/flash-attention-2.md)，参见其中关于 GQA 头分组与工作划分的实现说明
- **vLLM KV Cache**: [paged-attention.md](../inference/paged-attention.md)
- **后续 MLA (DeepSeek-V2)**: [notes/algorithms/mla-deepseek.md](../../notes/algorithms/mla-deepseek.md)
