# 最新模型与结构：从基础设施视角读懂大模型架构

> 目标：不是背模型名字，而是能对着 HuggingFace config 读懂一个模型的 attention、norm、位置编码、FFN/MoE、KV cache 和推理代价。
> 定位：理论线“模型架构”主笔记，配套 [MoE 推理挑战](moe-inference.md)、[MLA](mla-deepseek.md)、[Flash Attention 机制](flash-attention-mechanism.md)。
> 状态：生成稿，需要逐节消化后才能算进度。

---

## 目录

1. [为什么 Infra 工程师要学模型结构](#1-为什么-infra-工程师要学模型结构)
2. [Transformer block 骨架](#2-transformer-block-骨架)
3. [组件拆解：Embedding / Norm / 位置编码](#3-组件拆解)
4. [Attention 家族：MHA / MQA / GQA / MLA](#4-attention-家族)
5. [FFN 与 MoE](#5-ffn-与-moe)
6. [SSM / Mamba 与混合架构](#6-ssm--mamba-与混合架构)
7. [现代模型常见“新零件”](#7-现代模型常见新零件)
8. [KV cache 深挖](#8-kv-cache-深挖)
9. [模型家族地图](#9-模型家族地图)
10. [用 HuggingFace config 解码模型](#10-用-huggingface-config-解码模型)
11. [显存与 FLOPs 手算](#11-显存与-flops-手算)
12. [对推理系统的五个影响](#12-对推理系统的五个影响)
13. [怎么学：从 config 到 benchmark](#13-怎么学从-config-到-benchmark)
14. [面试高频题库](#14-面试高频题库)
15. [参考与配套材料](#15-参考与配套材料)
16. [2026 新架构：混合注意力与稀疏注意力](#16-2026-新架构混合注意力与稀疏注意力)

---

## 1. 为什么 Infra 工程师要学模型结构

面试和日常优化里，真正被反复追问的不是“这个模型叫 XX”，而是“这个结构会带来什么计算、显存、带宽代价”。

举例：

- GQA 和 MLA 为什么流行？因为 Decode 阶段是 memory-bound，KV cache 越小，每 token 的 HBM 搬运越少。
- MoE 为什么难服务？因为激活参数少，但全部 expert 权重仍占显存；跨卡路由还会引入 AllToAll 通信。
- Mamba 为什么被讨论？因为 Attention 是 O(seq²)，SSM 是线性扫描，长序列时访存模式完全不同。
- 为什么量化总是围绕 FFN 和 KV cache？因为权重读取和 KV cache 是 Decode 阶段最大的带宽来源。

所以这一篇不按“模型发布会”写，而是按“组件 + 模型家族 + 对推理系统的影响”写。

---

## 2. Transformer block 骨架

绝大多数 LLM 还是 Decoder-only Transformer。一个典型 block 可以画成：

```text
输入 x
  -> RMSNorm / LayerNorm
  -> Attention（Q/K/V；推理时维护 KV cache）
  -> residual: x = x + attn_out
  -> RMSNorm / LayerNorm
  -> FFN 或 MoE（常见 SwiGLU）
  -> residual: x = x + ffn_out
```

整体模型通常由几十到上百个这样的 block 堆叠而成，前面有 token embedding，最后有 lm_head。

从 Infra 角度，这个 block 产生三类核心成本：

| 成本 | 来自哪里 | 典型瓶颈 |
|------|---------|---------|
| 激活计算 | Attention 的 QK^T / PV，FFN 的矩阵乘 | Prefill 时 compute-bound |
| 权重读取 | 每层 FFN / attention 的权重 | Decode 时 memory-bound |
| KV cache | 每层、每 token、每 head 存 K/V | 长上下文或并发高时显存爆炸 |

记牢这个模型，后面所有组件都是往这几个位置“换零件”。

---

## 3. 组件拆解

### 3.1 Token Embedding 与 lm_head

- 输入 token 通过 embedding 变成 `[seq_len, hidden_size]` 的向量。
- 最后用 lm_head 把 hidden state 映射回 vocab 概率。
- 很多模型做 tied embedding：embedding 和 lm_head 共享权重，省显存，但也不是所有模型都这么做。

Infra 影响：

- embedding/lm_head 的矩阵大小约 `vocab_size * hidden_size`，vocab 很大的模型（如多语言、多模态）权重占比不小。
- 推理时要算 `hidden @ lm_head^T`，输出层可能是 memory-bound。

### 3.2 Norm：LayerNorm vs RMSNorm

LayerNorm 会减去均值再除以方差；RMSNorm 只做缩放，不做减均值。

```text
LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
RMSNorm:   y = x / sqrt(mean(x^2) + eps) * gamma
```

为什么主流模型都爱 RMSNorm：

- 少一次 mean 计算，reduce 更简单。
- 经验上训练稳定性够用。
- 算子侧是典型的 row-wise reduce，和 Softmax 的 memory-bound 特征很像。
- RMSNorm 少存/少算一个统计量，对长序列训练的数值开销略低。

Infra 影响：

- Norm 是 memory-bound，优化点是少读几遍 HBM、用 warp reduce、和 residual 融合。
- 手写 kernel 时，LayerNorm 需要算 mean 和 var；RMSNorm 只算 mean(x²)，代码更短。

### 3.3 位置编码：从绝对位置到 RoPE

Transformer 本身不知道 token 顺序，需要位置信息。

位置编码主要分几类：

| 类型 | 思路 | 例子 |
|------|------|------|
| Learned absolute | 学一个 position embedding | BERT、GPT-2 |
| Sinusoidal | 固定频率的三角函数 | 原始 Transformer |
| ALiBi | 在 attention score 上加线性偏置 | BLOOM、部分 MPT |
| RoPE | 把 Q/K 按位置旋转 | LLaMA、Qwen、Mistral、DeepSeek |

RoPE 的思路：把 Q/K 的向量在高维空间里按位置旋转，让两个位置的距离体现在点积上。

```text
score(q, k, pos_q, pos_k) 里，实际计算 q^T R(pos_q - pos_k) k
```

Infra 影响：

- RoPE 只影响 Q/K，不需要额外大矩阵。
- 支持“外推”：训练短上下文，推理长上下文，但要配合插值/NTK 等方法。
- 很多长上下文优化都和 RoPE 的缩放策略有关。
- 实现上 RoPE 是 elementwise 旋转 + 合并，是简单但容易写错的算子。

### 3.4 Attention 家族

数学核心都一样：

```text
score = QK^T / sqrt(d)
weight = softmax(score)
out = weight @ V
```

区别在 K/V 是否被多个 query head 共享：

| 变体 | K/V head 数 | KV cache 趋势 | 代表 |
|------|------------|--------------|------|
| MHA | 每个 query head 一组 K/V | 最大 | 早期 GPT、BERT |
| MQA | 所有 query head 共享一组 K/V | 最小，但质量略降 | PaLM 等 |
| GQA | 若干 query head 共享一组 K/V | 折中 | LLaMA 2/3、Qwen、Mistral |
| MLA | 把 K/V 压成低秩 latent | 比 GQA 更省 | DeepSeek-V2/V3 |

KV cache 大小估算：

```text
KV_bytes = 2 * num_kv_heads * head_dim * seq_len * num_layers * dtype_bytes
```

举例：LLaMA-70B 类模型，GQA 的 KV head 比 MHA 少很多，长上下文下能省几十 GB。

对推理的意义：

- Decode 是逐 token 生成，每个 token 都要读已生成的 KV。
- KV cache 越小，单卡能服务的并发越高，PagedAttention 的 block 管理也越轻松。
- MLA 的 low-rank 设计很好，但解码时还需要做投影展开，属于“省显存、换一点计算”。

### 3.5 FFN 与 SwiGLU

标准 FFN：

```text
FFN(x) = act(x @ W_up) @ W_down
```

SwiGLU 是当前最常用的门控变体：

```text
SwiGLU(x) = (x @ W_gate) * silu(x @ W_up) @ W_down
```

为什么 FFN 重要：

- FFN 通常占模型参数的一半以上。
- 每个 token 都要过 FFN，是 Decode 阶段权重读取的主要来源。
- 量化、稀疏化、MoE 都先拿 FFN 开刀，因为收益最大。

### 3.6 MoE

MoE 把一个大 FFN 拆成多个 expert：

```text
router(x) 选出 top-k expert
输出 = sum(router_prob[i] * expert_i(x))
```

MoE 的 Infra 关键点：

- 激活计算少：每个 token 只走 top-k 个 expert。
- 显存不小：所有 expert 权重都要加载到显存。
- 路由不均衡：有些 expert 忙、有些闲，需要 auxiliary loss 或调度手段。
- 跨卡推理：expert 放在不同 GPU 上时，token 要 AllToAll 到对应卡。

详细内容见 [MoE 推理挑战](moe-inference.md)。

### 3.7 SSM / Mamba

状态空间模型把序列更新写成：

```text
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

优势：

- 推理复杂度与序列长度近似线性，不像 Attention 是 O(seq²)。
- 显存不随上下文二次增长。

代价：

- 状态容量有限，长距离记忆不一定比 Attention 强。
- 当前主流的做法是混合架构，例如部分层用 Attention，部分层用 SSM。
- Mamba 的 selective scan 在 GPU 上不是天然易写，需要专门 kernel。

代表：Mamba / Mamba-2、Jamba（SSM + Attention 混合）、RWKV、xLSTM。

### 3.8 现代模型常见“新零件”

| 概念 | 一句话 | Infra 影响 |
|------|--------|-----------|
| MTP / Multi-Token Prediction | 一次预测多个未来 token | 训练和投机解码相关 |
| MoD / Mixture-of-Depths | 不是每个 token 都经过每一层 | 减少激活计算，但调度和 kernel 更复杂 |
| Sliding Window Attention | 只看附近窗口，限制 attention 范围 | 显存和计算降为 O(seq)，但长距离能力下降 |
| Cross-Attention | query 来自一个模态/序列，K/V 来自另一个 | 多模态、RAG、encoder-decoder 都会用到 |
| KV cache 压缩/量化 | 把 K/V 用更低精度或低秩存 | 省显存，可能掉精度 |
| Speculative Decoding | 小模型先 draft，大模型再 verify | 降低 Decode 延迟，见 [投机解码](speculative-decoding.md) |
| Ring Attention | 长序列时把 attention 分块并流水通信 | 解决单卡放不下完整序列 |

---

## 4. Attention 家族

### 4.1 形状对照

假设 `H` 个 query head，每个 head 维度 `d`，GQA 组数 `G`：

| 变体 | Q shape | K/V head 数 | 每 token KV 参数量 |
|------|---------|-------------|---------------------|
| MHA | `[H, d]` | `H` | `2 * H * d` |
| MQA | `[H, d]` | `1` | `2 * d` |
| GQA | `[H, d]` | `G` | `2 * G * d` |
| MLA | `[H, d]` | latent `d_c` | `2 * d_c + 少量投影` |

### 4.2 KV cache 计算公式

```text
KV_bytes = 2 * num_kv_heads * head_dim * seq_len * num_layers * dtype_bytes
```

例子：`seq=4096, layers=32, head_dim=128, fp16`

| 变体 | num_kv_heads | KV |
|------|-------------|-----|
| MHA | 32 | `2*32*128*4096*32*2 = 2 GiB` |
| GQA(8) | 8 | `2*8*128*4096*32*2 = 512 MiB` |
| MQA | 1 | `2*1*128*4096*32*2 = 64 MiB` |
| MLA(latent=512) | 1 latent | 约 `2*512*4096*32*2 = 256 MiB`（未含展开投影） |

> 这里的 MLA 例子只是示意，实际 DeepSeek 还有额外缓存和投影参数，不要直接拿这个数当论文值。

### 4.3 为什么 GQA/MLA 对 Decode 重要

Decode 每生成一个 token：

- 只需要新 token 的 Q。
- 需要读取全部历史 K/V。
- 因此 KV 越小，每 token HBM 读取越少，吞吐越高。

这也是“KV cache 是并发上限”的原因。

---

## 5. FFN 与 MoE

### 5.1 参数占比

一个 7B 模型里，FFN 通常占大头：

```text
attention 参数 ≈ 4 * hidden_size^2
FFN 参数（SwiGLU）≈ 3 * hidden_size * intermediate_size
```

`intermediate_size` 通常是 `hidden_size` 的 2-4 倍，所以 FFN 更重。

### 5.2 MoE 的推理账本

假设模型有 `E` 个 expert，每个 token 激活 top-k：

| 项目 | 数量 |
|------|------|
| 总权重 | `E * expert_size`，全部要放显存 |
| 每个 token 激活 | `k * expert_size` |
| 跨卡路由 | AllToAll，通信量与 token 数量、expert 分布有关 |
| 负载均衡 | 需要 auxiliary loss 或调度约束 |

面试重点：

- “MoE 参数量大”不等于“推理计算量大”。
- 显存瓶颈仍在权重，计算瓶颈只在激活 expert。
- Expert parallelism 和 Tensor Parallelism 的取舍要看通信拓扑。

### 5.3 路由公式

```text
logits = x @ W_router
probs = softmax(logits)
topk_indices = argmax_k(probs)
output = sum_i probs[topk_i] * expert_i(x)
```

---

## 6. SSM / Mamba 与混合架构

### 6.1 状态方程

```text
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

- `A` 是状态转移矩阵，`B` 是输入投影，`C` 是输出投影。
- 推理时每个 token 只更新 hidden state，不保存完整 KV cache。
- 长序列复杂度线性，但状态容量有限。

### 6.2 Mamba 的关键点

- S4 是线性时不变系统；Mamba 的关键是 selective scan，让 `A/B/C` 依赖输入。
- Mamba-2 把状态方程和 attention 的关系讲得更清楚，提出结构化状态空间对偶。
- 混合架构（如 Jamba）交替使用 Attention 和 SSM，兼顾长距离和表达能力。

### 6.3 Infra 影响

- SSM 在长序列下省显存和带宽，但 kernel 不是普通 GEMM，需要专门 scan 实现。
- 现有 serving 框架对 SSM 的支持比 Attention 少，需要看具体后端。
- 面试不要只说“线性复杂度”，要能讲状态容量和 kernel 复杂度。

---

## 7. 现代模型常见“新零件”的实现影响

这些结构经常一起出现，不能只背名字：

- **MTP / Multi-Token Prediction**：训练时一次预测多个未来 token；推理时可配合 speculative decoding 提高吞吐，代价是训练目标和服务结构更复杂。
- **MoD / Mixture-of-Depths**：router 决定 token 是否跳过某层；省激活计算，但 serving 调度和 kernel 负载更难预测。
- **Sliding Window Attention**：每个 token 只看固定窗口，KV cache 和计算线性化；长距离信息靠多层堆叠或额外全局 token 补偿。
- **Cross-Attention**：query 来自当前序列，K/V 来自上下文/另一个模态；多模态和 RAG 常见，KV cache 管理和 multimodal batching 更复杂。
- **KV cache 压缩/量化**：GQA/MLA 之外的低成本手段，但每层敏感度不同，需要校准和精度测试。
- **Speculative Decoding**：draft 模型生成多个候选，目标模型一次验证；收益依赖接受率，见 [投机解码](speculative-decoding.md)。
- **Ring Attention**：长序列分块后跨设备流水传递 KV，把显存压力转成通信压力。

---

## 8. KV cache 深挖

### 8.1 为什么 KV cache 是瓶颈

- Prefill 阶段一次算完所有 token，适合并行、compute-bound。
- Decode 阶段逐 token 生成，每次都要把全部历史 K/V 读一遍，memory-bound。
- KV cache 决定单卡并发、长上下文能力和 prefix cache 的收益。

### 8.2 PagedAttention 与 block table

- 传统 KV cache 是连续分配，容易碎片化。
- PagedAttention 把 KV 切成固定 block，用 block table 管理虚拟到物理映射。
- 支持 prefix sharing 和 copy-on-write，让多轮对话/RAG 复用前缀。

详细见 [PagedAttention 论文笔记](../../papers/inference/paged-attention.md)。

### 8.3 KV cache 优化方向

| 方向 | 思路 | 代价 |
|------|------|------|
| GQA/MQA | 减少 KV head | 表达能力略降 |
| MLA | 低秩压缩 | 增加投影计算 |
| 量化 | INT8/FP8 存 KV | 精度损失，需校准 |
| 驱逐/压缩 | 丢弃旧 token、摘要历史 | 信息丢失 |
| prefix cache | 复用相同前缀 | 管理复杂度 |

---

## 9. 模型家族地图

> 截止 2026-08，闭源模型内部细节以官方技术报告为准。下面的表格重点列“公开可验证的组件”和“推理影响”，不猜未公开实现。

| 家族 | 公开度 | 关键结构 | 对 Infra 的启示 |
|------|--------|---------|----------------|
| LLaMA | 开源权重 | Dense Transformer；GQA、RoPE、SwiGLU、RMSNorm；新一代开始走 MoE/多模态 | 最常用的学习样本，config 简单，适合手写推理和对比 kernel |
| Qwen | 开源权重 | Dense 和 MoE 路线都有；GQA、SwiGLU、RoPE | 适合练多尺寸模型、量化、vLLM 部署 |
| DeepSeek | 开源权重/论文 | MLA + DeepSeekMoE；V3 用 FP8 训练；R1 用强化学习 | MLA 和 MoE 是当前最重要的两个结构考点 |
| Mistral / Mixtral | 开源权重 | GQA、sliding window attention、MoE | sliding window 和 MoE 是面试常考实现 |
| GPT / o-series | 闭源 | Decoder-only Transformer；o-series 增加推理时思考；公开结构信息有限 | 面试关注训练后 RL、推理时 token 预算、服务 API，不猜内部 |
| Claude | 闭源 | 未见完整结构公开；核心是 Transformer 路线 + 训练后对齐 | 关注长上下文、工具调用、服务形态 |
| Gemini | 闭源为主 | 多模态 Transformer；规模化 MoE/TPU 路线 | 关注多模态输入、长上下文、分布式训练视角 |
| Mamba/Jamba/RWKV | 部分开源 | SSM 或 SSM+Attention 混合 | 关注线性时间推理、长序列、混合架构取舍 |

学习建议：不要平均用力。先精读 LLaMA 和 DeepSeek，因为它们把 GQA、SwiGLU、MLA、MoE 都摊开了；闭源模型只用来做“业界趋势”和面试叙事。

---

## 10. 用 HuggingFace config 解码模型

打开模型 config，重点看这些字段：

| 字段 | 含义 | 影响 |
|------|------|------|
| `hidden_size` | 每层 hidden 维度 | 显存、FLOPs 基础 |
| `num_hidden_layers` | 层数 | KV cache、延迟 |
| `num_attention_heads` | query head 数 | 计算并行度 |
| `num_key_value_heads` | KV head 数 | KV cache 大小 |
| `head_dim` | 每个 head 维度 | QK^T 计算量 |
| `intermediate_size` | FFN 中间维度 | 权重、显存 |
| `num_experts` / `num_experts_per_tok` | MoE 配置 | 权重、路由、通信 |
| `rope_theta` | RoPE 频率 | 长上下文外推 |
| `vocab_size` | 词表大小 | embedding/lm_head 大小 |

最小练习：

```python
# 用 HF 读结构，不训练只分析
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
print(cfg.num_attention_heads, cfg.num_key_value_heads, cfg.num_hidden_layers)
print(cfg.hidden_size, cfg.intermediate_size)
```

然后手算：

```text
seq = 4096, layers = 32, GQA 8 组, head_dim = 128, fp16
KV = 2 * 8 * 128 * 4096 * 32 * 2 bytes = 512 MB
```

这个数能直接解释“为什么长上下文必须做 KV 优化”。

---

## 11. 显存与 FLOPs 手算

### 11.1 推理显存三部分

```text
总显存 ≈ 权重 + KV cache + 激活/临时 buffer
```

### 11.2 Decode 单 token 读取量

```text
每 token HBM 读取 ≈ 全部权重 + 全部 KV
```

所以 Decode 的优化就是两条路：

- 把权重变小：量化、稀疏、MoE 激活稀疏。
- 把 KV 变小：GQA/MLA、KV 量化、prefix cache。

### 11.3 Prefill FLOPs 估算

```text
Attention FLOPs ≈ 4 * seq^2 * head_dim * layers（粗略）
FFN FLOPs ≈ 2 * token * hidden * intermediate * layers * 3（SwiGLU 粗略）
```

面试只要能说清“Attention 随 seq²，FFN 随 token 数线性”就够了。

---

## 12. 对推理系统的五个影响

### 12.1 KV cache 决定并发上限

```text
单请求 KV = 每层 KV head 数 * head_dim * seq_len * 2 * layer 数 * dtype_bytes
```

GQA/MLA 能显著提高单卡并发，也因此影响 PagedAttention、prefix cache、PD 分离的收益。

### 12.2 MoE 决定权重显存和通信

MoE 模型“总参数大，激活参数小”。服务时：

- 单卡放不下所有 expert，通常要 tensor parallel 或 expert parallel。
- 跨卡路由有 AllToAll，延迟取决于通信拓扑。
- 量化 MoE 时，router 精度要保，expert 可以更激进。

### 12.3 Attention 变体决定 Decode 带宽

Decode 阶段每个 token 主要瓶颈是：

```text
读 KV cache + 读权重 > 计算量
```

所以 MHA -> GQA -> MLA 的演进本质是“减少每 token 必须读的 KV 字节数”。

### 12.4 长序列结构决定是否能线性扩展

标准 Attention 的 prefill 计算和 KV 显存都随 seq² 增长。要支持百万级上下文，需要：

- Flash Attention / Ring Attention
- 分块 prefill / chunked prefill
- 线性注意力或 SSM 结构

### 12.5 位置编码和归一化决定算子形状

RoPE 引入 Q/K 的旋转操作，RMSNorm 引入 row-wise reduce。两者都会落到具体 kernel 优化，也解释了为什么 Triton/CUDA 里 softmax、reduce、elementwise 算子这么重要。

---

## 13. 怎么学：从 config 到 benchmark

推荐一条可执行路径：

1. 打开一个开源模型 config，找出 `num_attention_heads`、`num_key_value_heads`、`num_hidden_layers`、`hidden_size`、`intermediate_size`、`num_experts`。
2. 读模型代码，定位 attention、norm、FFN/MoE 的调用链。
3. 手算三层账：KV cache 大小、激活显存、单 token Decode 的 HBM 读取量。
4. 用一个小模型跑 forward，对比 PyTorch / vLLM 的输出和显存。
5. 对其中一个组件写简化实现：GQA 或 MoE 路由或 RoPE。
6. 用 vLLM benchmark 记录 TTFT / TPOT / 并发。

---

## 14. 面试高频题库

### 14.1 GQA 为什么比 MHA 省显存？

“KV cache 按 KV head 数增长。GQA 让多个 query head 共享一组 K/V，所以 KV cache 大致从 H 降到 H/G，Decode 阶段每 token 读取的 KV 字节也按比例减少。”

### 14.2 MLA 和 GQA 的区别？

“GQA 仍按 token 存 K/V，只是减少 head 数；MLA 把 K/V 压到低维 latent，再在计算时投影回来。省的是 KV cache 存储，代价是多一次低秩投影。”

### 14.3 MoE 为什么推理难？

“MoE 激活参数少、计算省，但全部 expert 权重都要在显存里；token 路由到不同 expert 后，跨卡场景要 AllToAll，且负载不均衡会拖慢整体。”

### 14.4 Mamba 能替代 Attention 吗？

“在长序列和低内存上有优势，但状态容量有限，不能简单认为所有场景都赢；当前更多是 Attention + SSM 混合。”

### 14.5 RoPE 为什么能外推？

“RoPE 把位置信息编码成旋转角度，相对位置差可以写成旋转矩阵差；直接外推会掉精度，所以要配合插值、NTK 或 YaRN。”

### 14.6 为什么 Decode 是 memory-bound？

“Decode 每次只算一个新 token，但必须读全部权重和全部历史 KV；计算量小，HBM 读取量大。”

### 14.7 为什么 MoE 权重量化比 Dense 更敏感？

“router 决定 token 去哪个 expert，router 错了整个输出就偏了；expert 内部可以更激进量化，router 需要保精度。”

### 14.8 长上下文为什么难？

“Attention 显存和计算随 seq²，KV cache 也随 seq 线性增长；需要 Flash Attention、Ring Attention、chunked prefill、KV 优化一起配合。”

---

## 15. 参考与配套材料

- [MoE 推理挑战](moe-inference.md)
- [MLA（DeepSeek）](mla-deepseek.md)
- [Flash Attention 机制](flash-attention-mechanism.md)
- [剩余理论主题速览](remaining-theory-primer.md)
- [模型追踪表](model-tracker.md)
- AIInfraGuide 的 Transformer 与 GPU 章节，见 [roadmap/ai-infra-curriculum.md](../../roadmap/ai-infra-curriculum.md)

---

## 16. 2026 新架构：混合注意力与稀疏注意力

2026 年四个可学的生产案例，目标都是同一个：**让长上下文变便宜**。

### Qwen3.5：GDN 线性注意力混合（省 KV + 近线性长序列）

- 60 层 = 15 组 ×（3 GDN + 1 full attention）；512+1 experts、每 token 激活 10 个；397B 总参 / 17B 激活
- GDN 层没有传统 KV cache，只有固定大小的 recurrent state + conv state；full attention 层保留精确召回
- 推理影响：长上下文成本接近线性，但 recurrent state 对 prefill 批量不友好（需要 chunked / parallel scan）
- 详细：[GDN（Qwen3.5）](gdn-linear-attention.md)

### DeepSeek-V3.2：MLA + DSA（压 KV + 只算 top-k）

- ~685B 总参 / 37B 激活；MLA 压缩 KV 存储，DSA 用 indexer 只选 K=2048 个 key 做完整 attention
- 复杂度 O(L²) → O(L·K)；推理栈（vLLM / TRT-LLM / SGLang / TileRT）都有 sparse MLA / fp8_ds_mla / MTP-3 支持
- 详细：[DSA](dsa-sparse-attention.md)

### GLM-5：DSA 架构复用

- 78 层、256 experts、8 active、激活约 44B；代码层面复用 DeepSeek 的 DSA 实现
- 说明稀疏注意力已成为"标准件"，模型之间互相复用实现
- 详细：[DSA](dsa-sparse-attention.md)

### DeepSeek-V4：CSA + HCA（MLA 骨架 + 压缩稀疏）

- 发布线：2026-04-24 预览开源（V4-Pro ~1.6T 总参 / ~49B 激活；V4-Flash 284B / ~13B 激活；原生 1M 上下文）；2026-07-31 V4-Flash 正式；2026-08-13 V4-Pro-0813 正式（架构不变，后训练大幅提升）
- 注意力 = **CSA + HCA 混合**，保留 MLA 低秩 latent 骨架：
  - **CSA（压缩稀疏注意力）**：每 m=4 个 token 用重叠窗口（2m）+ joint softmax 压成 1 个 compressed entry；Lightning Indexer 打分选 top-k（Pro k=1024 / Flash k=512）；核心 attention 是 shared-KV 多 query，所有 head 共用同一组压缩 KV
  - **HCA（压缩稠密注意力）**：m'=128、无 indexer、无 overlap，对全部压缩 entry 做 dense attention，兜底召回
  - 每层另留最近 128 个未压缩 token 的滑窗分支（`sliding_window=128`）
- 层排布：V4-Pro 61 层 + MTP，前 2 层纯 HCA，之后 CSA/HCA 交替（`compress_ratios=[128,128,4,128,4,...,4,0]`）；V4-Flash 同模式 43 层
- 1M 上下文账（对比 V3.2）：V4-Pro prefill FLOPs ≈ 27%、KV cache ≈ 10%；V4-Flash ≈ 10% / 7%；对比 BF16 GQA-8 基线 KV 仅约 2%
- 同代配套：partial RoPE（只旋最后 64 维）、attention sink（每 head 可学习 logit）、grouped output projection（组内 bottleneck 1024）、mHC 流形约束超连接（4 条残差流 + 双随机 B 矩阵）、Muon 优化器（替代 AdamW 大矩阵部分）、MoE 权重 MXFP4（E2M1）QAT + 非专家 FP8 + KV 混合精度
- 推理侧：MegaMoE 波次调度（1.5–1.73×，已进 DeepGEMM）、TileLang/DeepGEMM 替代手写 CUDA/cuBLAS、磁盘 KV + SWA 三档策略
- 注意：V4 config 与 V3.2 不兼容（无 `kv_lora_rank`，改用 `compress_ratios`），不要把 V3.2 手算公式直接套 V4
- 详细：[DeepSeek-V4（CSA + HCA）](deepseek-v4.md) · 对照基准：[DeepSeek-V3.2 / DSA](dsa-sparse-attention.md)

### 注意力实现侧同步更新

- FA4 + FlexAttention：可编程注意力（score_mod + block 稀疏），Blackwell 上默认后端推进中 → [fa4-flexattention.md](fa4-flexattention.md)
- SageAttention3：Q/K 量化到 FP4（microscaling），带宽降 4x → [attention-2026-sage3-kascade.md](attention-2026-sage3-kascade.md)
- Kascade：anchor layer 算 exact top-k、跨层复用索引，免训练 → [attention-2026-sage3-kascade.md](attention-2026-sage3-kascade.md)