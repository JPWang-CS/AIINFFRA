# 大模型板块：模型结构

> 子板块目标：能对着 HuggingFace config 讲清一个模型的 attention、norm、位置编码、FFN/MoE、KV cache 和推理代价。
> 状态：🚧 草稿，需要逐节消化。

---

## 1. 先建立骨架

绝大多数 LLM 是 Decoder-only Transformer：

```text
输入 token -> embedding
  -> [ RMSNorm -> Attention -> residual ] x N 层
  -> [ RMSNorm -> FFN/MoE -> residual ] x N 层
  -> final norm -> lm_head
```

从 Infra 角度，成本主要来自三处：

| 成本 | 来源 | 典型瓶颈 |
|------|------|---------|
| 计算 | QK^T、PV、FFN | Prefill 时 compute-bound |
| 权重读取 | FFN/attention 权重 | Decode 时 memory-bound |
| KV cache | 每层每 token 的 K/V | 长上下文或高并发时显存爆炸 |

---

## 2. 核心组件

### 2.1 Embedding 与 lm_head

- embedding 把 token 变成 `[seq, hidden_size]`。
- lm_head 把 hidden state 映射回 `vocab_size`。
- 很多模型 embedding 和 lm_head 共享权重，省显存。

### 2.2 Norm

```text
LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
RMSNorm:   y = x / sqrt(mean(x^2) + eps) * gamma
```

主流模型用 RMSNorm：少算 mean，reduce 更简单，数值够用。

### 2.3 位置编码

| 类型 | 思路 | 例子 |
|------|------|------|
| Learned absolute | 学 position embedding | BERT、GPT-2 |
| Sinusoidal | 固定频率三角函数 | 原始 Transformer |
| ALiBi | score 加线性偏置 | BLOOM、部分 MPT |
| RoPE | Q/K 按位置旋转 | LLaMA、Qwen、Mistral、DeepSeek |

RoPE 是当前主流。长上下文通常需要 RoPE 插值、NTK 或 YaRN。

### 2.4 Attention 变体

```text
score = QK^T / sqrt(d)
weight = softmax(score)
out = weight @ V
```

| 变体 | KV head 数 | KV cache | 代表 |
|------|-----------|----------|------|
| MHA | 每个 Q head 一组 | 最大 | GPT-2、BERT |
| MQA | 共享 1 组 | 最小 | PaLM |
| GQA | 分 G 组共享 | 折中 | LLaMA、Qwen、Mistral |
| MLA | 低秩 latent | 最省 | DeepSeek |

### 2.5 FFN 与 MoE

```text
标准 FFN: y = act(x W_up) W_down
SwiGLU:  y = (x W_gate) * silu(x W_up) W_down
MoE:     y = sum router_i * expert_i(x)
```

MoE 激活参数少，但全部 expert 权重都要占显存；跨卡推理有 AllToAll。

### 2.6 SSM / Mamba

```text
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

长序列线性复杂度，但状态容量有限；常见做法是 Attention + SSM 混合。

---

## 3. KV cache 手算

```text
KV_bytes = 2 * num_kv_heads * head_dim * seq_len * num_layers * dtype_bytes
```

例：`seq=4096, layers=32, head_dim=128, fp16`

| 变体 | num_kv_heads | KV |
|------|-------------|-----|
| MHA | 32 | 2 GiB |
| GQA | 8 | 512 MiB |
| MQA | 1 | 64 MiB |
| MLA | 1 latent | 约 256 MiB（示意） |

这解释了为什么长上下文和 Decode 阶段对 KV cache 优化这么敏感。

---

## 4. 模型家族

| 家族 | 关键结构 | 学习重点 |
|------|---------|---------|
| LLaMA | GQA、RoPE、SwiGLU、RMSNorm；4 代有 MoE/多模态 | 最完整的 Dense 样本 |
| Qwen | Dense/MoE 都有，GQA、SwiGLU | 量化、vLLM 部署实验 |
| DeepSeek | MLA、DeepSeekMoE、MTP、FP8 | 当前最重要的结构考点 |
| Mistral/Mixtral | GQA、sliding window、MoE | sliding window 和 MoE |
| GPT/o | 闭源，decoder-only + 推理时思考 | 关注服务形态，不猜内部 |
| Claude | 闭源，Transformer + 对齐 | 长上下文、工具调用 |
| Gemini | 多模态、MoE/TPU 路线 | 多模态、长上下文 |
| Mamba/Jamba | SSM / SSM+Attention | 线性时间、混合架构 |

---

## 5. 用 HF config 解码

重点字段：

| 字段 | 含义 | 影响 |
|------|------|------|
| `hidden_size` | hidden 维度 | 显存、FLOPs |
| `num_hidden_layers` | 层数 | KV、延迟 |
| `num_attention_heads` | Q head | 并行度 |
| `num_key_value_heads` | KV head | KV cache |
| `head_dim` | head 维度 | QK^T 计算 |
| `intermediate_size` | FFN 中间维度 | 权重 |
| `num_experts` | MoE | 权重、路由 |
| `rope_theta` | RoPE 频率 | 长上下文 |
| `vocab_size` | 词表 | embedding/lm_head |

```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
print(cfg.num_attention_heads, cfg.num_key_value_heads, cfg.num_hidden_layers)
```

---

## 6. 学习任务

1. 选一个开源模型，读 config 和模型代码。
2. 手算 KV cache，和 `torch` 或 vLLM 实测显存对比。
3. 写一个简化 GQA 或 MoE router。
4. 用 vLLM 跑 benchmark，记录 TTFT/TPOT。
5. 更新 [模型追踪表](../algorithms/model-tracker.md) 学习状态。

## 7. 关联材料

- [最新模型与结构](../algorithms/latest-model-architectures.md)：更完整的主笔记。
- [模型追踪表](../algorithms/model-tracker.md)：模型家族状态表。
- [MLA（DeepSeek）](../algorithms/mla-deepseek.md)
- [MoE 推理挑战](../algorithms/moe-inference.md)
- [Flash Attention 机制](../algorithms/flash-attention-mechanism.md)
