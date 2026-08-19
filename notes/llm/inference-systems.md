# 大模型板块：推理系统

> 子板块目标：理解 LLM serving 从请求到 token 的完整链路，能解释 vLLM 的核心机制。
> 状态：🚧 草稿，需要结合 vLLM 源码和 benchmark 消化。

---

## 1. 推理的两阶段

| 阶段 | 行为 | 瓶颈 | 优化重点 |
|------|------|------|---------|
| Prefill | 一次处理完整 prompt | compute-bound | Flash Attention、chunked prefill |
| Decode | 逐 token 生成 | memory-bound | 权重量化、KV 优化、speculative decoding |

面试口径：

> Prefill 计算量大但可并行，Decode 计算量小但每次都要读全部权重和 KV cache，所以是 memory-bound。

### Prefill / Decode 的 Q 关系

来源相同：

```text
Q_i = x_i @ W_Q，K_i = x_i @ W_K，V_i = x_i @ W_V
```

差别：

| 维度 | Prefill | Decode |
|------|---------|--------|
| Q 公式 | x_i @ W_Q | x_i @ W_Q，相同 |
| Q 数量 | 整段 prompt，一批 | 每次 1 个新 token |
| Q 形状 | [S, d] | [1, d] |
| QK^T | 矩阵乘矩阵 | 向量乘矩阵 |
| 瓶颈 | compute-bound | memory-bound |
| Q 是否缓存 | 不缓存 | 不缓存，现算现用 |

衔接：prefill 一次算完整段 prompt 的 K/V 存入 cache；decode 每步只算新 token 的 Q/K/V，新 Q 查历史 K/V，新 K/V 追加进 cache。

---

## 2. KV cache 与显存

```text
KV_bytes = 2 * num_kv_heads * head_dim * seq_len * num_layers * dtype_bytes
```

KV cache 决定：

- 单卡能承载多少并发。
- 长上下文能否放得下。
- prefix cache 和 PD 分离的收益。

优化方向：

- GQA/MQA：减少 KV head。
- MLA：低秩压缩。
- KV 量化：INT8/FP8。
- prefix cache：复用相同前缀。
- 驱逐/摘要：丢旧 token。

### decode 越长，压力越大

decode 第 t 步：

```text
Q_t = [1, d]
K_cache = [t, d]，V_cache = [t, d]
scores = Q_t @ K_cache^T   # [1, t]
```

所以：

- 每步 attention 范围 O(t)，t 越大越慢。
- KV cache 显存 O(t)。
- 生成 T 个 token 的总 attention 计算量 ≈ 1+2+...+T = O(T^2)。

压力主要来自显存、每步 HBM 读取、单 token 延迟，不是权重。

缓解手段里，GQA/MLA/量化主要省存储和带宽，稀疏/窗口/线性注意力才能减少每步要看的长度。

---

## 3. PagedAttention

传统 KV cache 连续分配容易碎片化。PagedAttention 把 KV 切成固定 block，用 block table 做虚拟到物理映射。

关键点：

- block 大小固定，例如 16 token。
- 每个请求维护 block table。
- 支持 prefix sharing 和 copy-on-write。

```text
逻辑 block:  [0] [1] [2] [3]
block table: [8] [3] [11] [6]
物理 block:  8 -> 3 -> 11 -> 6
```

---

## 4. Continuous Batching

传统静态 batching 要等整批完成，GPU 利用率低。Continuous batching 在 iteration 级调度：

- 请求完成即退出。
- 新请求随时加入。
- Prefill/Decode 可以在同一 iteration 混合。

Chunked prefill 把长 prefill 切成小块，避免长时间霸占 GPU。

---

## 5. Prefix Cache / RadixAttention

- vLLM：hash 精确匹配前缀。
- SGLang RadixAttention：radix tree 最长前缀匹配。
- 共同点：复用重复 KV，减少 prefill。
- 代价：显存管理、refcount、eviction 更复杂。

---

## 6. 量化

推理量化主要目标：减小权重读取和 KV cache 带宽。

| 方法 | 量化对象 | 核心 |
|------|---------|------|
| GPTQ | 权重 | 逐层 Hessian 校准 |
| AWQ | 权重 | 按激活重要性保护 channel |
| SmoothQuant | 权重+激活 | 激活困难迁移到权重 |
| KV cache 量化 | KV | 低精度存 KV |

MoE 量化时，router 精度要保，expert 可以更激进。

---

## 7. Speculative Decoding

```text
小模型 draft -> 大模型 verify -> 接受一致部分
```

收益取决于：

- draft 模型接受率。
- draft 模型额外显存。
- 目标模型一次验证多个 token 的并行度。

---

## 8. PD 分离

Prefill 和 Decode 放在不同实例：

- Prefill 实例处理长 prompt。
- Decode 实例保持低延迟。
- 之间传输 KV cache 或中间状态。

挑战：KV 传输开销、调度、故障恢复。

---

## 9. vLLM 链路

```text
请求 -> Scheduler -> Worker -> ModelRunner -> Attention -> KV Cache -> 返回
  |        |            |           |             |            |
排队    抢占/调度     权重管理    forward     PagedAttention  block 管理
```

学习任务：

1. 读 `vllm/core/scheduler.py`，画出调度循环。
2. 读 `vllm/attention/ops/paged_attn.py`，理解 block table。
3. 跑一次 `vllm bench`，记录 TTFT/TPOT/throughput。
4. 对比 FP16 / INT8 / FP8 的显存和延迟。

## 10. 关联材料

- [vLLM 源码深挖](../../roadmap/vllm.md)
- [剩余理论主题速览](../algorithms/remaining-theory-primer.md)
- [PagedAttention 论文](../../papers/inference/paged-attention.md)
- [投机解码](../algorithms/speculative-decoding.md)
- [PD 分离](../algorithms/pd-disaggregation.md)
