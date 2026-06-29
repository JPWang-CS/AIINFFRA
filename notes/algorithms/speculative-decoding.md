# Speculative Decoding（投机解码）

> 推理系统技术类 · 用小草稿模型加速大模型的 decoding

---

## 解决了什么问题

LLM decoding 是**逐 token 串行**的——每次只生成一个 token，然后将其加入上下文再生成下一个。这个过程是 **memory-bandwidth bound**：每次生成都要从 HBM 加载全部模型权重，计算量却只有矩阵-向量乘（极低的 arithmetic intensity）。

**结果**：A100 80GB 跑 LLaMA-2-70B（FP16），batch=1 时吞吐只有 ~20 tokens/s，而理论算力（312 TFLOPS）几乎空转，实际 GPU 利用率 <5%。

**核心问题**：能否在一次 forward pass（即一次全量模型调用）中生成多个 token？

---

## 核心思路

### 基本版本：Draft-Verify Loop

```
1. 用小模型（draft model）快速自回归生成 k 个 token
   draft_tokens = small_model.generate(context, k=5)  # 超快，~5× 批量 forward

2. 将 [context + draft_tokens] 送入大模型（target model）并行验证
   target_logits = large_model.forward(context + draft_tokens)  # 一次 forward
   # 注意：forward 是并行的（不是串行），因为 context + draft 作为完整序列输入

3. 逐 token 检验（rejection sampling）：
   for i in range(k):
       p = target_model.prob(draft_tokens[i])
       q = draft_model.prob(draft_tokens[i])
       if random() < p/q:          # 接受
           accepted_tokens.append(draft_tokens[i])
       else:                        # 拒绝：从 target 分布重采样
           accepted_tokens.append(sample(max(0, p - q)))
           break  # 后续 token 都丢弃

4. 净结果：一次 target forward 接受了 ~2-4 个 token（取决于 acceptance rate）
```

**为什么可行**：大模型 forward 是并行的（Transformer 可以同时计算所有位置），计算量是 O(k) 倍，但只算了一次 KV cache 加载；小模型 forward 很快（~1/10 时间）。

### 加速比分析

设大模型耗时 $T_L$，小模型耗时 $T_S$，接受率 $\alpha$：

$$
\text{理想加速比（当 } T_S \to 0 \text{ 时）} \approx \frac{k \cdot \alpha}{1 - \alpha^{k}} \quad\text{对于 k 个草稿 token}
$$

$$
\text{实际加速比} \approx \frac{1 + \alpha + \alpha^{2} + \dots + \alpha^{k-1}}{1 + k \cdot T_S/T_L} \times \frac{T_L}{T_L}
             = \frac{(1-\alpha^{k})/(1-\alpha)}{1 + k \cdot T_S/T_L}
$$

**典型数据**（Llama 2 70B 为 target，Llama 2 7B 为 draft，k=4）：
- $\alpha \approx 0.75$（代码生成）, $0.55$（通用文本）
- $T_S/T_L \approx 0.1$（7B vs 70B 参数比≈1/10）
- 加速比：~2.5× 代码，~1.8× 通用文本

---

## 关键数据/取舍

| 场景 | 接受率 α | 加速比 | 原因 |
|------|:--------:|:------:|------|
| 代码生成（结构化） | 0.7-0.85 | 2-3× | 代码高度可预测 |
| 数学推理 | 0.65-0.75 | 1.8-2.5× | 推导步骤有规律 |
| 通用文本（聊天） | 0.5-0.65 | 1.5-2× | 多样性高 |
| 创意写作 | 0.3-0.5 | 1.0-1.5× | 大小模型分布差异大 |

**限制条件**：
- **Batch size 小时有效**：大 batch 时 target model 本身已经 compute-bound，加速比接近 1
- **Draft model 质量重要**：同家族小模型效果好（Llama 7B for Llama 70B），不相关模型 α 很低
- **Sampling temperature**：确定性采样（greedy, temperature=0）α ≈ 1，随机采样 α 降低

---

## 进阶变体

### Tree Attention（SpecInfer）

不是单链 draft，而是树状 draft（draft model 每步保留 top-k 个候选，形成 token 树）：

```
            [C]
           /   \
         [A]   [B]
        / \   / \
      [X] [Y] [Z] [W]
```

Target model 一次验证整棵树（用 tree attention mask），接受的最长路径作为输出。  
理论上能更高效地"猜"正确 token，尤其适合 beam search 场景。

### Medusa

不用独立的小模型，而是在大模型顶层加几个轻量 draft head（各预测 +1, +2, +3 步的 token），共享大模型的 hidden state：

```python
# 大模型最后一层 hidden state: h_t
draft_1 = MedusaHead1(h_t)   # 预测 token_{t+1}
draft_2 = MedusaHead2(h_t)   # 预测 token_{t+2}
draft_3 = MedusaHead3(h_t)   # 预测 token_{t+3}
```

优点：不需要独立部署小模型；缺点：需要 fine-tune 大模型加这些 head，部署复杂度增加。

### Self-Speculative（草稿用大模型自身的早期层）

用大模型前几层作为 draft（跳过后几层），再用完整大模型验证。适合单卡场景（不需要额外小模型）。

---

## 在 Ascend 的对应

Speculative decoding 是调度逻辑层面的优化，不依赖特定硬件指令。  
Ascend NPU 上同样可以实现：CANN / MindSpore 已有实验性支持。  
关键 kernel 挑战：**tree attention mask**（稀疏的 causal mask，每个 token 的 visible range 不同）需要 custom attention kernel。

---

## 与我何干

**C 线推理系统**：vLLM 0.3+ 支持 speculative decoding（`--speculative-model`），理解这个技术才能读懂相关代码。

**[面试]**：
- "Speculative decoding 的原理？" → 小模型 draft k 个 token，大模型一次 verify，rejection sampling 保证分布等价
- "什么时候有效？" → batch size 小 + 任务有规律性 + 大小模型同家族
- "加速比多少？" → 代码生成约 2-3×，通用文本约 1.5-2×，大 batch 下无效

## 参考

- 原论文: [Speculative Decoding (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192)
- SpecInfer (Tree attention): [arxiv 2305.09781](https://arxiv.org/abs/2305.09781)
- Medusa: [arxiv 2401.10774](https://arxiv.org/abs/2401.10774)
- vLLM 实现: `vllm/spec_decode/`
