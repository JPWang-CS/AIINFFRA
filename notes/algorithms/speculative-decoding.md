# Speculative Decoding（投机解码）

> 推理系统技术类 · 用草稿模型减少目标模型的串行解码轮次

---

## 解决了什么问题

LLM decoding 通常按 token 递推：每一步的输出决定下一步输入，因此目标模型需要多次串行调用。小 batch 时，权重和 KV 的读取可能使每步算术强度较低；但具体瓶颈仍取决于 batch、缓存驻留、融合和目标模型实现。

**核心问题**：能否让一个较小的草稿模型先提出一段候选，再让目标模型并行验证，从而用一次目标前向提交多个已验证 token？

---

## 核心思路

### 基本版本：Draft-Verify Loop

```
1. 用小模型（draft model）快速自回归生成 k 个候选 token
   draft_tokens = small_model.generate(context, k=5)

2. 将 [context + draft_tokens] 送入大模型（target model）并行验证
   target_logits = large_model.forward(context + draft_tokens)  # 一次块前向
   # 验证 draft[i] 使用尚未包含 draft[i] 的前缀处的 next-token logits

3. 逐 token 检验（rejection sampling）：
   for i in range(k):
       p_vec = target_model.distribution_at(i)  # 目标在该位置的完整词表分布
       q_vec = draft_model.distribution_at(i)
       y = draft_tokens[i]
       if random() < min(1, p_vec[y] / q_vec[y]):  # 按目标/草稿概率接受
           accepted_tokens.append(draft_tokens[i])
       else:                        # 拒绝：从目标分布的残差重采样
           residual = normalize(max_vector(0, p_vec - q_vec))
           accepted_tokens.append(sample(residual))
           break  # 后续 token 都丢弃

4. 净结果：提交通过的草稿前缀，以及失败位置采样出的一个纠正 token；失败位置后的草稿状态回滚。若 k 个候选全通过，再按算法规则处理 bonus token。
```

**为什么可行**：因果 Transformer 可以在一次块前向中并行计算候选位置的 logits；目标模型仍需读取候选前缀对应的状态，但把多次串行提交改成一次验证。接受后提交通过的草稿前缀，并在首个失败位置提交目标分布采样出的纠正 token；失败位置后的草稿状态回滚。草稿、验证和提交成本都属于同一轮开销。

### 加速比分析

设每个候选位置在前缀存活条件下的接受概率为 $p_i$，草稿长度为 $k$；$T_D(k)$、$T_T(k)$、$T_C(k)$ 分别表示本轮草稿生成、目标验证、提交/回滚及状态整理的耗时。草稿接受数的期望为：

$$
E[L_{\mathrm{acc}}]=\sum_{i=1}^{k}\prod_{j=1}^{i}p_j.
$$

在每轮验证都会提交一个纠正或 bonus token、且没有 EOS/长度截断时，目标上下文的期望前进量是 $1+E[L_{\mathrm{acc}}]$；遇到 EOS 或预算边界时应按实际提交规则截断。

$$
\text{token\ throughput\ proxy}=\frac{1+E[L_{\mathrm{acc}}]}{T_D(k)+T_T(k)+T_C(k)}.
$$

这个表达式只是选择 k 的成本模型；端到端速度还受调度、batch 形状、KV 临时空间和请求混合影响。

---

## 关键数据/取舍

| 场景特征 | 影响接受概率的因素 | 需要观察的结果 |
|------|:--------:|:------:|
| 结构化输出 | 草稿与目标的分布是否匹配 | 接受长度、验证时间、质量 |
| 高随机性采样 | temperature/top-p 改变目标分布 | 接受率、残差采样成本 |
| 大 batch 服务 | 目标模型可能已有更高并行度 | 请求级 TTFT/TPOT 与吞吐 |

草稿模型质量、采样温度、目标 batch 和验证形状共同决定收益。greedy 验证可以按 token ID 比较；随机采样必须使用目标分布与草稿分布的接受/残差规则，不能把 greedy 的接受率直接套用。

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

Target model 一次验证整棵树（用 tree attention mask），再按目标分布的验证规则选择可提交路径。最长路径只有在特定 greedy/候选策略下才可能成为输出；树结构本身不保证 exact sampling 或分布保持。

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

Speculative decoding 主要是调度与采样算法，不绑定某一种硬件指令；迁移到其他设备时仍需确认目标框架是否提供草稿/目标模型协同、树 mask、KV 回滚和随机采样实现。关键 kernel 挑战是 **tree attention mask**（每个 token 的可见范围不同）及其临时 KV 布局，具体支持情况应以目标版本源码和文档为准。

---

## 与我何干

**推理系统代码阅读**：具体 vLLM 或其他引擎的 speculative decoding 参数、KV 回滚和调度 API 随版本变化，应以目标 checkout 和官方文档为准。

## 求职追问与示范回答

以下是教学示例，不代表个人实测：
- “怎样说明投机解码保持目标分布？”——给出逐位置的完整目标分布 `p_vec` 与草稿分布 `q_vec`，对候选索引 y 以 `min(1,p_vec[y]/q_vec[y])` 接受；拒绝时从归一化的 `max(0,p_vec-q_vec)` 向量采样。若只比较 token ID，回答范围应限定为 greedy 验证。
- “怎样选择草稿长度 k？”——对候选长度测量草稿、目标验证、提交/回滚成本，并按前缀条件计算期望接受长度；再放回真实 batch 与请求混合，比较 TTFT、TPOT、吞吐和显存。
- “为什么接受率提高仍可能变慢？”——接受率只影响分子；更长候选也会增加验证矩阵、临时 KV、调度和提交成本，甚至把目标 batch 推出高效形状。

## 参考

- 原论文: [Speculative Decoding (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192)
- SpecInfer (Tree attention): [arxiv 2305.09781](https://arxiv.org/abs/2305.09781)
- Medusa: [arxiv 2401.10774](https://arxiv.org/abs/2401.10774)
- vLLM 实现: `vllm/spec_decode/`
