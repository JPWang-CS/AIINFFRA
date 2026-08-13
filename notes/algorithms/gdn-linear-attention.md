# Gated Delta Network（GDN）与 Qwen3.5 混合注意力

> 模型架构类 · 线性注意力 + 门控 + delta rule · Qwen3.5 是大规模生产案例之一
> 挂靠：主线 B Qwen3.5 第 1-3 步 · 长上下文方案对比

---

## 解决了什么问题

- 标准 attention 计算 O(N²)，KV cache 随序列长度线性增长——长上下文推理很贵
- 早期线性注意力（Linear Transformer 等）用固定状态，复杂度 O(N) 但记忆容量有限，长程召回弱
- GDN 的思路：用"门控 + 按误差更新的记忆"提升线性注意力的容量，同时保持 O(N)；再和 full attention 按 3:1 混合，让 full attention 兜底精确召回

## 核心思路

### 1. 线性注意力的递推形式

把 attention 写成状态递推（和 SSM/Mamba 同族）：

```text
每个 head 维护一个记忆矩阵 S: [d_v, d_k]

对每个 token:
  k, v = 输入 x 的投影
  S ← 门控 × S + v ⊗ k^T     # 写入记忆（外积更新）
  o = S @ q                  # 读出结果
```

关键性质：

- **复杂度 O(N)**：每 token 只做固定量的矩阵运算，不需要和全部历史做点积
- **没有传统 KV cache**：decode 时只需携带固定大小的 S（外加 conv 状态），序列多长显存都不涨

### 2. Gated Delta Network 的改进

普通线性注意力是"无脑累加记忆"，GDN 加了三样东西：

- **门控（gate）**：控制"遗忘旧记忆 / 写入新记忆"的比例，类似 LSTM 的选择性——不重要的 token 少写，重要的多写
- **delta rule**：不是直接加 `v ⊗ k`，而是按"当前记忆对真实 v 的预测误差"更新——只修正记错的部分，记忆容量更大（这是 DeltaNet 那支工作的核心）
- **数值稳定**：Q/K 做 L2 normalize，避免递推中值爆炸；前面接 conv1d + 深度卷积，捕捉局部 token 模式；输出再接 gate 控制读出的信息量

### 3. Qwen3.5 的混合方案

```text
60 层 = 15 组 × (3 × Gated DeltaNet + 1 × Full Attention)

15 层 full attention（标准注意力变体）→ 负责精确召回
45 层 GDN（线性注意力）→ 负责长程 + 成本
```

- 总参 397B、激活 17B；MoE：512 routed experts + 1 shared，每 token 激活 10 个
- full attention 层才有 KV cache；GDN 层只有固定大小的 recurrent state + conv state
- 结果：长上下文成本接近线性，同时保留 full attention 的召回能力

## 关键数据与取舍

| 方案 | 长序列计算 | KV cache | 长程召回 |
|---|---|---|---|
| Full attention | O(N²) | 随 N 涨 | 强 |
| 纯线性注意力 | O(N) | 固定 | 弱-中 |
| Qwen3.5 混合 3:1 | 近 O(N) | 只有 1/4 的层有 KV | 强（full attention 兜底）|

取舍：

- **比例 3:1 不是免费的**：full attention 层决定长程质量上限；如果 GDN 记忆不够用，多放 full attention 层，成本又涨回去
- **recurrent state 对批量不友好**：GDN 的递推是序列相关的，prefill 要高效就得用 chunked / parallel scan；这是 serving 系统（vLLM/SGLang/ONNX Runtime 都专门支持了 Qwen3.5 的 state 管理）新增的工作量
- **架构层选择，不是规模特例**：Qwen3.5 从 0.8B 小模型到 397B 都用这套混合，说明"线性注意力 + 定期 full attention"已经是一个被验证的架构模板

## 与我何干

- **理论线**：把"线性注意力"从 Mamba/SSM 一路接到 GDN；和 [MLA](mla-deepseek.md) 对比记：**MLA 压缩 KV，GDN 消灭 KV**（改成固定状态）
- **C 阶段**：面试问"长上下文方案有哪些"，标准答法覆盖四类：KV 压缩（GQA/MLA/KV 量化）、稀疏注意力（DSA/Kascade）、线性注意力（Mamba/GDN）、系统技巧（chunked prefill/PD 分离）
- **算子线**：GDN 的 conv1d + 门控 + 状态递推是典型的 Triton 可写算子（B4 之后可以试），也是理解"attention 之外的算子"的好样本

---

*配套：[模型追踪表](model-tracker.md) · [最新模型与结构](latest-model-architectures.md) · [DSA](dsa-sparse-attention.md)*