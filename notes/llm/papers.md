# 大模型板块：论文

> 子板块目标：把大模型相关论文按学习顺序整理，读一篇更新一篇，并关联到对应子板块。
> 状态：🚧 草稿。

---

## 阅读顺序

```text
1. FlashAttention（推理算子基础）
2. GQA / MLA / MoE（模型结构）
3. PagedAttention / vLLM（推理系统）
4. ZeRO / FSDP（训练系统）
5. 量化（AWQ/GPTQ）
6. 长序列/SSM（可选深钻）
```

---

## 核心论文清单

| 论文/技术 | 对应子板块 | 仓库状态 | 学习重点 |
|-----------|-----------|:--:|---------|
| FlashAttention 1 | 推理 / 结构 | [已有](../../papers/attention/flash-attention.md) | tiling + online softmax |
| FlashAttention 2 | 推理 | [已有](../../papers/attention/flash-attention-2.md) | work partitioning |
| GQA | 结构 | [已有](../../papers/attention/gqa.md) | KV cache |
| PagedAttention | 推理 | [已有](../../papers/inference/paged-attention.md) | block table |
| ZeRO | 训练 | [已有](../../papers/training/zero-paper.md) | 显存分片 |
| DeepSeek-V2 MLA | 结构 | 已有笔记 [MLA](../algorithms/mla-deepseek.md) | KV 压缩 |
| DeepSeekMoE | 结构 | 已有笔记 [MoE](../algorithms/moe-inference.md) | expert 路由 |
| Mamba | 结构 | 待建 | SSM |
| AWQ | 推理/量化 | 待建 | 激活感知 |
| GPTQ | 推理/量化 | 待建 | 权重量化 |
| Speculative Decoding | 推理 | 已有笔记 | draft-verify |
| Ring Attention | 推理/长序列 | 待建 | 序列分块 |

---

## 精读输出模板

每篇论文至少写：

```text
1. 解决什么问题
2. 核心思路
3. 关键公式/数据结构
4. 性能数字
5. 局限和取舍
6. 面试口径
```

完整格式见 [papers/process.md](../../papers/process.md)。

---

## 建议路径

### 第一周：算子

- FlashAttention 1
- FlashAttention 2

### 第二周：结构

- GQA
- DeepSeek-V2（MLA）
- DeepSeekMoE

### 第三周：推理

- PagedAttention
- vLLM 技术报告
- Speculative Decoding

### 第四周：训练

- ZeRO
- FSDP 文档
- Megatron TP/PP 文档

### 第五周以后：可选

- Mamba
- AWQ / GPTQ
- Ring Attention
- SGLang / RadixAttention

---

## 关联材料

- [论文索引](../../papers/README.md)
- [模型结构](architectures.md)
- [推理系统](inference-systems.md)
- [训练系统](training-systems.md)
