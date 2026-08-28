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
| FlashAttention 2 | 推理 | [统一笔记](../algorithms/flash-attention-2.md) | 公式、online softmax、代码映射、work partitioning |
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

## 2026-08-20 外部资料更新

以下条目已按公开论文/官方项目入口核对，作为现有阅读路线的增量，不改变当前“先 Triton、后 CUDA 深钻”的执行顺序：

| 论文/技术 | 入口 | 挂靠 | 学习重点 |
|-----------|------|------|---------|
| FlashAttention-4 | [arXiv:2603.05451](https://arxiv.org/abs/2603.05451) | 主线 A 注意力实现侧 / B4 之后 | Blackwell 非对称硬件、异步 MMA、2-CTA MMA、CuTe-DSL |
| DeepSeek-V3.2 / DSA | [arXiv:2512.02556](https://arxiv.org/abs/2512.02556) | 主线 A 第 2 步 | sparse attention、indexer、长上下文计算量 |
| SageAttention3 | [arXiv:2505.11594](https://arxiv.org/abs/2505.11594) | 注意力实现侧 / 量化枝干 | Blackwell FP4 attention、8-bit attention 的精度与训练取舍 |
| Kascade | [arXiv:2512.16391](https://arxiv.org/abs/2512.16391) | DSA 后、serving 前 | anchor layer、跨层 top-k 复用、prefill/decode 稀疏化 |
| DeepSeek-V4 | [官方 Hugging Face 模型集合](https://huggingface.co/collections/deepseek-ai/deepseek-v4) · [技术报告入口](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) | 主线 A 第 3 步 | CSA/HCA、mHC、Muon、MXFP4、1M context 的 KV/FLOPs 账 |

阅读原则：先读 FA2/MLA/DSA 这条主线，FA4、SageAttention3、Kascade 作为实现侧增量；论文中的硬件数字必须区分 GPU 型号、dtype、baseline 和是否包含编译/调度开销。

持续更新入口：[2026 AI Infra 论文与项目观察池](../../papers/watchlist-2026.md)；自动抓取只进入 [papers/inbox](../../papers/inbox/README.md)，筛选前不进入学习计划。

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
