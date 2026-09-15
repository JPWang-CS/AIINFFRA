# Algorithms — 独立论文与理论学习线

> 论文线只负责阅读、公式推导、作者关键代码阅读和回答能力；它与实践线同级但独立，不要求每篇论文实现，也不与LeetGPU、服务器、当前算子或vLLM互设前置。当前主题由 [NOW.md](../../NOW.md) 指定，权威进度在 [PATH.md](../../PATH.md)。

## 1. 当前顺序

```text
已完成：Online Softmax / Parallel Reduce / FlashAttention-1 / FlashAttention-2
当前：MLA（DeepSeek-V2/V3）
下一篇：DSA（DeepSeek-V3.2）
之后：DeepSeek-V4增量、MoE、量化、推理系统、分布式与最新论文
```

FA2于2026-09-02完成阅读，只代表理论学习完成，不代表Triton实现或GPU验证。MLA仍为`WIP`。

## 2. 阅读分级

| 级别 | 要求 |
|---|---|
| 速读 | 问题、前代、贡献、主要结果、是否值得精读 |
| 精读 | 符号表、关键公式逐步推导、算法、实验、适用条件、局限 |
| 精读+代码 | 将公式、伪代码和数据结构映射到作者关键代码、执行路径与硬件假设 |

核心论文和重要最新论文至少达到“精读+代码”。代码阅读是理解作者实现，不等于自己复现；只有用户明确选择复现时，才在实践线创建任务。

## 3. 每篇论文的完成标准

必须能够脱离笔记回答：

1. 解决什么问题，前代瓶颈是什么；
2. 核心贡献和关键假设是什么；
3. 各符号、张量shape和关键公式如何推导；
4. 算法或系统流程如何执行；
5. 作者关键代码如何对应公式和伪代码；
6. 实验使用什么硬件、dtype、baseline、shape和指标；
7. 收益在哪些条件下成立；
8. 局限、代价和可能失败的场景；
9. 与前代、同期和后续方法有什么区别；
10. 面试中如何在1分钟和5分钟两个尺度讲清楚。

论文精读存入 `papers/`；跨多篇论文的技术机制、推导和问答存入本目录。二者可以互链，但不复制两份相同正文。

## 4. 经典主干

### 4.1 数学与GPU算法基础

| 主题 | 入口 | 状态 |
|---|---|---|
| Online Softmax | [online-softmax.md](online-softmax.md) | ✅ 已掌握 |
| Parallel Reduce | [parallel-reduce.md](parallel-reduce.md) | ✅ 已掌握 |
| Attention基础 | [Flash Attention机制](flash-attention-mechanism.md) | ✅ 已掌握FA1机制 |

### 4.2 Attention演进

| 主题 | 入口 | 状态 |
|---|---|---|
| FlashAttention-1 | [论文笔记](../../papers/attention/flash-attention.md) · [机制](flash-attention-mechanism.md) | ✅ 已消化并读CUDA |
| FlashAttention-2 | [统一精读](flash-attention-2.md) | ✅ 2026-09-02阅读完成 |
| MHA→MQA→GQA | [GQA论文](../../papers/attention/gqa.md) · [速览](remaining-theory-primer.md) | 🚧 待系统精读 |
| MLA | [mla-deepseek.md](mla-deepseek.md) | 🚧 当前 |
| DSA | [dsa-sparse-attention.md](dsa-sparse-attention.md) | 🚧 下一篇 |
| FA3/FA4/FlexAttention | [FA4/FlexAttention](fa4-flexattention.md) · [观察池](../../papers/watchlist-2026.md) | 🚧 待读 |
| SageAttention3/Kascade | [attention-2026-sage3-kascade.md](attention-2026-sage3-kascade.md) | 🚧 草稿 |
| GDN/线性注意力/SSM | [gdn-linear-attention.md](gdn-linear-attention.md) · [速览](remaining-theory-primer.md) | 🚧 待读 |

### 4.3 模型架构与MoE

| 主题 | 入口 | 状态 |
|---|---|---|
| DeepSeek-V3.2手算 | [deepseek-v32-handcalc.md](deepseek-v32-handcalc.md) | ✅ 已完成 |
| DeepSeek-V4增量 | [deepseek-v4.md](deepseek-v4.md) | 🚧 草稿 |
| MoE路由、专家与推理 | [moe-inference.md](moe-inference.md) | 🚧 草稿 |
| 最新模型架构 | [latest-model-architectures.md](latest-model-architectures.md) | 🚧 字典 |
| 模型追踪 | [model-tracker.md](model-tracker.md) | 持续维护 |

### 4.4 量化与低精度

| 主题 | 入口 | 状态 |
|---|---|---|
| INT8/FP8数值基础 | [quantization-int8-fp8.md](quantization-int8-fp8.md) | 🚧 草稿 |
| SmoothQuant/KV量化 | [remaining-theory-primer.md](remaining-theory-primer.md) | 🚧 速览 |
| GPTQ/AWQ | [remaining-theory-primer.md](remaining-theory-primer.md) | 🚧 待建正式精读 |
| FP4/SageAttention3 | [attention-2026-sage3-kascade.md](attention-2026-sage3-kascade.md) | 🚧 草稿 |

量化论文线要覆盖算法公式、calibration、scale粒度、误差机制、实验与作者代码；不因为实践线另有量化主课而省略论文学习。

### 4.5 推理系统

| 主题 | 入口 | 状态 |
|---|---|---|
| PagedAttention/vLLM | [PagedAttention精读](../../papers/inference/paged-attention.md) | 🚧 待读 |
| Continuous/Chunked Prefill | [remaining-theory-primer.md](remaining-theory-primer.md) | 🚧 速览 |
| Speculative Decoding | [speculative-decoding.md](speculative-decoding.md) | 🚧 草稿 |
| PD分离 | [pd-disaggregation.md](pd-disaggregation.md) | 🚧 草稿 |
| RadixAttention等 | [remaining-theory-primer.md](remaining-theory-primer.md) | 🚧 速览 |

### 4.6 训练与分布式

| 主题 | 入口 | 状态 |
|---|---|---|
| Adam/AdamW/Muon | [optimizers-adam.md](optimizers-adam.md) | 🚧 草稿 |
| ZeRO/FSDP | [ZeRO论文](../../papers/training/zero-paper.md) · [速览](remaining-theory-primer.md) | 🚧 待精读 |
| TP/PP/EP/CP与通信 | [remaining-theory-primer.md](remaining-theory-primer.md) · [观察池](../../papers/watchlist-2026.md) | 🚧 待读 |

## 5. 最新论文线

最新论文不必等待实践挂载点。候选先进入 [2026观察池](../../papers/watchlist-2026.md)，按价值决定速读、精读或精读+代码：

- 是否提出新的算法、数据结构、调度或硬件映射；
- 是否与GPU算子、低精度、Attention、MoE、推理系统或分布式直接相关；
- 实验和baseline是否可信；
- 是否有作者代码可读；
- 是否改变已有认识，而不只是刷新单个模型数字。

新论文阅读可以独立前进，但`NOW.md`一次只保留一个当前论文主题。

## 5.1 本地报告来源

[本地课程资料页](../../roadmap/curriculum/materials.md)收录了两份按页码定位的本地 PDF。DeepSeek 报告条目仅用于关联量化、KV Cache 与 Prefill/Decode 的资料核对；它不改变上面的论文顺序和状态，也不替代论文精读要求。

- [DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression（本地 PDF）](../../downloads/DeepSeek_V41_Tech_Report.pdf)：本地文件 51 页；封面署名 DeepSeek-AI。资料页标注了 §2 架构、CED、CSA2、跨层 KV/index 复用、层次索引、FP4 Main KV、推理系统、PersistentKV 与 SWA BoundedReplay 的 PDF 页码。
- [大模型推理实践（本地讲义）](../../downloads/大模型推理实践.pdf)：本地文件 84 页；资料页按 16 个主题提供页码入口，并区分昇腾/盘古案例与 NVIDIA CUDA 课程依据。

## 6. 文件边界

- `papers/`：以一篇论文为单位的精读；
- `notes/algorithms/`：跨论文机制、公式教程、模型专题和问答；
- `papers/inbox/`：未筛选候选；
- `papers/watchlist-2026.md`：确认值得观察的最新论文；
- `PATH.md`：论文进度权威；
- `NOW.md`：当前论文主题。
