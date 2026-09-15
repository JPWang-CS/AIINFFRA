# 大模型论文学习入口

> 论文线独立于实践线，纯阅读、公式推导和作者关键代码阅读；不要求每篇实现，不以LeetGPU、服务器、当前算子或vLLM进度为前置。权威状态见 [PATH.md](../../PATH.md)，详细索引见 [Algorithms README](../algorithms/README.md)。

## 当前进度

```text
FA1：已消化并读CUDA
FA2：2026-09-02阅读完成
MLA：当前WIP
DSA：下一篇
```

## 阅读地图

| 专题 | 经典主干 | 扩展与最新 |
|---|---|---|
| Attention | Attention、GQA、FlashAttention-1/2、MLA | FA3/4、DSA、SageAttention、Kascade、GDN/线性注意力 |
| 模型/MoE | Transformer、DeepSeekMoE、DeepSeek-V2/V3 | DeepSeek-V4、新MoE与混合架构 |
| 量化 | SmoothQuant、GPTQ、AWQ、FP8 | FP4、MXFP8/MXFP4/NVFP4、KV量化、低比特Attention |
| 推理系统 | PagedAttention/vLLM、Continuous Batching | Chunked Prefill、RadixAttention、PD分离、投机解码 |
| 分布式 | ZeRO、Megatron、FSDP | EP/CP、通信计算重叠、超大规模训练/推理系统 |
| 编译/Kernel | Triton、FlashAttention实现体系 | CUTLASS/CuTe、TileLang、新编译与kernel生成工作 |

## 核心仓库入口

| 论文/技术 | 本地入口 | 当前状态 |
|---|---|---|
| FlashAttention-1 | [论文笔记](../../papers/attention/flash-attention.md) | ✅ 已消化 |
| FlashAttention-2 | [统一精读](../algorithms/flash-attention-2.md) | ✅ 阅读完成 |
| GQA | [论文笔记](../../papers/attention/gqa.md) | 🚧 待系统精读 |
| MLA | [机制与公式](../algorithms/mla-deepseek.md) | 🚧 当前 |
| DSA | [机制笔记](../algorithms/dsa-sparse-attention.md) | 🚧 下一篇 |
| PagedAttention | [论文笔记](../../papers/inference/paged-attention.md) | 🚧 待读 |
| ZeRO | [论文笔记](../../papers/training/zero-paper.md) | 🚧 待读 |
| MoE | [机制笔记](../algorithms/moe-inference.md) | 🚧 草稿 |
| 量化 | [量化基础](../algorithms/quantization-int8-fp8.md) | 🚧 草稿；GPTQ/AWQ/SmoothQuant正式精读待建 |
| 最新论文 | [2026观察池](../../papers/watchlist-2026.md) | 持续更新 |

## 完成要求

普通论文按价值选择速读或精读；核心和重要最新论文至少做到“精读+代码”：问题与前代、符号、关键公式、算法、实验设置与结果、局限、作者关键代码映射，以及1分钟/5分钟回答。完整记录格式见 [papers/process.md](../../papers/process.md)。

论文代码阅读不等于实践完成；是否复现由用户单独决定。
