# 低精度与量化主题索引

量化算法、编码、kernel 与实践已在[量化算子正文](../operators/07-quantized-operators/README.md)连续展开。本页保留主题索引，不是额外必修路线；更深入的论文阅读独立推进。

| 单元 | 内容 | 现有入口 | 最终验收 |
|---|---|---|---|
| 第一章 | FP32/TF32/FP16/BF16、INT8/INT4、E4M3/E5M2、FP4、MX/NV格式、round/saturate | [量化基础](../../../notes/algorithms/quantization-int8-fp8.md) | 手算范围、编码与误差 |
| 第二章 | symmetric/asymmetric、per tensor/channel/token/group/block、static/dynamic、clipping/outlier | [理论速览](../../../notes/algorithms/remaining-theory-primer.md) | scale/zero point与metadata账本 |
| 第三章 | RTN、PTQ/QAT、SmoothQuant、GPTQ、AWQ | [论文入口](../../../notes/algorithms/README.md) | 公式、校准、误差补偿、作者代码 |
| 第四章 | Weight/Activation：W8A8、W4A16、W4A8、dynamic activation | 量化算子正文与后续模型实验 | 精度、显存、吞吐对比 |
| 第五章 | FP8/FP4：current/delayed/block scaling、MXFP8、NVFP4 | [DeepSeek相关](../../../notes/algorithms/) | 计算/累加路径、layout、硬件边界 |
| 第六章 | KV Cache量化与GQA/MLA叠加 | [MLA](../../../notes/algorithms/mla-deepseek.md) | cache bytes、TPOT、长上下文质量 |
| 第七章 | packing、quant/dequant、低精度GEMM、fused dequant、quant attention | [量化算子](../operators/README.md) | kernel-only与含量化开销端到端分测 |
| 第八章 | checkpoint/scale metadata、backend、TP/EP分片 | [系统落地](../systems/README.md) | Mini Transformer/框架接入 |

准确性按任务使用abs/rel、MSE/SQNR、cosine、perplexity或下游质量；记录校准集、失败层和长上下文退化。官方入口：[Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/) · [TensorRT-LLM Quantization](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html)
