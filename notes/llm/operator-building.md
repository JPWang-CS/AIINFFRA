# LLM 算子构建能力地图

> 本文件是内容索引，不是第三条学习线。现行顺序和状态以 [PATH.md](../../PATH.md) 为准，统一验收见 [总执行计划](../../roadmap/ai-infra-curriculum.md)。

## 1. 定位

目标是把模型结构拆成可实现、可测量、可极致优化的 GPU 算子，并在 Mini Transformer 中验证；vLLM只负责真实系统映射。论文阅读进度独立，不因这里的实现状态改变。

## 2. 从 Decoder 到算子

```text
Decoder
├── Embedding / layout
├── RMSNorm / fused residual norm
├── QKV projection / RoPE / QK norm
├── Prefill or Decode Attention
├── output projection
├── SwiGLU / fused MLP
├── MoE router / grouped experts（可选架构）
└── LM head / logits / sampling
```

## 3. 构建路线

| 算子族 | 代表任务 | 性能核心 |
|---|---|---|
| Memory/Layout | transpose、gather/scatter、embedding、KV copy、pack | coalescing、cache、transaction、GB/s |
| Reduction/Norm | Softmax、Online Softmax、RMSNorm、LayerNorm | reduce层级、SFU、数值、occupancy |
| GEMM | GEMV/GEMM、batched/grouped、split-K、persistent | Tensor Core、tile、layout、pipeline |
| Fusion | RoPE、SwiGLU、residual norm、QKV、fused MLP | 中间traffic、launch、register |
| Prefill Attention | naive、FA1/FA2、causal、varlen、GQA/MLA | HBM bytes、MMA与非MMA、CTA并行 |
| Decode Attention | single-query、split-KV、Paged/MLA decode | KV带宽、batch、page、TPOT |
| Quant | quant/dequant、W8A8、W4A16、FP8/FP4 | packing、scale、低精度MMA、误差 |
| MoE | top-k、permute、dispatch/combine、grouped GEMM | 负载、traffic、专家并行 |
| Serving | logits、sampling、cache维护 | 小kernel、launch、动态shape |

## 4. 统一完成定义

有LeetGPU题面的算子：原理 → LeetGPU通过并归档原始代码 → 真实GPU正确性与baseline → 性能分析。核心锚点继续走Nsight、PTX/SASS、多shape和模型接入；没有平台题面的系统primitive使用reference门。

每个构建任务至少留下：代码、reference对齐、硬件/shape/dtype、性能数字、瓶颈解释、下一实验。只有读懂论文或保存reference不算实践完成。

## 5. 模型级验证

Mini Transformer从标准Decoder开始，逐步替换RMSNorm、RoPE、Attention、MLP、GQA和量化实现；分别测kernel、layer、Prefill和Decode。必须解释单算子收益为何能或不能传播到模型。

## 6. 关联入口

- [GPU架构与性能工程](../../roadmap/gpu-foundations.md)
- [总执行计划](../../roadmap/ai-infra-curriculum.md)
- [Triton代码](../../solutions/triton/README.md)
- [模型结构](architectures.md)
- [推理系统](inference-systems.md)
- [论文入口](papers.md)
