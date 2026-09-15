# 性能实验检查表

优化正文位于各算子自己的章节中。本页只供整理实验和检查证据时查阅，不设独立学习顺序，也不要求离开算子章节完成这里的项目。

## 适用对象

GEMM、Reduction/Norm、Fused MLP、Prefill FlashAttention、Decode/PagedAttention、Quantized GEMM、MoE Grouped GEMM。各章按自身瓶颈选择优化方法，不机械套用全部项目；核心算子的最终报告应覆盖正确性、强基线、资源与指令分析、多形状结果和收益边界。

## 按实验需要核对的事项

| 检查项 | 核心问题 | 证据 |
|---|---|---|
| 检查一 | 数学、shape、stride、dtype、accumulator和误差语义？ | 公式与correctness contract |
| 检查二 | 公平strong baseline是谁？ | PyTorch/官方Triton/cuBLAS(Lt)/CUTLASS等同语义结果 |
| 检查三 | FLOPs、bytes、AI和各层roof？ | compute/memory/latency预判 |
| 检查四 | tile、warp、stage、grid如何影响？ | 单变量sweep与失败配置 |
| 检查五 | coalescing、cache、shared、layout和复用？ | memory workload/transaction |
| 检查六 | register、shared、resident CTA、occupancy和spill？ | 编译资源、occupancy、stall |
| 检查七 | copy/compute能否流水重叠？ | stage、async、timeline |
| 检查八 | CPU/launch/stream还是kernel？ | Nsight Systems/NVTX |
| 检查九 | 哪条pipe或memory层限制？ | Nsight Compute/roofline/warp state |
| 检查十 | 是否生成预期load/MMA/barrier？ | PTX/SASS |
| 检查十一 | fusion、split、persistent、work partition是否更优？ | 算法级实验 |
| 检查十二 | 是否只赢单一shape/架构？ | 多shape/batch/dtype/GPU与模型接入 |

## 报告样例

[MatMul 配置分析](../../../notes/triton/matmul-performance-analysis.md)展示同精度、同形状的候选比较；[流水深度对比](../../../notes/triton/matmul-nsys-p0-lite-2026-08-30.md)展示如何从原始 trace 筛选同一工作负载。报告中的数字只适用于其设备、形状、精度和软件环境，不作为其他 GPU 的性能预期。

整理自己的报告时，把假设、代码改动、预期指标、实测与结论放在一起；缺少硬件计数器时，明确区分资源推导与实际测量。
