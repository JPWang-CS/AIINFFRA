<!-- Archived source: roadmap/curriculum/operators/README.md; retained for maintenance, not a published lesson. -->

# 第二篇 完整 GPU / LLM 算子体系

这一篇从数据搬运和布局开始，逐步进入归约、矩阵计算、融合、Attention、量化与MoE。目标不是记住几份kernel模板，而是面对一个新的形状和布局，能够写出正确实现，判断主要成本，再用实验解释优化为什么有效。

每个算子先讲清公式、数据布局与实现，紧接着在同页实践：核对题目要求、编写代码、验证结果，再比较基线与优化版本。必要代码直接放片段，便于对照公式与实验；不通过源码文件跳转组织阅读。

## 全景概览：九类算子怎样连接

| 学习内容 | 主要关注点 |
|---|---|
| 访存与布局 | 数据的位置、步长、重排与有效传输 |
| Reduction与Norm | 归约组织、数值稳定性与多行并行 |
| GEMM全体系 | 计算复用、矩阵指令、分块与调度 |
| Activation与Fusion | 中间结果、寄存器复用与物化边界 |
| Prefill Attention | 多Query并行、score tile与长序列处理 |
| Decode Attention | KV读取、分页、小计算量与批量扩展 |
| 量化kernel | 编码、scale、低精度计算与误差 |
| MoE算子 | 路由、重排、专家计算与负载 |
| Sampling与KV辅助算子 | 选择、随机性、cache更新与动态形状 |

学习顺序不是等到写完全部算子才开始优化。每类算子先固定数学、shape、stride和精度，建立独立参考与真实baseline，再根据所需知识逐步深入。**GEMM、Norm、Fused MLP、Prefill、Decode和量化/MoE是持续研究的对象，不是一次性练习题。**

> [!IMPORTANT] 始终保留同一个比较前提
> **相同输入输出、相同数值要求和相同计时边界，才构成有意义的性能比较。** 接近某个框架实现，不自动等于接近硬件上界。

## 1. 访存与布局算子

这类算子算术少，但可能反复出现在模型的数据准备、布局转换和cache更新中。它们也是理解“逻辑张量”和“物理存储”差异最直接的入口。

### 知识点

- **Copy、Transpose、Permute**：区分view与真实数据重排，展开shape、stride、storage offset和字节地址。
- **Gather/Scatter与Embedding**：研究索引局部性、重复读取、重复写入以及冲突语义。
- **KV Copy与Pack**：处理page边界、位布局、有效元素尾部和buffer的使用生命周期。
- **性能优化**：合并访问、向量化copy、shared tile与padding、缓存工作集、异步搬运；先确认实际数据移动，再计算有效带宽。

### 推荐资料与代码

[访存与布局首章](./01-memory-and-layout/README.md)从Copy、float4对齐和真实Transpose开始，包含LeetGPU入口、服务器命令与代码对照。[GPU存储系统](../gpu/03-registers-and-memory-system/README.md)用于回查sector、bank、寄存器和local memory等机制。

### 掌握标准

能手算一个矩形张量的输入输出地址，解释尾部mask；能分辨转置view和物化结果；能在相同工作量下比较标量、向量化与分块实现，并说明缓存条件对带宽数字的影响。

## 2. Reduction 与 Norm

本章正文入口：[并行归约、Softmax 与归一化](./02-reduction-and-norm/README.md)。

归约不只是把元素相加。输入规模、归约轴和独立行数决定怎样分工；累加顺序与数据类型则影响误差。Norm和Softmax在归约基础上增加统计量与逐元素变换，需要同时考虑数值和数据复用。

### 知识点

- **Sum/Max与树归约**：thread、warp、block与多阶段归约，各自的同步和中间结果成本。
- **Softmax**：稳定形式、Online Softmax、尾部mask、长行与多行的实现选择。
- **RMSNorm、LayerNorm与Cross Entropy**：明确归约统计量、精度和输出公式，再研究读写次数。
- **性能优化**：warp shuffle、online reduction、persistent row、融合、长序列，以及低精度误差预算。

Scan（前缀扫描）和Histogram（直方图）作为支撑模式补入：前者用于索引生成、压缩和数据重排，后者用于理解原子竞争与局部聚合。它们补足基础能力，不替代九类算子的任何一项。

### 推荐资料与代码

[CUDA Softmax](../../../lessons/04-softmax.md)保留归约实现和讨论；[Triton Softmax](../../../lessons/08-triton-fused-softmax.md)连接平台代码与服务器实验；本章正文继续到 [并行归约、Softmax 与归一化](./02-reduction-and-norm/README.md)，覆盖 CUDA 两级归约、row-wise Triton、RMSNorm/LayerNorm 和多 shape 性能分析。已有原理不重新当作入门任务，重点转向形状适配、融合和性能解释。

### 掌握标准

能根据短行、长行、多行和小batch选择并行组织；能解释单位元、warp参与mask与barrier的关系；能在独立参考下判断数值误差，并说明某种布局为什么减少或增加数据移动。

## 3. GEMM 全体系

GEMM需要同时研究数据复用和计算管线。先把已有矩阵乘的分块、地址和数值要求讲清楚，再进入Tensor Core与更复杂的调度，不把“更大tile”当作通用优化规则。

### 知识点

- **基础形态**：GEMV、GEMM，以及row-major、显式stride、累加精度和M/N/K尾块。
- **扩展形态**：batched/grouped GEMM、split-K、persistent kernel、epilogue和quantized GEMM。
- **深入优化**：Tensor Core tile、double buffering、warp specialization、寄存器与shared预算、指令生成和多shape适配。
- **低精度矩阵计算**：FP8、FP4、W4A16等路径的输入布局、scale位置与误差要求，不能与IEEE FP32混作一张优化表。

### 推荐资料与代码

从[第三章 GEMM：从分块实现到性能优化](./03-gemm/README.md)、[Naive GEMM](../../../lessons/02-gemm-naive.md)、[Tiled GEMM](../../../lessons/03-gemm-tiled.md)与[现有MatMul代码和实测](../../../solutions/triton/README.md)继续。已有baseline不清零；后续项目用它追踪每项优化改变了哪些资源和指令。

### 掌握标准

能计算FLOPs、算法bytes与tile的复用关系；能区分输入、累加和输出精度；能读寄存器/shared报告，并用Profiler和PTX/SASS解释一次收益或退化，而不是只报告最快值。

## 4. Activation 与 Fusion

融合的价值在于避免不必要的中间结果物化，但融合也会延长值的活跃区间、提高寄存器压力。判断是否值得融合，需要先明确生产者与消费者的数据依赖。

### 知识点

- **独立算子**：Activation、RoPE、QK Norm、SwiGLU、Residual Norm、QKV与MLP中的张量变换。
- **融合边界**：哪些中间值可以直接在寄存器或shared中消费，哪些需要跨block或跨kernel协作。
- **深入优化**：producer-consumer fusion、寄存器复用、融合epilogue、减少global流量与launch数量。
- **模型收益**：局部kernel更快，不保证模型层更快；还要检查布局转换、框架调用和额外同步。

### 推荐资料与代码

[第四章 Activation 与 Fusion](./04-activation-and-fusion/README.md)给出 dense MLP、SwiGLU、packed layout、Triton pointwise kernel 和中间激活流量账本；[算子构建地图](../../../notes/llm/operator-building.md)保留模型模块与算子的对应关系。阅读时先还原独立算子公式和中间张量，再分析融合是否改变数值顺序或接口合同。

### 掌握标准

能列出融合前后的中间张量与读写字节；能说明寄存器和并行度的代价；能同时报告单kernel与算子链的实际收益。

## 5. Prefill Attention

Prefill中有多个Query位置可供并行，长序列又使score矩阵的物化成本突出。重点是理解分块如何保持结果正确，以及Q/K/V与归约状态怎样分配到线程和存储。正文与 CPU/GPU correctness harness 见[第五章 Prefill Attention](./05-prefill-attention/README.md)。

### 知识点

- **基础实现**：naive、tiled、FlashAttention 1/2，先用小矩阵核对Q/K/V、mask、Softmax和输出。
- **形状变化**：causal、varlen、GQA/MLA，不能把一种固定head/序列布局推广到全部输入。
- **深入优化**：Online Softmax、片上tile、Q方向并行、warp分工、异步流水与长序列流量。
- **结果边界**：精确算法、低精度实现和近似方法分开讨论，明确误差来自哪里。

### 推荐资料与代码

[FlashAttention读码](../../../lessons/05-flash-attn-reading.md)与[FA2笔记](../../../notes/algorithms/flash-attention-2.md)保留已有讨论。实现课程把公式、张量形状和作者代码对应起来，不要求重新证明已经掌握的全部基础公式。当前方形 `[B,H,S,D]` 教学 baseline 与矩形 #6、无 batch GQA #80、split-half RoPE #61 分开核对，不互相冒充题面实现。

### 掌握标准

能画出一个Q tile扫描K/V tile的数据流，解释Online Softmax状态更新；能说明causal与变长输入改变了什么；能分析计算、片上资源和显存流量各自的限制。

## 6. Decode Attention

Decode每次新增的Query很少，却可能需要读取大量历史KV。它与Prefill不是仅仅换一个序列长度，瓶颈和并行拆分都可能不同。完整正文见[第六章 Decode / PagedAttention](./06-decode-paged-attention/README.md)。

### 知识点

- **基础形态**：single-query、batch/head并行、KV dtype与长度。
- **分页与拆分**：PagedAttention、page table、split-KV及合并、FlashDecoding类实现。
- **模型变体**：GQA/MLA、KV量化，以及对应的读取和还原成本。
- **深入优化**：page locality、batch利用、KV bytes、split-KV粒度与合并代价，连接TPOT和吞吐。

### 推荐资料与代码

[PagedAttention](../../../papers/inference/paged-attention.md)与[MLA](../../../notes/algorithms/mla-deepseek.md)用于核对cache表示和访问路径。先算清楚每次Decode读取什么，再考虑分块和量化。

### 掌握标准

能给定batch、上下文长度、head配置与dtype，计算KV容量和一次Decode的主要读取量；能解释拆分如何增加并行度，又为什么会增加中间结果与合并成本。

## 7. 量化 Kernel

[量化算子正文](07-quantized-operators/README.md)：编码、scale 布局、低精度 GEMM、算法与实践在同一章展开。

量化算法决定近似方式，kernel实现决定这种近似能否转换为带宽、容量或计算收益。两者要连接起来，但不能把位打包当作完整量化算法。

### 知识点

- **基础操作**：amax、scale、quant/dequant，区分权重、激活与KV。
- **数据格式**：W8A8、W4A16、FP8/FP4，packing/unpacking与scale的索引关系。
- **计算路径**：低精度MMA、累加精度、反量化位置与融合dequant。
- **评价方法**：误差、显存、带宽、kernel耗时和端到端收益；Prefill与Decode分别分析。

### 推荐资料与代码

量化正文在同一页给出编码、数据布局、计算路径和练习。先定义量化粒度，再核对目标指令实际消费的格式，并将反量化成本纳入对比。

### 掌握标准

能说明每个量化块的scale服务哪些元素；能还原packed编码并评估误差；能把反量化与矩阵计算的成本纳入同一份性能分析。

## 8. MoE 算子

[MoE 正文](08-moe/README.md)：路由、重排、专家计算与合并，附基础 GPU 路由及完整 CPU 对照。

MoE把token分配给不同专家，计算之前和之后都需要数据组织。实际成本既包含专家GEMM，也包含路由、重排、负载与通信。

### 知识点

- **基础路径**：router/top-k、permute、dispatch/combine、grouped GEMM和expert fusion。
- **正确性合同**：token-to-expert索引、输出顺序，以及具体算法采用的容量、重复或丢弃策略。
- **单GPU优化**：dispatch流量、不同专家形状、Grouped GEMM利用率、路由与融合。
- **多GPU扩展**：负载均衡、通信—计算重叠与EP（Expert Parallelism，专家并行），不把它当作单卡算子实现的前置。

### 推荐资料与代码

[MoE推理笔记](../../../notes/algorithms/moe-inference.md)提供模型侧背景。先在可控输入上核对每个token的去向与回收，再研究专家负载不均怎样影响kernel形状。

### 掌握标准

能从原始token还原派发与合并索引，解释Grouped GEMM的形状集合；能分别计算数据重排、专家计算和通信的成本，避免只按激活参数量估算整条执行路径。

## 9. Sampling、KV 与 Serving 辅助算子

[Sampling 与 KV 辅助算子正文](09-sampling-kv/README.md)：候选选择、随机采样、投机验证与缓存提交，在页内对照代码和实践。

这些操作可能单次很短，却会高频出现在Decode循环中。重点不是孤立追求某个小kernel的峰值，而是同时关注工作量、调用次数与动态形状。

### 知识点

- **Logits与选择**：top-k、top-p、sampling，明确随机数与采样分布合同。
- **Cache操作**：KV写入、page/cache复用、所有权和生命周期。
- **动态工作负载**：小vocabulary、小batch、动态长度及shape分桶。
- **深入优化**：选择算法、融合launch、减少布局转换，以及与batch调度的实际关系。

### 推荐资料与代码

[推理系统内容](../../../notes/llm/inference-systems.md)帮助理解调用位置。系统知识用于解释这些算子的工作负载，不能反过来把部署框架变成整条课程的中心。

### 掌握标准

能验证小规模选择与采样结果，说明随机性测试方法；能定位高频小kernel的launch gap与中间数据成本，并判断局部加速是否传递到请求延迟。

## 推荐的实践方法

有LeetGPU题面的算子，按题面从空白实现，原样保存平台solve/kernel，再建立服务器包装。没有对应题面的硬件与系统实验，也要有独立参考和明确测量条件，但不伪造平台成绩。

每次baseline固定shape、stride、dtype、输入输出布局与计时边界。用 $2NE/t$ 计算copy类有效带宽、用 $2MNK/t$ 表示GEMM的运算吞吐时，先写清变量、单位与实际工作。算法流量和Profiler观测流量也应分别保留。

进入极致优化后，围绕具体瓶颈提出假设，一次改变一个可解释因素，再做多shape回归。寄存器、shared、缓存、同步与指令吞吐会互相制约，不能只用一个occupancy或GB/s数字替代完整分析。

## 推荐资料与原始记录

- [GPU架构与CUDA课程](../gpu/README.md)：回查线程、存储、同步和性能模型。
- [性能实验检查表](../performance/README.md)：整理报告时按需核对，不是独立学习单元。
- [自己的代码总览](../../../solutions/README.md)：查看原始实现及实验来源。
- [参考实现目录](../../../reference/)：完成自写版本后对照成熟实现。
- [AIInfraGuide学习路线正文](https://caomaolufei.github.io/AIInfraGuide/guides/ai-infra%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/)：参考学习内容的组织方式。

论文线独立推进，实践线则沿着当前算子持续深入。已有代码、问题和失败实验都是下一次分析的依据，不因课程结构调整而重新开始。
