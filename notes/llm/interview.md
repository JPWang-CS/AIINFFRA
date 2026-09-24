# 大模型性能优化面试笔记

> 本文聚焦模型结构、量化和推理系统。GPU 架构、CUDA/Triton 与算子回答主线见[GPU 高性能算子面试指南](../../roadmap/interviews.md)。教材覆盖不是个人掌握、平台成绩或真实设备验证的证明；没有亲自完成的工作要明确说是分析、复现计划或待验证假设。

## 回答方法与准备范围

每题都沿着“模型/请求条件 → 公式或算法 → 代码与数据流 → 正确性和性能证据 → 反例”展开。先说清模型形状、batch、prompt/output 长度、dtype、量化配置和测量边界，再讨论 bottleneck。一个 kernel 加速不等于服务端端到端提升；mean throughput 也不能替代 TTFT、TPOT 与尾延迟。无实测时不要填推测倍数。

本笔记的准备重点是 LLM Prefill/Decode、KV cache、Attention、量化和推理服务；GPU 体系结构及算子优化仍是第一核心。分布式训练、编译器及多 vendor 深入程度应根据岗位样本定向准备。岗位要求、个人资格年限、课程内容和个人成绩必须分开陈述。

| 能力 | 课程/材料入口 | 可出示的证据 | 分类 |
|---|---|---|---|
| Prefill Attention 与 IO 优化 | [Attention 算子章节](../../roadmap/curriculum/operators/05-prefill-attention/README.md)、[FlashAttention 论文笔记](../../papers/attention/flash-attention.md)、[Attention 代码阅读](../cuda/flash-attn-reading.md) | 论文推导、对应版本关键实现、同条件 correctness/benchmark；说明是否只读码 | 教材已覆盖；GPU 复现需实测 |
| Decode/PagedAttention 与 KV | [Decode/PagedAttention 章节](../../roadmap/curriculum/operators/06-decode-paged-attention/README.md)、[推理系统课程](../../roadmap/curriculum/systems/README.md)、[GQA 论文笔记](../../papers/attention/gqa.md) | KV 容量账本、块映射、真实请求负载的 TTFT/TPOT/吞吐和尾延迟 | 教材已覆盖；部署/压测需实测 |
| 量化精度和效率 | [量化路线](../../roadmap/curriculum/quantization/README.md)、[量化算子](../../roadmap/curriculum/operators/07-quantized-operators/README.md)、[INT8/FP8 笔记](../algorithms/quantization-int8-fp8.md) | 校准集与质量指标、dtype/scale/granularity、低精度指令路径和相同负载性能 | 教材已覆盖；质量/速度结论需实验 |
| Mini Transformer、系统路径 | [Mini Transformer](../../roadmap/curriculum/model-analysis/mini-transformer/README.md)、[推理系统章节](../../roadmap/curriculum/systems/README.md) | 从模型结构追踪到算子、cache、调度和端到端 trace 的可复现实验 | 教材覆盖；服务实证需部署 |
| 分布式训练 | [多 GPU 章节](../../roadmap/curriculum/systems/multi-gpu/README.md) | 拓扑、显存账本、collective 路径和 step-time 证据 | 按岗位需要补充 |
| 论文公式与关键代码 | [FlashAttention](../../papers/attention/flash-attention.md)、[GQA](../../papers/attention/gqa.md)、[量化论文入口](../algorithms/quantization-int8-fp8.md) | 公式符号/维度/推导与论文算法/代码变量的一一对应；说明所读版本 | 阅读需本人演示 |

## 模型结构和 KV cache

### MHA、GQA、MQA 与 KV 容量

MHA 为每个 query head 配置相应的 K/V head；MQA 让多个 query head 共享一个 KV head；GQA 则把 query heads 分组，每组共享一个 KV head。假设模型有 $L$ 层、batch 中共 $B$ 条缓存序列、每条缓存 $S$ 个 token、$H_{KV}$ 个 KV heads、每 head 维度 $D$、每元素 $b$ bytes，则不含分页对齐和元数据时：

$$
\mathrm{KV\ bytes}=2\,L\,B\,S\,H_{KV}\,D\,b.
$$

系数 2 来自 K 和 V。固定其他维度时，GQA 相对 MHA 的 KV 元素量比例约为 $H_{KV}/H_Q$；节省幅度取决于模型配置，不能把单个模型的 head 数或 MiB 数推广到所有模型。Decode 要读取历史 KV 计算注意力，但每步的瓶颈还受 batch、KV 长度、cache 层级、权重流量和调度影响。GQA 降低 KV 容量/流量的同时，也影响模型结构与质量，应按目标模型讨论。

追问“怎么验证”时，可先用配置值算理论字节数，再检查框架分配与 block table，最后用不同 batch/context length 的显存和请求 trace 对照。若实际分配高于公式，检查对齐、碎片、scale/metadata、临时 buffer、beam/copy-on-write 与实现布局，而不是修改理论公式来凑数。

### MLA 的低维缓存与权重吸收

MLA（Multi-head Latent Attention，多头潜在注意力）将每个 token 的内容 K/V 信息共同编码为低维向量。设压缩缓存为 $c_s\in\mathbb R^{d_c}$，第 h 个 head 的内容 Key 为 $k_{h,s}=W_h^{UK}c_s$，其中 $W_h^{UK}\in\mathbb R^{d_k\times d_c}$。则点积可改写为

$$
q_h^T k_{h,s}=\bigl((W_h^{UK})^Tq_h\bigr)^T c_s.
$$

右边先变换当前 Query，再与缓存中的低维向量计算分数，不必为所有历史 token 恢复各个 head 的完整内容 Key。Value 路径也可利用线性变换的结合律，将恢复映射与输出投影合并。位置分支采用解耦 RoPE，其旋转 Key 还需按具体模型保存，不能只计算内容缓存。

若每层保存 $d_c$ 个联合 latent 元素和 $d_r$ 个位置 Key 元素，等长 B 条序列、长度 S、L 层、每元素 b 字节的有效缓存为

$$
M_{\mathrm{MLA}}=LBS(d_c+d_r)b.
$$

这一个联合 latent 同时承载内容 K/V，不能按普通 MHA 公式再给 $d_c$ 乘 2；分页、量化 scale 和其他元数据另算。计算量与读取量还取决于权重吸收方式和 kernel 实现，缓存变小不等于全部算子同比例加速。

MLA 是模型结构中学习到的压缩路径；LoRA 是为权重更新引入低秩增量的适配方法。两者都用到低秩矩阵，但优化对象和推理数据流不同。复习时应能从[MLA 的公式与代码](../algorithms/mla-deepseek.md)说明缓存内容、点积维度、RoPE 分支，以及直接恢复与吸收两种执行方式。

### RoPE、RMSNorm 与 MoE

RoPE 将位置相关旋转作用于 query/key 的成对维度，使点积包含相对位置关系。讨论长度外推时，要区分位置频率缩放/插值策略、训练长度、模型适配和评测任务；不能把一个方法名当作所有模型都有效的解决方案。准备时沿公式说明旋转频率、位置索引与维度，并指出长上下文效果需要具体模型和数据验证。

对 d 维向量 $x$，RMSNorm 的典型表达如下。$\epsilon$ 是有限正数，用于稳定分母；$\gamma_i$ 是第 i 个特征的可学习缩放系数：

$$
y_i=\frac{x_i}{\sqrt{\epsilon+\frac{1}{d}\sum_{j=1}^{d}x_j^2}}\,\gamma_i.
$$

它按均方根缩放，不像 LayerNorm 那样先减均值；算子实现仍需做 reduction、归一化和逐元素缩放。融合与精度行为依输入 dtype、累加 dtype、向量长度和 kernel 实现验证。

MoE 的稀疏路由不代表实际计算和通信成本都按激活专家比例缩小。要同时考虑 router、token-to-expert dispatch/combine、专家负载不均、容量/丢 token 策略、all-to-all、权重放置和小 GEMM 效率。面试时画出 token 路由和返回的路径，结合专家并行拓扑与 token 分布说明瓶颈；不能只用“每 token 激活 k 个专家”推算端到端加速。

## Attention 与长上下文

### FlashAttention 的 IO-aware 计算

标准注意力为 $QK^T$、缩放与 softmax，再乘 $V$。直接物化完整 score/probability 矩阵会产生 $N\times N$ 的中间数据及相应 HBM 写读。FlashAttention 用 tile 遍历 K/V 块，在片上保存当前 query tile 的运行最大值、softmax normalizer 和输出累加；合并新块时重标定已有统计量，因此不必把完整注意力矩阵写入 HBM，并保持精确注意力计算（浮点舍入顺序可能不同）。

它仍然要读取输入 Q/K/V 并输出结果；实际 HBM 流量受 tile、片上存储预算、head dimension、causal/sparse mask、反向实现和硬件影响。论文的 IO 复杂度结论有其 SRAM/问题规模假设，不能说“所有 HBM IO 都是 O(N)”或宣称端到端固定 3–4 倍。FlashAttention-2 还调整并行划分与工作分配；回答“为什么更快”时要说出论文版本和对应机制，不能把 FA1/FA2 或某个库的实现混为一谈。

可拿来追问的验证链是：对齐参考实现和数值容差 → 对不同序列长度/head_dim/batch 做正确性 → 分别测 kernel 与完整模型 → 查看显存峰值和 Nsight 指标 → 用服务请求指标确认端到端受益。短序列、很小 batch、不同融合/库基线或非 attention 主导的模型，可能无法显著获益。

### PagedAttention、Continuous Batching 与 Chunked Prefill

PagedAttention 的块表把逻辑 token 区间映射到 KV cache 的物理块，避免要求每条序列拥有一整段连续最大容量，并便于调度和前缀共享。实现中的 block 大小、分配、复制和抢占策略会变化；面试时描述机制并核对目标 vLLM 版本，避免说固定默认 block 数或固定碎片率。

Continuous batching 在 decode iteration 边界移除完成请求、纳入新请求，减少静态 batch 被慢请求拖住的情况；其吞吐和尾延迟依请求长度分布、调度策略、KV 空间及并发而变。Chunked prefill 把长 prompt 切成块，使调度器有机会穿插 decode 工作；它改变公平性和 TTFT/TPOT 权衡，不能保证消除所有干扰。验证需使用包含长短混合请求的负载，并报告请求级分位数，而不仅是 GPU 利用率。

### Prefill 与 Decode 的瓶颈不是定义

Prefill 常能形成较大的矩阵计算，Decode 每步生成较少 token；但“Prefill 必然 compute-bound、Decode 必然 memory-bound”过于绝对。小 batch/短 prompt、上下文长度、GQA、权重驻留、cache 命中、并发、融合、采样和 launch 开销都会影响瓶颈。

单序列 decode 的权重读取估计可从参数个数乘每参数有效存储字节开始，但这只是流量/带宽下界的粗模型，不代表每一步必定从 HBM 重新读取全部权重。批量矩阵计算可在请求间复用权重，cache 也可能改变外部流量；同时还要加上 KV 扫描与其他读写。用 Nsight 和端到端 trace 观察真实 DRAM/L2 流量、kernel 时间和空隙，再用不同 batch/context/output length 找交叉点。

TTFT、TPOT、tokens/s 和 P95/P99 的测量边界必须说明。TTFT 含排队、调度、Prefill、首 token 处理等阶段；kernel event 时间不是 TTFT。请求完成时间的分位数应从逐请求样本计算，不要把多个阶段各自的 P95 直接相加。

### Speculative Decoding 与 Prefill/Decode 分离

Speculative decoding 用较小 draft 模型提出候选，再由 target 验证。接受/拒绝采样在算法条件满足时保持 target 输出分布；性能取决于 draft 开销、候选长度、接受率、target 的 batching 和验证 kernel，不能用固定 1.5–3 倍或“同家族必高接受率”作结论。测量需记录接受率、吞吐、延迟和输出分布/质量检查。

Prefill/Decode disaggregation 可把两类阶段放到独立资源池，减少互相干扰，但要支付 KV 状态迁移和调度成本。是否有效取决于负载、拓扑、传输量、带宽/拥塞和资源供给；不存在固定的 RDMA 时间，也不能先验说传输可忽略。比较时同时记录 TTFT、TPOT、吞吐、尾延迟、设备利用率和 KV 传输字节/耗时。

## 量化：从算法名回到运行路径

先明确格式：W4A16 是 4-bit 权重、16-bit 激活的常见记法；W8A8 表示权重和激活均为 8-bit，但具体 scale 粒度、累加 dtype、zero-point 和编码布局要继续交代。低 bit 可能降低权重存储和读取量，实际速度却受解包、scale 读取、反量化、矩阵形状、batch、融合和硬件支持影响。不能把 W4A16 固定说成“小 batch 优选”，或 W8A8 固定说成“大 batch 优选”。

- **AWQ**（Activation-aware Weight Quantization）是训练后权重量化。它使用校准输入的激活统计识别敏感通道，通过缩放选择来保护重要权重；“activation-aware”不是把激活做训练，也不是 QAT。面试时区分使用激活统计与量化对象是权重。
- **GPTQ** 使用近似二阶信息和逐步误差补偿对权重进行训练后量化。讨论时指出校准样本、分组/位宽、重构目标和实现差异。
- **SmoothQuant** 利用等价缩放把激活 outlier 导致的困难部分迁移给权重，以支持 W8A8 路径；它是 training-free/PTQ 方法，但质量和速度结论仍依校准、模型与 kernel。
- **QAT 与 PTQ** 是训练/后训练流程的区别，不是低精度位宽的别名。某个项目可以只校准不微调、做轻量适配，或执行量化感知训练；回答前应确认项目真实流程。

要证明量化有效，至少在同一模型与评测集上记录质量指标/误差、校准设置、权重与激活格式、峰值显存、吞吐和端到端时延。确认 profiler 或生成代码显示目标 kernel 实际使用了预期低精度路径；权重文件变小或配置写着 INT8 本身不证明计算更快。

## 训练/分布式问题的边界

按模型形状、精度、优化器和并行维度估算参数、梯度、master weights、optimizer states、激活和临时通信 buffer。Adam 状态是否使用 FP32、参数是否分片/卸载以及 activation checkpointing 会改变账本，避免直接报固定的“每参数 16 bytes”。

DP、TP、PP、EP 的通信模式分别与梯度同步、层内切分、层间激活传输、专家 token 路由有关；策略由显存容量、节点拓扑、模型 shape 和通信实现共同决定。ZeRO/FSDP 不同阶段切分不同状态，参数 all-gather 与 gradient reduce-scatter 时机和额外 buffer 也影响峰值。Ring all-reduce 的每 rank 传输量在经典 ring 假设下可写为约 $2(P-1)/P$ 倍 payload，但不能据此断言所有 NCCL 配置都走 ring 或链路饱和。

这部分主要用于目标岗位涉及训练系统时定向复习。性能判断要结合框架 profiler、通信 trace 和每 step 时间；只会背并行名词或纸面显存账本，不能替代集群实测。

## 论文阅读与代码复述

读论文时把符号和维度先写清楚，再沿算法步骤走到代码变量。对于 FlashAttention，应能解释 online softmax 的 running max、normalizer 和输出累加如何合并；对于 GQA，应从 query/KV head 数推导 cache 大小；对于 AWQ/SmoothQuant，应说明校准统计或缩放如何改变量化误差分布。读代码时记录仓库/commit 或 release、关键函数和实际支持路径，不能把伪码、论文结论和当前库实现拼成同一个版本。

| 主题 | 论文/代码入口 | 复述重点 |
|---|---|---|
| FlashAttention | [论文](../../papers/attention/flash-attention.md)、[算子中的公式与代码分析](../../roadmap/curriculum/operators/05-prefill-attention/README.md) | tile 数据流、online softmax 状态、避免中间矩阵物化、IO 结论的条件 |
| GQA | [论文笔记](../../papers/attention/gqa.md) | head 共享关系、KV 容量比例、质量与带宽取舍 |
| Triton 编译模型 | [论文笔记](../../papers/compiler/triton-paper.md)、[编译实现笔记](../cuda/triton-under-the-hood.md) | block-level 抽象、编译映射、性能可移植性的边界 |
| 量化 | [INT8/FP8 资料](../algorithms/quantization-int8-fp8.md)、[量化章节](../../roadmap/curriculum/quantization/README.md) | 数值范围、校准/缩放、kernel 路径与精度评估 |

## 项目证据模板

每个项目准备一页：

```text
工作负载与问题：模型/算子、shape、dtype、设备与目标
正确性合同：reference、误差容限、通过方式
初始证据：baseline、计时方法、profiling 观察
实现变更：具体改动的函数/数据布局/调度路径及其理由
验证结果：未开启 profiler 的 benchmark、多 shape 与请求指标
限制和反例：没有验证的设备、shape、版本或性能假设
个人贡献：本人实际写/调试/测量的部分，与团队结果分开
```

没有数据的字段保留为“未测”或删去，不填计划值。仓库里的代码、论文分析、LeetGPU 成绩和真实设备日志应分别表述，不互相代替。

## 参考来源

- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)；[FlashAttention-2 论文](https://tridao.me/publications/flash2/flash2.pdf)。IO-aware exact attention 与工作划分的原始论述。
- [AWQ 论文](https://arxiv.org/abs/2306.00978)；[SmoothQuant 论文](https://arxiv.org/abs/2211.10438)。核对训练后量化、激活统计及等价缩放。
- [vLLM PagedAttention 论文](https://arxiv.org/abs/2309.06180)。核对 KV 分页与服务调度设计；版本实现仍须另行检查。
- [PyTorch Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)。核对自定义算子注册与集成，而不是仅从 kernel 函数推断框架接入完成。

*更新范围：面试复习内容；不维护学习进度、LeetGPU 成绩或 GPU 验证状态。*
