# 大模型性能优化面试笔记

> 本文聚焦模型结构、量化和推理系统。GPU 架构、CUDA/Triton 与算子回答主线见[GPU 高性能算子面试指南](../../roadmap/interviews.md)。教材覆盖不是个人掌握、平台成绩或真实设备验证的证明；没有亲自完成的工作要明确说是分析、复现计划或待验证假设。

## 回答方法与准备范围

每题都沿着“模型/请求条件 → 公式或算法 → 代码与数据流 → 正确性和性能证据 → 反例”展开。先说清模型形状（shape）、batch、prompt/output 长度、数据类型（dtype）、量化配置和测量边界，再判断瓶颈（bottleneck）。单个 kernel 变快只有在关键路径上才可能改善服务端时间；平均吞吐（mean throughput）也必须和首 token 时间（TTFT，Time To First Token）、每输出 token 时间（TPOT，Time Per Output Token）及尾延迟一起看。没有实测时只写待验证假设，不填推测倍数。

本笔记的准备重点是 LLM 的 Prefill（提示词阶段）、Decode（逐 token 阶段）、KV cache、Attention、量化和推理服务；GPU 体系结构及算子优化仍是第一核心。分布式训练、编译器及多厂商（multi-vendor）深入程度应根据岗位样本定向准备。岗位要求、个人资格年限、课程内容和个人成绩必须分开陈述。

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

多头注意力（MHA，Multi-Head Attention）为每个 query head 配置相应的 K/V head；多查询注意力（MQA，Multi-Query Attention）让多个 query head 共享一个 KV head；分组查询注意力（GQA，Grouped-Query Attention）则把 query heads 分组，每组共享一个 KV head。假设模型有 $L$ 层、batch 中共 $B$ 条缓存序列、每条缓存 $S$ 个 token、$H_{KV}$ 个 KV heads、每 head 维度 $D$、每元素 $b$ bytes，则不含分页对齐和元数据时：

$$
\mathrm{KV\ bytes}=2\,L\,B\,S\,H_{KV}\,D\,b.
$$

系数 2 来自 K 和 V。固定其他维度时，GQA 相对 MHA 的 KV 元素量比例约为 $H_{KV}/H_Q$；节省幅度取决于模型配置，不能把单个模型的 head 数或 MiB 数推广到所有模型。Decode 要读取历史 KV 计算注意力，但每步的瓶颈还受 batch、KV 长度、cache 层级、权重流量和调度影响。GQA 降低 KV 容量/流量的同时，也影响模型结构与质量，应按目标模型讨论。

验证路径是：先用模型配置计算理论字节数，再检查框架分配与 block table，最后用不同 batch/context length 的显存和请求 trace 对照。若实际分配更高，应沿对齐、碎片、scale/metadata、临时 buffer、beam/copy-on-write 与实现布局逐项定位。

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

MoE（Mixture of Experts，混合专家）的稀疏路由只减少部分专家计算，实际成本还由 router、token-to-expert dispatch/combine、专家负载、容量/丢 token 策略、all-to-all、权重放置和小 GEMM 效率共同决定。回答时可画出 token 从路由到专家再返回的路径，结合专家并行拓扑与 token 分布定位瓶颈；“每 token 激活 k 个专家”只能估算局部计算量，不能替代端到端测量。

## Attention 与长上下文

### FlashAttention 的 IO-aware 计算

标准注意力为 $QK^T$、缩放与 softmax，再乘 $V$。直接物化完整 score/probability 矩阵会产生 $N\times N$ 的中间数据及相应 HBM 写读。FlashAttention 用 tile 遍历 K/V 块，在片上保存当前 query tile 的运行最大值、softmax normalizer 和输出累加；合并新块时重标定已有统计量，因此不必把完整注意力矩阵写入 HBM，并保持精确注意力计算（浮点舍入顺序可能不同）。

它仍然要读取输入 Q/K/V 并输出结果；实际 HBM 流量受 tile、片上存储预算、head dimension、causal/sparse mask、反向实现和硬件影响。论文的 IO 复杂度结论有其 SRAM/问题规模假设，不能说“所有 HBM IO 都是 O(N)”或宣称端到端固定 3–4 倍。FlashAttention-2 还调整并行划分与工作分配；回答“为什么更快”时要说出论文版本和对应机制，不能把 FA1/FA2 或某个库的实现混为一谈。

可拿来追问的验证链是：对齐参考实现和数值容差 → 对不同序列长度/head_dim/batch 做正确性 → 分别测 kernel 与完整模型 → 查看显存峰值和 Nsight 指标 → 用服务请求指标确认端到端受益。短序列、很小 batch、不同融合/库基线或非 attention 主导的模型，可能无法显著获益。

解释 FlashAttention 的优化原理时，可以组织为以下完整表述：

> 标准稠密注意力需要计算各个 Query 与可见 Key 的相关性。FlashAttention 将这些计算分块，并用在线 Softmax 合并每行的最大值、指数和与输出累加，因此无需在显存中保存完整的分数和概率矩阵。它减少的是中间结果的存储与读写，稠密注意力的主要计算量仍随序列长度呈二次增长。实际收益取决于分块、片上资源和对照实现，应在相同形状、精度与掩码条件下验证。

这是一段算法说明；若继续介绍个人实现，应另外说明所用版本、代码改动及实际测量结果。

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

## 现场推导与工程追问

以下题目用于检验前文概念能否落实到计算和决策，并非某家公司的固定题库。先独立写出假设、维度和结果，再阅读解释；回答中出现的测量数字必须来自自己的实验。

### GQA 的缓存减少了多少，为什么显存没有同比下降

假设模型有 32 层、32 个 Query heads、8 个 KV heads，每个 head 维度为 128。4 条序列均缓存 8192 个 token，K/V 都用 BF16，暂不计分页和其他开销。

代入前文公式：

$$
M=2\times32\times4\times8192\times8\times128\times2
 =4\,294\,967\,296\ \mathrm{bytes}=4\ \mathrm{GiB}.
$$

如果改为 32 个 KV heads，其他条件不变，缓存为 16 GiB。减少的是这部分有效 KV 数据，不是模型权重、激活、工作区和预留缓存池；因此进程总显存不会自动降为四分之一。若按张量并行分片，还应核对 KV head 是否可整除并行数，以及框架是否复制了部分 heads。

进一步追问：分页块长为 16 时，长度为 17 的单条序列需要两个块，当前占用 32 个 token 槽，其中 15 个尚未使用。实际服务应逐请求计算向上取整，不能把“每请求最多浪费 15 槽”当作所有请求的平均浪费。算法关系见 [GQA 原论文](https://arxiv.org/abs/2305.13245)，块管理见 [PagedAttention 论文](https://arxiv.org/abs/2309.06180)。

### 两个 Attention 分块的输出能否直接取平均

不能。块 A、B 分别保留最大分数 $m_A,m_B$、指数和 $\ell_A,\ell_B$ 和未归一化 Value 累加 $u_A,u_B$。令 $m=\max(m_A,m_B)$，则

$$
\ell=e^{m_A-m}\ell_A+e^{m_B-m}\ell_B,\qquad
u=e^{m_A-m}u_A+e^{m_B-m}u_B,\qquad o=u/\ell.
$$

两个块的指数基准不同，必须先统一基准，再合并分母和分子。例如 A 只有分数 0、Value 2，B 只有分数 $\ln3$、Value 10，最终输出为 $(2+3\times10)/(1+3)=8$，不是两个块输出的算术平均 6。

全遮挡块需要单独处理：其归一化和为零，不应直接计算 $-\infty-(-\infty)$。至于整行没有可见 Key 时返回零、NaN 还是报错，应服从算子约定，并与参考实现对齐。推导和分块实现见[在线 Softmax](../algorithms/online-softmax.md)与 [FlashAttention](../algorithms/flash-attention-mechanism.md)。

### MLA 为什么不能把缓存公式直接乘二

普通 K/V 是两份张量；MLA 的内容 Key 和 Value 可以从同一个压缩向量恢复。假设每层缓存 512 个内容元素和 64 个位置元素、均为 BF16，则每 token 每层的有效数据是 $(512+64)\times2=1152$ 字节，而不是 $2\times512\times2+64\times2$。这是指定表示下的计算例子，不代表某个部署版本的实际分配。

追问不能停在容量：如果直接恢复所有历史 K/V，再执行普通 Attention，会增加哪些临时张量和读写？权重吸收为什么要求线性变换，位置旋转为什么要单独讨论？回答时应给出 $W^{UK}:[d_k,d_c]$、$q:[d_k]$ 和压缩缓存 $c:[d_c]$，确认 $(W^{UK})^Tq$ 的维度是 $d_c$。对应推导见 [MLA 课程笔记](../algorithms/mla-deepseek.md)。

### 投机解码为什么不是“目标模型概率最高就接受”

在随机采样设定下，令目标分布为 $p$、草稿分布为 $q$。从 $q$ 提出的候选 $x$ 按 $\min(1,p(x)/q(x))$ 接受；发生拒绝时，从归一化后的 $\max(p-q,0)$ 重新采样。它与“候选是否等于目标 argmax”的贪心验证不是同一个问题。

用只有 A、B 两个 token 的例子检验：$q=(0.8,0.2)$、$p=(0.5,0.5)$。A 的接受概率为 0.625，B 为 1；直接接受的概率质量分别是 0.5 和 0.2，剩余 0.3 在拒绝后补给 B，最终分布正好是 p。如果拒绝后仍直接从 p 采样，结果就会改变。

工程上还要确认温度、Top-K/Top-P 等处理后的分布一致，以及拒绝后的 KV 回滚、位置编号和 RNG 状态。分布等价不要求在任意不同实现中逐次产生相同 token。算法依据为 [Speculative Decoding 原论文](https://arxiv.org/abs/2211.17192)。

### 权重文件变小了，为什么 W4A16 反而更慢

先区分存储格式和计算格式。低位宽权重可能需要解包、加载分组 scale、反量化，再进入矩阵计算；这些成本未必能被节省的访存抵消。小矩阵还可能受 launch 或调度开销主导。比较时固定模型、输入、输出长度、质量要求和计时范围，再检查反量化是否融合、布局是否适合目标 kernel，以及实际调用了哪条指令路径。

理解缩放算法可以从 $Y=XW$ 出发。令可逆对角矩阵为 D，则 $(XD^{-1})(DW)=XW$；量化前这是等价变换，分别量化两侧后则不再保证数值完全一致。某个通道的激活范围被压缩时，相应权重范围会扩大，校准需要评估两侧误差，而不是只追求激活最大值变小。与此相关的原始方法见 [SmoothQuant](https://arxiv.org/abs/2211.10438) 和 [AWQ](https://arxiv.org/abs/2306.00978)。

### 吞吐提高而 P99 恶化，是否应上线

缺少目标和负载时不能回答。先说明指标是请求完成时间、TTFT、TPOT 还是单个 token 的间隔，再区分输入到达率、请求长度分布、成功率和超时策略。更大的 batch 可能提高设备利用率，同时延长排队或单轮执行；只统计完成请求，还可能漏掉已经超时的慢请求。

可以先固定到达过程和样本集，对照排队时间、Prefill、逐轮 Decode 和 KV 传输的 trace。若瓶颈来自长 Prefill 插入 Decode，考虑分块与调度；若来自 KV 容量不足和反复抢占，先检查缓存与并发；若来自跨资源池传输，则评估数据量和网络拥塞。不同原因需要不同实验，不能先决定用 PD 分离再寻找理由。

### ZeRO-3 分片以后，为什么仍然会 OOM

分片后的持久模型状态只是峰值显存的一部分。计算某层时可能要聚合参数，预取可能让下一层同时驻留，此外还有激活、通信 buffer、算子工作区、allocator 保留空间和未释放引用。应按执行时间线统计同时存活的对象，不能把整模型账本除以数据并行数就当作峰值。

现场可以先画出一层的“参数聚合—前向/反向—梯度归约—释放”过程，再解释预取深度与显存的权衡。检查工具应能够区分已分配与已保留空间，并把峰值定位到具体阶段。状态分片的基本语义见 [ZeRO 论文笔记](../../papers/training/zero-paper.md)；实际聚合和释放时机仍需结合框架版本。

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
