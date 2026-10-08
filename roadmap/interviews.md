# GPU 高性能算子与 LLM 系统面试准备

本文面向已有 NPU 算子经验、准备转向 GPU 高性能算子与大语言模型（LLM，Large Language Model）性能优化的工程师。复习时以 CUDA/Triton 实现、性能分析和真实项目为主，模型结构及推理问题结合[LLM 面试笔记](../notes/llm/interview.md)展开。

## 复习原则

回答一个优化问题时，沿着“工作负载与正确性要求 → 瓶颈假设 → 代码改动 → 可观测证据 → 反例与边界”连续展开。先说明输入形状（shape）、数据类型（dtype）、布局、误差容限和测量范围，再解释为何修改某段代码；最后说明用哪一种性能分析器（profiler）或基准验证，以及哪些条件会改变结论。数字只引用注明设备、版本、shape、精度和计时口径的真实记录；没有实验的内容标为待验证假设，不用峰值规格替代实测。

学习材料、LeetGPU 通过、真实设备验证、岗位要求和个人工作经历是不同证据。课程覆盖只说明仓库有相应教材；不能据此声称读者已掌握或已满足招聘条件。

## 能力与证据定位

2026-09-25 查阅的公开职位反映了几种不同的工作侧重。以下比较用于选择准备深度；职位可能更新，具体年限和地域要求以招聘页面为准。

| 公开岗位样本 | 主要技术要求 | 准备时应突出什么 |
|---|---|---|
| [NVIDIA：CUTLASS Kernels](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Software-Engineer--CUTLASS-Kernels-_JR2018988) | Tensor Core 算子、C++/Python DSL、计算机体系结构与汇编 | 分块与布局、矩阵指令、编译资源及不同形状下的性能 |
| [NVIDIA：AI Inference Performance](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Software-Engineer---AI-Inference-Performance_JR2024262) | 模型到服务的性能分析、可复现基准、推理引擎和通信 | 局部优化如何改变端到端指标，如何设置性能回归检查 |
| [腾讯：高性能算子优化](https://careers.tencent.com/jobdesc.html?postId=2072509101644103680) | CUDA/CUTLASS/Triton、集合通信、通信计算重叠 | 计算与通信的依赖关系、并行组织、正确性和工具证据 |
| [Inferact：Kernel Engineering](https://jobs.ashbyhq.com/inferact/384d9db8-c712-4caa-8091-444b4189e161) | GPU kernel、性能分析、低精度及推理引擎贡献 | 自写算子的实现细节、基准质量、与引擎的接入方式 |
| [Anthropic：Performance Engineer, GPU](https://job-boards.greenhouse.io/anthropic/jobs/4926227008) | 自定义算子、框架、分布式和大规模性能优化 | 如何跨层定位问题，以及优化在真实负载中的效果 |
| [Tilde：Kernel Engineer](https://jobs.ashbyhq.com/tilderesearch/bc4e4071-cf64-4460-8265-b1e5a603d6b8) | 算法能力、GPU 编程、技术表达及可展示产物 | 能解释的实现、公开贡献或技术记录，而非只列工具名 |

据此，算子岗位应重点准备线程与数据布局、同步、数值和性能分析；推理岗位还需说明模型状态、调度与请求指标。偏编译器或分布式的职位，则应进一步准备中间表示、代码生成或通信协议。下面将这些要求映射到课程中的具体材料。

| 能力 | 对应课程 | 面试中应能说明或展示 |
|---|---|---|
| 线程、存储与同步 | [GPU 与 CUDA](curriculum/gpu/README.md) | 具体线程到地址的映射、同步范围、资源限制，以及错误定位 |
| GEMM 与 Tensor Core | [GEMM 实现与优化](curriculum/operators/03-gemm/README.md) | tile、布局、累加精度、流水；同精度强基线和退化案例 |
| 性能诊断 | [计时与性能分析](curriculum/gpu/05-performance-analysis-and-optimization/README.md) | 时间线、硬件指标、PTX/SASS、重复测量和跨形状回归 |
| 框架算子接入 | [Mini Transformer 算子替换](curriculum/model-analysis/mini-transformer/README.md) | schema、FakeTensor、注册检查、数值与梯度、编译组合 |
| 量化算法与 kernel | [量化算子](curriculum/operators/07-quantized-operators/README.md) | scale、打包、校准与误差补偿；同模型的质量、容量和速度对照 |
| Prefill 与 Decode | [模型 GPU 执行分析](curriculum/model-analysis/README.md) | 从形状推 FLOPs/bytes，区分权重、KV、算子和请求时间 |
| 真实推理系统 | [vLLM 与系统应用](curriculum/systems/README.md) | 找到实际后端，解释缓存与调度，并验证端到端收益 |
| 论文推导与代码 | [论文与算法](../notes/algorithms/README.md) | 符号、维度、状态更新与作者实现的对应关系和适用条件 |
| C++/Linux 调试 | [执行与编译](curriculum/gpu/02-cuda-execution-and-scheduling/README.md) | 最小复现、编译/链接/装载错误、主机调用栈与设备错误定位 |
| 跨平台迁移 | 上述算子案例与本人 NPU 项目 | 两端共同的数学要求、不同的执行/存储/同步机制与真实测量 |
| 多 GPU 与专项 | [多 GPU](curriculum/systems/multi-gpu/README.md) | 根据岗位准备 collective、拓扑、负载与通信；训练/编译器开发另按岗位深入 |

本项目先形成 GPU 算子和性能分析能力，再用量化、模型接入及真实系统证明其作用。至少准备三项能完整解释基线、改动、正确性、性能和失败尝试的算子案例。论文阅读独立进行，不要求每篇都实现；岗位中的工作年限、生产经历和语言要求另行核对。

## GPU 架构与算子优化

### CUDA 线程、分歧与访存合并

CUDA kernel 按 grid、block 和 thread 表达并行工作。NVIDIA GPU 通常以 32 条 lane 构成 warp；warp 是调度与协作的重要粒度，但不能因此把所有指令行为简化为“32 线程永远锁步”。同一 warp 的线程若走不同控制路径，路径可能需要分别执行，有效 lane 利用率下降；实际影响取决于分支、编译生成代码和工作量。短分支可能被谓词化，未必比显式分支更好，应检查 PTX/SASS 并测量。

Coalescing 按同一条 warp 访存指令的地址合并请求。连续 32 个线程各读取一个 4 字节 float，总有效数据为 128 字节；起始地址按 32 字节对齐时，在 32 字节 sector 模型下覆盖 4 个 sector，错开起点则可能覆盖 5 个。散乱地址可能增加请求数和未使用字节，最终耗时还受缓存、访问宽度、并行请求数和设备影响。

面试中可以从地址式入手：FP32 读取的字节地址为 `base + 4*lane` 时，相邻 lane 读取相邻元素；行主序矩阵沿行方向跨步时，则应把行跨度代入公式。随后检查生成的访存指令、DRAM/L1/L2 流量和 sector 指标。数据已在缓存、规模很小或瓶颈不在访存时，减少请求数量未必缩短整体时间。

共享内存的 bank conflict 来自一次访问中对存储 bank 的竞争。对典型的 32 位字访问，连续字分布到不同 bank；多个 lane 读取同一 bank 的不同地址时，可能需要分次服务。同地址广播、宽类型和向量指令需要另按实际访问规则分析。矩阵转置中的 padding 或 swizzle 通过改变地址映射减少竞争，应再用冲突指标与耗时确认效果。[CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory)提供了相应的访问模型。

### 延迟、Occupancy 与性能指标

Latency hiding 的机制是：当一个 warp 等待数据或依赖时，SM 调度器可以选择其他就绪 warp 发射指令。Occupancy 描述活动 warp 相对硬件上限的比例；它受寄存器、shared memory、线程数和架构约束影响，但不等于吞吐率。高 occupancy 可能有助于隐藏延迟；增加 tile、独立累加器或 pipeline stage 也可能提高复用/指令并行度，同时消耗更多寄存器或 shared memory、降低驻留并引发 spill。

因此不能用单一 occupancy 或 SOL 百分比直接判定瓶颈。先用 Nsight Systems 找到端到端关键路径、launch 间隙、并发和拷贝；再用 Nsight Compute 结合计算管线、DRAM/L2 流量、指令与 stall 指标解释目标 kernel。将实测耗时与一个有明确假设的 Roofline/带宽下界比较，并对多种代表 shape 重复测量。Memory SOL 低可能是访问没有形成足够并行请求、工作量太小、cache 命中或其他原因，不能直接翻译成“memory-bound”；Compute SOL 低也不自动表示“latency-bound”。SOL 是诊断线索，不是单指标分类器。

### GEMM、Tensor Core 与整数点积

对 $A\in\mathbb{R}^{M\times K}$、$B\in\mathbb{R}^{K\times N}$，GEMM 计算 $C=AB$。朴素实现让每个输出元素重复读取输入；分块实现使 CTA/warp 复用子块，再沿 K 维累加。回答优化题时应说清线程/warp 如何映射到 tile，global→shared/register 的搬运和复用，累加精度，边界 mask，以及 M/N/K 变化如何影响资源与尾块。

Tensor Core 是矩阵乘加数据路径，不是任何 FP16/INT8 代码都会自动使用的标签。是否走该路径取决于目标 GPU、输入/累加 dtype、tile/layout、指令/API、对齐及编译器选择。验证应查看编译产物或 profiler 中的 MMA/Tensor 指标，并与允许误差的合同一致。NVIDIA 的 [`__dp4a` 文档](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html)将其定义为四对打包 8-bit 整数的点积并累加到 32-bit 结果；它不应说成 INT8 Tensor Core 路径。整数点积指令和 Tensor Core MMA 是不同指令/数据路径。整数 kernel 是否更快，还受反量化/缩放、布局、饱和规则、算子融合和目标设备支持影响。

### Attention 与 FlashAttention

标准 attention 的核心是 $S=QK^T/\sqrt{d}$、$P=\mathrm{softmax}(S)$、$O=PV$。朴素实现可能把 $N\times N$ 的分数或概率矩阵写到 HBM；FlashAttention 通过分块和 online softmax 在片上维护每行的运行最大值、归一化和输出累加，避免将完整中间矩阵物化到 HBM。它仍须读取 Q/K/V 并写 O，且 Q/K/V 的重读、tile 大小、SRAM 预算、因果 mask、反向重算和具体实现都会影响实际 HBM 流量。因此不能说“FlashAttention 的所有 HBM IO 都是 O(N)”或由此推出端到端固定倍数加速。应给出确切的 IO 模型假设，并测量对应 shape、dtype、head dimension、batch、设备和基线。

追问“改哪段”时，说明你改变的是 score/probability 的中间存储和 tile 循环，而不是注意力数学定义；若讨论 FlashAttention-2，再指出 work partitioning 对并行度、共享内存读写和 warp 间工作分配的调整。论文推导、论文伪码和所读代码版本要对应，不能把读论文说成已在 GPU 复现。

### Triton、CUDA 与 PyTorch 接入

Triton 提供块级张量编程与 JIT 编译；开发者写 tile 计算和指针表达式，编译器负责将逻辑张量映射到线程/warp，并选择/生成相应指令与数据布局。它不会对任意程序自动保证 coalescing、Tensor Core 使用或接近某固定百分比的峰值。面试可沿一个 MatMul 例子解释 `program_id`、tile、`tl.load` mask、`tl.dot`、编译期配置、layout 和生成代码，然后在多 shape 上与 PyTorch/cuBLAS 基线比较正确性和耗时。

Triton 的表达效率适合快速试验分块和融合，但运行性能仍取决于输入形状、布局、编译器和对照实现。接入 PyTorch 时，还需声明算子接口与输入修改行为，检查设备和类型，提供编译期间的形状推导，并按需实现反向传播。Python/Triton 接口与 C++/CUDA 扩展各有适用场景，应结合项目的构建、部署与版本要求选择。

### 什么时候讨论 Persistent Kernel

Persistent kernel 是让有限数量的 CTA/warp 长时间驻留并循环处理多批工作的一类调度方式；工作分配可在 GPU 侧完成，也可以由 kernel 的 program 结构预先分派，并不必然是“CPU 常驻工作队列”。Triton 的[教程目录](https://triton-lang.org/main/getting-started/tutorials/)把 Persistent Matmul 列作一种特定实现示例，而非通用调度要求。它可能摊薄重复 launch 或调度开销、改善特定工作负载的流水，但也可能占用 SM 资源、降低并发、造成负载不均或增加同步复杂度。是否适合取决于 grid 大小、任务粒度、设备驻留、其他 kernel 并发和服务延迟要求。对普通单次启动的 GEMM，persistent 主要改变 CTA 的工作分配与 tile 复用，并不自动减少 kernel 启动次数。应针对实际假设比较驻留、负载均衡、缓存流量和端到端时间。

## LLM 模型与推理性能

### Prefill、Decode 与 KV cache

Prefill（提示词阶段）一次处理多个输入位置并写入各层 KV cache；Decode（逐 token 阶段）通常每步处理一个新位置并读取历史 KV。Prefill 往往有较大的矩阵工作量，Decode 则更容易受到权重/KV 读取、batch、缓存和 launch 的影响，但真正的瓶颈仍由 shape、并发和实现路径决定。首 token 时间（TTFT，Time To First Token）包含排队、调度和 Prefill；每输出 token 时间（TPOT，Time Per Output Token）描述生成阶段的平均步进成本。应以请求级时间线和代表性的 batch、prompt length、decode length 组合验证，并同时报告吞吐与尾延迟。

对没有前缀共享、长度均为 S 的 B 条序列，L 层普通 MHA/GQA 的 KV 有效数据量为

$$
B_{KV}=2\,L\,S\,H_{KV}\,D\,B\,b.
$$

其中 $L$ 为层数，$S$ 为缓存 token 数，$H_{KV}$ 为 KV head 数，$D$ 为 head dimension，$B$ 为 batch/序列数，$b$ 为每元素字节数，系数 2 对应 K 和 V。分页、对齐、元数据、不同序列长度和量化 scale 会改变真实分配。GQA 在 head dimension 与序列长度等条件相同的情况下，将 KV heads 从 $H_Q$ 减少为 $H_{KV}$，理想 KV 元素数按 $H_{KV}/H_Q$ 缩小；没有模型配置时，只能说明这个比例，不能代入固定 MiB 数。

Decode 每步也不总是“完整读取全部权重”。单序列、较低 batch 且权重远大于 cache 时，权重流量常是重要项；但批处理复用、cache 层级、并发与实现会改变从 HBM 实际读取的字节。可先在“相关权重各读取一次”的假设下估算字节数，再用实际流量验证；MoE 则按该批次涉及的共享层和专家计算，不能直接把总参数容量当作每步读取量。

PagedAttention 通过块表将逻辑 token 块映射到物理 KV 块，降低连续大块预分配带来的碎片并支持灵活调度/共享；block size、复制/写时共享语义和调度策略由具体实现决定。Continuous batching 在迭代边界动态调整活动请求；其效果应在相同请求分布下用 TTFT、TPOT、吞吐、公平性和 P95/P99 验证。

回答“为什么 kernel 加速后，服务吞吐没有提高”时，可以先给出机制，再说明验证方法：

> 服务吞吐取决于请求处理过程中限制整体速度的环节。若优化的 kernel 占比较小，或节省的时间被布局转换、KV 搬运和调度等待抵消，局部加速就难以转化为服务收益。应先比较算子输出，再核对模型对应位置的 logits，随后在相同请求分布下检查时间线、TTFT、TPOT 和吞吐。局部算子的加速结果仍然有效，但其适用范围应与请求级结果分别陈述。

这段表述解释分析方法；用于项目回答时，再补充本人实际修改的位置、测量条件和结果。

### 量化：方法、误差与部署性能

量化答案至少区分权重与激活精度（如 W4A16、W8A8）、静态/动态 scale、per-tensor/per-channel/per-group 粒度、校准数据、误差指标和运行时 kernel。AWQ 是利用激活统计识别敏感权重通道并选择缩放策略的训练后权重量化方法；“activation-aware”不是“训练激活量化”，也不意味着它执行 QAT。GPTQ 是基于近似二阶信息逐步量化并补偿误差的训练后方法。SmoothQuant 通过等价缩放把激活 outlier 带来的量化难度迁移到权重，以支持 W8A8；方法效果取决于模型、校准数据、粒度和部署实现。

低 bit 权重可减少存储和潜在读取量，但解包、scale 处理、反量化、矩阵形状、batch 与硬件指令会改变吞吐；W4A16 不必然只适合小 batch，W8A8 也不必然只适合大 batch。量化评估要在同模型/任务集和相同请求分布下记录质量退化、显存、kernel/端到端延迟及吞吐，并确认运行路径真正使用目标低精度指令。训练量化方法需要按目标说明 QAT、PTQ 或混合流程，不能断言量化一定需要/不需要完整微调。

### Speculative decoding 与 PD 分离

Speculative decoding 用 draft 模型提出候选 token，再由 target 模型验证；接受/拒绝机制可保持目标分布（在算法前提成立时）。速度取决于 draft 成本、候选长度、接受率、目标模型批处理效率和调度开销，不能引用固定加速倍数或只凭同家族关系断言有效。测量需同时报告输出质量/分布检查、接受率、吞吐和延迟。

Prefill-Decode 分离把两阶段放在不同资源池以隔离服务目标或资源特征；是否值得取决于负载混合、调度、资源利用和 KV 传输成本。KV 搬迁时间受传输字节、链路拓扑、拥塞、协议和同步影响，不能说固定“远小于 prefill”。需要在真实部署上比较 TTFT/TPOT、尾延迟、吞吐、GPU 利用率及迁移流量。

## 分布式训练：回答时先声明假设

通信进程组为每个参与进程分配一个 rank 编号；collective 是所有相关 rank 按相同顺序共同完成的通信操作。Data Parallel（DP，数据并行）、Tensor Parallel（TP，张量并行）、Pipeline Parallel（PP，流水线并行）和 Expert Parallel（EP，专家并行）切分的对象与通信路径不同：DP 复制模型并同步梯度，TP 切分层内矩阵并交换部分结果，PP 按层分段传递激活，EP 为 MoE 专家分片并传递 token。并行选择取决于模型形状、显存、拓扑、batch/sequence、通信实现和目标延迟，不能套用固定的 3D 配方。

ZeRO/FSDP 按阶段切分 optimizer state、gradient 和 parameter，可减少每 rank 的持有量，但实际节省与参数/梯度/optimizer dtype、临时 buffer、激活、通信 bucket、预取和 checkpoint 策略有关。应先画单 rank 显存核算表，再说明一个同步窗口内何时 all-gather/reduce-scatter，并用实际配置与 trace 校验。Ring all-reduce 的每 rank 传输量可在经典 ring 算法假设下推导为约 $2(P-1)/P$ 个 payload 大小（$P$ 个 rank），但 NCCL 可按消息大小、拓扑和配置选择不同算法/协议，理论传输量不证明链路已饱和或通信必为瓶颈。

混合精度、loss scaling 与 activation checkpointing 的取舍也要注明训练目标和框架策略。checkpointing 以重算换激活存储，重算比例取决于选择哪些节点及实现；不能把某个理论渐近估计或常见开销百分比当作所有模型的实测。Nsight Systems、框架 profiler、通信库日志和 step time 一起回答“计算还是通信限制”。若应聘方向是 GPU kernel/inference，分布式训练可按岗位要求准备，不要让它替代 CUDA/Triton 核心实践。

## 求职场景中的问题与推理

公开面试经历可帮助了解问题如何展开。例如，一篇[算子开发实习面试自述](https://www.nowcoder.com/feed/main/detail/57033fe8898b4e85b97c48d8b2dcfb28)涉及矩阵乘、Softmax、bank conflict 和缓存；一篇[GPU 岗位技术筛选自述](https://www.reddit.com/r/qualcomm/comments/1sebf82/qualcomm_finished_gpu_screening_round_how_to/)涉及体系结构、存储层次与量化。这些是个人报告，无法据此断言某家公司固定考什么。以下练习由岗位要求、课程内容和官方技术说明综合设计，答案不采用面经中未经核实的硬件解释。

### 给定线程地址，如何判断访问是否合并

设 32 个 lane 各读取一个 FP32 元素，起始地址按 32 字节对齐。地址为 `base+4*lane` 时，128 个有效字节覆盖 4 个 sector；若整体偏移一个 float，范围变成字节 4–131，就覆盖 5 个 sector。换成 `base+128*lane`，每个 lane 落入不同 sector，这条 warp 指令需要覆盖 32 个 sector。

推导时先写地址集合，再计算它覆盖的事务单元；线程编号连续只是形成合并访问的一种方式。若这些地址命中缓存，实际 DRAM 流量又与请求覆盖范围不同。这一练习考查地址推理，不能仅用“行访问快、列访问慢”作答。访问模型见 [Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)。

### 增大 tile 后复用更多，为什么反而变慢

较大的输出 tile 使输入参与更多乘加，同时需要保存更多累加值；流水 stage 增加还会占用更多共享内存。若因此减少驻留 CTA、发生寄存器溢出，或让尾部无效工作增加，减少的访存未必足以补偿其他成本。

回答应明确改的是输出 tile、归约 tile 还是流水深度，再对照寄存器、共享内存、有效 warp、访存流量与耗时。已有 GEMM 实验中出现过 tiled 慢于 naive 的结果，可以用来说明优化需要验证；缺少计数器时，应将缓存与占用率解释列为假设，而不把它们当成已经确认的原因。

### 归约长度不是 block 大小的整数倍，尾部如何处理

先明确归约的数学单位元。无效位置在求和中贡献 0，在最大值归约中贡献负无穷。若后续会读取这些线程负责的共享内存位置，就需要先写入单位元，再按协作协议同步。归约范围、内存初始化和同步参与范围是三个分别检查的问题。

追问 warp 级实现时，还应说明参与掩码与源 lane 的有效性。`__shfl_sync` 用于寄存器交换，不替代通过共享内存交接数据所需的内存顺序；跨 warp 协作还需选择覆盖整个协作组的同步。具体规则见 [CUDA 线程协作原语](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)。

### 程序只在某些输入上报非法访问，应如何定位

保留最小失败形状和固定输入，先检查索引、stride、对齐、尾块及输出容量，再将首次错误与对应 kernel 联系起来。CUDA 启动通常异步，同步调用可能只是报告先前错误的位置。应同时检查启动返回状态和执行完成状态。

工具也要对应问题类型：Compute Sanitizer 的 `memcheck` 检查越界和未对齐访问，`racecheck` 主要检查共享内存访问冲突，`initcheck` 检查未初始化的设备全局内存读取，`synccheck` 检查同步原语的非法使用。某一工具通过，不代表所有竞争或数值错误都被排除。依据：[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。

### 普通计时与 Nsight Compute 的结果为何不一致

先对齐测量范围，再检查采集方式。Nsight Compute 可能为采集不同指标重放 kernel，并改变缓存、频率或并发条件；这些行为会使测量与应用的自然执行不同。在 profiler 运行期间包围程序的主机计时或 CUDA Event，也可能包含采集开销。

应保存一份不启用 profiler 的稳定基准，用 Nsight Systems 观察整体时间线，再对选中的 kernel 收集指标。比较不同配置时保持重放与缓存设置一致；需要研究真实缓存预热时，按工具支持的重放模式设计实验。依据：[Nsight Compute 的计时与可复现性说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。

### 同为 FP32，为何两个矩阵乘结果不一致

FP32 存储并不能独立决定乘法输入精度。应检查是否启用了 TF32、累加类型、FMA、归约顺序和输出转换，并排除索引或边界错误。比较时固定输入和容差，分别记录最大绝对误差与相对误差。

项目中的 Triton MatMul 保留了默认精度失败与 `input_precision='ieee'` 通过的两份记录，适合解释定位过程。回答时应说清修改改变了哪条数值路径，以及性能对照是否也采用同一精度要求；单凭输出 dtype 相同，不能将二者视为公平的性能对照。

### Kernel 正确，接入 torch.compile 后为何失败

独立调用只覆盖设备计算的一部分要求。编译器还需要算子 schema、输入输出别名与修改信息，以及 FakeTensor 下的形状、类型和设备推导。带梯度的算子还需正确注册反向传播，并检查保存状态与输入生命周期。

`torch.library.opcheck` 检查注册契约和相关组合行为，不证明梯度公式数学正确；梯度仍需独立参考或 `gradcheck`。应进一步覆盖非连续输入、边界尺寸、动态形状及目标编译路径。依据：[PyTorch 自定义算子教程](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)。

### C++ 对象离开作用域后，GPU 能否继续使用它的内存

主机函数返回与 GPU 工作完成是两个时刻。即使主机侧用 RAII 管理资源，若析构发生在异步使用结束之前，仍可能产生生命周期错误。需要说明指针由谁拥有、在哪个 stream 使用，以及释放如何依赖最后一次设备访问。

同样，另一个 stream 消费结果前，需要正确的事件或其他依赖关系。面试中可以画出“分配—生产—消费—释放”的顺序，再解释哪些操作由流内顺序保证、哪些需要跨流连接，而不是给每次调用都加全设备同步。对应代码在[同步与异步执行课程](curriculum/gpu/04-synchronization-and-asynchronous-execution/README.md)。

### 两卡张量并行应该拼接结果还是求和

考虑 `Y=XW`。若按 W 的输出列切分，各卡得到不同输出列，可以拼接成 Y；若按归约维切分 X 和 W，各卡得到同形状的部分和，需要求和。先根据矩阵维度推导输出语义，再选择 AllGather、AllReduce 或 ReduceScatter。

若每张卡都给部分和加上完整 bias，归约后就会重复添加。KV head 数较少时，也可能发生跨卡复制，不能仅用总 KV 容量除以卡数估算每卡占用。推导与代码见[多 GPU 课程](curriculum/systems/multi-gpu/README.md)。

### 没有 NVIDIA 生产经历，如何说明 NPU 经验的价值

选择一个亲自做过的算子，说明输入形状、数值要求、数据分块、片上复用、同步及实际工具结果，再解释迁移到 GPU 后哪些机制需要重新验证。能够清楚解释一次失败优化及其后续定位，通常比罗列硬件名词更有信息量。

简历与回答中的工作经历、个人实验和论文分析应注明来源。可展示已有代码、性能记录、复现步骤和明确的后续问题；尚未完成的服务器实验保留为计划。岗位对工作年限或生产经验的要求，需要在投递时单独核对。

## 系统设计与项目回答

设计推理服务时，先问清模型结构/权重格式、prompt 与输出长度分布、并发、吞吐目标和 TTFT/TPOT/P99 SLO；再算权重、KV、临时空间和副本数，设计 batching/KV 管理/量化/并行拓扑；之后用请求 trace 和压测调度。容量估算是初始约束，不是已实现的吞吐承诺。训练集群设计则从模型与优化器状态账本、激活、checkpoint、拓扑和容错约束出发，比较 TP/PP/DP/EP，并验证通信与 step time。

讲一个 kernel 项目时至少准备：基线定义、功能与精度约定、最初瓶颈证据、每次代码改动、正确性结果、多 shape benchmark、profiler 支持的解释、失败尝试和未解决问题。只报告实测数字；benchmark 的输入、GPU/计算能力、软件版本、warmup/repeat、计时方法和精度必须随结果给出。性能表里存在多个配置时不把不同时次/条件拼成单一曲线。

### Ascend NPU → GPU 迁移经历

可解释两端都需要权衡数据搬运、复用、并行度和流水，但 API、memory hierarchy、执行粒度、编译器和 profiler 不能一一等同。Cube 与 Tensor Core、L1/UB 与 shared memory、pipe 与 CUDA pipeline 可作为“功能目标相似”的比较起点，不是实现完全相同的证明。回答迁移经验时说出亲自完成的代码、具体 shape/dtype、验证与测量条件；若尚未在目标 GPU 完成实验，明确说这是原理映射或待验证工作，不编造上手时长、项目年限、性能提升和峰值比例。

可直接使用的自我介绍框架：

> 我在【真实平台与时间范围】做过【具体算子/系统任务】。迁移到 CUDA/Triton 时，我把问题拆成【计算/访存/同步/系统层机制】，用【仓库或个人项目中的代码与工具】验证了【真实结果及完整条件】。目前我还在补【具体缺口】，其中【NPU 与 GPU 的一个相同目标、一个不同机制】是我能用代码和实验说明的。

方括号必须由本人事实替换。若没有某项实绩，删除对应句子，不把模板当作个人经历。

## 参考资料

- NVIDIA, [CUDA C++ Programming Guide: coalesced global memory access](https://docs.nvidia.com/cuda/cuda-programming-guide/)。用于核对事务如何依 warp 地址请求合并；事务模型和硬件行为应按目标架构版本阅读。
- NVIDIA, [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。SOL/Roofline 与详细指标属于诊断材料，需结合 workload 和其他指标解释。
- Triton, [Matrix Multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) 与[教程索引](https://triton-lang.org/main/getting-started/tutorials/)。用于查看 tile、program 重排、调优等真实可运行例子，不构成跨场景性能保证。
- PyTorch, [Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)。用于核实 custom op schema、注册、测试及扩展构建路径。
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135)；Dao, [FlashAttention-2](https://tridao.me/publications/flash2/flash2.pdf)。IO 复杂度和并行划分应依论文条件理解。
- Lin et al., [AWQ](https://arxiv.org/abs/2306.00978)；Xiao et al., [SmoothQuant](https://arxiv.org/abs/2211.10438)。分别核对激活感知的训练后权重量化与 W8A8 的等价缩放方法。
- 岗位样本及核对日期见本文开头的岗位对照表。招聘页面可能更新或撤下；这些样本用于区分准备方向，不代表市场频率、个人资格或投递情况。
