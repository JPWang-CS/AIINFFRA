# GPU 高性能算子与 LLM 系统面试准备

本文面向已有 NPU 算子经验、准备转向 GPU 高性能算子与 LLM 性能优化的工程师。复习时以 CUDA/Triton 实现、性能分析和真实项目为主，模型结构及推理问题结合[LLM 面试笔记](../notes/llm/interview.md)展开。

## 复习原则

回答一个优化问题时，沿着“工作负载与正确性要求 → 瓶颈假设 → 代码改动 → 可观测证据 → 反例与边界”连续展开。先说明输入形状、dtype、布局、误差容限和测量范围，再解释为何改某段代码；最后说清用什么 profiler/基准验证、还有哪些情形可能让结论失效。数字只引用注明设备、版本、shape、精度和计时口径的真实记录。没有实验的内容标为待验证假设，不用峰值规格替代实测。

学习材料、LeetGPU 通过、真实设备验证、岗位要求和个人工作经历是不同证据。课程覆盖只说明仓库有相应教材；不能据此声称读者已掌握或已满足招聘条件。

## 能力与证据定位

以下采用招聘方公开的两个岗位作为样本，用于核对能力要求，不据此统计市场频率。课程以 GPU 架构、CUDA/Triton 和算子优化为主干，量化与模型性能分析为核心应用，框架接入和真实系统验证用于展示工程效果。RadixArk 样本另有 4 年以上经验要求；岗位年限、地域等资格条件需单独核对，不能用课程学习替代。

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

Shared memory bank conflict 也不能只背“32 banks，所有冲突都串行”。对通常的 32-bit word 访问，bank 映射会让连续 word 分散到不同 bank；同一请求中多个 lane 访问同一 bank 的不同地址时，可能需要多个服务阶段。[CUDA 编程指南的 shared-memory 章节](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory)说明了 bank conflict 与矩阵转置例子；广播同一地址等访问有不同规则，宽数据、向量化与具体代际也会影响行为。矩阵转置中常用 padding 或 swizzle 改变 shared-memory 布局；随后确认冲突指标和总耗时，不能只凭代码形状断言已解决。

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

Triton 便于表达和迭代，但“开发速度提升 10 倍”或“固定达到 CUDA 的 93–95%”没有跨 workload 的普适依据。性能受 shape、dtype、layout、编译器版本、调优空间与基线影响。PyTorch 集成也不等于 kernel 函数能被调用：生产级 custom op 还涉及 schema/dispatch、设备与 dtype 检查、fake/meta 行为、Autograd、编译/打包、测试和版本兼容。根据目标需求选 Python/Triton 接口或 C++/CUDA custom operator，并用官方 `torch.library`/`cpp_extension` 路径与实际项目约束验证。

### 什么时候讨论 Persistent Kernel

Persistent kernel 是让有限数量的 CTA/warp 长时间驻留并循环处理多批工作的一类调度方式；工作分配可在 GPU 侧完成，也可以由 kernel 的 program 结构预先分派，并不必然是“CPU 常驻工作队列”。Triton 的[教程目录](https://triton-lang.org/main/getting-started/tutorials/)把 Persistent Matmul 列作一种特定实现示例，而非通用调度要求。它可能摊薄重复 launch 或调度开销、改善特定工作负载的流水，但也可能占用 SM 资源、降低并发、造成负载不均或增加同步复杂度。是否适合取决于 grid 大小、任务粒度、设备驻留、其他 kernel 并发和服务延迟要求。对普通单次启动的 GEMM，persistent 主要改变 CTA 的工作分配与 tile 复用，并不自动减少 kernel 启动次数。应针对实际假设比较驻留、负载均衡、缓存流量和端到端时间。

## LLM 模型与推理性能

### Prefill、Decode 与 KV cache

Prefill 通常有较大的序列维度和矩阵工作量，常见情况下计算并行度较高；Decode 每步生成少量 token，可能更受权重/KV 读取、batch、cache 和 launch 影响。但这是工作负载倾向，不是由阶段名称决定的硬约束。短 prompt、小 batch、长上下文、并发请求、权重驻留或融合都会改变瓶颈。用请求级时间线和代表性的 batch、prompt length、decode length 矩阵验证，并分别报告 TTFT、TPOT、吞吐与尾延迟。

对没有前缀共享、长度均为 S 的 B 条序列，L 层普通 MHA/GQA 的 KV 有效数据量为

$$
B_{KV}=2\,L\,S\,H_{KV}\,D\,B\,b.
$$

其中 $L$ 为层数，$S$ 为缓存 token 数，$H_{KV}$ 为 KV head 数，$D$ 为 head dimension，$B$ 为 batch/序列数，$b$ 为每元素字节数，系数 2 对应 K 和 V。分页、对齐、元数据、不同序列长度和量化 scale 会改变真实分配。GQA 在 head dimension 与序列长度等条件相同的情况下，KV heads 从 $H_Q$ 减少为 $H_{KV}$，理想 KV 元素数按 $H_{KV}/H_Q$ 缩小；不要在缺少模型配置时引用固定 MiB 数。

Decode 每步也不总是“完整读取全部权重”。单序列、较低 batch 且权重远大于 cache 时，权重流量常是重要项；但批处理复用、cache 层级、并发与实现会改变从 HBM 实际读取的字节。可先在“相关权重各读取一次”的假设下估算字节数，再用实际流量验证；MoE 则按该批次涉及的共享层和专家计算，不能直接把总参数容量当作每步读取量。

PagedAttention 通过块表将逻辑 token 块映射到物理 KV 块，降低连续大块预分配带来的碎片并支持灵活调度/共享；block size、复制/写时共享语义和调度策略属于具体实现。不要把某个历史版本的默认 block token 数或碎片百分比说成所有 vLLM 版本的固定事实。Continuous batching 在迭代边界动态调整活动请求，也不保证任意负载获得固定倍数吞吐提升；应用相同请求分布测 TTFT、TPOT、吞吐、公平性和 P95/P99。

### 量化：方法、误差与部署性能

量化答案至少区分权重与激活精度（如 W4A16、W8A8）、静态/动态 scale、per-tensor/per-channel/per-group 粒度、校准数据、误差指标和运行时 kernel。AWQ 是利用激活统计识别敏感权重通道并选择缩放策略的训练后权重量化方法；“activation-aware”不是“训练激活量化”，也不意味着它执行 QAT。GPTQ 是基于近似二阶信息逐步量化并补偿误差的训练后方法。SmoothQuant 通过等价缩放把激活 outlier 带来的量化难度迁移到权重，以支持 W8A8；方法效果取决于模型、校准数据、粒度和部署实现。

低 bit 权重可减少存储和潜在读取量，但解包、scale 处理、反量化、矩阵形状、batch 与硬件指令会改变吞吐；W4A16 不必然只适合小 batch，W8A8 也不必然只适合大 batch。量化评估要在同模型/任务集和相同请求分布下记录质量退化、显存、kernel/端到端延迟及吞吐，并确认运行路径真正使用目标低精度指令。训练量化方法需要按目标说明 QAT、PTQ 或混合流程，不能断言量化一定需要/不需要完整微调。

### Speculative decoding 与 PD 分离

Speculative decoding 用 draft 模型提出候选 token，再由 target 模型验证；接受/拒绝机制可保持目标分布（在算法前提成立时）。速度取决于 draft 成本、候选长度、接受率、目标模型批处理效率和调度开销，不能引用固定加速倍数或只凭同家族关系断言有效。测量需同时报告输出质量/分布检查、接受率、吞吐和延迟。

Prefill-Decode 分离把两阶段放在不同资源池以隔离服务目标或资源特征；是否值得取决于负载混合、调度、资源利用和 KV 传输成本。KV 搬迁时间受传输字节、链路拓扑、拥塞、协议和同步影响，不能说固定“远小于 prefill”。需要在真实部署上比较 TTFT/TPOT、尾延迟、吞吐、GPU 利用率及迁移流量。

## 分布式训练：回答时先声明假设

Data Parallel、Tensor Parallel、Pipeline Parallel、Expert Parallel 切分的对象和通信路径不同。DP 复制模型并对梯度做 collective；TP 切层内矩阵并频繁交换部分结果；PP 按层分段传激活，可能有 pipeline bubble；EP 为 MoE 专家分片并涉及 token dispatch。并行选择取决于模型形状、显存、拓扑、batch/sequence、通信实现和目标延迟，不存在适用于所有集群的固定 3D 配方。

ZeRO/FSDP 按阶段切分 optimizer state、gradient 和 parameter，可减少每 rank 的持有量，但实际节省与参数/梯度/optimizer dtype、临时 buffer、激活、通信 bucket、预取和 checkpoint 策略有关。不要背固定“4×/8×”或未经推导的通信倍数。可以先画单 rank 显存账本，再说明一个同步窗口内何时 all-gather/reduce-scatter，并用实际配置与 trace 校验。Ring all-reduce 的每 rank 传输量可在经典 ring 算法假设下推导为约 $2(P-1)/P$ 个 payload 大小（$P$ 个 rank），但 NCCL 可按消息大小、拓扑和配置选择不同算法/协议，理论传输量不证明链路已饱和或通信必为瓶颈。

混合精度、loss scaling 与 activation checkpointing 的取舍也要注明训练目标和框架策略。checkpointing 以重算换激活存储，重算比例取决于选择哪些节点及实现；不能把某个理论渐近估计或常见开销百分比当作所有模型的实测。Nsight Systems、框架 profiler、通信库日志和 step time 一起回答“计算还是通信限制”。若应聘方向是 GPU kernel/inference，分布式训练可按岗位要求准备，不要让它替代 CUDA/Triton 核心实践。

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
- [Anthropic GPU Performance Engineer](https://job-boards.greenhouse.io/anthropic/jobs/4926227008)；[RadixArk Cross-Hardware Inference](https://job-boards.greenhouse.io/radixark/jobs/4343666009)。仅作两个岗位能力样本，不据此声称市场频率、个人资格或投递情况。
