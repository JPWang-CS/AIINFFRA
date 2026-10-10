# 通用整网分析方法

这套方法适用于不同模型和服务框架。每一项都按“先看什么→如何判断→用什么证据证伪”执行，并把 CPU/框架视角与真实 NPU 算子、设备和通信证据分开。框架 trace 不能自动证明 NPU 执行。

## 1. 环境、源码与模型识别

先看芯片、驱动、CANN、容器、框架、模型标识、源码 commit、启动参数和配置哈希，再读模型 `config` 区分 dense/MoE、MHA/GQA/MLA。MoE 的 active 参数不等于驻留权重；只有实际模型适用时才分析 EP 和路由不均。

判断是所有结论都能绑定到同一版本和配置；用版本清单、配置文件、权重加载日志、源码行号和启动日志证伪。缺任一绑定项就保留未知，不拿历史 7B 参数替代当前模型。

## 2. 模型结构与内存账本

先按实际配置估算权重、激活、KV cache、临时 buffer、通信 buffer、运行时预留和可用容量，并注明 dtype、序列长度、batch、量化和并行布局。可用粗略账本：权重驻留 + 激活峰值 + KV 容量 + 临时/通信开销 ≤ 可用设备内存。

判断加载是否成功、KV 增长是否解释显存变化；用配置、加载日志、设备内存快照、KV block 统计和 profile memory 证伪。参数规模只能生成假设，不能证明实际驻留量。

## 3. 启动链、请求链与关键路径

先从启动入口追到控制面、路由/代理、调度、tokenizer、Prefill、Decode、模型 forward、设备算子和返回路径，分别标记控制、数据、通信路径。再找端到端关键路径和可重叠区间：异步拷贝、通信、CPU 调度和设备计算可能重叠，不能把各 kernel 时长直接相加。

判断某段是否真正决定端到端延迟，要看关键路径上的等待和依赖；用请求日志、调度事件、设备 timeline、同步点和源码调用关系证伪。单个算子很慢但处于重叠区，不能直接称为端到端瓶颈。

## 4. 时间线、归属与时钟

先为每条请求记录 request id、rank、batch/step、到达/排队/首 token/完成时间、事件类型、时钟来源和时钟偏移，再关联服务日志、CPU 调度、设备 kernel、拷贝和通信 timeline。跨进程或跨主机 trace 若时钟未校准，timestamp 不能直接相减；不要为采 trace 修改共享系统时钟。

判断请求归属必须同时满足 request、batch/step 和 rank 关联；一个 batch 共享 kernel 不能随意归给单请求。用 correlation id、调度表、事件顺序和设备 trace 证伪。PyTorch HTA 可借鉴 CPU/GPU correlation 方法，但其 GPU 专属字段不能宣称已适用于 NPU。

## 5. Prefill/Decode 与 P/D 模式

先分别看 Prefill 计算、排队、首 token 等待，以及 Decode 每步、尾部 ITL、批处理和抢占。混部要找 Prefill 插入 Decode 的干扰；分离要找 P/D 资源、代理路由、KV 产生/传输/接收/释放和等待，并各保留原始日志与 trace。

判断阶段瓶颈要用分段时间线和请求状态，不能用整段端到端时间代替。Prefill 偏计算、Decode 偏带宽只是可证伪假设，需结合 batch、长度、量化和并行实测；用阶段事件、KV 字节和设备计数器证伪。

## 6. 并行、通信与性能上界

先记录 TP/PP/DP 等实际划分、rank 映射、collective、消息大小、等待、慢卡/慢链路和通信矩阵，再用 FLOPs、bytes、计算峰值和带宽上界形成假设，例如计算下界约为 FLOPs/峰值计算率，搬运下界约为 bytes/有效带宽。

判断受计算、带宽或通信限制只是一阶分类；用 CANN/设备计数器、通信 timeline、慢 rank 数据和端到端变化证伪。CANN 文档中的通信耗时、通信矩阵和 Host 指标需按现场版本核对，不能把上界当实测。

## 7. 负载扫描与稳态

先分开记录 cold start、权重加载、编译、预热和稳态推理，再扫描短/长输入×短/长输出、单请求/多并发/近饱和的长度和并发网格。观察吞吐—延迟转折、KV 压力、排队和尾延迟，缓存开关在框架支持时做单变量对照。

判断转折来自调度、内存还是设备瓶颈，要看多轮重复、异常样本和阶段 trace；用未启 profile 的正式 benchmark 证伪。profile run 单独记录开销，不能代替正式 benchmark。

## 8. 证据分层与故障归因

先把证据分为源码路径、服务日志、CPU/框架 trace、NPU/CANN 设备证据，再从最深已知层提出最小可证伪预测。异步 CPU 调用时间不等于设备执行时间；服务存活、请求完成和设备成功也分别判断。

判断结论边界要保留失败尝试、未验证项和适用版本；用新 run、原始日志、trace 和源码版本复验。没有更深证据就停止在最深已知层，不把“无异常日志”写成设备路径已执行。

## 指标口径

统一记录测量端、统计范围和时间点：TTFT、请求级 TPOT、ITL、端到端延迟、输入/输出吞吐、P50/P95/P99、完成/失败/超时/取消。流式 chunk 可能包含多个 token，须记录 chunk 粒度；输出 token 数为 1 时请求级 TPOT 不定义。公式和测量端参考 [vLLM Benchmark CLI](https://docs.vllm.ai/en/stable/benchmarking/cli/)。

## 公平比较

混部与分离须固定模型权重、精度、tokenizer、请求集合、实际输入/输出长度、停止条件、采样、流式方式和 cache 状态；记录总设备数、P/D 配额、拓扑、额外模型副本、可用 KV 容量、代理/路由/排队、KV 传输及等待。资源无法匹配时写清条件差异，不能直接称公平优胜，也不预设分离一定更快。

官方方法参考（访问日期：2026-10-08，目标版本需现场核对）：[vLLM Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/)、[vLLM-Ascend Service Profiling](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/performance_and_debug/service_profiling_guide.html)、[vLLM-Ascend PD Design](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/disaggregated_prefill.html)、[PyTorch Holistic Trace Analysis](https://docs.pytorch.org/tutorials/beginner/hta_intro_tutorial.html)。
