# 本地课程资料

本页按主题整理两份本地 PDF，提供页码直达入口，并说明它们与课程主体和论文线的关系。

## 资料说明

- [CUDA C++ Programming Guide（本地 PDF）](../../downloads/cuda-programming-guide.pdf) 是 NVIDIA GPU 硬件、CUDA 编程模型和 API 行为的依据；相关课程章节同时提供[在线官方文档](https://docs.nvidia.com/cuda/cuda-programming-guide/)入口。
- [大模型推理实践（本地讲义）](../../downloads/大模型推理实践.pdf) 共 84 页，封面标题为“大模型推理实践”；作者和正式出版版本未注明。
- [DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression（本地技术报告）](../../downloads/DeepSeek_V41_Tech_Report.pdf) 共 51 页；封面署名 DeepSeek-AI。
- 讲义中的昇腾系统、盘古案例和相关部署内容属于对应平台案例；与 NVIDIA CUDA 课程内容对照阅读时，应分开平台边界。Prefill 更偏计算受限、Decode 更偏带宽受限只能作为给定模型、形状、并发、缓存和硬件条件下的经验假设，需要实测确认。

## 《大模型推理实践》：16 章主题导航

表中“章”是讲义的章节序号，“PDF 页”是阅读器页码；两者不是同一个编号。

| 章 | 主题 | PDF 页 | 课程衔接 |
|---:|---|---:|---|
| 1 | 推理概览 | [第 1 页](../../downloads/大模型推理实践.pdf#page=1) | [实践课程顺序](README.md#实践线的阅读顺序)：建立从模型到系统的全景 |
| 2 | 性能指标 | [第 6 页](../../downloads/大模型推理实践.pdf#page=6) | [性能分析方法](gpu/05-performance-analysis-and-optimization/README.md)与[Decode / PagedAttention](operators/06-decode-paged-attention/README.md)：把吞吐、延迟、显存、并发和利用率落实到测量条件 |
| 3 | Scaling Law | [第 10 页](../../downloads/大模型推理实践.pdf#page=10) | [模型 GPU 分析](model-analysis/README.md)：理解模型规模与计算/存储预算 |
| 4 | 昇腾系统 | [第 14 页](../../downloads/大模型推理实践.pdf#page=14) | [GPU 与 CUDA](gpu/README.md)：作为 NPU→GPU 迁移的对照案例 |
| 5 | 推理框架 | [第 16 页](../../downloads/大模型推理实践.pdf#page=16) | [真实系统落地](systems/README.md)：理解算子进入真实调用链的方式 |
| 6 | RL | [第 23 页](../../downloads/大模型推理实践.pdf#page=23) | [模型 GPU 分析](model-analysis/README.md)：了解训练与推理工作负载边界 |
| 7 | 并行策略 | [第 33 页](../../downloads/大模型推理实践.pdf#page=33) | [真实系统落地](systems/README.md)：连接多 GPU、通信和计算 |
| 8 | 融合与图 | [第 39 页](../../downloads/大模型推理实践.pdf#page=39) | [完整 GPU 算子体系](operators/README.md)：连接算子融合、编译图和物化边界 |
| 9 | KV Cache | [第 49 页](../../downloads/大模型推理实践.pdf#page=49) | [模型 GPU 分析](model-analysis/README.md)、[Decode / PagedAttention](operators/06-decode-paged-attention/README.md)与[论文线](../../notes/algorithms/README.md)：连接缓存账本、Attention 和 Decode |
| 10 | 量化 | [第 52 页](../../downloads/大模型推理实践.pdf#page=52) | [低精度与量化](quantization/README.md)：连接算法、误差与 kernel |
| 11 | 投机解码 / MTP | [第 55 页](../../downloads/大模型推理实践.pdf#page=55) | [模型 GPU 分析](model-analysis/README.md)：补充 Decode 调度与推理优化 |
| 12 | 整网瓶颈 | [第 61 页](../../downloads/大模型推理实践.pdf#page=61) | [性能分析方法](gpu/05-performance-analysis-and-optimization/README.md)与[Decode / PagedAttention](operators/06-decode-paged-attention/README.md)：沿机制→代码→可观测行为→结果建立证据链 |
| 13 | 盘古案例 | [第 64 页](../../downloads/大模型推理实践.pdf#page=64) | [真实系统落地](systems/README.md)：阅读昇腾/盘古平台案例 |
| 14 | Agent 挑战 | [第 74 页](../../downloads/大模型推理实践.pdf#page=74) | [真实系统落地](systems/README.md)：理解工具调用和长上下文带来的系统约束 |
| 15 | RL Rollout | [第 78 页](../../downloads/大模型推理实践.pdf#page=78) | [真实系统落地](systems/README.md)：理解 rollout 服务的吞吐、并发与调度 |
| 16 | 自适应投机 | [第 81 页](../../downloads/大模型推理实践.pdf#page=81) | [模型 GPU 分析](model-analysis/README.md)：延伸阅读投机策略与运行时反馈，不把闲时边际成本写成无条件为零 |

## 融合顺序与事实边界

- 讲义 ch2/ch12 接入性能与模型分析；ch8 接入 Fusion；ch9 接入 Decode；ch10 接入量化；ch7/ch13 接入 MoE 与系统；ch11/ch16 接入 sampling/投机。训练、scaling law、RL 是扩展，不抢算子主干。
- 讲义 p49 的 MLA 数值按官方 DeepSeek-V3 absorbed cache 路径校正为 `61*(512+64)*2=70272 B/token`；p40、p52-p55、p61、p83 的核对和不采用项见[PDF 融合与 Decode 决策记录](decisions/2026-09-10-pdf-integration-and-decode.md)。
- [V4.1 论文正文](papers/deepseek-v41.md)连续解释 CED、CSA2、层次索引、Engram、DSpark、Sinkhorn 更新、训练共享状态与缓存重放；[Scaling Law、RL 与长任务](papers/inference-workloads.md)承接讲义的训练和工作负载主题。数值与代码示例用于检查机制，不替代作者实现、质量实验或实际 backend/allocator 证据。

## DeepSeek-V4.1-Flash：KV Cache Compression

这份报告围绕 KV Cache 压缩、注意力结构和推理系统展开，可作为[量化](quantization/README.md)、[KV Cache 与模型 GPU 分析](model-analysis/README.md)以及 Prefill/Decode 相关论文阅读的补充材料。

| 报告部分 | PDF 页 | 课程衔接 |
|---|---:|---|
| §2 架构 | [第 7 页](../../downloads/DeepSeek_V41_Tech_Report.pdf#page=7) | [论文线总入口](../../notes/algorithms/README.md)：建立报告中的架构语境 |
| §2.2 CED（Causal Encoder-Decoder，因果编码器-解码器） | [第 9 页](../../downloads/DeepSeek_V41_Tech_Report.pdf#page=9) | [模型 GPU 分析](model-analysis/README.md)：关联模型的 Prefill/Decode 路径 |
| §2.3 CSA2（Compressed Sparse Attention 2） | [第 9 页](../../downloads/DeepSeek_V41_Tech_Report.pdf#page=9) | [Attention 演进](../../notes/algorithms/README.md#42-attention演进)：关联稀疏注意力结构与访问路径 |
| 跨层 KV / index 复用 | [第 10 页](../../downloads/DeepSeek_V41_Tech_Report.pdf#page=10) | [KV Cache](#大模型推理实践16-章主题导航)：关联缓存容量、复用和访存路径 |
| 层次索引 | [第 11 页](../../downloads/DeepSeek_V41_Tech_Report.pdf#page=11) | [模型 GPU 分析](model-analysis/README.md)：关注索引结构对 Prefill/Decode 的影响 |
| 扩展 | [第 12 页](../../downloads/DeepSeek_V41_Tech_Report.pdf#page=12) | [真实系统落地](systems/README.md)：把结构设计放回系统扩展条件 |
| FP4 Main KV | [第 14 页](../../downloads/DeepSeek_V41_Tech_Report.pdf#page=14) | [量化主课](quantization/README.md)与[量化论文线](../../notes/algorithms/README.md#44-量化与低精度)：区分格式、误差和 kernel 支持 |
| §3.2 推理系统 | [第 18 页](../../downloads/DeepSeek_V41_Tech_Report.pdf#page=18) | [推理系统论文线](../../notes/algorithms/README.md#45-推理系统)：连接系统执行路径 |
| PersistentKV | [第 19 页](../../downloads/DeepSeek_V41_Tech_Report.pdf#page=19) | [KV Cache](#大模型推理实践16-章主题导航)：关联缓存驻留和生命周期 |
| SWA（Sliding-Window Attention，滑动窗口注意力）BoundedReplay | [第 20 页](../../downloads/DeepSeek_V41_Tech_Report.pdf#page=20) | [Prefill/Decode 模型分析](model-analysis/README.md)：关注窗口、重放和运行时带宽/计算条件 |

## 如何结合课程阅读

讲义提供从算子到整网的观察角度，技术报告提供具体模型的数据结构。CUDA Programming Guide 仍用于确定线程、存储、同步和 API 行为。三类材料各自回答不同问题，不按书的目录重新排列 GPU 算子主线。

| 材料主题 | 对应课程 | 阅读时要落实的问题 |
|---|---|---|
| 讲义第 2、12 章：指标与整网瓶颈 | [性能分析：从 kernel 时间走到请求时间](gpu/05-performance-analysis-and-optimization/README.md#从-kernel-时间走到请求时间) | TTFT/TPOT 的时间戳、P95 是否可相加、单算子收益怎样传递到请求 |
| 讲义第 8 章与报告 §2.4.1 | [Fusion：Single-Pass mHC](operators/04-activation-and-fusion/README.md#模型依赖怎样限制融合single-pass-mhc) | 依赖能否消除、哪些权重可折叠、减少遍历是否改变模型 |
| 讲义第 9 章与报告 §2.3、§3.2 | [Decode / PagedAttention](operators/06-decode-paged-attention/README.md) | 软件分页、缓存容量、有效读取、跨层复用和近似恢复的区别 |
| 讲义第 10 章与报告 §2.4.4 | [低精度与量化](quantization/README.md) | 数据值、scale、zero point、packing、反量化与指令路径分别如何计算 |
| 讲义第 7、13 章与报告 MoE/Engram 内容 | [算子体系](operators/README.md)与[系统扩展](systems/README.md) | 专家路由、重排、稀疏访问和通信成本；昇腾案例不直接等同 CUDA 实现 |
| 讲义第 11、16 章与报告 DSpark | [模型执行分析](model-analysis/README.md)与[独立论文线](../../notes/algorithms/README.md) | 接受率、草稿开销、验证长度和当前负载共同决定投机收益 |
| Scaling Law、训练、RL 和多模态数据 | [论文线](../../notes/algorithms/README.md) | 作为模型和工作负载背景，保持与算子实践独立的阅读节奏 |

课程引用模型数字时先展开数据结构。例如 MLA 的 latent 与 RoPE key 不重复计作两份完整 K/V；FP4 还要加 scale 字节；权重与 KV 量化应分别缩小对应容量项后相加，不把两个压缩比例直接相乘。报告中的 Bounded Replay 接受近似状态，不等价于完整前向恢复。上述区别分别在 Decode 和 Fusion 正文中用公式、代码及反例说明。

## 相关课程入口

- [GPU 与 CUDA](gpu/README.md)：把硬件机制落实到线程、warp、SM、存储、同步与性能证据。
- [完整 GPU 算子体系](operators/README.md)：把融合、KV、量化和 Attention 主题落到算子对象。
- [独立论文与理论学习线](../../notes/algorithms/README.md)：保留论文阅读顺序和完成标准。
