# GPU 高性能算子与 LLM 性能优化课程

课程面向已有 NPU 算子经验的工程师。主干是 NVIDIA GPU 架构、CUDA/Triton 和算子极致优化；低精度、量化与模型级 Prefill/Decode 分析是核心应用，Mini Transformer 与 vLLM 用来验证真实系统收益。

[打开课程网站](gpu/course-site/index.html?chapter=1) · [完整算子体系](operators/README.md) · [本地课程资料](materials.md) · [独立论文线](../../notes/algorithms/README.md)

<a id="实践线的五个主体"></a>

## 实践线的阅读顺序

| 顺序 | 内容 | 入口 |
|---|---|---|
| 一、GPU 与 CUDA 基础 | 架构、执行、存储、同步与通用性能分析方法 | [GPU 与 CUDA](gpu/README.md) |
| 二、算子实现与优化 | 每个算子连续讲解数学、布局、实现、验证和优化；覆盖量化、Attention 与 MoE | [算子课程](operators/README.md) |
| 三、模型与系统应用 | 组合算子，分析 Prefill/Decode、Mini Transformer，再进入 vLLM 与多 GPU | [模型分析](model-analysis/README.md) · [系统应用](systems/README.md) |

算子的性能分析、指令检查、多形状调优和失败案例都在对应算子内展开，不另设一条需要往返跳转的优化课程。通用硬件知识在基础部分集中讲解；算子章会就地说明所需条件，基础链接只供复习。低精度的格式、量化算法和 kernel 路径随量化主题展开，[量化目录](quantization/README.md)保留主题导航。

## 算子覆盖与优化深度

算子广度不能只靠MatMul、Softmax和FlashAttention三个例子。访存/布局、归约/Norm、GEMM全体系、Activation/Fusion、Prefill Attention、Decode Attention、量化kernel、MoE与Sampling/KV辅助算子都保留；Scan和Histogram作为基础并行模式补入，而不是拿基础六章替代完整覆盖。

深入优化重点包括GEMM、Reduction/Softmax/RMSNorm、Fused MLP、Prefill FlashAttention、Decode/PagedAttention，以及Quantized GEMM或MoE Grouped GEMM。量化与MoE的基础实现都属于算子覆盖，深入项目则按学习与硬件条件选择。

至少三个核心对象应形成完整报告：数值合同与目标形状、强baseline、FLOPs/bytes和理论上界、优化假设、代码变化、Nsight Systems/Compute证据、PTX/SASS、多形状回归，以及继续或停止优化的理由。不能把单一有利形状或不同精度之间的速度差直接算作优化成果。

## 从已有实践继续

已有Vector Add、MatMul、Softmax、CUDA实现、失败配置和性能报告继续作为案例。原始源码保留在solutions，课程解释关键片段并链接其出处；讨论中的疑问与认识放在相关知识点旁，实验数字保留设备、精度、形状与计时条件。

需要平台练习的算子按“理解原理 → LeetGPU自写与原始代码归档 → 服务器正确性/benchmark → 性能分析”推进。硬件知识、Profiler阅读和系统实验不伪造平台题目，也不要求把已经掌握的算法重新作为入门作业。

[访存与布局算子](operators/01-memory-and-layout/README.md)从Copy、向量化访问、矩阵转置和数据布局切入；它与GPU基础篇的机制解释互相引用，进一步进入可比较的算子实现和性能实验。

## 论文线独立推进

论文线覆盖经典与最新工作，保留动机、关键公式、张量形状、作者关键代码和适用边界。不要求每篇复现，不受实践线当前算子的节奏限制。

本地 PDF 的按主题页码入口见[本地课程资料](materials.md)；其中讲义案例与 DeepSeek 技术报告只作为关联阅读，不改变论文线顺序或课程章节数量。

## 阅读与资料原则

两份新增 PDF 的主题对应见[本地课程资料](materials.md#如何结合课程阅读)；来源核对、采纳边界和讨论结论保存在[资料融合记录](decisions/2026-09-10-pdf-integration-and-decode.md)。课程正文与维护记录分开阅读。

CUDA本地手册用于固定教材版本，在线文档用于核查最新API与能力；章节编号要注明版本。专业缩写在首次出现时解释，公式给出变量与适用条件，代码标出关键行，图解明确是推导示意还是实测数据。

工程判断沿着“机制 → 代码 → 可观测行为 → 结果”展开。文档中已知、推测与待测的技术结论必须区分；课程页面不显示学习状态或编写流程。
