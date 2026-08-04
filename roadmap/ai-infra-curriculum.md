# PATH 执行参考：AI Infra 学习内容细化（2026-08）

> 定位：把 [PATH.md](../PATH.md) 的知识地图展开成可执行的任务和验收标准，解决当前计划“干货密度不足”的问题。
> 本文件不是另一条学习线。进度仍以 [PATH.md](../PATH.md) 为唯一权威源，当前焦点由 [NOW.md](../NOW.md) 指定，这里只做内容组织和验收细化。
> 参考：[AIInfraGuide](https://github.com/caomaolufei/AIInfraGuide)（AI Infra 全栈从 0 入门）的 4 大模块结构，同时保留本仓库的 Ascend→CUDA 映射、Triton-first、跑通才算数的原则。

---

## 干货标准

每个学习单元完成时，至少产出三样东西，否则不算完成：

1. **跑通的代码或可演示脚本**：CUDA 进 `solutions/cuda/`，Triton/Python 进 `solutions/triton/`（按需新建），理论笔记进 `notes/algorithms/`。
2. **一个可量化的结论**：GFLOPS、有效带宽 GB/s、延迟、显存、通信量、kernel launch 次数、代码行数，任选能说明问题的。
3. **一段能讲 1 分钟的面试口径**：问题 → 结论 → 取舍，例如“为什么 tiling 不一定赢”“为什么 Flash Attention 不落 N×N 矩阵”。

理论笔记必须是“你能讲清”，不是“Agent 生成过就算”。仓库里已有但你没消化的草稿，要逐条重新过一遍。

---

## 文件归属

| 文件 | 职责 |
|------|------|
| [PATH.md](../PATH.md) | 唯一进度源，记录全貌和状态 |
| [NOW.md](../NOW.md) | 当前焦点，决定下一步学什么 |
| [lessons/](../lessons/) · [solutions/](../solutions/) · [reference/](../reference/) | 算子线内容 |
| [notes/algorithms/](../notes/algorithms/) | 理论线内容 |
| 本文件 | PATH 执行参考，只细化任务和验收，不重复维护进度 |

## PATH 执行阶段

| 模块 | 主题 | 状态 | 建议顺序 |
|------|------|:--:|------|
| M0 | PATH A4/A5 收尾：softmax benchmark + Flash Attn 读码 | 🚧 当前 | 第 1-2 周 |
| M1 | PATH A CUDA/算子优化：Reduce / GEMM / Softmax / Flash Attn / LayerNorm / Profiling | ⏳ | 第 2-5 周 |
| M1.5 | PATH 理论线·模型架构配套：LLaMA/Qwen/DeepSeek/GPT/Claude/Gemini/MoE/SSM | 🚧 草稿 | 理论线滚动，第 3-7 周 |
| M2 | PATH B Triton 主力：matmul / fused softmax / flash attention / autotune | ⏳ | 第 5-9 周 |
| M3 | PATH C 推理系统：vLLM / PagedAttention / scheduling / quant / speculative / PD | ⏳ | 第 9-14 周 |
| M4 | PATH D 分布式训练：NCCL / DDP / FSDP / ZeRO / TP / PP / EP | ⏳ | 第 14-17 周 |
| M5 | PATH 求职/面试冲刺：面试题库 + 项目叙事 + 复盘 | ⏳ | 最后 2-3 个月 |

> 顺序是建议不是枷锁。每完成一个模块，更新 [PATH.md](../PATH.md) 状态并写一篇 [weekly/](../weekly/) 回顾。

---

## M0：收尾当前主线（2026-08）

| 任务 | 最小产出 | 验收 |
|------|---------|------|
| A5 读 Flash Attn CUDA | 逐行/逐块注释版笔记 | 能标出每个 `__syncthreads` 的作用，能解释 Q/K/V tile、online softmax、register accumulator 如何协作 |
| Softmax 三版 benchmark | `softmax_naive` / `softmax_online` / `softmax_opt` 的带宽对比 | 每个版本有 GB/s，能说出瓶颈是 HBM 还是 launch 开销 |
| 1-pass true online 落盘 | `solutions/cuda/softmax/softmax_1pass.cu` | 仓库里要有 LeetGPU 上实践过的 per-thread `(m,s)` scan 版本，而不是只有文档 |
| 补 warp shuffle | `softmax_opt.cu` 真正用 `__shfl_down_sync` | 有可跑代码 + benchmark 数字 |

## M1：CUDA/算子优化（干货主线）

| 算子 | 必须做到 | 可选深钻 |
|------|---------|---------|
| Reduce | block tree reduce → warp shuffle → 多级归约 | grid-wide reduce / atomic |
| GEMM | naive → tiled → fp16 | vec4、double buffer、bank conflict 消除 |
| Softmax | 3-pass → 2-pass fused → 1-pass online | 与 cuBLAS/PyTorch 对比 |
| Flash Attention | V1 读码 → V2 对比 | Decode 阶段优化、PagedAttention CUDA |
| LayerNorm/RMSNorm | 读 `reference/cuda/layernorm/layernorm.cu`，手写一版 | 融合 bias/残差 |
| Profiling | Nsight Systems/Compute 跑通，能报 roofline | PyTorch Profiler |

验收：每个算子至少有一个 correctness check 和一个性能数字；至少 3 个算子与 cuBLAS/cuDNN/PyTorch 对照；能说出“这个算子是 memory-bound 还是 compute-bound，依据是什么”。

## M1.5：理论线配套（PATH 模型架构子类）

理论线不再只列“待写”。先读两份入口笔记，再按任务逐个消化：

- [最新模型与结构](../notes/algorithms/latest-model-architectures.md)：组件、模型家族、KV cache/MoE/SSM 对推理的影响。
- [模型追踪表](../notes/algorithms/model-tracker.md)：记录每个模型家族的学习状态和结构观察点。
- [剩余理论主题速览](../notes/algorithms/remaining-theory-primer.md)：Norm、Flash-2 work partitioning、线性注意力、量化、batching、RadixAttention、ZeRO/FSDP、TP/PP/EP 等。

| 任务 | 最小产出 | 验收 |
|------|---------|------|
| LLaMA/Qwen config 分析 | 读 HF config + 模型代码 | 能讲清 attention head、GQA 组数、norm、位置编码、FFN/MoE |
| DeepSeek MLA/MoE | 读 [MLA](../notes/algorithms/mla-deepseek.md) + [MoE](../notes/algorithms/moe-inference.md) | 能手算 KV cache，能讲 expert 并行和负载均衡 |
| GPT/Claude/Gemini 趋势图 | 基于公开资料写一页 | 能区分“公开事实”和“传闻”，并说出推理影响 |
| Mamba/SSM 对比 | 读 [最新模型与结构](../notes/algorithms/latest-model-architectures.md) | 能对比 Attention 和 SSM 的复杂度、状态容量、GPU 实现难度 |
| 剩余理论逐条深钻 | 每主题一页独立笔记 | 不再只靠速览，能讲原理 + 验证方法 |

验收：能对着一个最新开源模型的 config 讲清结构，并解释 GQA/MLA/MoE/位置编码对推理系统的影响。

## M2：Triton 主力

| 任务 | 产出 | 验收 |
|------|------|------|
| vec add / matmul | `solutions/triton/` | GPU/CPU 模拟跑通，和 PyTorch 对齐 |
| fused softmax | Triton kernel | 对比 PyTorch 正确 + 提速 |
| flash attention v1/v2 | Triton kernel | 对比 PyTorch ref 正确，记录显存与速度 |
| GQA / fused MLP | Triton kernel | 正确性 + autotuning |
| Triton 底层 | 笔记更新 | 能讲 `tl.dot`、自动 shared memory、`tl.load/store` 对应什么 CUDA 机制 |

验收：能解释“为什么 Triton 是主力”：开发速度、可读性、接近手写 CUDA 性能，同时能说出它替你做了什么、没替你做什么。

## M3：推理系统

| 主题 | 核心问题 | 参考 |
|------|---------|------|
| Prefill vs Decode | 两阶段瓶颈为什么不同 | [AIInfraGuide: LLM 推理基础](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/11-llm推理基础/) |
| PagedAttention / KV Cache | block table 如何做虚拟→物理映射 | [AIInfraGuide: PagedAttention](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention/) |
| Continuous Batching | iteration-level scheduling 如何提高利用率 | [AIInfraGuide: Continuous Batching](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/22-continuous-batching/) |
| Prefix Cache / RadixAttention | 前缀共享的取舍 | [AIInfraGuide: Prefix Cache](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/23-prefix-cache-与-radixattention/) |
| Chunked Prefill | 如何减少 prefill/decode 干扰 | [AIInfraGuide: Chunked Prefill](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/24-chunked-prefill-与统一调度/) |
| Quantization | SmoothQuant / AWQ / GPTQ / FP8 / KV cache | [AIInfraGuide: 推理优化模块](https://caomaolufei.github.io/AIInfraGuide/inference/) |
| Speculative Decoding | draft-verify 的收益条件 | [AIInfraGuide: 推理优化模块](https://caomaolufei.github.io/AIInfraGuide/inference/) |
| PD 分离 | 为什么 prefill/decode 要拆开 | 已有 [notes/algorithms/pd-disaggregation.md](../notes/algorithms/pd-disaggregation.md) |

验收：能画出 vLLM 的请求→scheduler→worker→attention→KV cache→response 链路；能解释 block table；跑一次 vLLM benchmark 并记录 TTFT/TPOT。

## M4：分布式训练

| 主题 | 最小产出 |
|------|---------|
| NCCL / 集合通信 | 能画 AllReduce / ReduceScatter / AllGather / All-to-All 通信量 |
| DDP / FSDP | 能手算优化器状态、梯度、参数的显存账本 |
| ZeRO-1/2/3 | 能讲清各阶段切什么、通信代价 |
| TP / PP / EP / CP | 能画通信图并说清适用场景 |
| 3D parallel | 能设计一个 TP×PP×DP×EP 拓扑 |

验收：在有 GPU 环境时跑一个最小 DDP/FSDP demo；没有多卡环境就把通信量和显存账本推导到能口算。

## M5：求职冲刺

- 用 `solutions/`、`weekly/`、`notes/algorithms/` 整理 3-5 个可讲项目。
- 按模块刷 [interviews.md](interviews.md)，重点：bank conflict、tiling、Flash Attention、vLLM scheduler、quantization、分布式显存账本。
- 面试叙事固定为“Ascend → GPU 跨平台优化者”：异构计算的本质是计算与访存的权衡、数据搬运开销、并行度挖掘。

---

## 建议推进节奏（不是死课表）

| 周次 | 焦点 | 最小产出 |
|------|------|---------|
| W1 | A5 读码 + softmax benchmark | 注释版 Flash Attn + 3 个 GB/s 数字 |
| W2 | Reduce / LayerNorm + Triton vec add | 两个算子代码 + 性能记录 |
| W3 | Triton matmul / fused softmax + 模型结构速览 | 正确性 + autotune + 能讲清一个模型组件 |
| W4 | Flash Attention V1/V2 + DeepSeek MLA/MoE | 能讲 V2 改了什么 + KV cache 账本 |
| W5 | Profiling + LLaMA/Qwen config 分析 | 每个算子有 roofline/瓶颈结论 + 一张结构图 |
| W6 | 推理基础 + vLLM 快速入门 | 跑一个 vLLM 服务 |
| W7 | PagedAttention + scheduler 源码 | 画出 block table 和调度循环 |
| W8 | continuous batching + prefix cache + 剩余理论速览 | 能讲清三个机制怎么配合 + 速览笔记 |
| W9 | quantization + speculative decoding | 每种方案一张取舍表 |
| W10 | PD 分离 + benchmark | 端到端 TTFT/TPOT 数据 |
| W11 | NCCL / DDP / FSDP | 显存账本 + 最小 demo |
| W12 | ZeRO / TP / PP / EP | 通信图 + 面试口径 |

> 一周内如果没跑完，就留到下一周；不要为了赶进度把验收标准降成“看过文档”。

---

## 外部参考

- [AIInfraGuide](https://github.com/caomaolufei/AIInfraGuide) — CUDA、分布式训练、推理优化、面试宝典，是本次计划的主要对照。
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — 官方权威文档。
- [Triton Language Docs](https://triton-lang.org/) — Triton 语法和编程模型。
- [vLLM](https://github.com/vllm-project/vllm) — 推理系统源码和文档。
- [CUDA MODE Lectures](https://github.com/cuda-mode/lectures) — 社区课程和讨论。

## 更新记录

- 2026-08-03：新增 PATH 执行参考，参考 AIInfraGuide 的 4 大模块结构，收紧每阶段的产出与验收标准；不替代 PATH/NOW。
- 2026-08-03：补充最新模型与结构主笔记 + 剩余理论主题速览，接入 PATH 与理论线索引。
