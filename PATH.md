# PATH — 知识地图 + 进度

> **这是知识地图,也是唯一的进度源。** 想找任何东西(某算子、某理论、某篇论文)——这里一定有条目,一跳到位。
> 进来先看 → [NOW.md](./NOW.md)（现在做什么 + 接下来）。这里是全貌。
> 方向：ML 系统工程师 · Triton 为主力 · CUDA 为底层 · 约 1 年
> 密集课表与每阶段验收标准 → [roadmap/ai-infra-curriculum.md](./roadmap/ai-infra-curriculum.md)

图例：✅ 完成　🚧 进行中　⏳ 待做　⚠️ 证据/归档缺口　⭐ 可选/进阶

---

## 两条平行路径

地位一样,每周并行各推一点。一条动手、一条理解。

- **算子线**（动手写代码，学 GPU/算子）→ 产出能跑的代码，LeetGPU/本地跑通算数
- **理论线**（学算法/理论）→ 产出一页笔记进 [notes/algorithms/](./notes/algorithms/)，能讲清原理算数

```
算子线   A CUDA打底 ─→ B Triton ─→ C 推理系统 ─→ D 分布式 ─→ E Agent
理论线   GPU优化算法 · 量化 · 注意力演进 · 模型架构 · 推理系统技术 · 训练/并行
         （两条模型主线：DeepSeek-V3.2 / Qwen3.5，概念作枝干挂载）
```

---

## 验收基线（每个阶段都算数）

- **代码**：完成版本必须跑通并进 `solutions/`；允许保存明确标记为 `WIP` 的用户草稿，但它不算完成；只读参考、只有 Agent 生成的草稿不算完成。
- **数字**：每个算子至少留一个性能/正确性数字（GB/s、GFLOPS、误差、耗时），不能只写“跑通了”。
- **面试**：每条理论要能讲清“是什么、为什么、取舍、怎么验证”，对应 [roadmap/interviews.md](./roadmap/interviews.md)。

# 算子线（动手）

## A — CUDA 打底（B 级，"读得懂并能解释性能"）

> 目标：能写 tiled GEMM、读懂 Flash Attn CUDA、知道 Triton 底层在干什么。Tensor Core 的数值路径、tile 和 profiler 证据属于必学；手写 MMA/WGMMA/TCGen05 PTX 仍是可选深钻。

| 阶段 | 课 | 自己写 | 验收 | 笔记 | 参考 | 状态 |
|:-:|----|--------|------|------|------|:--:|
| A1 | [01 cuda-basics](./lessons/01-cuda-basics.md) | Vector Add | LeetGPU 跑通 + 能算 bandwidth；原始 LeetGPU `solve`/本地代码未单独归档 ⚠️ | [memory-model](./notes/cuda/memory-model.md) · [warp-and-sync](./notes/cuda/warp-and-sync.md) | 参考→[vector_add.cu](./reference/cuda/vector_add.cu) · 我的代码归档缺失 | ⚠️ |
| A2 | [02 gemm-naive](./lessons/02-gemm-naive.md) | `gemm_naive` (float) | LeetGPU 跑通（2026-06-16） | [memory-model](./notes/cuda/memory-model.md) | [gemm.cu](./reference/cuda/gemm/gemm.cu) · 我的→[gemm_naive.cu](./solutions/cuda/gemm/naive_float.cu) | ✅ |
| A2+ | — | `gemm_fp16_naive` | LeetGPU fp16 跑通（2026-06-22）·[review](./notes/cuda/code-review-gemm-fp16-naive.md) | — | 我的→[gemm_fp16_naive.cu](./solutions/cuda/gemm/naive_fp16.cu) | ✅ |
| A3 | [03 gemm-tiled](./lessons/03-gemm-tiled.md) | `gemm_fp16_tiled` | LeetGPU 跑通（2026-06-22）· 4090 实测 K=2048/8192 tiled 0.6x naive（L2 cache + occupancy，详见 benchmark） | [memory-model §3.3](./notes/cuda/memory-model.md) | 我的→[tiled_fp16.cu](./solutions/cuda/gemm/tiled_fp16.cu) · 参考→[gemm.cu](./reference/cuda/gemm/gemm.cu) | ✅ |
| A3+ | — | `gemm_tiled` (float) | 计划项，当前没有独立归档产物 | [memory-model §3.3](./notes/cuda/memory-model.md) | 参考→[gemm.cu](./reference/cuda/gemm/gemm.cu) | ⏳ |
| A4 | [04 softmax](./lessons/04-softmax.md) | `softmax_naive` → `softmax_online` → `softmax_1pass` | 3-pass baseline ✅（2026-07-01）· 2-pass fused ✅ · 1-pass true online 待落盘 · warp shuffle/benchmark 待做 | [warp-and-sync §4](./notes/cuda/warp-and-sync.md) | [softmax.cu](./reference/cuda/softmax/softmax.cu) · 我的→[softmax_naive.cu](./solutions/cuda/softmax/softmax_naive.cu) · [softmax_online.cu](./solutions/cuda/softmax/softmax_online.cu) | 🚧 |
| A5 | [05 flash-attn-reading](./lessons/05-flash-attn-reading.md) | 读代码（不手写） | 能标注每个 `__syncthreads` 作用（2026-08-10 ✅，[阅读笔记](./notes/cuda/flash-attn-reading.md)，发现 2 个真实 bug） | [triton-under-the-hood](./notes/cuda/triton-under-the-hood.md) | [flash_attn.cu](./reference/cuda/flash_attention/flash_attn.cu) · [论文](./papers/attention/flash-attention.md) | ✅ |

**阶段出口**：A5 完成 = CUDA B 级达成，切 B 线。

## B — Triton 算子（主力工具）

> 目标：用 Triton 写常见 ML 算子，并把硬件知识转化成性能优化。MatMul、Softmax/Norm、FlashAttention、Fused MLP/GQA 是极致性能锚点；从这里 Triton 成主力。

| 阶段 | 课 | 自己写 | 验收 | 参考 | 状态 |
|:-:|----|--------|------|------|:--:|
| B1 | [06 triton-intro](./lessons/06-triton-intro.md) | Vector Add：LeetGPU 通过 + AutoDL RTX 3090 benchmark（840.1 GB/s），但原始 LeetGPU `solve` 尚未单独归档 ⚠️；MatMul LeetGPU 原始代码已 `LEETGPU_PASS`，服务器适配版已 `GPU_VALIDATED` | MatMul 当前基线出口已完成：RTX 3090 最佳 20.830 ms / 19,794.1 GFLOPS / `torch.mm` 80.3%；Nsight Systems P0-lite：[s3/s2 详细分析](./notes/triton/matmul-nsys-p0-lite-2026-08-30.md) 与完整 raw logs 已归档。剩余 P0–P8 极致优化延期至 [GPU 优化篇](./roadmap/gpu-foundations.md#matmul-优化债务-deferred-backlog)，不再阻塞 B2 | 我的→[vector_add.py](./solutions/triton/vector_add.py) · LeetGPU最终版→[matmul_leetgpu.py](./solutions/triton/matmul_leetgpu.py) · 服务器版→[matmul.py](./solutions/triton/matmul.py) · P0-lite→[分析](./notes/triton/matmul-nsys-p0-lite-2026-08-30.md) · 参考→[matmul.py](./reference/triton/matmul/matmul.py) | GPU_VALIDATED |
| B2 | [08 triton-fused-softmax](./lessons/08-triton-fused-softmax.md) | Triton fused softmax：当前无用户代码 | **LeetGPU #5 正确性与原始代码归档 → 服务器 row-wise softmax 对齐 PyTorch并记录 ms/GB/s** | [CUDA Softmax](./lessons/04-softmax.md) · [Online Softmax](./notes/algorithms/online-softmax.md) · [Lesson 07 调试](./lessons/07-triton-debugging.md) | WIP 当前 |
| B3 | _按需生成_ | Triton flash attention | 对比 PyTorch ref 正确 | [flash_attn.py](./reference/triton/flash_attention/flash_attn.py) | ⏳ |
| B4 | _按需生成_ | Triton GQA / fused MLP | 正确性 + autotuning | [activations.cuh](./reference/cuda/include/activations.cuh)（料） | ⏳ |

核心锚点通过 LeetGPU 并归档原始实现后，服务器阶段可按需执行 [P0–P8 极致性能阶梯](./roadmap/gpu-foundations.md#32-核心算子的极致性能阶梯)；当前 MatMul 已完成基线出口，剩余极致优化延期至 GPU 优化篇，不阻塞算子主线。

## C — 推理系统

> 目标：弄明白 LLM serving 核心机制。从 vLLM PagedAttention 入手。详细计划 → [roadmap/vllm.md](./roadmap/vllm.md)

| 阶段 | 主题 | 出口 | 论文 | 状态 |
|:-:|------|------|------|:--:|
| C1 | Prefill vs Decode | 能讲清两者瓶颈不同 | — | ⏳ |
| C2 | PagedAttention / KV Cache | 读懂 block table 虚→实映射 | [paged-attention](./papers/inference/paged-attention.md) | ⏳ |
| C3 | 调度 continuous batching | 能讲 vLLM 调度循环 | — | ⏳ |
| C4 | 量化通路 AWQ/GPTQ/FP8 | 知道量化权重如何加载+调用 | _理论线_ | ⏳ |

## D / E — 了解概念即可

| 线 | 范围 | 出口 | 计划 | 状态 |
|:-:|------|------|------|:--:|
| D 分布式 | DP/FSDP/TP/PP/CP/EP + 多机网络 | 能跑单机/多机 baseline，画 topology/通信图并定位 NCCL/RDMA 问题 | [基础](./roadmap/distributed.md) · [多机多卡专项](./roadmap/multi-node-multi-gpu.md) | ⏳ |
| E Agent | MCP/Tool Use/RAG | 熟悉 + 1 个 demo | [roadmap/agents.md](./roadmap/agents.md) | ⏳ |

---

# 理论线（理解）

> 模型驱动主干 + 枝干 + 字典：主干决定学什么（DeepSeek-V3.2 / Qwen3.5），必要小模块是挂在主干上的枝干（按挂载点学，不推迟），字典只做概念速查；推进顺序按主干走，不按子类推进。产出一页笔记进 [notes/algorithms/](./notes/algorithms/)。有标志性论文的,论文精读放 [papers/](./papers/)、这里写"机制+怎么实现"并互链（边界规矩见 [algorithms/README](./notes/algorithms/README.md)）。

## GPU 优化算法
| 主题 | 笔记 | 状态 |
|------|------|:--:|
| GPU 底层架构与全栈性能优化（G0–G8） | [课程](./roadmap/gpu-foundations.md) · [架构知识图](./notes/cuda/gpu-architecture-layers.md) | `WIP`，随当前算子挂载；B1 当前解锁 SM/内存/Tensor Core/roofline |
| online softmax（Flash 的心脏） | [online-softmax.md](./notes/algorithms/online-softmax.md) | ✅ |
| parallel reduce / prefix sum | [parallel-reduce.md](./notes/algorithms/parallel-reduce.md) | ✅ |
| Norm 的 reduce 模式（LayerNorm/RMSNorm） | [速览](./notes/algorithms/remaining-theory-primer.md) · 料→[layernorm.cu](./reference/cuda/layernorm/layernorm.cu) | 🚧 |
| work partitioning（Flash 2 的思路） | [速览](./notes/algorithms/remaining-theory-primer.md) | 🚧 |

## 量化
| 主题 | 笔记 | 论文 | 状态 |
|------|------|------|:--:|
| 数值格式 INT8 / FP8 | [quantization-int8-fp8.md](./notes/algorithms/quantization-int8-fp8.md) | — | 🚧 |
| AWQ | [速览](./notes/algorithms/remaining-theory-primer.md) | _待建_ | 🚧 |
| GPTQ | [速览](./notes/algorithms/remaining-theory-primer.md) | _待建_ | 🚧 |
| SmoothQuant / KV Cache 量化 | [速览](./notes/algorithms/remaining-theory-primer.md) | — | 🚧 |

## 注意力演进
| 主题 | 笔记 | 论文 | 状态 |
|------|------|------|:--:|
| MHA→MQA→GQA→MLA | [速览](./notes/algorithms/remaining-theory-primer.md) + [最新模型](./notes/algorithms/latest-model-architectures.md) | [gqa.md](./papers/attention/gqa.md) | 🚧 |
| Flash Attention 1→2→3 | [flash-attention-mechanism.md](./notes/algorithms/flash-attention-mechanism.md) · [FA2 统一笔记](./notes/algorithms/flash-attention-2.md) | [FA1](./papers/attention/flash-attention.md) · FA2 已并入左侧统一笔记 | 🚧 → FA1 ✅（2026-08-10 A5），FA2 🚧 用户阅读约 50%（仍未读完），FA3 待补 |
| MLA（DeepSeek-V2/V3） | [mla-deepseek.md](./notes/algorithms/mla-deepseek.md) | DeepSeek-V2 | 🚧 |
| FA4 / FlexAttention | [fa4-flexattention.md](./notes/algorithms/fa4-flexattention.md) | 官方博客/PR | 🚧 草稿 2026-08-13 |
| DSA（DeepSeek-V3.2 / GLM-5） | [dsa-sparse-attention.md](./notes/algorithms/dsa-sparse-attention.md) | DeepSeek-V3.2 · GLM-5 | 🚧 草稿 2026-08-13 |
| SageAttention3 / Kascade | [attention-2026-sage3-kascade.md](./notes/algorithms/attention-2026-sage3-kascade.md) | SageAttention3 · Kascade | 🚧 草稿 2026-08-13 |
| 线性注意力 / [GDN（Qwen3.5）](./notes/algorithms/gdn-linear-attention.md) / Ring Attention | [速览](./notes/algorithms/remaining-theory-primer.md) | — | 🚧 |

## 模型架构
| 主题 | 笔记 | 状态 |
|------|------|:--:|
| 最新模型架构地图（LLaMA/Qwen/DeepSeek/GPT/Claude/Gemini/MoE/SSM） | [latest-model-architectures.md](./notes/algorithms/latest-model-architectures.md) | 🚧（DeepSeek-V3.2 config + 手算 ✅ 2026-08-20） |
| 模型追踪表（最新模型/结构/学习状态） | [model-tracker.md](./notes/algorithms/model-tracker.md) | 🚧 |
| MoE 推理挑战 | [moe-inference.md](./notes/algorithms/moe-inference.md) | 🚧 |
| Mamba / SSM | [最新模型与结构](./notes/algorithms/latest-model-architectures.md) + [速览](./notes/algorithms/remaining-theory-primer.md) | 🚧 |

## 推理系统技术
| 主题 | 笔记 | 状态 |
|------|------|:--:|
| continuous batching | [速览](./notes/algorithms/remaining-theory-primer.md) | 🚧 |
| PD 分离 | [pd-disaggregation.md](./notes/algorithms/pd-disaggregation.md) | 🚧 |
| 投机解码 speculative decoding | [speculative-decoding.md](./notes/algorithms/speculative-decoding.md) | 🚧 |
| RadixAttention | [速览](./notes/algorithms/remaining-theory-primer.md) | 🚧 |

## 训练 / 并行
| 主题 | 笔记 | 论文 | 状态 |
|------|------|------|:--:|
| 优化器：Adam / AdamW（含显存账） | [optimizers-adam.md](./notes/algorithms/optimizers-adam.md) | 论文散 | 🚧 枝干 A1（主线 A serving 之后） |
| ZeRO / FSDP | [速览](./notes/algorithms/remaining-theory-primer.md) | [zero-paper](./papers/training/zero-paper.md) | 🚧 |
| TP / PP / EP 通信 | [速览](./notes/algorithms/remaining-theory-primer.md) | — | 🚧 |

> 主题池随业界更新增删。看到新东西（X/arxiv/公众号）随时加一行。

---

## ⭐ 可选 / 进阶（B 级用不到，有余力或面试需要再碰）

| 方向 | 内容 | 去哪 |
|------|------|------|
| CUDA 指令深钻 | 手写 MMA/WGMMA/TCGen05、CuTe layout；只在锚点的 Triton/CUDA 优化完成后进入 | [roadmap/leetgpu-ladder.md](./roadmap/leetgpu-ladder.md) |
| LeetGPU 刷题 | 75 题完整索引 + 难度分级 | [notes/cuda/leetgpu-challenges.md](./notes/cuda/leetgpu-challenges.md) |
| 统一实验流程 | 知识→平台验收→服务器→profiler→归档 | [roadmap/execution-system.md](./roadmap/execution-system.md) |
| 多机多卡 | NCCL、RDMA、DeviceMesh、混合并行与排障 | [roadmap/multi-node-multi-gpu.md](./roadmap/multi-node-multi-gpu.md) |
| 论文观察 | 最新论文/官方项目筛选池 | [papers/watchlist-2026.md](./papers/watchlist-2026.md) |

---

## 里程碑

- [ ] 算子线：A4 1-pass 后置收尾；B1 Vector Add 技术验收完成但 LeetGPU 原始代码归档有缺口；MatMul 当前基线已 `GPU_VALIDATED` 并阶段性冻结，剩余 P0–P8 优化延期至 GPU 优化篇
- [ ] 算子线：B1-B3 完成，Triton 写出 Flash Attention 并记录性能差距
- [ ] 模型结构：能对着 HF config 讲清一个最新模型的 GQA/MoE/位置编码
- [ ] 算子线：C1-C4 完成，跑通 vLLM benchmark 并讲清 PagedAttention/scheduling
- [ ] 理论线：用户能讲清 ≥ 12 条（Agent 草稿不算）
- [ ] 论文：精读关键 ≥ 10 篇，每篇有一页可讲的口径（→ [papers/](./papers/)）

---

## 背景

Ascend C 算子经验 → 转 GPU。长板是算子优化方法论（tiling、内存层级、并行策略）。面试叙事："跨平台优化者，理解异构计算本质"，详见 [roadmap/interviews.md](./roadmap/interviews.md)。

*路径定稿 2026-06-06，重组为知识地图 2026-06-24*
