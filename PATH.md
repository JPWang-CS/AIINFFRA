# PATH — 知识地图 + 进度

> **这是知识地图,也是唯一的进度源。** 想找任何东西(某算子、某理论、某篇论文)——这里一定有条目,一跳到位。
> 进来先看 → [NOW.md](./NOW.md)（现在做什么 + 接下来）。这里是全貌。
> 方向：ML 系统工程师 · Triton 为主力 · CUDA 为底层 · 约 1 年

图例：✅ 完成　🚧 进行中　⏳ 待做　⭐ 可选/进阶

---

## 两条平行路径

地位一样,每周并行各推一点。一条动手、一条理解。

- **算子线**（动手写代码，学 GPU/算子）→ 产出能跑的代码，LeetGPU/本地跑通算数
- **理论线**（学算法/理论）→ 产出一页笔记进 [notes/algorithms/](./notes/algorithms/)，能讲清原理算数

```
算子线   A CUDA打底 ─→ B Triton ─→ C 推理系统 ─→ D 分布式 ─→ E Agent
理论线   GPU优化算法 · 量化 · 注意力演进 · 模型架构 · 推理系统技术 · 训练/并行
         （6 子类滚动挑，跟业界动态走）
```

---

# 算子线（动手）

## A — CUDA 打底（B 级，"读得懂"）

> 目标：能写 tiled GEMM、读懂 Flash Attn CUDA、知道 Triton 底层在干什么。**不深入 tensor core**（那是 ⭐ 可选，见底部）。

| 阶段 | 课 | 自己写 | 验收 | 笔记 | 参考 | 状态 |
|:-:|----|--------|------|------|------|:--:|
| A1 | [01 cuda-basics](./lessons/01-cuda-basics.md) | Vector Add | LeetGPU 跑通 + 能算 bandwidth | [memory-model](./notes/cuda/memory-model.md) · [warp-and-sync](./notes/cuda/warp-and-sync.md) | [vector_add.cu](./reference/cuda/vector_add.cu) · 我的→[solutions](./solutions/) | ✅ |
| A2 | [02 gemm-naive](./lessons/02-gemm-naive.md) | `gemm_naive` (float) | LeetGPU 跑通（2026-06-16） | [memory-model](./notes/cuda/memory-model.md) | [gemm.cu](./reference/cuda/gemm/gemm.cu) · 我的→[gemm_naive.cu](./solutions/cuda/gemm_naive.cu) | ✅ |
| A2+ | — | `gemm_fp16_naive` | LeetGPU fp16 跑通（2026-06-22）·[review](./notes/cuda/code-review-gemm-fp16-naive.md) | — | 我的→[gemm_fp16_naive.cu](./solutions/cuda/gemm_fp16_naive.cu) | ✅ |
| A3 | [03 gemm-tiled](./lessons/03-gemm-tiled.md) | `gemm_tiled` (float) | GFLOPS 比 naive ≥ 5× | [memory-model §3.3](./notes/cuda/memory-model.md) | [gemm.cu](./reference/cuda/gemm/gemm.cu) | ✅ |
| A3+ | — | `gemm_fp16_tiled` | LeetGPU fp16 跑通（2026-06-22）·TILE=32·⚠️ K=16 太小无加速，待 4090 大 K 验证 | — | 我的→[gemm_fp16_tiled.cu](./solutions/cuda/gemm_fp16_tiled.cu) | ✅ |
| A4 | [04 softmax](./lessons/04-softmax.md) | `softmax_naive` | LeetGPU 跑通，结果正确 | [warp-and-sync §4](./notes/cuda/warp-and-sync.md) | [softmax.cu](./reference/cuda/softmax/softmax.cu) | ⏳ |
| A5 | [05 flash-attn-reading](./lessons/05-flash-attn-reading.md) | 读代码（不手写） | 能标注每个 `__syncthreads` 作用 | [triton-under-the-hood](./notes/cuda/triton-under-the-hood.md) | [flash_attn.cu](./reference/cuda/flash_attention/flash_attn.cu) · [论文](./papers/attention/flash-attention.md) | ⏳ |

**阶段出口**：A5 完成 = CUDA B 级达成，切 B 线。

## B — Triton 算子（主力工具）

> 目标：用 Triton 写常见 ML 算子，接近手写 CUDA 性能。从这里 Triton 成主力。

| 阶段 | 课 | 自己写 | 验收 | 参考 | 状态 |
|:-:|----|--------|------|------|:--:|
| B1 | [06 triton-intro](./lessons/06-triton-intro.md) | Triton vec_add + matmul | GPU/CPU 模拟跑通 | [matmul.py](./reference/triton/matmul/matmul.py) | ⏳ |
| B2 | _按需生成_ | Triton fused softmax | 对比 PyTorch 正确 + 提速 | — | ⏳ |
| B3 | _按需生成_ | Triton flash attention | 对比 PyTorch ref 正确 | [flash_attn.py](./reference/triton/flash_attention/flash_attn.py) | ⏳ |
| B4 | _按需生成_ | Triton GQA / fused MLP | 正确性 + autotuning | [activations.cuh](./reference/cuda/include/activations.cuh)（料） | ⏳ |

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
| D 分布式 | DP/FSDP/TP/PP | 够面试，能画通信图 | [roadmap/distributed.md](./roadmap/distributed.md) | ⏳ |
| E Agent | MCP/Tool Use/RAG | 熟悉 + 1 个 demo | [roadmap/agents.md](./roadmap/agents.md) | ⏳ |

---

# 理论线（理解）

> 每周一条,产出一页笔记进 [notes/algorithms/](./notes/algorithms/)。有标志性论文的,论文精读放 [papers/](./papers/)、这里写"机制+怎么实现"并互链（边界规矩见 [algorithms/README](./notes/algorithms/README.md)）。

## GPU 优化算法
| 主题 | 笔记 | 状态 |
|------|------|:--:|
| online softmax（Flash 的心脏） | [online-softmax.md](./notes/algorithms/online-softmax.md) | ✅ |
| parallel reduce / prefix sum | [parallel-reduce.md](./notes/algorithms/parallel-reduce.md) | ✅ |
| Norm 的 reduce 模式（LayerNorm/RMSNorm） | _待写_ · 料→[layernorm.cu](./reference/cuda/layernorm/layernorm.cu) | ⏳ |
| work partitioning（Flash 2 的思路） | _待写_ | ⏳ |

## 量化
| 主题 | 笔记 | 论文 | 状态 |
|------|------|------|:--:|
| 数值格式 INT8 / FP8 | [quantization-int8-fp8.md](./notes/algorithms/quantization-int8-fp8.md) | — | ✅ |
| AWQ | _待写_ | _待建_ | ⏳ |
| GPTQ | _待写_ | _待建_ | ⏳ |
| SmoothQuant / KV Cache 量化 | _待写_ | — | ⏳ |

## 注意力演进
| 主题 | 笔记 | 论文 | 状态 |
|------|------|------|:--:|
| MHA→MQA→GQA→MLA | ⏳ | [gqa.md](./papers/attention/gqa.md) | ⏳ |
| Flash Attention 1→2→3 | [flash-attention-mechanism.md](./notes/algorithms/flash-attention-mechanism.md) | [FA1](./papers/attention/flash-attention.md) · [FA2](./papers/attention/flash-attention-2.md) | ✅ |
| MLA（DeepSeek-V2/V3） | [mla-deepseek.md](./notes/algorithms/mla-deepseek.md) | DeepSeek-V2 | ✅ |
| 线性注意力 / Ring Attention | _待写_ | — | ⏳ |

## 模型架构
| 主题 | 笔记 | 状态 |
|------|------|:--:|
| MoE 推理挑战 | [moe-inference.md](./notes/algorithms/moe-inference.md) | ✅ |
| Mamba / SSM | _待写_ | ⏳ |

## 推理系统技术
| 主题 | 笔记 | 状态 |
|------|------|:--:|
| continuous batching | _待写_ | ⏳ |
| PD 分离 | [pd-disaggregation.md](./notes/algorithms/pd-disaggregation.md) | ✅ |
| 投机解码 speculative decoding | [speculative-decoding.md](./notes/algorithms/speculative-decoding.md) | ✅ |
| RadixAttention | _待写_ | ⏳ |

## 训练 / 并行
| 主题 | 笔记 | 论文 | 状态 |
|------|------|------|:--:|
| ZeRO / FSDP | _待写_ | [zero-paper](./papers/training/zero-paper.md) | ⏳ |
| TP / PP / EP 通信 | _待写_ | — | ⏳ |

> 主题池随业界更新增删。看到新东西（X/arxiv/公众号）随时加一行。

---

## ⭐ 可选 / 进阶（B 级用不到，有余力或面试需要再碰）

| 方向 | 内容 | 去哪 |
|------|------|------|
| CUDA 深钻 | GEMM vec4 / double buffer / tensor core，各算子钻到峰值 | [roadmap/leetgpu-ladder.md](./roadmap/leetgpu-ladder.md) |
| LeetGPU 刷题 | 75 题完整索引 + 难度分级 | [notes/cuda/leetgpu-challenges.md](./notes/cuda/leetgpu-challenges.md) |

---

## 里程碑

- [ ] 算子线：A3-A5 完成，能读懂 Flash Attn CUDA
- [ ] 算子线：B1-B3 完成，Triton 写出 Flash Attention
- [ ] 算子线：C1-C2 完成，讲清 PagedAttention
- [ ] 理论线：积累 ≥ 12 条笔记
- [ ] 论文：精读关键 ≥ 10 篇（→ [papers/](./papers/)）

---

## 背景

Ascend C 算子经验 → 转 GPU。长板是算子优化方法论（tiling、内存层级、并行策略）。面试叙事："跨平台优化者，理解异构计算本质"，详见 [roadmap/interviews.md](./roadmap/interviews.md)。

*路径定稿 2026-06-06，重组为知识地图 2026-06-24*
