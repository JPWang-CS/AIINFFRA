# Lessons — 主题课

> 按主题组织的讲解内容。**不绑定"第几周"**——你走到哪学到哪，进度记在 [PATH.md](../PATH.md)。
> 规则：先看完原理，再直接去 LeetGPU 题目编辑器写题；通过后同步到本地 `solutions/`，再对照 [reference/](../reference/)。
> **所有可执行算子/内核实验的统一顺序**：看完原理 → 直接去 LeetGPU 题目编辑器写题并通过 → 同步本地 solutions → 真实 GPU benchmark → 性能分析。LeetGPU 未通过时不进入真实卡；纯阅读课不适用。
> 每课完成标准：跑通代码 + 性能/正确性数字 + 能讲清核心机制；完整课表见 [roadmap/ai-infra-curriculum.md](../roadmap/ai-infra-curriculum.md)。

## 课程列表

| # | 课 | 主题 | 配套代码 | 状态 |
|:-:|----|------|---------|------|
| 01 | [cuda-basics](01-cuda-basics.md) | CUDA 编程模型 + Vector Add | [reference/cuda/vector_add.cu](../reference/cuda/vector_add.cu) | ✅ |
| 02 | [gemm-naive](02-gemm-naive.md) | Naive GEMM + 瓶颈分析 | [solutions/cuda/](../solutions/cuda/) | ✅ |
| 03 | [gemm-tiled](03-gemm-tiled.md) | Shared memory tiling + bank conflict | [reference/cuda/gemm/](../reference/cuda/gemm/gemm.cu) | ✅ |
| 04 | [softmax](04-softmax.md) | Softmax + warp shuffle reduce | [reference/cuda/softmax/](../reference/cuda/softmax/softmax.cu) | 🚧 |
| 05 | [flash-attn-reading](05-flash-attn-reading.md) | 读懂 Flash Attention CUDA 代码 | [reference/cuda/flash_attention/](../reference/cuda/flash_attention/flash_attn.cu) | 🚧 |
| 06 | [triton-intro](06-triton-intro.md) | 第一个 Triton kernel：vec_add → matmul（B1 当前主线） | [reference/triton/](../reference/triton/) | 🚧 |

> 01-05 是 CUDA 打底阶段（"能读懂 CUDA 代码"的 B 级深度）。当前主线已切到 06 Triton 实现阶段；详细任务见 [roadmap/ai-infra-curriculum.md](../roadmap/ai-infra-curriculum.md)。

## 这些课怎么来的

源自早期按周组织的教程（week-01~04）。重组时按主题拆开：week-03 拆成 GEMM tiled（03）+ Softmax（04），week-04 拆成 Flash Attn 阅读（05）+ Triton 入门（06）。原始周文件的历史在 git 里。

## 配套

- **理论线** → [notes/algorithms/](../notes/algorithms/) — 每周一条：量化、新算法、GPU 优化算法
- **知识库** → [notes/cuda/](../notes/cuda/) · [notes/triton/](../notes/triton/) — 速查表 + 深入笔记
- **我写的代码** → [solutions/](../solutions/)
- **回顾周报** → [weekly/](../weekly/)
