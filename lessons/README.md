# Lessons — 主题课

> 按主题组织的讲解内容。**不绑定"第几周"**——你走到哪学到哪，进度记在 [PATH.md](../PATH.md)。
> 规则：每个可执行章节固定两段：**LeetGPU：正确性与代码归档**、**服务器：真实性能**。通过前不进入服务器；`reference/` 只在自己的版本提交后对照。
> 每课必须有一张当前单元卡：题目入口、代码快照/路径、平台状态、服务器状态、下一步；完成状态仍以 [PATH.md](../PATH.md) 为准。
> 每课完成标准：跑通代码 + 性能/正确性数字 + 能讲清核心机制；完整课表见 [roadmap/ai-infra-curriculum.md](../roadmap/ai-infra-curriculum.md)。

## 课程列表

| # | 课 | 主题 | 配套代码 | 状态 |
|:-:|----|------|---------|------|
| 01 | [cuda-basics](01-cuda-basics.md) | CUDA 编程模型 + Vector Add | [Lesson code](01-cuda-basics.md#22-在-leetgpu-上跑推荐) · 本地归档缺失 | ⚠️ |
| 02 | [gemm-naive](02-gemm-naive.md) | Naive GEMM + 瓶颈分析 | [Lesson](02-gemm-naive.md) · [solutions/cuda/](../solutions/cuda/) | ✅ |
| 03 | [gemm-tiled](03-gemm-tiled.md) | Shared memory tiling + bank conflict | [Lesson](03-gemm-tiled.md) · [solutions/cuda/](../solutions/cuda/) | ✅ |
| 04 | [softmax](04-softmax.md) | Softmax + warp shuffle reduce | [reference/cuda/softmax/](../reference/cuda/softmax/softmax.cu) | 🚧 |
| 05 | [flash-attn-reading](05-flash-attn-reading.md) | 读懂 Flash Attention CUDA 代码 | [reference/cuda/flash_attention/](../reference/cuda/flash_attention/flash_attn.cu) | 🚧 |
| 06 | [triton-intro](06-triton-intro.md) | 第一个 Triton kernel：vec_add → matmul（B1 当前主线） | [Lesson code](06-triton-intro.md#本课代码与进度索引从这里一眼查看) · [solutions/triton/](../solutions/triton/) | 🚧 |
| 07 | [triton-debugging](07-triton-debugging.md) | Triton 调试：interpreter、打印、断言、sanitizer、数值误差 | [Lesson 06 MatMul](06-triton-intro.md#56-服务器真实性能) · [Triton Debugging](07-triton-debugging.md) | 配套 |

> 01-05 是 CUDA 打底阶段（"能读懂 CUDA 代码"的 B 级深度）。当前主线已切到 06 Triton 实现阶段；详细任务见 [roadmap/ai-infra-curriculum.md](../roadmap/ai-infra-curriculum.md)。

## 这些课怎么来的

源自早期按周组织的教程（week-01~04）。重组时按主题拆开：week-03 拆成 GEMM tiled（03）+ Softmax（04），week-04 拆成 Flash Attn 阅读（05）+ Triton 入门（06）。原始周文件的历史在 git 里。

## 配套

- **理论线** → [notes/algorithms/](../notes/algorithms/) — 每周一条：量化、新算法、GPU 优化算法
- **知识库** → [notes/cuda/](../notes/cuda/) · [notes/triton/](../notes/triton/) — 速查表 + 深入笔记
- **我写的代码** → [solutions/](../solutions/)
- **回顾周报** → [weekly/](../weekly/)
