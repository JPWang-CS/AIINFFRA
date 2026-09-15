# PATH — AIINFFRA唯一进度源

> 当前动作看 [NOW.md](./NOW.md)，模块化课程看 [roadmap/curriculum/README.md](./roadmap/curriculum/README.md)，历史证据看 [HISTORY.md](./HISTORY.md)。本文件只维护状态和路由，不承载课程正文。

## 1. 总目标与两条线

目标：从Ascend NPU算子开发转向NVIDIA GPU高性能LLM算子与性能优化；vLLM是系统落地加分项。

- 实践线：GPU 与 CUDA 基础 → 算子实现与优化（包含低精度与量化）→ 模型分析与系统应用。优化在对应算子内连续展开，详见[总路线](./roadmap/curriculum/README.md)。
- [论文线](./notes/algorithms/README.md)：独立阅读、公式推导与作者关键代码，不默认实践。

## 2. 状态与门槛

`WIP → LEETGPU_PASS → GPU_VALIDATED → COMPLETE`。

有LeetGPU题面的算子：原理 → 平台从题面实现并归档原始代码 → 真实GPU → 性能优化。无平台题面的架构、profiling和系统实验使用official/reference门，不伪造`LEETGPU_PASS`。

## 3. 当前确认事实

| 项目 | 状态 | 已确认事实 | 下一门槛 |
|---|---|---|---|
| Triton Vector Add | `GPU_VALIDATED`，归档有缺口 | RTX3090 840.1 GB/s；`torch.add` 843.0 GB/s；平台原始`solve`未单独归档 | 有机会补原始版本 |
| Triton MatMul | `GPU_VALIDATED`，阶段性收口 | LeetGPU原始版已归档；RTX3090 20.830 ms / 19,794.1 GFLOPS / `torch.mm` 80.3%；已有Nsys P0-lite | 在 GEMM 章内继续 NCU、PTX/SASS、低精度与多 shape |
| Triton Softmax | `LEETGPU_PASS`，当前实践 | LeetGPU #5，0.29 ms，47.0th percentile，原始版已归档 | RTX3090二维row-wise correctness、ms、effective GB/s |
| FlashAttention-1 | 理论/读码完成 | CUDA读码与问题记录已保存 | 后续进入第二篇Attention课程实践 |
| FlashAttention-2 | 阅读完成（2026-09-02） | 仅论文学习完成 | 实践状态独立，尚未开始 |
| MLA | 论文线`WIP` | 统一笔记已形成 | 公式、cache账本、权重吸收、作者代码回答 |

旧CUDA实现、Agent草稿、失败实验、weekly和reference继续保留，详见[HISTORY](./HISTORY.md)与各代码README。

## 4. 实践课程状态

| 模块 | 路由 | 状态 |
|---|---|---|
| GPU 架构与性能 | [课程路由](./roadmap/curriculum/gpu/README.md) | 五章正文；用户阅读与真实实验状态不因教材生成而改变 |
| 算子实现与优化 | [算子路由](./roadmap/curriculum/operators/README.md) | 九类算子正文及页内实践已形成；原有实验状态保持独立 |
| 算子内的深入优化 | 各算子正文；[实验检查表](./roadmap/curriculum/performance/README.md)按需使用 | MatMul已有部分证据；未另设优化课 |
| 量化与模型 GPU 分析 | [量化](./roadmap/curriculum/quantization/README.md) · [Prefill/Decode/Mini Transformer](./roadmap/curriculum/model-analysis/README.md) | 量化及模型分析/Mini Transformer 正文已形成；组合实测未新增 |
| 系统应用 | [vLLM/多GPU/MoE](./roadmap/curriculum/systems/README.md) | vLLM 与多 GPU 正文已形成；服务与通信实验未在本机执行 |

### 第一篇当前入口

[第一篇路由](./roadmap/curriculum/gpu/README.md)按整体结构、执行、存储、协作、性能证据五章组织；教材路由已更新，当前实践与论文焦点不变，未新增用户阅读完成或GPU验证事实。

## 5. 论文线状态

| 主题 | 状态 | 入口 |
|---|---|---|
| Online Softmax / Parallel Reduce | ✅ 已掌握 | [理论入口](./notes/algorithms/README.md) |
| FlashAttention-1 | ✅ 已消化并读CUDA | [FA机制](./notes/algorithms/flash-attention-mechanism.md) |
| FlashAttention-2 | ✅ 阅读完成 | [FA2](./notes/algorithms/flash-attention-2.md) |
| MLA | 🚧 当前 | [MLA](./notes/algorithms/mla-deepseek.md) |
| DSA | 下一篇 | [DSA](./notes/algorithms/dsa-sparse-attention.md) |
| 最新论文 | 持续观察 | [watchlist](./papers/watchlist-2026.md) |

核心与重要最新论文至少达到精读+作者关键代码；是否复现由用户另行决定。

## 6. 里程碑

- [ ] 第一章真实服务器验收；
- [ ] GPU基础五章逐章阅读与必要实践验证；
- [ ] 主要算子族各有实现与真实GPU baseline；
- [ ] 至少三个核心锚点完成完整极致性能阶段；
- [ ] 量化完成算法、kernel、精度—性能闭环；
- [ ] Mini Transformer完成Prefill/Decode分析和自写算子接入；
- [ ] vLLM中解释并验证至少一个真实优化；
- [ ] 多GPU或MoE通信—计算实验。
