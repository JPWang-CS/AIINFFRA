# CUDA / GPU 底层知识入口

> 路线和实验顺序以 [GPU 底层架构与性能优化课程](../../roadmap/gpu-foundations.md) 为准；本目录保存可跨算子复用的知识，不维护独立进度。

## 知识地图

| 问题 | 笔记 | 用在何处 |
|------|------|----------|
| GPU 从整机到 SM、指令和架构代际 | [gpu-architecture-layers.md](gpu-architecture-layers.md) | 全阶段总地图 |
| global/shared/register 与访问模式 | [memory-model.md](memory-model.md) | Vector Add、MatMul、Softmax |
| warp、同步、reduction、divergence | [warp-and-sync.md](warp-and-sync.md) | Softmax、Norm、Attention |
| Triton 到 GPU 的编译和映射 | [triton-under-the-hood.md](triton-under-the-hood.md) | B1–B5 |
| LeetGPU 题目地图 | [leetgpu-challenges.md](leetgpu-challenges.md) | 正确性门与代码归档 |
| CUDA 常用语法速查 | [cuda-cheatsheet.md](cuda-cheatsheet.md) | 写题时查询 |
| FlashAttention CUDA 读码 | [flash-attn-reading.md](flash-attn-reading.md) | B3/FA2 |

旧的 [gpu-architecture.md](gpu-architecture.md) 是早期 NVIDIA/昇腾速记；涉及架构事实时以新版分层笔记和 NVIDIA 官方文档为准。

## 固定使用方式

```text
当前算子提出问题
-> 从本页找到对应底层知识
-> 回到 LeetGPU 从空题面实现并归档原始代码
-> 真实服务器建立 baseline
-> Nsight / PTX / SASS 找证据
-> 单变量优化
```

知识笔记不能替代实验状态；只有对应 lesson 单元卡和 `PATH.md` 可以表达当前进度。

## 外部资料职责

- NVIDIA Programming Guide / Tuning Guide：架构和编程语义的权威事实。
- Nsight 文档：counter、roofline 和诊断方法。
- GPU MODE / NVIDIA samples：最小实验和讲解。
- SGEMM_CUDA / CUTLASS / CCCL：通过后读码，对照生产级优化组织方式。
- LeetGPU：有对应题面时的正确性门；真实 GPU 服务器负责性能结论。
