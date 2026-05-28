# Triton Kernels (Phase 2: Week 11-18)

OpenAI Triton 实现的 ML 算子，用于对比 CUDA 版本，理解 block-level 编程模型。

## 学习目标

- 理解 Triton 的 programming model（block-level、no warp-level control）
- 能用 Triton 写常见 ML kernel
- 理解 Flash Attention 的 Triton 实现

## 目录

| 目录 | 内容 | LeetGPU 对应 | 状态 |
|------|------|-------------|------|
| [matmul/](./matmul/) | Triton 版 GEMM | Matrix Multiplication (Easy) | ⏳ |
| [fused_mlp/](./fused_mlp/) | Fused SiLU/GeGLU + Linear | - | ⏳ |
| [flash_attention/](./flash_attention/) | Triton 版 Flash Attention | Flash Attention (Hard) | ⏳ |
| [notes/](./notes/) | Triton 学习笔记 | - | ⏳ |

## Triton vs CUDA 关键差异

| 概念 | CUDA | Triton |
|------|------|--------|
| 编程粒度 | Thread-level (每个 thread 独立) | Block-level (操作 tile) |
| 内存管理 | 显式 shared memory / register | 编译器自动管理 |
| 同步 | `__syncthreads()` | 隐式（block 边界） |
| 优化重点 | Occupancy、bank conflict、coalescing | Tile size、autotuning |
| 编译 | nvcc → PTX → SASS | Triton → MLIR → PTX → SASS |

## 环境

```bash
pip install triton
# 注意：Triton 需要 NVIDIA GPU，LeetGPU 支持在线运行 Triton
```

## 相关资源

- [Triton 官方文档](https://triton-lang.org/)
- [Triton 教程（含中文）](https://github.com/dsl-learn/LeetGPU)
