# Reference — Triton Kernels

> **参考实现，看不抄。** 学的时候对着 [lessons/](../../lessons/) 从空文件自己写，写完跑通存进 [solutions/](../../solutions/)。
> 笔记和速查在 [notes/triton/](../../notes/triton/)，学习路径在 [PATH.md](../../PATH.md)。

## 目录

| 目录 | 内容 | 对应课 |
|------|------|--------|
| [matmul/](./matmul/) | Triton 版 GEMM（L2 cache 优化、GROUP_M swizzle） | [Lesson 06](../../lessons/06-triton-intro.md) |
| [flash_attention/](./flash_attention/) | Triton 版 Flash Attention（online softmax） | [Lesson 05](../../lessons/05-flash-attn-reading.md) → B3 |

## Triton vs CUDA 关键差异

| 概念 | CUDA | Triton |
|------|------|--------|
| 编程粒度 | Thread-level (每个 thread 独立) | Block-level (操作 tile) |
| 内存管理 | 显式 shared memory / register | 编译器自动管理 |
| 同步 | `__syncthreads()` | 隐式（block 边界） |
| 优化重点 | Occupancy、bank conflict、coalescing | Tile size、autotuning |
| 编译 | nvcc → PTX → SASS | Triton → MLIR → PTX → SASS |

完整对比 → [notes/triton/triton-vs-cuda.md](../../notes/triton/triton-vs-cuda.md)

## 环境

```bash
pip install triton
# 注意：Triton 需要 NVIDIA GPU，LeetGPU 支持在线运行 Triton
```

## 相关资源

- [Triton 官方文档](https://triton-lang.org/)
- [Triton 速查表](../../notes/triton/triton-cheatsheet.md)
