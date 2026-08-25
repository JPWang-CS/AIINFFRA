# Solutions — 我自己写的代码

> 这里放**我在 LeetGPU 题目编辑器亲手写并通过验收**、再同步到本地的代码。
> 跟 [reference/](../reference/) 的区别：reference 是预置参考实现（看不抄），solutions 是我的产物。

## 规则

1. **先看原理，再去 LeetGPU 写题** —— 对着 [lessons/](../lessons/) 的框架理解题意，在题目编辑器完成实现，不复制 reference
2. **跑通才算数** —— 先在 LeetGPU 提交通过，再同步本地并做真实 GPU；本地 CPU baseline 只用于理解和排错
3. **跑通后 commit 到这里** —— 按算子分目录，命名 `{版本}_{精度}.cu`
4. 进度记到 [PATH.md](../PATH.md)，不在这里重复记

## 目录

```
solutions/
├── cuda/
│   └── gemm/             GEMM 系列
│       ├── naive_float.cu    浮点 naive
│       ├── naive_fp16.cu     fp16 naive + alpha/beta
│       ├── tiled_fp16.cu     fp16 shared memory tiling (TILE=32)
│       └── benchmark.cu      naive vs tiled 性能对比 (CPU验证 + GPU计时)
└── triton/               Triton 算子（写到 B 线时生长）
```

## 已完成

| 算子 | 文件 | 平台 | 日期 | 备注 |
|------|------|------|------|------|
| GEMM naive (float) | [cuda/gemm/naive_float.cu](cuda/gemm/naive_float.cu) | LeetGPU `2_matrix_multiplication` | 2026-06-16 | 2D grid 16×16 |
| GEMM fp16 naive | [cuda/gemm/naive_fp16.cu](cuda/gemm/naive_fp16.cu) | LeetGPU `22_gemm` | 2026-06-22 | alpha/beta BLAS |
| GEMM fp16 tiled | [cuda/gemm/tiled_fp16.cu](cuda/gemm/tiled_fp16.cu) | LeetGPU `22_gemm` | 2026-06-25 | TILE=32 shared mem |
| GEMM benchmark | [cuda/gemm/benchmark.cu](cuda/gemm/benchmark.cu) | RTX 4090 (AutoDL) | 2026-06-25 | K=2048/8192 naive vs tiled |
