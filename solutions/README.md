# Solutions — 我自己写的代码

> 这里放**我亲手写、并在 LeetGPU（或本地）跑通**的代码。
> 跟 [reference/](../reference/) 的区别：reference 是预置参考实现（看不抄），solutions 是我的产物。

## 规则

1. **从空文件开始自己写** —— 对着 [lessons/](../lessons/) 的框架和 TODO，不复制 reference
2. **跑通才算数** —— LeetGPU 提交通过 / 本地对比 CPU baseline 正确
3. **跑通后 commit 到这里** —— 文件名标注题目/版本
4. 进度记到 [PATH.md](../PATH.md)，不在这里重复记

## 目录

```
solutions/
├── cuda/      CUDA 算子（自己写的版本）
└── triton/    Triton 算子（写到 B 线时生长）
```

## 已完成

| 算子 | 文件 | 平台 | 日期 | 备注 |
|------|------|------|------|------|
| GEMM naive (float) | [cuda/gemm_naive.cu](cuda/gemm_naive.cu) | LeetGPU `2_matrix_multiplication` | 2026-06-16 | 2D grid 16×16 |
| GEMM fp16 naive | [cuda/gemm_fp16_naive.cu](cuda/gemm_fp16_naive.cu) | LeetGPU `22_gemm` fp16 | 2026-06-22 | alpha/beta，[code review](../notes/cuda/code-review-gemm-fp16-naive.md) |
| GEMM fp16 tiled | [cuda/gemm_fp16_tiled.cu](cuda/gemm_fp16_tiled.cu) | LeetGPU `22_gemm` fp16 | 2026-06-22 | TILE=32，⚠️ K=16 太小待 4090 验证加速比 |
