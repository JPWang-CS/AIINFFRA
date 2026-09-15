# Solutions — 我的代码与 WIP 草稿

> 这里放我的可执行代码，以及明确标记为 `WIP` 的当前草稿。只有通过正确性门并完成所需证据的版本，才算完成产物。
> 跟 [reference/](../reference/) 的区别：reference 是预置参考实现（看不抄），solutions 是我的产物。

## 规则

1. **先看原理，再去 LeetGPU 写题** —— 对着 [lessons/](../lessons/) 的框架理解题意，在题目编辑器完成实现，不复制 reference
2. **状态分层** —— `WIP` 可以保存进展但不算完成；`LEETGPU_PASS` 必须有题号、语言、日期和原始 `solve`；`GPU_VALIDATED` 还必须有真实 GPU 正确性/性能证据
3. **跑通后 commit 到这里** —— 按算子分目录，命名 `{版本}_{精度}.cu`
4. 进度记到 [PATH.md](../PATH.md)；每个当前单元的代码入口、快照和验收证据同时在对应 lesson 一眼可见

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

## 已知归档缺口

| 单元 | 现状 |
|------|------|
| CUDA Vector Add（A1） | Lesson 01 有 LeetGPU 代码快照，但没有单独的本地 `solve` 文件 |
| Triton Vector Add（B1） | 已有 AutoDL wrapper 和 benchmark，但没有单独的 LeetGPU 原始 `solve` 文件 |

## 当前产物

| 算子 | 文件 | 状态 |
|------|------|------|
| Triton MatMul LeetGPU 最终版 | [triton/matmul_leetgpu.py](triton/matmul_leetgpu.py) | `LEETGPU_PASS`：LeetGPU #02，Triton，2026-08-28；SuccessPublicTrace（A100-80GB，24.54 ms，55.3th percentile） |
| Triton MatMul服务器适配版 | [triton/matmul.py](triton/matmul.py) | `GPU_VALIDATED`：RTX3090 IEEE FP32当前最佳20.830 ms / 19,794.1 GFLOPS / `torch.mm` 80.3%；22.033 ms为早期sweep历史结果 |

## 历史 WIP

| 算子 | 文件 | 说明 |
|------|------|------|
| Triton MatMul | [triton/matmul_leetgpu_wip.py](triton/matmul_leetgpu_wip.py) | 历史 TF32 默认精度失败版本；4×4 case 最大绝对误差为 `0.1275177001953125`，保留用于复盘，不代表当前状态 |

MatMul单元总体为`GPU_VALIDATED`，尚未`COMPLETE`；现有数字保留，后续按[第三篇极致性能课程](../roadmap/curriculum/performance/README.md)补Nsight Compute、PTX/SASS、低精度和多shape。当前实践动作是Triton Softmax服务器闭环；Vector Add原始LeetGPU `solve`归档缺口保持不变。
