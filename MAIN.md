# MAIN — AI Infra 学习入口

> 方向：ML 系统工程师 · Triton 为主力 · CUDA 为底层 · 推理系统并行
> 平台：LeetGPU 入门，真机 GPU 后续搭建
> 代码：从空文件自己写，仓库成品仅作参考

---

## 路线总览

```
Phase 1 (Week 1-4)        Phase 2 (Week 5-?)         Phase 3                  持续
CUDA 基础                  Triton 算子 + 推理系统      分布式 + Agent            
████████                   ████████████████           ████                     ████
                                                                              
学会写 tiled GEMM          用 Triton 写 GEMM          了解概念即可              Agent
能读 Flash Attn CUDA 代码   vLLM 源码分析              够面试用                 贯穿
知道 Triton 底层在干什么    PagedAttention/KV Cache                             
```

---

## 每周教程

| 周 | 主题 | 教程 | 自己写什么 | 检验 |
|:--:|------|------|-----------|------|
| 1 | CUDA 概念 + 第一个 kernel | [week-01.md](./weekly/week-01.md) | Vector Addition | ✅ LeetGPU 跑通 |
| 2 | GEMM naive + 搭 GPU 服务器 | [week-02.md](./weekly/week-02.md) | `gemm_naive` | ✅ LeetGPU 跑通 |
|   | ✅ 4090 已购 + 环境已配 + vector_add 跑通 | — | nvcc/CUDA 12.4/PyTorch 2.5.1 | 696 GB/s, 0 error |
|   | ⚠️ SSH 被公司防火墙拦截，用 Jupyter Web Terminal | — | — | — |
|   | ✅ GEMM naive LeetGPU 跑通（2026-06-16） | — | `2_matrix_multiplication` float 版 | 2D grid, atomicAdd |
|   | ✅ GEMM fp16 naive LeetGPU 跑通（2026-06-22） | [gemm_fp16_naive.cu](./cuda-kernels/gemm/gemm_fp16_naive.cu) | half 精度 + alpha/beta | [Code Review](./cuda-kernels/notes/code-review-gemm-fp16-naive.md) |
|   | ✅ GEMM fp16 tiled LeetGPU 跑通（2026-06-22） | [gemm_fp16_tiled.cu](./cuda-kernels/gemm/gemm_fp16_tiled.cu) | TILE=32 shared mem | LeetGPU 测试 K=16 太小无加速，待 4090 大 K 验证 |
| 3 | GEMM tiled + Softmax | week-03.md | `gemm_tiled`, `softmax_naive` | GFLOPS 提升 5×+ |
| 4 | 读 Flash Attn CUDA + Triton 入门 | week-04.md | 读代码 + Triton matmul | 能讲清 tiling 流程 |
| 5+ | Triton 算子 + 推理系统 | 待定 | 待定 | 待定 |

> 最后两周（Week 3-4）收束到"能读懂 CUDA 代码"即可，不深入 tensor core。

---

## 参考索引

| 找什么 | 路径 |
|--------|------|
| 方向讨论结论 | [Memory: direction-triton-first] |
| 环境/工作流约定 | [Memory: environment-and-workflow] |
| 优化阶梯方案（旧，CUDA 深钻用） | [leetgpu-roadmap.md](./leetgpu-roadmap.md) |
| LeetGPU 75 题索引 | [cuda-kernels/notes/leetgpu-challenges.md](./cuda-kernels/notes/leetgpu-challenges.md) |
| GPU 架构对比（Ascend ↔ NVIDIA） | [cuda-kernels/notes/gpu-architecture.md](./cuda-kernels/notes/gpu-architecture.md) |
| 编译配置 | [cuda-kernels/CMakeLists.txt](./cuda-kernels/CMakeLists.txt) |
| 工具库 | [cuda-kernels/include/](./cuda-kernels/include/) |
| 论文清单 | [papers/README.md](./papers/README.md) |
| 面试大纲 | [interviews/README.md](./interviews/README.md) |
| 项目概览 | [README.md](./README.md) |

---

## 规则

1. **自己写**——仓库 `.cu` 是参考，从空文件开始
2. **LeetGPU 为主**——跑通才算完成
3. **每周围绕一个主题**——不并行
4. **每周日 commit**：`week-X: <实际做了什么>`

---

*定稿于 2026-06-06*
