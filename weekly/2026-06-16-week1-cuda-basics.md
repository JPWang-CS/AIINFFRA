# Week 1 — CUDA 基础 + Vector Add

> 2026-06-16 前后 · 算子线 A1 · 平台：LeetGPU

## 做了什么

### 理论学习
- 理解了 CUDA 编程模型：thread → warp → block → grid 层次
- 理解了 Ascend vs CUDA 的核心差异：硬件控制数据搬运 vs 你告诉每个线程算哪个元素
- 掌握了 GPU 内存层级：Register → Shared Memory → L2 → Global Memory (HBM)
- 理解了 warp 概念（CUDA 独有的 32-thread 执行单元）和 warp divergence
- 能用 Ascend 对应关系解释 CUDA 概念（L1 Buffer ≈ Shared Memory, Cube ≈ Tensor Core）

### 动手写代码
- **Vector Add kernel**：在 LeetGPU 上跑通，`C[i] = A[i] + B[i]`
- 掌握了 grid-stride loop：`for (int i = idx; i < N; i += stride)` 处理 N > grid 的情况
- 加了 CUDA Error Check（`CUDA_CHECK` 宏）
- 用 `cudaEvent` 计时 + 算 bandwidth

### 环境搭建
- 购买 4090 GPU + 配置 CUDA 12.4 + PyTorch 2.5.1
- Vector Add 本地跑通：**696 GB/s bandwidth, 0 error**
- ⚠️ SSH 被公司防火墙拦截 → 改用 Jupyter Web Terminal 操作

## 关键数据

| 指标 | 值 |
|------|-----|
| Vector Add Bandwidth (4090) | 696 GB/s |
| 线程配置 | 256 threads/block |
| 数据量 | 1M elements (FP32) |

## 卡点 / 怎么解决的

- **SSH 被公司防火墙拦截**：SSH 端口被封 → 用 Jupyter Web Terminal（浏览器操作，不需要 SSH）
- **理解 warp divergence**：这是 CUDA 独有的概念，Ascend 没有 → 通过对比 Ascend Vector Unit 的 SIMD 宽度和 CUDA 的分支代价来理解

## 面试可用点

- 能说清 thread/block/grid/warp 四个概念
- 能解释为什么 Vector Add 是 memory-bound（3N float 读写 ÷ 耗时 = bandwidth）
- 能说 Ascend 和 CUDA 编程模型的 3 个关键差异
- Warp divergence 的原理（面试高频题）

## 产出物

- [x] LeetGPU Vector Addition 跑通
- [x] 4090 本地 vector_add 跑通，696 GB/s
- [x] 笔记：GPU 架构对比、内存模型、CUDA API 速查
