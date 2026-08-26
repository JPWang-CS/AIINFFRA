# 实验记录模板

## 目标

- 问题：
- 类型：算子 / 通信 / 训练 / 推理
- 预计瓶颈：
- 硬件假设：哪个层次/资源限制性能？
- 可控旋钮：tile / warp / stage / layout / dtype / fusion / stream / collective
- 验收标准：

## 代码与状态索引

- 当前状态：`WIP` / `LEETGPU_PASS` / `GPU_VALIDATED` / `COMPLETE`
- 对应 lesson：
- LeetGPU 题目/题号/语言：
- LeetGPU 原始 `solve`/kernel 归档：
- 本地可执行代码：
- reference（仅对照）：
- 代码归属：用户 / Agent review / harness

## 环境

| 项 | 值 |
|----|----|
| 日期 / commit | |
| GPU / CC / 数量 | |
| Driver / CUDA | |
| PyTorch / Triton / NCCL | |
| Topology / interconnect | |

## 性能上限与强基线

- 数值语义：IEEE FP32 / TF32 / FP16 / BF16 / FP8 / 其他
- FLOPs / 最少 bytes / arithmetic intensity：
- 理论峰值与理论带宽：
- 实测 copy/compute roof：
- 同语义强 baseline：cuBLAS / PyTorch / 官方 Triton / FlashAttention / 其他
- 目标 shape 集：不能只选一个有利 shape

## 正确性门

- LeetGPU 题号或自建 reference：
- 平台通过日期/提交版本：
- shape / dtype / seed：
- atol / rtol / max error：
- 结果：

## Benchmark

- warmup / repeats / synchronization：
- baseline：

| 版本 | 配置 | 时间 | GB/s / GFLOPS / tokens/s | 相对 baseline | roof/reference % |
|------|------|------|--------------------------|---------------|------------------|
| | | | | | |

## Profiler 证据

- 工具/命令：
- 主要瓶颈：
- 关键指标：
- 源码对应位置：
- PTX/SASS 目标指令是否出现：

## 单变量实验

| 改动 | 硬件假设 | 预期 counter 变化 | 实测 counter / 性能 | 结论 |
|------|----------|---------------------|----------------------|------|
| | | | | |

## 停止判断

- 是否接近同语义强 baseline 或实测 roof：
- 最近两轮收益是否超过噪声：
- 尚存差距及证据：
- 当前架构限制：
- 下一步：继续 / 跨架构复测 / 归档

## 失败版本

- 现象：
- 原因：
- 如何定位：

## 一分钟口径

是什么 → 为什么慢 → 做了什么 → 数字证据 → 取舍。
