# 实验记录模板

## 目标

- 问题：
- 类型：算子 / 通信 / 训练 / 推理
- 预计瓶颈：
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

## 正确性门

- LeetGPU 题号或自建 reference：
- 平台通过日期/提交版本：
- shape / dtype / seed：
- atol / rtol / max error：
- 结果：

## Benchmark

- warmup / repeats / synchronization：
- baseline：

| 版本 | 配置 | 时间 | GB/s / GFLOPS / tokens/s | 相对 baseline |
|------|------|------|--------------------------|---------------|
| | | | | |

## Profiler 证据

- 工具/命令：
- 主要瓶颈：
- 关键指标：
- 源码对应位置：

## 单变量实验

| 改动 | 假设 | 结果 | 结论 |
|------|------|------|------|
| | | | |

## 失败版本

- 现象：
- 原因：
- 如何定位：

## 一分钟口径

是什么 → 为什么慢 → 做了什么 → 数字证据 → 取舍。
