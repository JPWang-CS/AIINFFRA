# 实验记录模板

## 目标

- 问题：
- 类型：算子 / 通信 / 训练 / 推理
- 预计瓶颈：
- 验收标准：

## 环境

| 项 | 值 |
|----|----|
| 日期 / commit | |
| GPU / CC / 数量 | |
| Driver / CUDA | |
| PyTorch / Triton / NCCL | |
| Topology / interconnect | |

## 正确性门

- LeetGPU 题号或 reference：
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
