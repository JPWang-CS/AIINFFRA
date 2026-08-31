# 2026-09-01 — Triton Softmax LeetGPU 归档

## 问题与修复

LeetGPU 反馈的问题是尾部越界写（OOB store）。修复方式是让输出写回使用与输入 load 相同的 `mask`，确保无效 offset 不写入 `output`。

## 实现结构

用户通过版保留三阶段数据流：

1. `softmax_partial`：分块计算 partial `(max, sum)`；
2. `softmax_reduce`：归并 partial pair，得到全局 max/sum；
3. `softmax_sum`：使用全局统计量完成 normalize 并 masked store。

归档文件：[solutions/triton/fused_softmax.py](../solutions/triton/fused_softmax.py)。这是 LeetGPU 一维题的原始通过代码；服务器阶段仍需单独实现二维 row-wise baseline。

## 平台结果

- 题目：LeetGPU Softmax #5
- 状态：`LEETGPU_PASS`
- 证据：`SuccessPublicTrace`
- 时间：2026-09-01 00:37:33
- 结果：0.29 ms，47.0th percentile

## 下一步

服务器二维 row-wise baseline 尚未开始。下一步在 RTX 3090 完成正确性、耗时和 effective GB/s 记录；在此之前不标记 `GPU_VALIDATED` 或 `COMPLETE`。FA2 理论进度约 50% 保持不变。
