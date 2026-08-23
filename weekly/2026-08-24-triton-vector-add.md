# Triton Vector Add — 2026-08-23～2026-08-24

> 算子线 · PATH B1 · AutoDL RTX 3090

## 做了什么

- 完成 Triton Vector Add 的边界正确性验证：`N=1/256/257/1000/2^20`。
- 在 AutoDL RTX 3090 上完成真实 GPU 运行和带宽 benchmark。
- 修正 benchmark harness：GPU 环境使用 CUDA tensor，使用 `triton.testing.do_bench`，本地 CPU 环境只做解释器正确性验证。
- 全盘复盘并统一 README、PATH、NOW、HISTORY、Lesson 06 的进度记录。
- 开始阅读 Triton MatMul（Lesson 06 Part 5），尚未写代码。

## 关键数据

- `N=2^25`，`BLOCK_SIZE=256`。
- Triton：`0.479 ms`，`840.1 GB/s`。
- `torch.add`：`0.478 ms`，`843.0 GB/s`。
- Triton 达到 `torch.add` 的 `99.7%`；相对 RTX 3090 理论带宽约 `89.8%`。
- 结论：Vector Add 是 memory-bound，继续微调的收益很小。

## 卡点 / 怎么解决的

- 原 benchmark 在 GPU 机器上错误地创建了 CPU tensor，导致 Triton 报 `cpu tensor` pointer error。
- 将性能段改为 CUDA tensor，并用 `do_bench` 处理 warmup 和 GPU 同步计时。

## 面试可用点

- Vector Add 算术强度极低，瓶颈是 HBM 带宽而不是计算吞吐。
- 纯 element-wise 算子要进一步提速，重点通常是 kernel fusion、减少中间结果写回，而不是继续堆算力。

## 下一步

- 下一次从空文件开始写 `solutions/triton/matmul.py`：单 tile → K 循环 → GFLOPS → autotune。
