# 2026-08-28 — Triton MatMul LeetGPU 归档复盘

> 对应 PATH：B1 · 代码：[LeetGPU 最终版](../solutions/triton/matmul_leetgpu.py) · [服务器适配版](../solutions/triton/matmul.py)

## 本次完成

- LeetGPU #02 Matrix Multiplication 的原始 Triton `solve`/kernel 已归档，状态为 `LEETGPU_PASS`。
- 平台证据：`SuccessPublicTrace`，A100-80GB，2026-08-28 22:23:16，24.54 ms，55.3th percentile。
- 最终归档以 `matmul_leetgpu_wip.py` 为精确基线，只把 `tl.dot(tile_a, tile_b)` 改为 `tl.dot(tile_a, tile_b, input_precision='ieee')`；WIP 保留为历史 TF32 失败案例。

## 关键复盘

默认 TF32 在 4×4 case 的最大绝对误差为 `0.1275177001953125`，不能满足平台精度要求；指定 IEEE 输入精度后通过。这个修正是数值语义修正，不是 tile 或性能优化。

服务器适配版的既有 RTX 3090 记录保持不变：FP32 `8192×6144×4096` 最佳 `22.033 ms / 18,713.5 GFLOPS`，`torch.mm` 为 `24,083.3 GFLOPS`，相对性能 `77.8%`；初版为 `24.681 ms / 16,706.0 GFLOPS`。`128×64×128` 因 shared memory `131072B > 101376B` 编译失败。

## 当前口径

MatMul 单元总体为 `GPU_VALIDATED`，不是 `COMPLETE`。B1 仍是当前主线；下一步用 Nsight Compute 按 P0–P8 补 profiler 证据、优化结论和最终口径。Vector Add 原始 LeetGPU `solve` 的归档缺口保持不变。
