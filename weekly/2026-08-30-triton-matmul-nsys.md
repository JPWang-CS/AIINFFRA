# Triton MatMul Nsight Systems P0-lite — 2026-08-30

## 本轮完成

- 在 AutoDL RTX 3090 上用 Nsight Systems 采集 `128×32×256, w8` 的 s3/s2 两组 timeline。
- 将两份约 90 KB 原始日志完整归档，并用 SHA256 核对源文件与仓库副本一致。
- 从 `cuda_gpu_trace` 按 Grid=64×16、Block=256 精确筛选 60 次大 shape，排除 4 次 correctness 小 kernel。
- 分析 `cuda_gpu_kern_sum`、`cuda_api_sum`、Reg/Trd、dynamic shared memory 和静态驻留上限。

## 关键数字

| 配置 | Triton mean / median | GFLOPS | Reg/Trd | DymSMem | 同次 CUTLASS mean | 相对 CUTLASS |
|---|---:|---:|---:|---:|---:|---:|
| `128×32×256, w8, s3` | 21.208 / 21.163 ms | 19,441.3 | 255 | 0.098 MB | 16.716 ms | 78.8% |
| `128×32×256, w8, s2` | 22.362 / 22.317 ms | 18,438.6 | 255 | 0.049 MB | 16.682 ms | 74.6% |

## 结论

s2 将 dynamic shared memory 减半，但没有降低 255 registers/thread。按静态资源推导，两组仍近似受限为 1 block/SM；s2 反而慢 5.44%，说明更浅 pipeline 的损失大于 shared-memory 减少的收益。`cudaLaunchKernel` median 只有微秒量级，相对 21–22 ms kernel 可忽略；Event/DeviceSynchronize 的累计时间主要是等待 GPU，不是独立计算瓶颈。

这只是 P0-lite：没有 NCU hardware counters，不能声称测得 achieved occupancy、warp stall、L2 hit 或 DRAM throughput。下一步保持 w8/s3，只把输出列 tile从 256 降到 128，验证 accumulator/register pressure。

## 证据

- [详细逐块分析](../notes/triton/matmul-nsys-p0-lite-2026-08-30.md)
- [s3 完整 raw log](../notes/triton/logs/2026-08-29-matmul-k256-s3-nsys.txt)
- [s2 完整 raw log](../notes/triton/logs/2026-08-30-matmul-k256-s2-nsys.txt)
