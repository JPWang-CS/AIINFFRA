# Triton MatMul Nsight Systems P0-lite 记录

> 日期：2026-08-30  
> 对象：RTX 3090 上 `128×32×256, w8` 的 Triton MatMul；比较 `num_stages=3`（s3）与 `num_stages=2`（s2）  
> 状态：`P0-lite` 完成（timeline + launch metadata）；MatMul 总体仍为 `GPU_VALIDATED`，不是 `COMPLETE`

## 1. 实验背景与命令

此前 IEEE FP32 sweep 在 RTX 3090 上选出 `BLOCK_M=128, BLOCK_N=32, BLOCK_K=256, num_warps=8, num_stages=3`。本次只改变 pipeline stage：s3 → s2；M/N/K、tile、warp、FP32/IEEE 语义和 benchmark shape 保持不变。目标 shape 是 `M=8192, N=6144, K=4096`，因此 Triton 大 shape 的 launch 为 `Grid=64×16×1, Block=256×1×1`。

服务器执行 Nsight Systems 统计的命令为：

```bash
nsys stats \
  --report cuda_gpu_trace \
  --report cuda_gpu_kern_sum \
  --report cuda_api_sum \
  /tmp/matmul_nsys_run1.nsys-rep       # s3

nsys stats \
  --report cuda_gpu_trace \
  --report cuda_gpu_kern_sum \
  --report cuda_api_sum \
  /tmp/matmul_k256_s2.nsys-rep         # s2
```

两份报告都来自 NVIDIA GeForce RTX 3090，且都包含 correctness、小 kernel、Triton 大 shape kernel 和 CUTLASS 对照。本文所有性能统计均重新从 `cuda_gpu_trace` 的逐次调用行计算；不直接采用被 correctness 调用混入的 `cuda_gpu_kern_sum` Triton summary 平均值。

## 2. 日志结构与解析口径

每份 raw 日志依次包含：

1. `cuda_gpu_trace`：逐次 GPU 活动，字段包括 `Start`、`Duration`、`CorrId`、`GrdX/Y/Z`、`BlkX/Y/Z`、`Reg/Trd`、`StcSMem`、`DymSMem`、设备和 kernel 名称。
2. `cuda_gpu_kern_sum`：按 kernel 名称聚合的总时间、实例数、均值、中位数、最小/最大值和标准差。
3. `cuda_api_sum`：CUDA Runtime/API 在主机侧的调用聚合。

解析脚本对 `cuda_gpu_trace` 做了以下筛选：

```text
Triton：Name == matrix_multiplication_kernel
        GrdX/Y/Z == 64/16/1
        BlkX/Y/Z == 256/1/1
        取到 N=60

CUTLASS：Name 包含 cutlass::Kernel2
         取到 N=60
```

`Duration` 是 trace 中该次 GPU kernel 的持续时间，单位为 ns；下文的 ms 只是 ns ÷ 1,000,000。`std` 使用这 60 个观测值的样本标准差（分母 `N-1`），以便和 Nsight summary 的 StdDev 口径对齐。

## 3. CUDA GPU Trace 详解

### 3.1 Triton 目标调用

Triton 的输出 tile 是 `128×256`，沿归约维 N 以 `BLOCK_N=32` 循环；对 `M=8192,K=4096`，program grid 为 64×16。Block 中 256 个线程对应 8 个 warp。trace 中的大 shape 调用全部呈现同一组 Grid/Block 和资源字段，适合做逐次统计。

### 3.2 correctness 污染的位置

每份 trace 里 `matrix_multiplication_kernel` 一共 64 次，但只有后 60 次是目标大 shape。前 4 次 correctness 的 Grid 是 `1×1×1`、`1×1×1`、`1×1×1`、`3×1×1`，仍然使用同一个 kernel 名称；因此不能把同名 kernel 的 64 次直接当作 benchmark 样本。

这 4 次的 duration 也明显不是目标 workload：

| 日志 | 4 次 correctness duration (ns) |
|---|---:|
| s3 | 10,561；12,514；29,829；163,769 |
| s2 | 12,219；14,361；32,560；185,736 |

它们数量少但会改变 `cuda_gpu_kern_sum` 的总时间、均值和标准差，所以本文的结论以 Grid/Block 精确筛出的 60 次为准。

## 4. Kernel Summary 去污染统计

`cuda_gpu_kern_sum` 的 Triton 行包含 64 个实例，不能直接使用：

| 日志 | Summary instances | Summary total (ns) | Summary Avg (ns) | Summary Med (ns) | Summary Min/Max (ns) |
|---|---:|---:|---:|---:|---:|
| s3 | 64 | 1,272,713,734 | 19,886,152.1 | 21,162,962.5 | 10,561 / 22,227,448 |
| s2 | 64 | 1,341,939,689 | 20,967,807.6 | 22,273,469.5 | 12,219 / 22,934,017 |

重新筛选大 shape 的 60 行后，得到以下统计。均值、中位数、最小值、最大值和样本标准差均为 trace duration：

| 运行 | N | mean (ms) | median (ms) | min (ms) | max (ms) | std (ms) |
|---|---:|---:|---:|---:|---:|---:|
| s3 Triton | 60 | 21.208284350 | 21.163139000 | 19.016006000 | 22.227448000 | 0.488482101 |
| s2 Triton | 60 | 22.361580217 | 22.317032000 | 21.870078000 | 22.934017000 | 0.268108583 |

同一结果用 ns 表示为：

```text
s3: N=60, mean=21208284.35, median=21163139,
    min=19016006, max=22227448, std=488482.101027 ns
s2: N=60, mean=22361580.216667, median=22317032,
    min=21870078, max=22934017, std=268108.582503 ns
```

因此在这组固定 workload 上，s2 的 Triton mean 比 s3 慢约 5.4379%，median 慢约 5.4524%。这不是“动态共享内存更少所以更快”的结果；s2 虽然资源字段更小，但观测到的 kernel 时间更长。

## 5. CUDA API Summary：等待与 launch 的边界

日志中的 API summary 是主机侧 API 时间聚合，不等于 GPU kernel 的执行时间。关键行如下：

| API | s3 total / calls | s3 median | s2 total / calls | s2 median |
|---|---:|---:|---:|---:|
| `cudaEventSynchronize` | 1,899,920,121 ns / 2 | 949,960,060.5 ns | 1,951,736,384 ns / 2 | 975,868,192 ns |
| `cudaDeviceSynchronize` | 373,313,899 ns / 6 | 73,182.5 ns | 388,369,646 ns / 6 | 88,264.5 ns |
| `cudaLaunchKernel` | 87,464,734 ns / 70 | 7,904 ns | 99,375,972 ns / 70 | 13,090 ns |

`cudaEventSynchronize` 和 `cudaDeviceSynchronize` 的高累计时间主要是在等待已经发射的 GPU 工作完成；它们是测量/正确性流程的同步点，不应被解读成一个独立的 GPU 算术瓶颈。s2 的 event/device wait 累计时间略高，与 s2 kernel 更慢的事实一致，但不能仅凭 API summary 证明原因。

`cudaLaunchKernel` 的 median 只有 s3 的 7.904 μs、s2 的 13.090 μs，而对应 Triton 大 shape kernel median 是 21.163139 ms、22.317032 ms。launch median 只占 kernel median 约 0.037348%（s3）和 0.058655%（s2），分别相差约 2,678× 和 1,705×；本次 21–22 ms workload 的主要时间不在 launch。

## 6. s3 vs s2：资源与性能

### 6.1 Triton 资源/形状字段

| 运行 | Reg/Trd | StcSMem (MB) | DymSMem (MB) | Grid | Block |
|---|---:|---:|---:|---|---|
| s3 (`w8-s3`) | 255 | 0.000 | 0.098 | 64×16×1 | 256×1×1 |
| s2 (`w8-s2`) | 255 | 0.000 | 0.049 | 64×16×1 | 256×1×1 |

唯一稳定的资源变化是 `DymSMem` 从 0.098 MB 降到 0.049 MB，约减少一半；`Reg/Trd`、Grid 和 Block 没有变化。性能却从 s3 的 21.208284350 ms mean 退化到 s2 的 22.361580217 ms mean。合理但尚未被 counter 证实的解释是 s2 减少了 pipeline buffering，可能降低延迟隐藏；在没有 warp stall、memory traffic 或 achieved occupancy counter 的情况下，这只能标记为推测，不能写成因果结论。

### 6.2 CUTLASS 对照

CUTLASS 的 trace kernel 名称为：

```text
void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1>(T1::Params)
```

它的 60 次调用统计为：

| 运行 | N | mean (ms) | median (ms) | min (ms) | max (ms) | std (ms) |
|---|---:|---:|---:|---:|---:|---:|
| s3 CUTLASS | 60 | 16.716166717 | 16.623331500 | 15.759053000 | 17.443282000 | 0.316561043 |
| s2 CUTLASS | 60 | 16.682412133 | 16.612236500 | 15.749290000 | 17.425223000 | 0.333787347 |

CUTLASS 资源和形状字段在 s3/s2 两份日志中相同：`Reg/Trd=202`、`StcSMem=0.000 MB`、`DymSMem=0.049 MB`、`Grid=512×2×2`、`Block=256×1×1`。s2 相对 s3 的 CUTLASS mean 只快约 0.2019%，落在本次观测噪声/运行差异的量级，不能归因于 Triton 的 stage 改动。相比 CUTLASS，Triton mean 在 s3 慢约 26.87%，在 s2 慢约 34.04%；这说明库 kernel 的执行映射/指令路径不同，但本日志没有足够硬件 counter 去解释具体差距。

## 7. 理论驻留推导：静态上限，不是 achieved occupancy

按 RTX 3090 近似资源：每 SM 约 65,536 registers、48 warps、1,536 threads。目标 block 为 256 threads，即 8 warps。只用这些上限做静态资源取最小值：

| Kernel | Reg/Trd | registers/block | register 上限 | warp 上限 | thread 上限 | 静态驻留上限 |
|---|---:|---:|---:|---:|---:|---:|
| Triton s3/s2 | 255 | 255×256=65,280 | floor(65,536/65,280)=1 | floor(48/8)=6 | floor(1,536/256)=6 | **1 block/SM** |
| CUTLASS | 202 | 202×256=51,712 | floor(65,536/51,712)=1 | 6 | 6 | **1 block/SM** |

这个结果只能说明按给定近似寄存器/warp/thread 限制时，理论最小资源上限为 1 block/SM；它不是 profiler 的 achieved occupancy，也没有证明实际任何时刻都只有一个活跃 block。真实驻留还会受寄存器分配粒度、shared-memory 配置、编译器生成代码和调度状态影响。尤其不能把本节的静态 1 block/SM 当成 achieved occupancy 或性能因果证据。

## 8. 证据边界与 P0-lite 结论

本次已经有证据的内容：

- `cuda_gpu_trace` timeline 中 60 次目标 Triton kernel 的逐次 duration；
- Triton/CUTLASS 的 Grid、Block、Reg/Trd、静态/动态 shared-memory metadata；
- CUDA launch 与同步 API 的调用数量、累计时间和 median；
- s3/s2 的资源变化与固定 shape 下的性能结果。

本次没有证据的内容：

- 没有 warp stall reason；
- 没有 L2 hit/traffic、DRAM traffic 或实际 memory throughput；
- 没有 achieved occupancy/active warps；
- 没有 PTX/SASS 指令混合、MMA/FP32 pipeline 利用率或 spill 证据；
- 没有 NCU hardware counters。AutoDL 环境的 `RmProfilingAdminOnly=1`，且缺少 `CAP_SYS_ADMIN/CAP_PERFMON`，因此 NCU 阶段被权限阻塞。

所以当前最稳妥的结论是：s2 确实降低了动态 shared-memory metadata，但没有降低 Reg/Trd，静态寄存器上限仍为 1 block/SM；在这次 60 次 trace 中，s2 性能反而比 s3 差约 5.44%。不能进一步断言是 occupancy、stall、L2、DRAM 或某条指令造成的退化。

### 下一单变量实验

s2 已把 dynamic shared memory 减半，但 `Reg/Trd` 仍为 255，静态驻留上限没有改善；因此下一步不再增加 stage。保持 `M/N/K`、IEEE FP32、`BLOCK_M=128`、归约 tile `BLOCK_N=32`、`num_warps=8`、`num_stages=3` 不变，只把输出列 tile `BLOCK_K` 从 256 降为 128，新增 `k128-128x32x128-w8-s3`。目标是观察 accumulator 缩小后 `Reg/Trd` 是否下降，以及更低 register pressure 能否抵消较小输出 tile 的开销。若环境权限恢复，再用 NCU 对当前 s3 与 k128 配置采同一组 counters。

## 9. 一分钟面试口径

“我在 RTX 3090 上对同一个 IEEE FP32 Triton GEMM 只改变 `num_stages`，用 Nsight Systems 从 GPU trace 精确筛了 Grid=64×16、Block=256 的 60 次调用，而不是直接用混入 4 次 correctness 的 kernel summary。s3 的 Triton kernel mean 是 21.208 ms，s2 是 22.362 ms，s2 约慢 5.44%；资源上 Reg/Trd 都是 255，但动态 shared memory 从 0.098 MB 降到 0.049 MB。按 65,536 registers/SM、48 warps/SM、1,536 threads/SM 做静态推导，两者都被寄存器近似限制到 1 block/SM，但这不是 achieved occupancy。CUTLASS 约 16.7 ms，明显更快。当前证据能说明资源变化和性能结果，不能说明具体 stall/L2/DRAM 原因，因为 NCU counters 被 AutoDL profiling 权限阻塞；下一步缩小输出列 tile，验证 accumulator/register pressure。”

## 10. 完整原始日志

- [2026-08-29 s3 raw Nsight Systems log](./logs/2026-08-29-matmul-k256-s3-nsys.txt)
- [2026-08-30 s2 raw Nsight Systems log](./logs/2026-08-30-matmul-k256-s2-nsys.txt)

两份 raw 文件均为源日志的逐字节机械复制；源/目标 SHA256 已在落盘后独立核对。
