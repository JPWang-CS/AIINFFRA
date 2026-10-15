# Triton Matrix Transpose V2 单元卡

- 题目：[LeetGPU Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose)
- 平台归档代码：[transpose_v2_leetgpu.py](../../solutions/triton/transpose_v2_leetgpu.py)
- 历史本地快照：[transpose_v2.py](../../solutions/triton/transpose_v2.py)
- 原始平台结果：[2026-10-13-transpose-v2-leetgpu.txt](./logs/2026-10-13-transpose-v2-leetgpu.txt)
- 状态：`GPU_VALIDATED`；LeetGPU `LEETGPU_PASS` 保留。平台 `SuccessPublicTrace` 时间为 2026/10/13 23:21:59，耗时 `0.33 ms`，`50.5th` percentile。归档代码对应用户提供的运行版本，成绩取自用户回传的页面结果。
- 服务器验证：新私有环境中的五个小尺寸 FP32 正确性检查及 4096×4096 FP32 profiler workload 正确性均通过。NCU kernel replay 报告完成、退出码 0；性能计数器是诊断采集，不能当正式 benchmark。
- 服务器实验记录：[Transpose V2 性能验证](./transpose-performance-2026-10-15.md)。
- 服务器入口：[transpose_bench.py](../../solutions/triton/transpose_bench.py)，调用平台通过版 `solve`，不复制 kernel。
- 下一步：本阶段验证已通过并收口。后续模型实现及多算子组合确定真实工作负载后，再结合工作负载开展极致优化；当前不单独继续转置参数扫描。

| 验收项 | 正确性 | 结果 |
|------|--------|------|
| LeetGPU V2 平台归档 | 通过 | 0.33 ms，50.5th percentile |
| 服务器小尺寸 FP32 | 通过 | `1×1`、`3×5`、`32×32`、`37×65`、`65×37` |
| 服务器 4096×4096 FP32 | 通过 | NCU replay 目标完成；报告时长 32.224 μs，仅作 profiler 诊断 |

## 当前性能证据

NCU 诊断观察到 L2 throughput 91.071394%、DRAM throughput 66.773761%、SM throughput 27.449247%、LSU 指令吞吐 21.752234%，achieved occupancy 89.170925%。这些比例分别按各自峰值/持续活跃口径归一化，不能相加；L2 throughput 不是命中率，SM throughput 不是 occupancy。L2 是后续检查方向，尚不能据此断言瓶颈。完整条件及指标来源见[性能验证记录](./transpose-performance-2026-10-15.md)。

## 服务器运行

以下命令是复现实验入口；本单元的服务器正确性与 NCU 采集已在独立记录中完成：

```bash
python solutions/triton/transpose_bench.py --rows 37 --cols 65 --correctness-only
python solutions/triton/transpose_bench.py --rows 4096 --cols 4096 --warmup 20 --repeats 100
```

正确性覆盖 `1×1`、`3×5`、`32×32`、`37×65`、`65×37`，并检查指定的测试尺寸。性能命令比较 V2 和预分配输出的 PyTorch 物化转置，报告 CUDA event 区间均值和有效带宽；本条复现命令尚未在本地执行。

## 输入输出合同

实现面向 contiguous `[rows, cols]` 输入和独立存储的 `[cols, rows]` 输出，dtype 相同。`stride_ir`、`stride_ic`、`stride_or`、`stride_oc` 被传入 kernel，但当前地址表达式未使用它们，因此不支持一般非连续输入或输出。

## Tile 坐标与 mask

一个 program 覆盖输入的 32×32 tile，其中 `BR` 表示 tile 行数、`BC` 表示 tile 列数，本版两者均为 32。`offsetA`、`maskA` 和 `a` 的逻辑形状为 `[BR, BC]`；`offsetB`、`maskB` 和 `b` 的逻辑形状为 `[BC, BR]`。`tl.trans(a)` 交换值的逻辑坐标，输出坐标和输出 mask 也按转置后的维度生成。即使 tile 是方形、两个形状看起来相同，元素位置仍然不同。

平台结果只证明本次题面运行通过。服务器验证时可用 `1×1`、`3×5`、`32×32`、`37×65` 和 `65×37` 检查边界与坐标关系；这些形状是否属于平台题面，以题面为准。

## 当前性能证据边界

`tl.trans` 表示逻辑坐标变换。具体线程交换代价取决于前后布局转换。本次服务器 NCU 报告记录了 kernel 资源和吞吐计数器，见性能验证记录；这组 profiler 结果不替代 CUDA event benchmark，也不足以单独确定优化结论。

物化转置的逻辑读写字节数为 `2 * rows * cols * element_size`。比较基线应预先分配输出，并计量 `output.copy_(input.T)` 完成的写入；只创建 `input.T` view 不构成同等工作量。当前可参考的 AutoDL CUDA-event 结果记录在性能验证说明中，和服务器 profiler 数据分开呈现。
