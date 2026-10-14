# Triton Matrix Transpose V2 单元卡

- 题目：[LeetGPU Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose)
- 平台归档代码：[transpose_v2_leetgpu.py](../../solutions/triton/transpose_v2_leetgpu.py)
- 历史本地快照：[transpose_v2.py](../../solutions/triton/transpose_v2.py)
- 原始平台结果：[2026-10-13-transpose-v2-leetgpu.txt](./logs/2026-10-13-transpose-v2-leetgpu.txt)
- 状态：`LEETGPU_PASS`。平台 `SuccessPublicTrace` 时间为 2026/10/13 23:21:59，耗时 `0.33 ms`，`50.5th` percentile。归档代码对应本会话贴给用户运行的版本，成绩取自用户回传的页面结果。
- 服务器真实性能：待测；本次没有 GPU 型号、shape、dtype、重复次数或服务器结果，因此不填写有效带宽。
- 服务器入口：[transpose_bench.py](../../solutions/triton/transpose_bench.py)，调用平台通过版 `solve`，不复制 kernel。
- 下一步：在真实 GPU 上运行服务器脚本，记录 GPU、shape、dtype、正确性、耗时和有效带宽；完成前不推进到 `GPU_VALIDATED`。

| 验收项 | 正确性 | 耗时（ms） | percentile | 有效带宽（GB/s） |
|------|--------|-----------:|-----------:|-----------------:|
| LeetGPU V2 平台归档 | 通过 | 0.33 | 50.5th | 未提供 |
| 服务器 V2 | 待测 |  |  |  |

## 服务器运行

从仓库根目录、已安装 CUDA 版 PyTorch 和 Triton 的环境执行：

```bash
python solutions/triton/transpose_bench.py --rows 37 --cols 65 --correctness-only
python solutions/triton/transpose_bench.py --rows 4096 --cols 4096 --warmup 20 --repeats 100
```

正确性覆盖 `1×1`、`3×5`、`32×32`、`37×65`、`65×37`，并检查指定的测试尺寸。性能比较 V2 和预分配输出的 PyTorch 物化转置，报告 CUDA event 区间均值和有效带宽；本机尚未执行设备验证。

## 输入输出合同

实现面向 contiguous `[rows, cols]` 输入和独立存储的 `[cols, rows]` 输出，dtype 相同。`stride_ir`、`stride_ic`、`stride_or`、`stride_oc` 被传入 kernel，但当前地址表达式未使用它们，因此不支持一般非连续输入或输出。

## Tile 坐标与 mask

一个 program 覆盖输入的 32×32 tile，其中 `BR` 表示 tile 行数、`BC` 表示 tile 列数，本版两者均为 32。`offsetA`、`maskA` 和 `a` 的逻辑形状为 `[BR, BC]`；`offsetB`、`maskB` 和 `b` 的逻辑形状为 `[BC, BR]`。`tl.trans(a)` 交换值的逻辑坐标，输出坐标和输出 mask 也按转置后的维度生成。即使 tile 是方形、两个形状看起来相同，元素位置仍然不同。

平台结果只证明本次题面运行通过。服务器验证时可用 `1×1`、`3×5`、`32×32`、`37×65` 和 `65×37` 检查边界与坐标关系；这些形状是否属于平台题面，以题面为准。

## 当前性能证据边界

`tl.trans` 表示逻辑坐标变换。具体线程交换代价取决于前后布局转换；当前没有服务器编译结果或 profiler 数据，不能填写活跃线程数、shared-memory 用量、有效带宽或优化结论。

物化转置的有效读写字节数为 `2 * rows * cols * element_size`。服务器基线应预先分配输出，并计量 `output.copy_(input.T)` 完成的实际写入；只创建 `input.T` view 不构成同等工作量。服务器记录在真实运行后填写。
