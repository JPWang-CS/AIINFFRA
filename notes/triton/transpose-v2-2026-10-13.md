# Triton Matrix Transpose V2 单元卡

- 题目：[LeetGPU Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose)
- 代码：[本地对话快照 `transpose_v2.py`](../../solutions/triton/transpose_v2.py)，与平台编辑器代码的同步状态未知。
- 状态：`WIP`；Python 语法静态检查通过。Triton JIT、LeetGPU 和 GPU 验证尚未运行。
- 下一步：由用户在 LeetGPU 和服务器实际运行后补充结果；需要服务器测试时再按需补测试包装。

| 项目 | 正确性 | 耗时（ms） | 有效带宽（GB/s） |
|------|--------|-----------:|-----------------:|
| LeetGPU |  |  |  |
| 服务器正确性 |  |  |  |
| V1 转置 |  |  |  |
| V2 转置 |  |  |  |
| 预分配输出的 PyTorch copy 基线 |  |  |  |

## 输入输出合同

实现面向 contiguous `[rows, cols]` 输入和独立存储的 `[cols, rows]` 输出，dtype 相同。`stride_ir`、`stride_ic`、`stride_or`、`stride_oc` 被传入 kernel，但当前地址表达式未使用它们，因此不支持一般非连续输入或输出。

## Tile 坐标与 mask

`maskA`、`offsetA` 和 `a` 的逻辑形状为 `[BR, BC]`；`maskB`、`offsetB` 和 `b` 的逻辑形状为 `[BC, BR]`。`tl.trans(a)` 交换值的逻辑坐标，输出 mask 也必须按转置后的坐标对应。即使 tile 是方形、两个形状看起来相同，元素位置仍然不同。

可用 `1×1`、`3×5`、`32×32`、`37×65` 和 `65×37` 检查边界与坐标关系；这些是建议的本地/平台检查形状，不代表题面支持矩形形状，也不代表已经通过。

## 当前性能证据边界

`tl.trans` 表示逻辑坐标变换。具体线程交换代价取决于前后布局转换；目前没有编译结果或 profiler 数据，不能据此填写活跃线程数、shared-memory 用量或速度结论。

物化转置的有效读写字节数为 `2 * rows * cols * element_size`。基线应预先分配输出，并计量 `output.copy_(input.T)` 完成的实际写入；只创建 `input.T` view 不构成同等工作量。结果表在真实运行后填写。
