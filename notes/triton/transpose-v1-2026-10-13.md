# Triton Matrix Transpose V1 单元卡

- 来源：用户 2026-10-13 对话中的初版；本地快照为 [transpose_v1.py](../../solutions/triton/transpose_v1.py)。Agent 只做两处 API 修正，未把这份代码标成平台原始通过版本。
- 题目：[LeetGPU Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose)
- 适用范围：仅支持 contiguous `input[rows, cols]`，输出为独立存储、相同 dtype 的 contiguous `output[cols, rows]`。内核保留四个 stride 形参及调用，但当前地址公式没有使用它们。
- 状态：`WIP`。Python 语法静态检查通过；尚未运行 Triton JIT、LeetGPU、GPU 正确性或性能测试。
- LeetGPU：未验证。
- 服务器：未运行。
- 下一步：将 `1×1`、`3×5`、`32×32`、`37×65` 作为建议的小 case 检查，确认平台题面合同后再决定是否提交。这里不代表平台已经验证矩形输入。

源码保留 `solve` 和 kernel，供题目编辑器调用；没有服务器测试或计时入口。Python 语法检查不会执行这些函数，也不会触发 Triton JIT。

V2 已保存为本地对话快照：[Triton Matrix Transpose V2 单元卡](./transpose-v2-2026-10-13.md)。该链接只表示代码已归档，不表示平台或服务器验证通过。

## API 修正记录

以下两行保留为未运行的原始错误调用示例：

```python
a = tl.load(input + offsetA, other=0.0)
tl.store(output + offsetB)
```

本地 V1 只将无 mask 的 `tl.load` 改为不带 `other`，并给 `tl.store` 补上待写入值 `a`。没有进行其他代码整理或优化。

## V2 优化思路（仅建议）

V1 每个 program 对应一个矩阵元素，`grid=(rows, cols)`，所以 37×65 会启动 2405 个 program。可以从 32×32 tile 开始讨论 V2：`grid=(triton.cdiv(rows, 32), triton.cdiv(cols, 32))`，37×65 对应 6 个 program。这只是 program 数量对比，不代表速度倍数。

二维 tile 需要分别处理输入和输出边界，并设计读写布局。不能把两个 `tl.arange(0, 1)` 直接改成 `tl.arange(0, 32)`：若输入沿列方向读 `A[r, c+k]`，输入偏移随 `k` 增加 1；转置输出是 `B[c+k, r]`，其连续地址步进为 `rows` 个元素。需要重新安排 tile 内的值与写出顺序，使读写两侧都获得合适的连续访问。

下一次实现可以先写下面的坐标与 mask 骨架，不急着填写完整读写：

```python
r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
valid = (r[:, None] < rows) & (c[None, :] < cols)
```

`r` 是这一块覆盖的行号，`c` 是列号；广播后 `valid` 的形状是 `[BLOCK_R, BLOCK_C]`。从 `BLOCK_R=BLOCK_C=32` 开始，先确认每个逻辑元素 `A[r,c]` 写到 `B[c,r]`，再讨论实际线程的读写排列。32×32 的逻辑 tile 不等于 1024 个硬件线程。

后续应结合实际连续读写布局、编译器生成指令、寄存器和 shared memory 资源，再选择 tile 或 `num_warps`。CUDA 的 `tile[32][33]` 是特定 shared-memory bank 映射下的做法，不等价于 Triton tile 简单加 33。当前没有测量 lane 分配、DRAM 饱和程度或速度数字。

性能对照要完成同样的实际搬运：使用预分配的输出，与 `output.copy_(input.T)` 比较，不能只计创建 `input.T` 视图的时间。平台验证通过后，按同一 GPU、shape、dtype 和计时范围记录 ms 与有效 GB/s；读取和写入合计的有效数据量为 `2 * rows * cols * input.element_size()`。
