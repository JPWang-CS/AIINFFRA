# Lesson 08 — Triton Softmax 迁移实战 / 检查点

> 当前主线：B2 Triton Softmax 语言迁移
> 前置：Softmax 理论与 CUDA 实现已掌握；Triton Vector Add / MatMul 已完成阶段性基线
> 状态：`WIP`；尚无用户编写的 Triton 版本
> 本课目的：用一个小检查点把 CUDA 的地址、mask、reduce 心智迁移到 Triton，然后立即进入 B3 FlashAttention

## 当前单元卡

| 项目 | 当前状态 |
|---|---|
| 题目 | [LeetGPU Softmax](https://leetgpu.com/challenges/softmax) · 题库 #5 |
| 当前代码 | 尚未编写；必须从 LeetGPU 题面提供的模板开始 |
| LeetGPU 归档 | 通过后原样保存平台 `solve`/kernel 到 `solutions/triton/fused_softmax.py` |
| 服务器版本 | 尚未创建；不能覆盖上面的原始平台归档 |
| LeetGPU 状态 | `WIP` |
| 服务器状态 | 未开始 |
| 现在做什么 | 先做 10 分钟 CUDA → Triton 映射，再打开 LeetGPU #5 写题 |
| 下一站 | LeetGPU 通过并完成 RTX 3090 row-wise baseline 后，立即进入 B3 FlashAttention |
| 调试入口 | 遇到编译、mask、越界或数值错误时查 [Lesson 07 — Triton Debugging](07-triton-debugging.md) |

## 已掌握：本课不重新教学

只保留入口，不在本课展开定义、稳定性、online 公式或 CUDA reduce：

- [Lesson 04 — CUDA Softmax](04-softmax.md)
- [Online Softmax](../notes/algorithms/online-softmax.md)
- [Parallel Reduce](../notes/algorithms/parallel-reduce.md)
- [Lesson 05 — Flash Attention CUDA 读码](05-flash-attn-reading.md)
- [Triton 入门与 MatMul](06-triton-intro.md)

旧 CUDA 1-pass 重写、三版 benchmark、warp-shuffle 深钻和 Softmax P0–P8 都是可选优化债务，不是 B2 的前置条件。

## 唯一新增：10 分钟 CUDA → Triton 映射

只做语言和执行模型的对应，不重新解释 Softmax 算法：

| CUDA 心智 | Triton 写法 | 本次检查点 |
|---|---|---|
| `blockIdx.x` 选择任务 | `tl.program_id(0)` | 一个 program 负责题面规定的一个向量或一行 |
| `threadIdx.x` 加循环索引 | `tl.arange(0, BLOCK_SIZE)` | 用索引向量表达一组 lane 的列偏移 |
| `base + tid * stride` | 指针加 element offset | 先确认题面布局和 stride，Triton 偏移按元素不是按字节 |
| `if (idx < N)` | `mask = offsets < N` | 所有尾部 load/store 都沿用有效元素 mask |
| 越界 load 的安全哨兵 | `tl.load(..., mask=..., other=...)` | max 的无效值与 sum 的无效值按已有数值语义选择 |
| shared/warp reduce | `tl.max` / `tl.sum` | 明确 reduce axis；不手写线程同步 |
| kernel launch grid | `grid` 与 `meta` 参数 | 先按题面规模映射 program 数量，不凭感觉改 grid |
| `__global__` 参数与 block 常量 | 普通参数与 `tl.constexpr` | block 宽度属于编译期配置，输入 shape 属于运行时数据 |
| 每线程写回 | `tl.store(..., mask=...)` | store mask 必须和合法输出地址一致 |

迁移完成的判断很窄：能把题面中的输入、输出、长度/stride 和平台签名映射到上表，并能指出一个尾部元素如何经过 mask 到达 load、reduce、store。到此就开始写题，不再增加理论章节。

## LeetGPU：正确性与代码归档

入口：[LeetGPU Softmax](https://leetgpu.com/challenges/softmax)（#5）。这是本课的唯一实现入口；平台当前题面、参数和 `solve` 签名优先于仓库中的旧 CUDA 描述。

执行约束：

1. 用 10 分钟完成上面的映射，然后在平台选择 Triton，从空题面/平台模板开始写。
2. 不复制 `reference/triton/`，不把课程说明当成完整 kernel 或 skeleton；本课不提供可直接提交的实现。
3. 根据真实题面决定是一维向量还是按行处理，并保留平台要求的函数签名。
4. 写出与已掌握语义一致的稳定计算、正确 reduce axis、尾部 mask 和 masked store。
5. 先用最小输入、非 2 次幂长度、全负数、极端值和相等输入排查；以平台判题作为最终正确性证据。
6. 只有平台通过后，才把当次原始 `solve`/kernel 原样归档到 `solutions/triton/fused_softmax.py`，并记录题号、语言、日期和平台结果。

状态规则：平台未通过或原始代码未归档时保持 `WIP`；平台通过且原始代码归档后才改为 `LEETGPU_PASS`。平台通过不等于服务器已验证，也不等于 `COMPLETE`。

## 服务器：真实性能

前置条件：LeetGPU 已为 `LEETGPU_PASS`，且原始平台代码已经归档。服务器代码/benchmark 必须与原始平台文件分开记录，不能用服务器 wrapper 反向冒充平台归档。

服务器只做一个可靠的二维 row-wise baseline：

```text
X: [rows, cols]
Y[row, :] = softmax(X[row, :])
```

目标环境是 RTX 3090；运行时必须记录 `torch.cuda.get_device_name(0)`，不能把 A100 或其他 GPU 的数字写成 RTX 3090 结果。最低正确性集合包含 `(1, 1)`、`(3, 257)`、`(128, 1000)`、`(4096, 1024)` 和 `(1024, 4096)`，并覆盖极端值与相等输入。

服务器验收必须留下：

- 与 `torch.softmax(x, dim=-1)` 的 `assert_close` 结果，以及 dtype、容差和 shape；
- 排除 Triton 首次 JIT 后固定的 warmup、repeats、CUDA 同步方式；
- Triton 与 `torch.softmax` 的同 shape、同 dtype 耗时（ms）和相对速度；
- 按理想一读一写口径计算的 effective GB/s：`2 × rows × cols × element_size / time`；
- 实际 GPU、配置、输入范围和失败 case；明确区分算法口径 GB/s 与 profiler 的真实 DRAM traffic。

这一阶段的停止条件是：row-wise baseline 正确、数字可复现、瓶颈有一句话解释。不要在 B2 追加 Softmax P0–P8、warp shuffle、三版 CUDA benchmark 或 1-pass 用户重写；它们统一留在可选优化债务池。

## 退出到 B3

满足 `LEETGPU_PASS` 并完成 RTX 3090 row-wise baseline 后，B2 结束，直接打开 B3 Triton FlashAttention。B3 的实现主线是 tiling、online softmax 数据流和与 PyTorch/reference 对齐；理论侧 FA2 目前约 50% WIP，继续在 B3 相关任务中消化，不回头补 Softmax 旧债务。

本课最终状态路径：

```text
WIP
→ LEETGPU_PASS（平台通过 + 原始代码归档）
→ GPU_VALIDATED（RTX 3090 row-wise 正确性 + baseline 数字）
→ B3 FlashAttention
```

## 资源

- [Lesson 04 — CUDA Softmax](04-softmax.md)
- [Lesson 07 — Triton Debugging](07-triton-debugging.md)
- [Triton 语法速查](../notes/triton/triton-cheatsheet.md)
- [LeetGPU 题库索引](../notes/cuda/leetgpu-challenges.md)
- [GPU 底层架构与性能优化课程](../roadmap/gpu-foundations.md)
- [Triton 官方 Fused Softmax 教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
