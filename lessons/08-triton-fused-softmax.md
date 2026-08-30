# Lesson 08 — Triton Fused Softmax

> 当前主线：B2 Triton Fused Softmax  
> 前置：CUDA Softmax、Online Softmax、Parallel Reduce、Triton Vector Add / MatMul  
> 范围：先完成正确、可解释的 fused baseline；极致资源调优进入 [GPU 优化篇](../roadmap/gpu-foundations.md)  
> 状态：`WIP`，尚无用户代码；本课 skeleton 只是 TODO，不算学习进度

## 当前单元卡

| 项目 | 当前状态 |
|---|---|
| 题目 | [LeetGPU Softmax](https://leetgpu.com/challenges/softmax) · [仓库题库索引](../notes/cuda/leetgpu-challenges.md)（#5） |
| LeetGPU 代码 | 尚未编写；通过后原样归档到 `solutions/triton/fused_softmax.py` |
| 服务器代码 | 尚未创建；必须在 LeetGPU 通过后开始 |
| 当前状态 | `WIP` |
| 今天从哪里开始 | 读完 Part 0–4，然后打开 LeetGPU #5，从平台模板写第一版 |
| 调试手册 | 遇到编译、mask、越界或数值错误时查 [Lesson 07 — Triton Debugging](07-triton-debugging.md) |
| 完成门槛 | `LEETGPU_PASS` → `GPU_VALIDATED`；没有正确性和性能证据不标 `COMPLETE` |

本课固定只有两个验收段：

```text
LeetGPU：正确性与代码归档
→ 服务器：真实性能
```

---

# Part 0：这次到底写什么

Softmax 对一个向量 $x$ 的定义是：

$$
y_i = \frac{e^{x_i}}{\sum_j e^{x_j}}.
$$

数值稳定版本先减最大值：

$$
m = \max_j x_j,
\qquad
y_i = \frac{e^{x_i-m}}{\sum_j e^{x_j-m}}.
$$

这不改变数学结果，但避免大正数指数溢出。

本课有两个接口层次，不能混写：

| 阶段 | 语义 | 目的 |
|---|---|---|
| LeetGPU #5 | 以平台当前题面和 `solve` 签名为准，可能是一维向量 | 先过正确性门，保存用户原始代码 |
| 服务器版本 | 二维 row-wise softmax：每一行独立归一化 | 对齐模型中的 attention / classifier 常见布局并测真实性能 |

如果 LeetGPU 题面、shape 或参数与本课描述不同，以平台页面为准，并在通过后回填本课。

---

# Part 1：为什么叫 Fused Softmax

概念上 Softmax 包含：

```text
max → subtract → exp → sum → divide
```

若拆成多个独立 kernel，中间向量需要反复经过显存。Triton baseline 的目标是让一个 program 在片上完成一行：

```text
HBM load 一行
→ 片上 max / exp / sum / divide
→ HBM store 一行
```

理想数据流量（FP32）约为：

```text
1 次读取 + 1 次写回
= 2 × rows × cols × 4 bytes
```

这是理想 fused 流量，不等于 GPU 实际 memory transaction；实际值还受 padding、cache、对齐和编译器实现影响。`torch.softmax` 是优化库 baseline，不应简单描述成 Python 表达式里的多个 kernel。

模型中的典型位置：

- Attention score 每行归一化；
- 分类输出概率；
- sampling 前的 logits 处理。

---

# Part 2：Triton 并行映射

二维输入 $X[rows, cols]$ 的第一版映射：

```text
一个 Triton program
→ 负责一整行
→ tl.arange 生成该行所有列 offset
→ 在同一个 program 内完成 reduce
```

地址形式：

```python
row = tl.program_id(0)
offsets = tl.arange(0, BLOCK_SIZE)
ptrs = x_ptr + row * stride_x + offsets
mask = offsets < n_cols
```

Ascend 对照：

| Triton | Ascend / Da Vinci 心智映射 |
|---|---|
| 一个 program 处理一行 | 一个核内处理一个 tile / 数据块 |
| `tl.load` 到 tensor | GM → UB 搬运后的片上视图 |
| `tl.max/tl.sum` | 片上向量 reduce |
| `BLOCK_SIZE` padding | tiling 后补齐并用 mask 隔离无效元素 |
| register pressure | UB / 本地资源过大导致并行度下降的同类问题 |

区别是 Triton 不要求手写线程级 reduce；编译器根据 tensor shape 和 layout 生成 GPU 实现。

---

# Part 3：mask、负无穷和数值稳定

列数通常不是 2 的幂，所以使用：

```python
BLOCK_SIZE = triton.next_power_of_2(n_cols)
```

尾部 padding 在求最大值时必须表现为“永远不可能成为最大值”：

```python
x = tl.load(ptrs, mask=mask, other=-float("inf"))
```

不能使用 `other=0.0`。当有效元素全是负数时，padding 的 0 会错误成为最大值。

正确顺序：

```text
row_max = max(x)
numerator = exp(x - row_max)
denominator = sum(numerator)
output = numerator / denominator
```

padding 位置满足：

```text
exp(-inf - row_max) = 0
```

最终 `tl.store` 仍必须使用原 mask，不能写到行尾之外。

---

# Part 4：从这个 skeleton 开始

下面只展示结构，不包含完整答案。用户需要在 LeetGPU 题目模板中自己补齐 TODO。

```python
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,
    y_ptr,
    n_cols,
    stride_x,
    stride_y,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x_ptrs = x_ptr + row * stride_x + offsets
    y_ptrs = y_ptr + row * stride_y + offsets
    x = tl.load(x_ptrs, mask=mask, other=-float("inf"))

    # TODO 1: reduce max
    # TODO 2: exp(x - max)
    # TODO 3: reduce sum
    # TODO 4: normalize
    # TODO 5: masked store
```

写之前先徒手回答：

1. `tl.max` 和 `tl.sum` 应沿哪个 axis？
2. 为什么 padding 必须是 `-inf`？
3. 为什么输出 store 还需要 mask？
4. 一行过宽时，片上 tensor 会消耗什么资源？

这段 skeleton 是 Agent 教学材料，不是用户代码，不改变 PATH 状态。

---

# Part 5：LeetGPU — 正确性与代码归档

入口：[Softmax](https://leetgpu.com/challenges/softmax)（题库 #5）。

执行顺序：

1. 在平台选择 Triton，保留平台给出的 `solve` 签名；
2. 根据真实题面判断是一维还是按行处理；
3. 完成稳定 max trick、mask、reduce 和 store；
4. 提交平台判题；
5. 通过后将原始 `solve`/kernel 原样保存到 `solutions/triton/fused_softmax.py`；
6. 记录题号、语言、通过日期、关键边界 case 和平台结果。

最低边界检查：

- 长度为 1；
- 非 2 次幂长度，例如 257 / 1000；
- 全负数；
- 大正数与大负数混合；
- 所有输入相等，此时输出应接近均匀分布；
- 输出之和接近 1。

若平台不提供这些 case，以平台通过为第一证据，服务器阶段再补自建 reference。

通过前状态只能是 `WIP`；只有平台通过且原始代码已归档，才是 `LEETGPU_PASS`。

---

# Part 6：服务器 — 真实性能

前置：LeetGPU 已 `LEETGPU_PASS`。

服务器版本建议实现二维 row-wise softmax：

```text
X: [rows, cols]
Y[row, :] = softmax(X[row, :])
```

必须动态记录：

```python
torch.cuda.get_device_name(0)
```

正确性 reference：

```python
expected = torch.softmax(x, dim=-1)
torch.testing.assert_close(actual, expected, rtol=..., atol=...)
```

建议覆盖：

```text
(rows, cols) =
(1, 1)
(3, 257)
(128, 1000)
(4096, 1024)
(1024, 4096)
```

具体 shape 要结合显存和平台约束调整。还要测试极端值与相等输入。

Benchmark 要求：

- 排除 Triton JIT 首次编译；
- 固定 warmup、repeats 和同步方式；
- 对比同 shape / dtype 的 `torch.softmax`；
- 同时记录 ms、相对速度和理想 effective GB/s；
- 只在真实 GPU 上报告性能；解释器只用于排错。

理想 effective bandwidth：

$$
BW = \frac{2 \times rows \times cols \times element\_size}{time}.
$$

这个 GB/s 是按理想一读一写计算的算法口径，必须明确标注，不能冒充 profiler 实际 DRAM traffic。

---

# Part 7：性能和资源边界

第一版最重要的旋钮：

| 旋钮 | 影响 |
|---|---|
| `BLOCK_SIZE` | padding、向量宽度、register 使用 |
| `num_warps` | reduce 并行度与调度开销 |
| `n_cols` | 单 program 工作量和片上资源 |
| dtype | 精度、带宽和 exp/reduce 路径 |

一行一个 program 的限制：

- 列数越大，片上 tensor 越大；
- padding 到 2 次幂会放大浪费；
- register pressure 可能降低 occupancy 或产生 spill；
- 超宽行可能需要分块、两阶段 reduce 或 online softmax，而不是无限增大 BLOCK。

本课先做可靠 baseline。极致优化、Nsight counter、PTX/SASS 和跨架构比较转入 [GPU 优化篇](../roadmap/gpu-foundations.md)，不阻塞 B2 主线出口。

---

# Part 8：常见错误

| 错误 | 现象 | 根因 |
|---|---|---|
| padding 使用 `other=0` | 全负输入结果错误 | 0 污染 max |
| 忘记减最大值 | inf / NaN | `exp` 溢出 |
| reduce axis 错 | 每列或整块结果异常 | tensor 维度理解错误 |
| row stride 写错 | 行之间串数据 | 地址公式不匹配布局 |
| store 不加 mask | 越界写 | padding 位置不是合法输出 |
| 只测 2 次幂列数 | 尾块 bug 未暴露 | mask 没被真正验证 |
| 不排除 JIT | 第一次耗时极大 | 把编译时间算入 kernel |
| 不同 shape/dtype 排名 | 得出错误快慢结论 | benchmark 语义不一致 |
| 超宽行仍强行单 program | 编译失败或很慢 | register / occupancy / block 限制 |

调试顺序：

```text
最小输入
→ 非 2 次幂
→ 全负数 / 极端值
→ assert_close + 行和
→ 真实 GPU benchmark
→ 最后才调 BLOCK/warps
```

---

# Part 9：验收清单

## LeetGPU

- [ ] 用户从题目模板完成 Triton kernel
- [ ] 平台通过
- [ ] 原始 `solve`/kernel 归档到 `solutions/triton/fused_softmax.py`
- [ ] 记录题号、语言、日期和边界 case

## 服务器

- [ ] 和 `torch.softmax(dim=-1)` 对齐
- [ ] 非 2 次幂、极端值和相等输入通过
- [ ] 记录实际 GPU、shape、dtype、warmup/repeats
- [ ] 记录 ms、effective GB/s 和相对 `torch.softmax`
- [ ] 写清瓶颈与停止条件

## 状态

```text
当前：WIP
平台通过且代码归档：LEETGPU_PASS
真实 GPU 正确性和性能验证：GPU_VALIDATED
证据、失败案例、分析与口径齐全：COMPLETE
```

---

# Part 10：一分钟面试口径

> Fused Softmax 把 max、subtract、exp、sum 和 normalize 放在一个 Triton program 中，让一行数据尽量只从 HBM 读一次、写一次，中间状态留在片上。为了稳定性先减 row max；尾部 padding 用 `-inf`，避免污染 max，并在 store 时继续 mask。第一版通常一行一个 program，性能主要受列宽、padding、register pressure、reduce 映射和 `num_warps` 影响。验证时先和 `torch.softmax` 对齐，再在同 shape/dtype 下比较耗时和按一读一写计算的 effective bandwidth；超宽行需要分块或多阶段算法，不能无限扩大 BLOCK。

自测问题：

1. 为什么 Softmax 必须减最大值？
2. 为什么 mask 的 `other` 是 `-inf` 而不是 0？
3. 一行一个 program 为什么能省 HBM？
4. `BLOCK_SIZE` 为什么通常取 next power of two？
5. `n_cols` 很大时为什么可能变慢或编译失败？
6. effective GB/s 和 profiler DRAM throughput 有什么区别？
7. LeetGPU 一维接口如何迁移到服务器二维 row-wise 版本？
8. 什么证据齐全后才能标记 `GPU_VALIDATED`？

---

# 资源

- [Lesson 04 — CUDA Softmax](04-softmax.md)
- [Online Softmax](../notes/algorithms/online-softmax.md)
- [Triton 语法速查](../notes/triton/triton-cheatsheet.md)
- [LeetGPU 题库索引](../notes/cuda/leetgpu-challenges.md)
- [GPU 优化篇](../roadmap/gpu-foundations.md)
- [Triton 官方 Fused Softmax 教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
