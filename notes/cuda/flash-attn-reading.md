# Flash Attention CUDA 逐段阅读笔记（A5）

> 代码：[flash_attn.cu](../../reference/cuda/flash_attention/flash_attn.cu) · 课程：[Lesson 05](../../lessons/05-flash-attn-reading.md) · 完成：2026-08-10
> 口径：单头、causal、BR=BC=32、d=64；一个 block = 32 线程 = 32 个 Q 行

## 1. 结构总览

1. **Q tile 加载（39-47 行）**：32 个线程协作把 `32×64=2048` 个元素搬进 shared memory。`i += blockDim.x` 只在**本 block 的 tile 内**步进，不跨 block；`i/d`、`i%d` 把一维下标拆成 (行, 列)，`d` 同时是全局内存的行跨度。47 行 `__syncthreads()` 保证整块 Q 就绪。
2. **计算（57-126 行）**：`threadIdx.x` 直接当 Q 行号（两个 `qr` 别混：加载阶段是坐标，计算阶段是线程 id）。每行在寄存器里持有 `m`、`l` 和 `acc[128]`；K/V 按 32 行一个 tile 流转。
3. **写回（129-131 行）**：`O = acc / l`，每线程写自己那行。

## 2. 三个 `__syncthreads()` 的作用

| 行 | 时机 | 作用 |
|----|------|------|
| 47 | Q tile 搬完后 | 所有线程的 Q_tile 写入完成，才能开始读 |
| 94 | K/V tile 搬完后 | 所有线程的 K_tile/V_tile 写入完成，才能开始算 |
| 125 | 一个 kv tile 算完后 | 所有线程都读完 K/V tile，下一轮才能安全覆盖 |

## 3. Online 更新（113-123 行）

```text
score = Q·K/√d                      # xi，只由 Q/K 算，V 不参与
m_new = max(m, score)
scale = exp(m - m_new)              # 旧权重整体打折
p     = exp(score - m_new)          # 当前 key 的未归一化分子
l     = l*scale + p                 # 分母累加和：Σp      形状 []
acc   = acc*scale + p*V             # 分子加权 V 的和：ΣpV 形状 [d]
O     = acc / l
```

两个累加器是对同一批权重求和的两种载荷：`l` 对权重求和，`acc` 对权重×V 求和。因为不物化 `S/P` 矩阵，只能按 tile 在线累加，max 变化时必须用 `scale` 把历史贡献折算到新基准。

## 4. Tiling 策略

- grid 按 Q 行分块：block `b` 管 `q_start = b*32` 起 32 行；Q 全网格只读一遍。
- K/V 每个 block 都会重扫一遍（简化版 FA 的代价），换来不物化 O(N²) 的 score/prob 矩阵。
- 一行一个线程、acc 全程寄存器：省 shared 带宽，但锁死 `d ≤ 128`。

## 5. 阅读中发现的问题（真实 bug / 边界）

1. **尾 block 提前 return 会死锁**：58 行 `if (qr >= q_end-q_start) return;` 发生在 94/125 两个 `__syncthreads()` 之前。N 不是 32 的倍数时，尾 block 部分线程退出，剩余线程永远等不到 barrier。当前测试 N=128 恰好不触发。修法：不 return，全部线程参与 barrier，用 `active` 标志跳过计算和写回。
2. **`m_i`/`l_i` 是死代码**：52-53 行分配了 shared 数组但从未使用，实际状态在寄存器 `m/l` 里；阅读指引（NOW 旧版）和代码不一致。
3. **`acc[128]` 的寄存器约束**：d>128 直接数组越界 + 寄存器溢出；真实实现需要一行多线程拆列（thread tile）。
4. **写回 O 无越界守卫**：129-131 行没有 `global_qr < N`，靠提前 return 掩盖；删 return 时必须补守卫。

## 6. 面试口径

- 为什么省 HBM：不物化 `S=QKᵀ` 和 `P=softmax(S)`，Q 只读一遍，O 在寄存器滚动。
- 为什么能滚动：online softmax 的 `(m, l, acc)` 三量，max 变化时旧贡献乘 `scale` 修正。
- 为什么一行一线程：简化、寄存器直达；代价是 d 上限和 occupancy 靠多 block 补。
- 踩过的坑：提前 return 绕过 `__syncthreads` = 死锁；空归并哨兵 NaN（A4 同款）。
