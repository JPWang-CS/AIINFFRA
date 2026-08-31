# FlashAttention-2：从公式、Online Softmax 到 Triton 代码

> 唯一入口：本文合并 FA2 论文精读与算法笔记。
> 前置：[Online Softmax](online-softmax.md) · [FlashAttention 机制](flash-attention-mechanism.md)
> 代码映射：[Triton 教学参考](../../reference/triton/flash_attention/flash_attn.py) · [CUDA 教学参考](../../reference/cuda/flash_attention/flash_attn.cu)
> 论文：[FlashAttention-2, arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
> 状态：🚧 Agent 整合稿，仍需在 B3 亲手实现和 GPU 验证

---

## 0. 一张图先看懂 FA1 → FA2

```text
标准 Attention
  QK^T -> 写出 N×N 的 S
  softmax -> 再读 S、写出 N×N 的 P
  PV -> 再读 P
        |
        v
FlashAttention-1
  Q/K/V 分块放入片上存储
  用 online softmax 边算分母、边累计 PV
  不把完整 S/P 落到 HBM
  主要收益：少做 HBM IO，激活显存从二次降为线性
        |
        v
FlashAttention-2
  数学结果和 FA1 相同，IO-aware 主体也相同
  进一步：
  1. 减少 non-matmul 标量运算
  2. 把单个 head 的多个 Q tile 分给不同 thread block
  3. 改变 block 内 warp 分工，减少 shared-memory 通信
```

一句话：

- **FA1 主要解决“不要反复搬运和保存 $N\times N$ 中间矩阵”。**
- **FA2 主要解决“省掉 IO 后，怎么让 GPU 的并行度和 Tensor Core 利用率更高”。**

FA2 不是新的注意力定义，也不是稀疏或近似算法。

---

## 1. 标准 Attention 到底算什么

先只看一个 batch、一个 attention head：

| 符号 | shape | 含义 |
|---|---:|---|
| $Q$ | $[N,d]$ | $N$ 个 query，每个维度为 $d$ |
| $K$ | $[N,d]$ | $N$ 个 key |
| $V$ | $[N,d_v]$ | $N$ 个 value，通常 $d_v=d$ |
| $S$ | $[N,N]$ | attention score |
| $P$ | $[N,N]$ | 每行 softmax 后的概率 |
| $O$ | $[N,d_v]$ | 最终输出 |

公式：

$$
S = \frac{QK^T}{\sqrt d} + M,
\qquad
P = \operatorname{softmax}_{\text{row}}(S),
\qquad
O = PV.
$$

$M$ 是可选 mask。causal attention 中：

$$
M_{rs} =
\begin{cases}
0, & s \le r,\\
-\infty, & s > r.
\end{cases}
$$

对第 $r$ 行展开：

$$
O_r =
\frac{
\sum_{s=1}^{N}\left(\exp(S_{rs})\,V_s\right)
}{
\sum_{s=1}^{N}\exp(S_{rs})
}.
$$

“分子是加权 $V$，分母是指数和”的形式非常关键，因为它允许我们分块后继续增量合并。

### 1.1 普通 PyTorch 写法

```python
scale = 1.0 / math.sqrt(d)
scores = q @ k.transpose(-2, -1) * scale  # [N, N]
scores = scores.masked_fill(mask, float("-inf"))
prob = torch.softmax(scores, dim=-1)      # [N, N]
out = prob @ v                            # [N, d_v]
```

概念上会产生两个 $N\times N$ 中间量：softmax 前的 $S$ 和 softmax 后的 $P$。

当 $N=128\text{K}$ 时：

$$
N^2 = 131072^2 \approx 1.72\times10^{10}.
$$

即使每个元素只有 2 bytes，一个矩阵也约为 32 GiB；训练还需要保存更多状态。

### 1.2 不要混淆三种复杂度

| 问题 | 标准 Attention | FlashAttention |
|---|---|---|
| 数学计算量 | $O(N^2d)$ | 仍是 $O(N^2d)$ |
| 显式 $S/P$ 激活显存 | $O(N^2)$ | 不保存完整矩阵，额外状态近似 $O(N)$ |
| HBM 读写 | 多次读写 $S/P$ | 通过 tiling 显著减少；精确 IO 复杂度取决于片上 SRAM 和 tile |

所以“激活显存从二次降为线性”是对的；但不能因此说计算量也变成 $O(N)$，也不要把所有 HBM IO 不加条件地简写成 $O(N)$。

---

## 2. 分块后如何保持 exact attention

把 $Q$ 切成行块，把 $K,V$ 切成列块：

$$
Q_i\in\mathbb{R}^{B_r\times d},\quad
K_j\in\mathbb{R}^{B_c\times d},\quad
V_j\in\mathbb{R}^{B_c\times d_v}.
$$

其中：

- $B_r$：一个 Q tile 的行数；
- $B_c$：一个 K/V tile 的行数；
- $i$：Q tile 编号；
- $j$：K/V tile 编号。

一个 thread block/program 固定负责 $Q_i$，依次扫描 $K_j,V_j$：

$$
S_{ij}=\frac{Q_iK_j^T}{\sqrt d}+M_{ij},
\qquad
S_{ij}\in\mathbb{R}^{B_r\times B_c}.
$$

由于一次只看到一部分 key，要为 $Q_i$ 的每一行维护三个运行状态：

| 状态 | shape | 含义 |
|---|---:|---|
| $m_i$ | $[B_r]$ | 当前看过 scores 的逐行最大值 |
| $\ell_i$ | $[B_r]$ | 以当前最大值为基准的逐行指数和 |
| $U_i$ | $[B_r,d_v]$ | 尚未除以 $\ell_i$ 的输出分子 |

最终输出：

$$
O_i=\frac{U_i}{\ell_i[:,\text{None}]}.
$$

这里维护的是 online softmax 与 $PV$ 融合后的状态，不需要保存独立的 $P$。

---

## 3. Online Softmax 的完整分块公式

处理第 $j$ 个 K/V tile 前，已有：

$$
m_i^{(j-1)},\quad \ell_i^{(j-1)},\quad U_i^{(j-1)}.
$$

先算当前 score tile：

$$
S_{ij}=\frac{Q_iK_j^T}{\sqrt d}+M_{ij}.
$$

逐行更新最大值：

$$
m_i^{(j)}
=
\max\left(
m_i^{(j-1)},
\operatorname{rowmax}(S_{ij})
\right).
$$

定义旧状态的重标定系数：

$$
\alpha_i^{(j)}
=
\exp\left(m_i^{(j-1)}-m_i^{(j)}\right).
$$

当前 tile 在新最大值下的未归一化概率：

$$
\widetilde P_{ij}
=
\exp\left(S_{ij}-m_i^{(j)}[:,\text{None}]\right).
$$

更新分母：

$$
\ell_i^{(j)}
=
\alpha_i^{(j)}\ell_i^{(j-1)}
+
\operatorname{rowsum}(\widetilde P_{ij}).
$$

更新输出分子：

$$
U_i^{(j)}
=
\alpha_i^{(j)}[:,\text{None}]\odot U_i^{(j-1)}
+
\widetilde P_{ij}V_j.
$$

全部 K/V tile 完成后才归一化：

$$
O_i
=
U_i^{(T_c)}
\oslash
\ell_i^{(T_c)}[:,\text{None}].
$$

其中 $\odot$ 是逐元素乘，$\oslash$ 是逐元素除。

### 3.1 为什么旧贡献必须乘 $\alpha$

旧状态以 $m_{old}$ 为指数基准：

$$
\ell_{old}=\sum_{x\in old}\exp(x-m_{old}).
$$

若新块产生更大的 $m_{new}$，旧项必须改写为：

$$
\exp(x-m_{new})
=
\exp(x-m_{old})\exp(m_{old}-m_{new}).
$$

因此旧分母 $\ell$ 和旧输出分子 $U$ 都必须乘：

$$
\alpha=\exp(m_{old}-m_{new}).
$$

漏掉这一步，新旧 tile 就处于不同指数尺度，结果必错。

### 3.2 一个手算例子

设一行 score 分两块到达：

```text
第 1 块 scores = [1, 2]，V = [10, 20]
第 2 块 scores = [3]，   V = [30]
```

第一块：

$$
m_1=2,
$$

$$
\ell_1=e^{1-2}+e^{2-2}=e^{-1}+1\approx1.3679,
$$

$$
U_1=e^{-1}\cdot10+1\cdot20\approx23.6788.
$$

第二块把最大值从 2 更新成 3：

$$
m_2=3,\qquad
\alpha=e^{2-3}=e^{-1}\approx0.3679.
$$

$$
\ell_2=0.3679\times1.3679+1\approx1.5032,
$$

$$
U_2=0.3679\times23.6788+30\approx38.711.
$$

$$
O=U_2/\ell_2\approx25.75.
$$

一次性直接计算的分母同样是：

$$
e^{1-3}+e^{2-3}+e^{3-3}=1.5032,
$$

分子也同样约为 38.711。分块改变执行顺序，没有改变数学结果。

---

## 4. Tiled Forward 的执行过程

定义：

$$
T_r=\left\lceil\frac{N}{B_r}\right\rceil,\qquad
T_c=\left\lceil\frac{N}{B_c}\right\rceil.
$$

```python
parallel for i in range(T_r):          # 一个 program/thread block 负责一个 Q tile
    q = load(Q_i)                      # [Br, d]
    m = full([Br], -inf)               # FP32
    l = zeros([Br])                    # FP32
    u = zeros([Br, dv])                # FP32 accumulator

    for j in valid_kv_tiles(i):
        k = load(K_j)                  # [Bc, d]
        v = load(V_j)                  # [Bc, dv]

        s = q @ k.T / sqrt(d)          # [Br, Bc]
        s = apply_mask(s, i, j)

        m_new = max(m, rowmax(s))
        alpha = exp(m - m_new)
        p = exp(s - m_new[:, None])    # 未归一化，不写回 HBM

        u = alpha[:, None] * u + p @ v
        l = alpha * l + rowsum(p)
        m = m_new

    o = u / l[:, None]
    store(O_i, o)
```

数据流：

```text
HBM: Q_i ───────┐
                ├─> SRAM/register: S_ij -> m/l/U 更新 -> O_i -> HBM
HBM: K_j, V_j ──┘          ^              |
                             └── 下一 KV 块 ┘

不会写出完整的 S[N,N] 或 P[N,N]。
```

---

## 5. 公式如何落到仓库 Triton 代码

代码：[reference/triton/flash_attention/flash_attn.py](../../reference/triton/flash_attention/flash_attn.py)，kernel 是 `flash_attn_kernel`。

它是一份 **forward 机制映射/教学参考**，不是经过完整验证和极致优化的官方 FA2 kernel。

### 5.1 Program ID 对应当前 $Q_i$

```python
pid = tl.program_id(0)
q_start = pid * BLOCK_Q
offs_q = q_start + tl.arange(0, BLOCK_Q)
```

对应：

$$
i=\text{pid},\qquad
Q_i=Q[iB_r:(i+1)B_r,:].
$$

`grid = (triton.cdiv(seq_len, BLOCK_Q),)` 表示不同 Q tile 可以由不同 program 并行处理。这正是 FA2 沿 Q 序列维度增加 thread-block 并行度的直接映射。

### 5.2 加载 $Q_i$，初始化 $m,\ell,U$

```python
q = tl.load(...)

m_i = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
acc = tl.zeros((BLOCK_Q, head_dim), dtype=tl.float32)
```

| 代码 | 公式 |
|---|---|
| `m_i` | $m_i$，逐行 running max |
| `l_i` | $\ell_i$，逐行 running exp sum |
| `acc` | $U_i$，未归一化输出分子 |

### 5.3 加载 K/V tile 并计算 score

```python
for kv_start in range(0, seq_len, BLOCK_KV):
    offs_kv = kv_start + tl.arange(0, BLOCK_KV)
    k = tl.load(...)
    v = tl.load(...)

    scores = tl.dot(q, tl.trans(k))
    scores = scores / math.sqrt(head_dim)
```

shape：

```text
q            [BLOCK_Q, d]
trans(k)     [d, BLOCK_KV]
scores       [BLOCK_Q, BLOCK_KV]
```

`scores` 只存在于当前 tile，不是完整 $N\times N$ 矩阵。

### 5.4 Online Softmax 更新

```python
m_new = tl.maximum(m_i, tl.max(scores, axis=1))
alpha = tl.exp(m_i - m_new)
p = tl.exp(scores - m_new[:, None])
```

对应：

$$
m_{new}=\max(m_{old},\operatorname{rowmax}(S_{ij})),
$$

$$
\alpha=\exp(m_{old}-m_{new}),
\qquad
\widetilde P_{ij}=\exp(S_{ij}-m_{new}).
$$

`p` 没有除以 $\ell$，所以是未归一化概率。

### 5.5 同时更新输出分子和分母

```python
acc = acc * alpha[:, None] + tl.dot(p, v)
l_i = l_i * alpha + tl.sum(p, axis=1)
m_i = m_new
```

正好对应：

$$
U_{new}=\alpha U_{old}+\widetilde P_{ij}V_j,
$$

$$
\ell_{new}=\alpha\ell_{old}+\operatorname{rowsum}(\widetilde P_{ij}).
$$

最常见 bug 是只重标定 `l_i`，忘了同步重标定 `acc`。

### 5.6 最终归一化和 store

```python
acc = acc / l_i[:, None]
tl.store(..., acc, mask=...)
```

对应：

$$
O_i=U_i/\ell_i.
$$

只有 $O_i\in\mathbb{R}^{B_r\times d_v}$ 写回 HBM。

### 5.7 当前 Triton 教学参考的边界

它还不是可直接声称“FA2 完整实现”的版本：

1. grid 只覆盖 sequence Q tile，没有完整映射 batch/head；
2. 没有 causal mask；
3. K/V 尾块 load 虽填 0，但 invalid score 应 mask 为 $-\infty$，否则会污染 softmax 分母；
4. 没有 backward；
5. 没有论文级 warp 分工、autotune、pipeline 和 profiler 证据。

因此准确状态是：**FA2 forward 机制映射/教学参考**。

---

## 6. CUDA 教学实现如何对应

代码：[reference/cuda/flash_attention/flash_attn.cu](../../reference/cuda/flash_attention/flash_attn.cu)，kernel 是 `flash_attn_forward`。

| CUDA 代码 | 算法含义 |
|---|---|
| `blockIdx.x * BR` | 当前 $Q_i$ |
| `Q_tile/K_tile/V_tile` | 片上 tile |
| `for (kv_start ...)` | 扫描 $K_j,V_j$ |
| `float m, l` | 当前行的 $m_i,\ell_i$ |
| `float acc[128]` | 当前行的 $U_i$ |
| `scale = expf(m - m_new)` | 重标定 $\alpha$ |
| `acc = acc * scale + p * V` | 更新输出分子 |
| `l = l * scale + p` | 更新分母 |
| `O = acc / l` | 最终归一化 |

这份 CUDA 文件为了教学用标量循环写 dot product，并不是 Tensor Core FA2 kernel；它展示算法控制流，不能代表论文性能。

---

## 7. Mask、边界与数值稳定性

### 7.1 Causal mask

对 query 位置 $r$ 和 key 位置 $s$：

$$
S_{rs}=
\begin{cases}
Q_rK_s^T/\sqrt d, & s\le r,\\
-\infty, & s>r.
\end{cases}
$$

分块后：

1. 整个 K tile 都在未来：直接跳过，不 load、不做 matmul；
2. 对角 tile 部分有效：逐元素把未来位置设为 $-\infty$。

“causal 省约一半”是大 $N$、规则方块下的近似结论；边界 tile、不同 $B_r/B_c$ 和调度会影响实测。

### 7.2 尾块 mask

若 $N$ 不是 tile 整数倍：

- Q 尾块：invalid 行不能 store；
- K 尾块：invalid 列对应 score 必须设为 $-\infty$；
- 只把 invalid K 填 0 不够，因为 score 0 的 $e^0=1$ 会污染分母。

### 7.3 为什么 $m,\ell,U$ 用 FP32

- $m$ 控制整行指数平移；
- $\ell$ 跨多个 tile 累加正数；
- $U$ 跨多个 tile 累加 $PV$。

即使 Q/K/V 是 FP16 或 BF16，softmax 状态和 accumulator 通常使用 FP32，再在 store 时转目标 dtype。

---

## 8. FA2 相对 FA1 为什么更快

论文主线有三项。它们优化执行效率，不改变 exact attention。

### 8.1 减少 non-matmul FLOPs

两块主计算：

$$
QK^T,\qquad PV
$$

可以走 Tensor Core；max、exp、加法、除法和 rescale 走标量/向量路径。后者 FLOPs 不一定多，但相对 Tensor Core 峰值吞吐低，可能占用显著时间。

FA2 调整 online-softmax 记账，维护未归一化输出状态，把归一化尽量推迟，减少每个 tile 周围不必要的缩放、除法和状态操作。

注意：**不是完全取消重标定。** running max 改变时，旧 $\ell$ 和旧 $U$ 仍必须乘 $\alpha$；减少的是额外归一化和 non-matmul 工作。

收益取决于 head dimension、tile、dtype 和 GPU，论文比例不能直接当作任意硬件的固定结果。

### 8.2 增加 thread block 级并行

若主要只沿 batch/head 启动 block：

$$
\text{blocks}\approx B\times H.
$$

小 batch、少 head 时，block 数可能不足。FA2 把 Q tile 加入 grid：

$$
\text{blocks}
\approx
B\times H\times
\left\lceil\frac{N}{B_r}\right\rceil.
$$

例如 $B=1,H=8,N=4096,B_r=64$：

$$
1\times8\times64=512\text{ blocks}.
$$

不同 $Q_i$ 独立维护自己的 $m,\ell,U$，所以不需要相互同步。

但 block 更多不保证更快：tile 太小会降低 matmul 效率，寄存器/shared memory 太大又会压低 occupancy，必须 profile。

### 8.3 改进 block 内 warp 分工

若多个 warp 分别负责同一输出 tile 的不同 K 片段，会产生部分结果，之后需要写 shared memory、同步、读取和归约。

FA2 更倾向于让 warp 各自拥有 Q tile 的不同行/输出子块，同时共同消费 K/V：

```text
同一个 thread block
  warp 0 -> Q_i 第 0 组行 -> 自己的 U/m/l
  warp 1 -> Q_i 第 1 组行 -> 自己的 U/m/l
  warp 2 -> Q_i 第 2 组行 -> 自己的 U/m/l
  warp 3 -> Q_i 第 3 组行 -> 自己的 U/m/l
             共同消费当前 K_j/V_j
```

每个 warp 对自己负责的输出行完成 $QK^T$、online softmax 和 $PV$，减少 warp 间中间输出通信。

预期 profiler 变化：

- shared-memory load/store 减少；
- barrier 压力下降；
- Tensor Core 活跃比例提高；
- 但寄存器压力可能上升，要同时检查 occupancy 和 spill。

---

## 9. FA1、FA2 与仓库代码对照

| 维度 | FA1 核心 | FA2 增量 | 当前仓库教学代码 |
|---|---|---|---|
| exact attention | 是 | 是 | 目标是 |
| 不保存完整 $S/P$ | 是 | 是 | 是 |
| online softmax | 是 | 调整记账、减少 non-matmul | 已映射 |
| Q tile 跨 block 并行 | 较受限 | 强化 | Triton grid 已体现 |
| warp 工作划分 | shared-memory 通信较多 | 减少 warp 间通信 | 未实现论文级策略 |
| causal | 支持 | 支持并优化 | CUDA 教学版有；Triton 教学版没有 |
| backward | 有 | 继续优化并行和分工 | 没有 |
| Tensor Core/pipeline | 优化实现具备 | 更强 | 教学代码不能代表 |
| 本仓库 GPU 验证 | — | — | 尚未验证 |

论文报告的约 $2\times$、A100 上 50%–73% 峰值利用率，以及训练最高 225 TFLOPS/s/72% MFU，都是特定硬件、dtype、shape 和端到端配置下的数据，不是所有环境的保证值。

---

## 10. 常见误解

1. **“计算量降成 $O(Nd)$”**：错。dense exact attention 仍是 $O(N^2d)$。
2. **“FlashAttention 是近似算法”**：错。浮点顺序会带来正常误差，但数学目标相同。
3. **“online softmax 只维护 running max”**：错。至少需要 $m,\ell$，融合 $PV$ 时还要维护 $U$。
4. **“最大值更新后只缩放分母”**：错。旧 $\ell$ 和旧 $U$ 必须一起乘 $\alpha$。
5. **“FA2 关键只是 tile 更大”**：错。主线是 non-matmul、block 并行和 warp 分工。
6. **“有 online softmax 就是完整 FA2”**：错。还要看 grid、mask、尾块、warp 分工、backward、正确性和性能证据。

---

## 11. 自检题

1. 为什么 $m_{new}>m_{old}$ 时，旧 $U$ 必须乘 $e^{m_{old}-m_{new}}$？
2. 为什么 invalid K 尾块不能只 load 为 0？
3. $U$ 与最终 $O$ 有什么区别？
4. 为什么不同 Q tile 可以由不同 thread block 独立处理？
5. FA2 的 warp 分工为什么能减少 shared-memory 通信？
6. 为什么激活显存可以线性，但计算量仍然二次？
7. A100 的 72% MFU 为什么不能直接推断 RTX 3090 也会达到 72%？

七题能不看笔记讲清楚，才算真正掌握机制。

---

## 12. 下一步：从看懂到自己写

进入 B3 时按项目统一流程：

1. 在 LeetGPU 从题面写正确版本；
2. 保存平台通过时的原始 `solve/kernel`；
3. 在真实 GPU 对齐 PyTorch reference；
4. 再做 tile、warp、stage 和 causal 优化；
5. 用 profiler 验证“硬件机制 → 代码旋钮 → counter 变化”。

写 kernel 时按这个顺序逐行对照：

```text
program_id -> 当前 Q tile
load Q
初始化 m/l/U
循环 load K/V tile
QK^T + scale + mask
更新 m 和 alpha
更新 U 与 l
最终 U/l
masked store O
```

配套：

- [Lesson 05：读懂 FlashAttention CUDA](../../lessons/05-flash-attn-reading.md)
- [Triton 教学参考](../../reference/triton/flash_attention/flash_attn.py)
- [CUDA 教学参考](../../reference/cuda/flash_attention/flash_attn.cu)
- [Online Softmax](online-softmax.md)
- [FlashAttention 机制](flash-attention-mechanism.md)

---

## 参考

- Tri Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691), 2023.
- Tri Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), 2022.
- Triton, [Fused Attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).
