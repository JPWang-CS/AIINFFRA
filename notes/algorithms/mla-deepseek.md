# MLA（Multi-head Latent Attention）：从论文公式到 Decode 代码

> 本笔记按 DeepSeek-V2 论文第 2.1 节和 Appendix C 重写。目标不是只记住“MLA 压缩 KV Cache”，而是能回答：压缩了什么、为什么 RoPE 要拆开、Decode 时为什么不用把历史 K/V 完整解压出来，以及代码中的 Cache 和矩阵乘法分别是什么形状。
>
> 代码状态：下面的代码是解释论文机制的 reference code，不是生产实现；当前没有 MLA Triton kernel、LeetGPU 结果或真实 GPU benchmark。

论文：[DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)

---

## 0. 先看结论：MLA 到底改了什么

标准 MHA 为每个历史 token 保存完整的 K 和 V：

~~~
每个历史 token
    ├── K：每个 head 一份
    └── V：每个 head 一份
~~~

MLA 改成：

~~~
每个历史 token
    ├── c_KV：一个共享的低维 latent
    └── k_R：一个较小的、携带 RoPE 的共享 key
~~~

其中：

- c_KV 保存无位置的 Key/Value 共同信息；
- k_R 保存位置相关的 RoPE 信息；
- 完整的 content Key 和 Value 不常驻在 KV Cache 中；
- Decode 时使用 weight absorption 后的矩阵，直接在 c_KV 上计算。

所以 MLA 不是简单的“减少 KV head”，也不是简单的“截短 d_k”。完整机制是：

~~~
低秩 KV 联合压缩
        +
解耦 RoPE
        +
Decode 阶段的 weight absorption
~~~

---

## 1. 符号和 shape 约定

为了避免转置混乱，公式使用列向量；代码使用 batch-first 的行存储张量。代码中的 x @ W.T 对应公式中的 W x。

| 符号 | 含义 |
|---|---|
| t | 当前 token 的位置 |
| j | 被查询的历史 token 位置 |
| d | hidden size |
| n_h | attention head 数 |
| d_h | 每个 head 的 content 维度 |
| d_h^R | 每个 head 的 RoPE 维度 |
| d_c | KV latent compression dimension |
| d_c' | Query latent compression dimension |
| h_t | 第 t 个 token 的 hidden state，h_t ∈ R^d |
| c_t^KV | KV 压缩 latent，c_t^KV ∈ R^{d_c} |
| c_t^Q | Query 压缩 latent，c_t^Q ∈ R^{d_c'} |

论文中的矩阵形状：

~~~
W_DKV: [d_c, d]
W_UK : [n_h * d_h, d_c]
W_UV : [n_h * d_h, d_c]
~~~

切到第 i 个 head 后：

~~~
W_UK_i: [d_h, d_c]
W_UV_i: [d_h, d_c]
~~~

---

## 2. 标准 MHA：KV Cache 为什么会成为瓶颈

### 2.1 MHA 的投影和注意力

给定 hidden state h_t：

$$
q_t = W_Q h_t, \qquad
k_t = W_K h_t, \qquad
v_t = W_V h_t
$$

其中：

$$
W_Q,W_K,W_V \in \mathbb{R}^{n_h d_h \times d}
$$

按 head 切分后，第 i 个 head 的输出为：

$$
o_{t,i}
=
\sum_{j=1}^{t}
\operatorname{softmax}_{j}
\left(
\frac{q_{t,i}^{\mathsf{T}} k_{j,i}}{\sqrt{d_h}}
\right)v_{j,i}
$$

最后拼接所有 head，再过输出投影：

$$
u_t = W_O[o_{t,1};o_{t,2};\ldots;o_{t,n_h}]
$$

### 2.2 Decode 时的访问模式

Decode 每次通常只生成一个新 token：

~~~
当前 Query： [1, H, d_h]
历史 K/V：  [T, H, d_h]
~~~

每个 head 都要读取从第 1 个 token 到第 t 个 token 的历史 K/V：

~~~
Q_new @ K_cache.T → score
softmax(score) @ V_cache → output
~~~

因此 Decode 的主要压力是历史 Cache 的读取，而不是只计算当前 token 的 Q/K/V 投影。

MHA 每个 token、每层要缓存：

$$
2n_h d_h
$$

个元素：一份 K 加一份 V。完整 Cache 还要乘上下文长度、层数和并发 batch。

---

## 3. MLA 第一部分：低秩 KV 联合压缩

### 3.1 压缩和恢复路径

MLA 先把 hidden state 压缩成一个 latent：

$$
c_t^{KV}=W_{DKV}h_t
$$

其中：

$$
c_t^{KV}\in\mathbb{R}^{d_c},
\qquad
W_{DKV}\in\mathbb{R}^{d_c\times d}
$$

再通过两个 up-projection 得到 content Key 和 Value：

$$
k_t^C=W_{UK}c_t^{KV}
$$

$$
v_t^C=W_{UV}c_t^{KV}
$$

其中：

$$
W_{UK},W_{UV}\in\mathbb{R}^{n_hd_h\times d_c}
$$

切到第 i 个 head：

$$
k_{t,i}^C=W_{UK,i}c_t^{KV},
\qquad
v_{t,i}^C=W_{UV,i}c_t^{KV}
$$

数据流是：

~~~
一个 token 的 h_t
       ↓ W_DKV
共享的 c_KV
       ├── W_UK_i → 第 i 个 head 的 content Key
       └── W_UV_i → 第 i 个 head 的 content Value
~~~

同一个 c_KV 同时承载 K 和 V 的信息，因此论文称为 low-rank Key-Value joint compression。

### 3.2 它不是训练后 SVD，也不只是 LoRA

不准确的说法是：

~~~
先训练完整 MHA，再对 W_K 和 W_V 做低秩分解。
~~~

MLA 从训练时就使用如下参数化：

$$
h_t
\xrightarrow{W_{DKV}}
c_t^{KV}
\xrightarrow{W_{UK},W_{UV}}
k_t^C,v_t^C
$$

更准确的表述是：

> MLA 用一个共享的低维 latent 对 Key 和 Value 做联合低秩参数化。它与低秩分解使用相似的数学结构，但不是训练完 MHA 后再做一次 LoRA/SVD 式替换。

### 3.3 Cache 为什么可以只保存 c_KV

未来 Query 需要的 content Key 和 Value 都可以由历史 token 的 c_j^KV 和固定模型权重得到。因此 Cache 的候选从：

~~~
k_j^C：n_h * d_h 个元素
v_j^C：n_h * d_h 个元素
~~~

变成：

~~~
c_j^KV：d_c 个元素
~~~

但是 Decode 时并不应该机械地把所有历史 c_j^KV 恢复成完整 K/V；weight absorption 会进一步消除这部分工作。

---

## 4. Query Compression：作用和边界

论文也对 Query 做低秩压缩：

$$
c_t^Q=W_{DQ}h_t
$$

$$
q_t^C=W_{UQ}c_t^Q
$$

其中：

$$
c_t^Q\in\mathbb{R}^{d_c'},
\qquad
W_{DQ}\in\mathbb{R}^{d_c'\times d}
$$

$$
W_{UQ}\in\mathbb{R}^{n_hd_h\times d_c'}
$$

Query 只属于当前 token，不会像历史 K/V 一样长期放入 KV Cache。因此：

~~~
KV compression → 直接减少 Decode KV Cache
Q compression  → 主要减少训练 activation memory
~~~

不能把 Q compression 说成减少 KV Cache。

---

## 5. 最关键的 Decode 优化：Inference Weight Absorption

### 5.1 Naive 路径的问题

直接按照低秩定义实现 Decode，需要对每个历史 token 做：

~~~
c_j^KV → k_j^C
c_j^KV → v_j^C
~~~

然后再做标准 attention。这样 Cache 虽然变小，但每一步又要对所有历史 latent 做 up-projection。

论文的关键优化是利用矩阵乘法的结合律，把固定的 up-projection 吸收到 Query 和 Output projection。

### 5.2 Key 侧吸收

第 i 个 head 的 content Query 和 Key：

$$
q_{t,i}^C=W_{UQ,i}c_t^Q
$$

$$
k_{j,i}^C=W_{UK,i}c_j^{KV}
$$

原始 content score：

$$
\begin{aligned}
(q_{t,i}^C)^{\mathsf{T}}k_{j,i}^C
&=
(W_{UQ,i}c_t^Q)^{\mathsf{T}}
(W_{UK,i}c_j^{KV}) \\
&=
(c_t^Q)^{\mathsf{T}}
W_{UQ,i}^{\mathsf{T}}W_{UK,i}
c_j^{KV}
\end{aligned}
$$

定义：

$$
W_{QK,i}
=
W_{UQ,i}^{\mathsf{T}}W_{UK,i}
\in
\mathbb{R}^{d_c'\times d_c}
$$

于是：

$$
(q_{t,i}^C)^{\mathsf{T}}k_{j,i}^C
=
(c_t^Q)^{\mathsf{T}}W_{QK,i}c_j^{KV}
$$

也可以先把当前 Query 变换到 latent-key 空间：

$$
a_{t,i}^{\mathsf{T}}
=
(c_t^Q)^{\mathsf{T}}W_{QK,i}
$$

然后直接读取 Cache：

$$
s_{t,j,i}^C
=
a_{t,i}^{\mathsf{T}}c_j^{KV}
$$

对比：

~~~
naive：     q_C · (W_UK c_KV_j)
absorbed： (W_UQᵀ W_UK c_Q) · c_KV_j
~~~

历史 token 不需要逐个恢复完整 content Key。

### 5.3 Value 侧吸收

令 attention 权重为 p_tji。原始 Value 输出：

$$
\begin{aligned}
\sum_jp_{t,j,i}v_{j,i}^C
&=
\sum_jp_{t,j,i}W_{UV,i}c_j^{KV} \\
&=
W_{UV,i}
\left(\sum_jp_{t,j,i}c_j^{KV}\right)
\end{aligned}
$$

定义 latent weighted sum：

$$
z_{t,i}
=
\sum_jp_{t,j,i}c_j^{KV}
$$

那么：

$$
o_{t,i}^C=W_{UV,i}z_{t,i}
$$

再把输出投影按 head 切开。设：

$$
W_O=[W_{O,1}\;W_{O,2}\;\ldots\;W_{O,n_h}]
$$

其中 $W_{O,i}\in\mathbb{R}^{d\times d_h}$，定义：

$$
W_{OV,i}=W_{O,i}W_{UV,i}
\in\mathbb{R}^{d\times d_c}
$$

最终：

$$
u_t=\sum_{i=1}^{n_h}W_{OV,i}z_{t,i}
$$

对比：

~~~
naive：     c_KV_j → v_C_j → 加权求和 → W_O
absorbed： c_KV_j 加权求和 → 使用已合并的 W_OV
~~~

因此 MLA 的 Decode 核心不是“先压缩、再全部解压”，而是：

~~~
W_UK 吸收到 Query 侧
W_UV 吸收到 Output projection 侧
~~~

这里只使用了结合律；矩阵乘法可以重新括号化，但不能交换顺序。

---

## 6. 为什么 RoPE 要解耦

### 6.1 直接给 content Key 加 RoPE 的冲突

假设直接对 content Key 使用位置旋转：

$$
\widetilde{k}_{j,i}^C=R_jW_{UK,i}c_j^{KV}
$$

Query 也使用当前位置旋转：

$$
\widetilde{q}_{t,i}^C=R_tW_{UQ,i}c_t^Q
$$

点积变成：

$$
(\widetilde{q}_{t,i}^C)^{\mathsf{T}}\widetilde{k}_{j,i}^C
=
(c_t^Q)^{\mathsf{T}}
W_{UQ,i}^{\mathsf{T}}
R_t^{\mathsf{T}}R_j
W_{UK,i}
c_j^{KV}
$$

关键问题：

~~~
R_tᵀ R_j 位于 W_UQᵀ 和 W_UK 之间
并且同时依赖当前 t 和历史 j
~~~

因此不能预先定义一个与位置无关的固定矩阵 $W_{QK,i}=W_{UQ,i}^{\mathsf{T}}W_{UK,i}$。如果不吸收，就要重新处理历史 Key，损失 Decode 效率。

### 6.2 Decoupled RoPE

MLA 把 Query/Key 拆成两个子空间：

~~~
content 子空间：大，不带 RoPE，走 latent compression
RoPE 子空间：小，带位置，单独保存
~~~

论文中的 RoPE 分支：

$$
q_t^R=\operatorname{RoPE}(W_{QR}c_t^Q)
$$

$$
k_t^R=\operatorname{RoPE}(W_{KR}h_t)
$$

其中 q_t^R 再切成每个 head 的 q_ti^R；k_t^R 在论文设计中跨 head 共享。

拼接：

$$
q_{t,i}=[q_{t,i}^C;q_{t,i}^R]
$$

$$
k_{j,i}=[k_{j,i}^C;k_j^R]
$$

因此 score 分成 content 和 RoPE 两项：

$$
q_{t,i}^{\mathsf{T}}k_{j,i}
=
(q_{t,i}^C)^{\mathsf{T}}k_{j,i}^C
+
(q_{t,i}^R)^{\mathsf{T}}k_j^R
$$

代入吸收后的 content score：

$$
s_{t,j,i}
=
(c_t^Q)^{\mathsf{T}}W_{QK,i}c_j^{KV}
+
(q_{t,i}^R)^{\mathsf{T}}k_j^R
$$

最终 attention：

$$
o_{t,i}
=
\sum_{j=1}^{t}
\operatorname{softmax}_{j}
\left(
\frac{s_{t,j,i}}{\sqrt{d_h+d_h^R}}
\right)v_{j,i}^C
$$

因此每个历史 token 的 Cache 是：

$$
\boxed{\operatorname{Cache}_j=[c_j^{KV};k_j^R]}
$$

RoPE 没有消失，而是被放到较小的独立分支中。

---

## 7. 论文公式和代码的对应关系

| 论文公式 | 含义 |
|---|---|
| Eq. 1–3 | 标准 MHA 的 Q/K/V 投影 |
| Eq. 7–8 | 标准 attention 输出与输出投影 |
| Eq. 9–11 | c_KV、content Key、content Value |
| Eq. 12–13 | Query compression |
| Eq. 14–15 | decoupled RoPE 的 Query/Key 分支 |
| Eq. 16–17 | content 与 RoPE 子向量拼接 |
| Eq. 18–19 | MLA attention 和输出投影 |
| Appendix C Eq. 37–47 | 完整 MLA 链路，并标注生成时需要缓存的向量 |

Appendix C 的工程含义是：

~~~
naive 公式会从 c_KV 恢复 k_C/v_C；
利用结合律后，W_UK 可吸收到 W_UQ，
W_UV 可吸收到 W_O，因此 Decode 不必为每个历史 token
重新算完整 Key 和 Value。
~~~

---

## 8. Prefill 和 Decode 数据流

### 8.1 Prefill

Prefill 一次处理整段输入：

~~~
h[0:T]
  ├── c_Q = W_DQ h
  ├── c_KV = W_DKV h
  ├── q_R = RoPE(W_QR c_Q)
  └── k_R = RoPE(W_KR h)
~~~

服务实现中长期保存的 Cache：

~~~
c_KV_cache: [T, d_c]
k_R_cache : [T, d_h_R]
~~~

训练时还存在 activation、attention score tile 和反向传播所需的中间量，不要把 Decode Cache shape 当成训练中所有 activation 的 shape。

### 8.2 Decode

新 token 到来时：

~~~
1. h_t → c_t^Q
2. h_t → c_t^KV，并追加到 c_KV_cache
3. c_t^Q → q_t^C
4. c_t^Q → q_t^R，再做 RoPE
5. h_t → k_t^R，再做 RoPE，并追加到 k_R_cache
6. content score = absorbed_query @ c_KV_cache.T
7. rope score = q_R @ k_R_cache.T
8. score = content score + rope score
9. softmax 后，在 c_KV_cache 上做 latent weighted sum
10. 使用 W_OV 把 latent output 合成 hidden output
~~~

关键点：

~~~
历史 Cache 只读 c_KV 和 k_R；
历史 token 不需要逐个恢复完整 k_C 和 v_C。
~~~

---

## 9. KV Cache 账本：元素数和 bytes 分开算

论文 Table 1 统计的是**每个 token、每层的元素数量**，不是直接统计 bytes。设层数为 l：

| 机制 | 每 token、每层缓存元素数 |
|---|---:|
| MHA | $2n_h d_h$ |
| GQA | $2n_g d_h$ |
| MQA | $2d_h$ |
| MLA | $d_c+d_h^R$ |

DeepSeek-V2 论文给出：

$$
d_c=4d_h,
\qquad
d_h^R=\frac{d_h}{2}
$$

因此：

$$
d_c+d_h^R=4.5d_h
$$

MLA 等效的 GQA group 数：

$$
\frac{d_c+d_h^R}{2d_h}
=
\frac{4.5d_h}{2d_h}
=2.25
$$

### 9.1 具体 bytes 例子

假设：

~~~
H = 128
d_h = 128
单层、单 token
FP16 或 BF16：2 bytes/element
~~~

| 机制 | 元素数 | bytes | KiB |
|---|---:|---:|---:|
| MHA | $2×128×128=32768$ | $65536$ | $64$ |
| GQA-8 | $2×8×128=2048$ | $4096$ | $4$ |
| MQA | $2×128=256$ | $512$ | $0.5$ |
| MLA | $512+64=576$ | $1152$ | $1.125$ |

所以不能把 MHA 的 32768 elements 写成 32 KB；在 2 bytes/element 下它是 65536 bytes，也就是 64 KiB。GQA-8 同理是 4 KiB，不是 2 KiB。

完整 Cache：

$$
\operatorname{total\ bytes}
=
\operatorname{bytes/token/layer}
\times T
\times l
\times B
$$

其中 T 是上下文长度，l 是层数，B 是并发 batch。若使用 INT8、FP8 等存储格式，bytes/element 需要按实际存储格式替换；论文 Table 1 只比较元素数量。

---

## 10. 小型 Decode 手算

设某个 head：

~~~
d_c = 3
d_h = 3
d_h_R = 1
~~~

两个历史 latent：

$$
c_1^{KV}=[1,0,2]^{\mathsf{T}},
\qquad
c_2^{KV}=[0,1,1]^{\mathsf{T}}
$$

假设当前 Query 经过 W_QK_i 后得到：

$$
(c_t^Q)^{\mathsf{T}}W_{QK,i}=[2,-1,0]
$$

content score：

$$
s_{t,1,i}^C=[2,-1,0]
\begin{bmatrix}1\\0\\2\end{bmatrix}=2
$$

$$
s_{t,2,i}^C=[2,-1,0]
\begin{bmatrix}0\\1\\1\end{bmatrix}=-1
$$

再假设：

~~~
q_R = 0.5
k_R[1] = 0.2
k_R[2] = -0.5
~~~

RoPE score：

$$
s_{t,1,i}^R=0.5×0.2=0.1,
\qquad
s_{t,2,i}^R=0.5×(-0.5)=-0.25
$$

合并并按 sqrt(d_h+d_h_R)=sqrt(4)=2 缩放：

$$
\left[
\frac{2+0.1}{2},
\frac{-1-0.25}{2}
\right]
=[1.05,-0.625]
$$

softmax 约为：

$$
p\approx[0.842,0.158]
$$

Value 直接在 latent 空间累加：

$$
\begin{aligned}
z_{t,i}
&=0.842c_1^{KV}+0.158c_2^{KV}\\
&=0.842[1,0,2]^{\mathsf{T}}
+0.158[0,1,1]^{\mathsf{T}}\\
&=[0.842,0.158,1.842]^{\mathsf{T}}
\end{aligned}
$$

最后才应用 W_OV_i。这个例子展示了：Cache 中读取的是 latent，Value 的加权和也先在 latent 空间完成。

---

## 11. Reference Code：naive 与 absorbed

下面使用行向量：

~~~
c_q       : [H, dc_q]
c_kv      : [T, dc]
W_UQ      : [H, dh, dc_q]
W_UK      : [H, dh, dc]
W_UV      : [H, dv, dc]
q_r       : [H, dr]       # 已经 RoPE
k_r       : [T, dr]       # 已经 RoPE，共享给各 head
W_O       : [d_model, H * dv]
~~~

代码只解释论文机制，省略 bias、RMSNorm、量化、paged cache 和 causal mask。

### 11.1 Decoupled RoPE reference helper

公式中的 RoPE 分支先产生未旋转的向量，再按位置旋转：

~~~python
def apply_rope(x, cos, sin):
    # x: [..., dr], cos/sin: [..., dr // 2]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos
    return torch.stack((y_even, y_odd), dim=-1).flatten(-2)


# q_r_raw comes from W_QR @ c_q and has shape [H, dr].
# k_r_raw comes from W_KR @ h for every cached token and has shape [T, dr].
# After applying position-specific cos/sin:
# q_r = apply_rope(q_r_raw, q_cos, q_sin)     # [H, dr]
# k_r = apply_rope(k_r_raw, k_cos, k_sin)     # [T, dr]
~~~

因此下面的 q_r 和 k_r 参数都表示已经完成 RoPE 的分支；它们没有和 content latent 合并，也没有被省略。

### 11.2 Naive：先恢复完整 content K/V

~~~python
import math
import torch


def mla_decode_naive(c_q, c_kv, W_UQ, W_UK, W_UV, q_r, k_r, W_O):
    # c_q: [H, dc_q], c_kv: [T, dc]
    # W_UQ: [H, dh, dc_q], W_UK: [H, dh, dc]
    # W_UV: [H, dv, dc], q_r: [H, dr], k_r: [T, dr]
    # W_O: [d_model, H * dv]
    q_c = torch.einsum("hdc,hc->hd", W_UQ, c_q)
    k_c = torch.einsum("hdc,tc->thd", W_UK, c_kv)
    v_c = torch.einsum("hdc,tc->thd", W_UV, c_kv)

    score_c = torch.einsum("hd,thd->ht", q_c, k_c)
    score_r = torch.einsum("hd,td->ht", q_r, k_r)
    scores = (score_c + score_r) / math.sqrt(q_c.shape[-1] + q_r.shape[-1])

    p = torch.softmax(scores, dim=-1)              # [H, T]
    head_out = torch.einsum("ht,thd->hd", p, v_c) # [H, dv]
    return head_out.reshape(-1) @ W_O.T             # [d_model]
~~~

这个版本会为每个历史 token 生成完整 k_c 和 v_c，适合验证公式，但不是 MLA 高效 Decode 路径。

### 11.3 Absorbed：直接在 compressed Cache 上计算

~~~python
def mla_decode_absorbed(c_q, c_kv, W_UQ, W_UK, W_UV, q_r, k_r, W_O):
    H, dh, dc_q = W_UQ.shape
    _, _, dc = W_UK.shape
    dv = W_UV.shape[1]

    # W_QK[h] = W_UQ[h].T @ W_UK[h]
    # [H, dc_q, dc]
    W_QK = torch.einsum("hdq,hdk->hqk", W_UQ, W_UK)

    # a[h] = c_q[h].T @ W_QK[h]
    # [H, dc]
    a = torch.einsum("hq,hqk->hk", c_q, W_QK)

    # 直接读取 c_kv_cache，不生成历史完整 k_c
    score_c = torch.einsum("hk,tk->ht", a, c_kv)
    score_r = torch.einsum("hd,td->ht", q_r, k_r)
    scores = (score_c + score_r) / math.sqrt(dh + q_r.shape[-1])
    p = torch.softmax(scores, dim=-1)

    # z[h] = sum_t p[h, t] * c_kv[t]
    z = torch.einsum("ht,tk->hk", p, c_kv)

    # W_OV[h] = W_O_head[h] @ W_UV[h]
    # W_O_heads: [H, d_model, dv]
    W_O_heads = W_O.reshape(W_O.shape[0], H, dv).permute(1, 0, 2)
    W_OV = torch.einsum("hmv,hvk->hmk", W_O_heads, W_UV)

    # u = sum_h W_OV[h] @ z[h]
    return torch.einsum("hmk,hk->m", W_OV, z)
~~~

两条路径在数学上等价，实际浮点结果可能因为运算顺序略有差异。随机小 shape 的等价性检查：

~~~python
torch.manual_seed(0)
H, T, dc_q, dc, dh, dv, dr, d_model = 2, 5, 4, 6, 8, 8, 4, 16

c_q = torch.randn(H, dc_q)
c_kv = torch.randn(T, dc)
W_UQ = torch.randn(H, dh, dc_q)
W_UK = torch.randn(H, dh, dc)
W_UV = torch.randn(H, dv, dc)
q_r = torch.randn(H, dr)
k_r = torch.randn(T, dr)
W_O = torch.randn(d_model, H * dv)

y0 = mla_decode_naive(c_q, c_kv, W_UQ, W_UK, W_UV, q_r, k_r, W_O)
y1 = mla_decode_absorbed(c_q, c_kv, W_UQ, W_UK, W_UV, q_r, k_r, W_O)
torch.testing.assert_close(y0, y1, rtol=1e-5, atol=1e-5)
~~~

这只是 explanatory reference code，不是已完成的 MLA 算子，不代表 LeetGPU 或真实 GPU 验收。

---

## 12. Triton / CUDA 实现落点

未来实现 MLA Decode 时，可以先把 Cache 抽象为：

~~~
c_kv_cache: [layers, batch, sequence, d_c]
k_r_cache : [layers, batch, sequence, d_h_R]
~~~

一个 query head 或 query head tile 处理一段 Cache tile：

~~~python
# 伪代码，不是当前可运行 kernel
q_latent = load_current_query_latent()       # [BLOCK_H, d_c_q]
c_kv = load(c_kv_cache + kv_offsets)         # [BLOCK_T, d_c]
k_r = load(k_r_cache + kv_offsets)           # [BLOCK_T, d_r]

q_in_latent = tl.dot(c_q, W_QK)            # [BLOCK_H, d_c]
content_score = tl.dot(q_in_latent, tl.trans(c_kv))  # [BLOCK_H, BLOCK_T]
rope_score = tl.dot(q_r, tl.trans(k_r))    # [BLOCK_H, BLOCK_T]
scores = (content_score + rope_score) * scale

p = online_softmax(scores)
latent_acc += p[:, :, None] * c_kv[None, :, :]
out += tl.dot(latent_acc, tl.trans(W_OV))
~~~

实现时要注意：

1. c_KV 的最后一维是 d_c，不是完整的 H × d_h；
2. k_R 是额外的 RoPE Cache，不能丢掉；
3. content score 使用 W_QK，避免 materialize 历史完整 k_C；
4. Value 先在 latent 空间累加，再使用 W_OV；
5. 投影和合适 tile 的 latent dot 可能使用 Tensor Core；softmax、RoPE、mask、地址计算和归约不是同一条路径；
6. tl.dot 只是矩阵点积语义，是否生成 MMA 指令要结合 dtype、layout、tile 和目标 GPU 的编译结果验证。

和 FA2 的关系：

~~~
FA2：优化 attention 的执行方式和 IO
MLA：改变 K/V 的表示方式和 KV Cache 组织
~~~

MLA 可以和 FlashAttention 的 tiled / online-softmax 思路结合，但 MLA 本身不是另一种 softmax。

---

## 13. MHA、MQA、GQA、MLA 对比

| 机制 | Cache 中存什么 | 主要压缩方式 | 主要代价 |
|---|---|---|---|
| MHA | 每个 head 的完整 K/V | 不压缩 | Cache 最大 |
| MQA | 一份共享 K/V | 所有 Query head 共享一组 K/V | 表达能力可能损失较大 |
| GQA | 每组一份 K/V | 多个 Query head 共享一组 K/V | Cache 和质量折中 |
| MLA | c_KV 加 k_R | KV joint compression + decoupled RoPE | 需要特殊 Decode 路径 |

MLA 的优势：

- 显著减少每 token 的 KV Cache；
- 不必像 MQA 那样让所有 Query head 共享一份完整 K/V；
- weight absorption 避免逐 token 恢复完整历史 K/V；
- decoupled RoPE 保留位置信息；
- 对长上下文和高并发 Decode 尤其有价值。

MLA 的代价：

- attention 结构比 MHA 复杂；
- 需要维护 content latent 和 RoPE 两个 Cache 分支；
- 需要实现 W_QK / W_OV 吸收；
- Decode 中会出现小矩阵 GEMM、Cache layout 和低 batch 效率问题；
- 不能只替换模型结构而不修改 serving/runtime 的 Cache 和 attention kernel。

---

## 14. 常见误解

### 误解 1：MLA 只是把 GQA 的 group 数继续减少

不准确。GQA 保存每组完整 K/V；MLA 保存联合 latent，并额外保存小的 RoPE Key，Cache 组织和计算路径都不同。

### 误解 2：MLA 只减少 d_k

不准确。MLA 是 KV joint compression、decoupled RoPE 和 inference weight absorption 的组合。

### 误解 3：Decode 时一定要把所有历史 K/V 解压出来

那是 naive reference 路径。论文的关键正是把 W_UK 和 W_UV 吸收掉，直接在 latent Cache 上计算。

### 误解 4：Q compression 也减少 KV Cache

不会。c_Q 主要属于当前 token，主要帮助训练 activation memory 和吸收后的 Query 计算。

### 误解 5：MLA Cache 只有一个向量

严格说是两个部分：

~~~
c_KV：低秩 content latent
k_R ：小的、位置相关的 RoPE key
~~~

### 误解 6：论文中的 576 是 576 bytes

论文 Table 1 的 576 是 elements。2 bytes/element 时是 1152 bytes，也就是 1.125 KiB。

---

## 15. 自检题和当前状态

自检：

1. MHA 的每 token、每层 KV Cache 为什么是 $2n_h d_h$ 个元素？
2. c_KV 与完整 K/V 的 shape 分别是什么？
3. 为什么 MLA 是 joint compression？
4. 为什么 Query compression 不等于 KV Cache compression？
5. 展开 $(W_{UQ,i}c_t^Q)^{\mathsf{T}}(W_{UK,i}c_j^{KV})$。
6. W_QK 为什么是 $W_{UQ,i}^{\mathsf{T}}W_{UK,i}$？
7. 为什么 Value 可以先在 latent 空间求加权和？
8. 为什么直接给 content K 加 RoPE 会破坏 weight absorption？
9. decoupled RoPE 中，为什么 k_R 可以跨 head 共享？
10. Decode Cache 为什么是 [c_KV, k_R] 而不是 [K, V]？
11. 给定 H=128、d_h=128、d_c=512、d_h^R=64，算 MHA、GQA-8 和 MLA 的 FP16 bytes。
12. 如果 kernel 对历史 token 逐个 materialize 完整 k_C/v_C，它还缺少 MLA 的哪个优化？

当前状态：

~~~
FA2 理论：已完成阅读
MLA 理论：本笔记作为当前学习节，待通过公式和代码自检
MLA Triton：尚未实现
MLA LeetGPU：尚未开始
MLA 真实 GPU：尚未验证
~~~

后续实现顺序：

~~~
PyTorch naive/reference
→ PyTorch absorbed 等价性检查
→ 明确 c_KV + k_R Cache layout
→ Triton compressed score + latent value accumulation
→ LeetGPU/正确性
→ 真实 GPU benchmark
→ DSA
~~~

---

## 参考

- DeepSeek-V2 论文：[arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- DeepSeek-V3 论文：[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- GQA 论文：[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- 仓库相关：[FlashAttention-2 统一笔记](./flash-attention-2.md)
- 仓库相关：[GQA 笔记](../../papers/attention/gqa.md)
