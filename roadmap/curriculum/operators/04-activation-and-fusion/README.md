# 第四章 Activation 与 Fusion：从 SwiGLU 点算子到 MLP 的物化边界

SwiGLU 包含两个上投影、一次激活与逐元素乘法，以及一个下投影：

$$
G=XW_{gate},\qquad U=XW_{up},\qquad
Z=\operatorname{SiLU}(G)\odot U,\qquad Y=ZW_{down}.
$$

$X\in\mathbb R^{T\times H}$，$G,U,Z\in\mathbb R^{T\times I}$，$Y\in\mathbb R^{T\times H}$。$T$ 是 token 数，$H$ 是 hidden size，$I$ 是中间宽度，$\odot$ 表示逐元素乘法。上投影沿 $H$ 归约，下投影沿 $I$ 归约。

融合要决定哪些中间值仍需存储：G/U 可以分开保存或打包，SiLU 与乘法可以同次读写，部分操作还可以放入 GEMM epilogue。省下的中间读写，要与新增寄存器、归约依赖和布局转换一起计算。

对应 A[M,N]×B[N,K] 的记号，两个上投影取 M=T、N=H、K=I；下投影取 M=T、N=I、K=H。符号改变后，归约轴和输出轴仍按各自矩阵的形状确定。

## 1. dense MLP 的形状和语义

### 1.1 Gated 与非 gated MLP 不是同一个公式

非 gated 的两层 MLP 可以写成

$$
Y=\operatorname{GELU}(XW_1+b_1)W_2+b_2.
$$

这里先对一个 `[T,I]` 的上投影结果做 GELU，再做下投影。Gated MLP 需要两路独立的上投影：

$$
G=XW_{gate}+b_{gate},\qquad U=XW_{up}+b_{up},
$$
$$
Z=\operatorname{SiLU}(G)\odot U,\qquad Y=ZW_{down}+b_{down}.
$$

`G` 是门控分支，`U` 是被门控的 value/up 分支；因此权重名不能只靠“左边/右边”猜。Llama 的官方实现可以直接对照 `self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))`：见 Hugging Face LlamaMLP。旧笔记中若把 `W_up` 和 `W_gate` 的激活角色调换，应以这里的权重角色为准；矩阵形状相同并不意味着语义可以互换。

如果使用 bias，`b_gate[I]`、`b_up[I]` 要在各自投影的输出列上广播；`b_down[H]` 要在下投影输出列上广播。bias 并不是 `[T,I]` 的一份新数组，正确的地址是 `b_gate[i]` 或 `b_up[i]`。若实现把 bias 写成按 token 的 `[T]`，即使 `T=I` 的测试碰巧通过，也已经改变函数。

门控角色不能靠交换变量名来“等价化”。例如 $g=-2,u=3$ 时，

$$
\operatorname{SiLU}(-2)\cdot3\approx-0.7152175,\qquad
\operatorname{SiLU}(3)\cdot(-2)\approx-5.7154448.
$$

两者相差很大；`W_gate` 先进入 activation，`W_up` 才作为逐元素被乘分支，必须在接口和 reference 中固定下来。

### 1.2 数字化一个小 MLP

取 `T=2,H=3,I=4`：`X` 有 6 个元素，两个上投影各产生 8 个元素，门控后的 `Z` 仍是 `[2,4]`，最后 `Y` 有 6 个元素。对 token `t=1`、中间列 `i=2`：

$$
G_{1,2}=\sum_{h=0}^{2}X_{1,h}W_{gate,h,2},\qquad
U_{1,2}=\sum_{h=0}^{2}X_{1,h}W_{up,h,2},
$$
$$
Z_{1,2}=G_{1,2}\,\sigma(G_{1,2})\,U_{1,2}.
$$

最后 `Y[1,0]` 再沿 `i=0..3` 归约。这个例子里 pointwise kernel 的逻辑元素数是 `T*I=8`，不是 `T*H=6`；把输入 hidden 维误当成 activation 宽度，会同时破坏索引和 bytes 账本。

## 2. 激活函数：公式、导数和 IEEE 边界必须分开

### 2.1 ReLU、GELU exact 与 GELU tanh approximation

ReLU（Rectified Linear Unit，修正线性单元）是

$$
\operatorname{ReLU}(x)=\max(0,x),\qquad
\operatorname{ReLU}'(x)=
\begin{cases}0,&x<0\\1,&x>0\end{cases}.
$$

在 `x=0` 处数学导数不存在，工程实现通常选 0 或 1 作为反向约定；这不应被写成函数在零点处有唯一导数。

GELU（Gaussian Error Linear Unit，高斯误差线性单元）的 exact erf 形式是

$$
\operatorname{GELU}_{erf}(x)=\frac{x}{2}\left(1+\operatorname{erf}\left(\frac{x}{\sqrt2}\right)\right).
$$

它的导数为

$$
\operatorname{GELU}_{erf}'(x)=\frac12\left(1+\operatorname{erf}\left(\frac{x}{\sqrt2}\right)\right)
 +\frac{x}{\sqrt{2\pi}}e^{-x^2/2}.
$$

常见的 tanh approximation 是另一个函数：

$$
\operatorname{GELU}_{tanh}(x)=\frac{x}{2}\left[1+\tanh\left(c(x+ax^3)\right)\right],
$$
$$
c=\sqrt{\frac2\pi},\qquad a=0.044715.
$$

若令 $q=c(x+ax^3)$，它的导数是

$$
\operatorname{GELU}_{tanh}'(x)=\frac12(1+\tanh q)+\frac{x}{2}(1-\tanh^2q)c(1+3ax^2).
$$

`erf` exact 与 `tanh` approximation 不能混称为同一个函数。以 $x=1.25$ 为例，exact erf 值约为 `1.117938`，tanh 近似约为 `1.117714`，差约 `0.00022358`。PyTorch `F.gelu` 的默认 `approximate='none'` 对应 erf 语义；调用方明确选择 `approximate='tanh'` 才是近似。仓库的 reference/cuda/include/activations.cuh 是可复用的 reference header，不是用户实践记录；它的函数体使用 tanh 近似，但注释把它写成“same as PyTorch's default”，这条注释不准确。下文引用的是 tanh 近似函数，应与 `approximate='tanh'` 对照。

### 2.2 SiLU、SwiGLU 和导数

SiLU（Sigmoid Linear Unit，也叫 Swish）为

$$
\operatorname{SiLU}(x)=x\sigma(x),\qquad
\sigma(x)=\frac1{1+e^{-x}}.
$$

用 $s=\sigma(x)$，导数可以写成

$$
\operatorname{SiLU}'(x)=s+x s(1-s).
$$

SwiGLU（Swish-Gated Linear Unit）是

$$
f(g,u)=\operatorname{SiLU}(g)u,\qquad
\frac{\partial f}{\partial g}=u\operatorname{SiLU}'(g),\qquad
\frac{\partial f}{\partial u}=\operatorname{SiLU}(g).
$$

因此 `gate` 和 `up` 都必须是同形状的独立逻辑输入。`up` 不是“激活后的第二个 gate”，而是被门控分支逐元素缩放的上投影值。

sigmoid 的直接式 `1/(1+exp(-x))` 在负向大幅值时可能构造 `exp(large positive)`。对有限 FP32 输入，`exp(x)` 的溢出阈值约为 `log(FLT_MAX)≈88.72`；有限不等于所有中间表达式都安全。教学 kernel 使用 $e^{-\lvert x\rvert}$：

$$
v=\exp(-\lvert x\rvert),\qquad
\sigma(x)=\begin{cases}(1+v)^{-1},&x\ge0\\v(1+v)^{-1},&x<0.
\end{cases}
$$

这不能把 NaN 或无穷输入变成合法输入：IEEE 特殊值仍应按合同处理，`NaN` 不应被“finite check 没有报错”掩盖，输出溢出也不能假定为 finite。这里的 kernel 合同是输入 finite 且幅度合理；正确性脚本先做一次 preflight，热路径 wrapper 不在每次调用中发起 `isfinite` 归约。实现时要注意 `tl.where` 的两个参数表达式都可能被求值，不能把它当作只执行选中分支；正因为两支都使用 `exp(-abs(g))`，表达式才保持数值安全。内存安全仍来自 `tl.load(..., mask=mask)` 与 `tl.store(..., mask=mask)`，不是来自 `tl.where`。

## 3. 地址、stride、广播和 packed 布局

### 3.1 两个独立 `[T,I]` buffer

contiguous row-major 的 `gate[t,i]` 元素偏移为

$$
o_{gate}(t,i)=t\cdot I+i,
$$

字节地址为 `gate_base + (t*I+i)*s`，其中 `s` 是元素字节数。`up[t,i]` 有同样的 stride 但不同的 base pointer，`out[t,i]` 也是第三个 base pointer。若存在 bias：

$$
o_{bias}(i)=i,\qquad o_{residual}(t,h)=t\cdot H+h.
$$

这就是广播的实际含义：`bias[I]` 在 token 轴没有乘法，`residual[T,H]` 则沿两个逻辑轴都变化。`bias` 和 `residual` 若使用不同 layout 或非 contiguous stride，kernel 必须把 stride 作为参数；本章 Triton 示例固定为 contiguous 2-D `[T,I]`，不把任意 view 偷换成 contiguous 合同。

本章的 `fused_swiglu` wrapper 只接受两路独立、contiguous 的 `[T,I]` buffer。对 packed `P[T,2I]` 来说，`P[:,:I]` 和 `P[:,I:]` 在 `T>1` 时通常带有 row stride `2I`，不是 contiguous view，不能直接传给这个 wrapper，也不能在计时循环内偷偷调用 `.contiguous()`。显式转换需要另列一次读写和 kernel/allocator 成本。#54 的 `T=1` 一维前后半切分可以 reshape 成 `[1,I]` 而不复制，但这不代表多行 packed `[T,2I]` 也满足本 kernel 的 contiguous 合同。

当 `n=T*I` 不被 `BLOCK` 整除时，第 `pid` 个 program 的索引是

$$
\text{offset}=\operatorname{int64}(pid)\times BLOCK+\operatorname{arange}(0,BLOCK),
$$

并用 `offset<n` 保护 load/store。使用 int64 是为了避免 `pid*BLOCK` 在较大展平张量上先发生 32-bit 乘法溢出；mask 只能阻止无效元素参与内存访问，不能修复已经溢出的索引。

### 3.2 `[T,2I]` packed 与一维题面的差异

如果一行采用 `[gate | up]` packed 布局，元素偏移是

$$
o_{packed}(t,branch,i)=t(2I)+
\begin{cases}i,&branch=gate\\I+i,&branch=up.
\end{cases}
$$

`T=2,I=3` 时，packed 顺序是 `[g00,g01,g02,u00,u01,u02,g10,g11,g12,u10,u11,u12]`。因此不能把整个二维 buffer flatten 后把前半段当成所有 gate、后半段当成所有 up；那样会把 token 0 的 up 误当成 token 1 的 gate，把 token 1 的 gate 误当成 token 0 的 up。每一行先切成两段才正确。

LeetGPU #54 的公开题面是一个不同的输入合同：一维 `input[N]`，`N` 为偶数，前 `N/2` 是 gate，后 `N/2` 是 up，输出是 `[N/2]`。它不能直接证明二维 `[T,2I]` packed 的地址公式；题面链接为 [官方题库目录](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/easy/54_swiglu) 与 [LeetGPU 题目](https://leetgpu.com/challenges)。

## 4. 可读的 Triton pointwise kernel

完整实现放在 examples/fused_swiglu.py。核心部分保持一个清楚的生命周期：每个 program 读取一段 `gate/up`，转成 FP32，计算 sigmoid、SiLU 和乘法，只把最终结果 cast/store 一次。一个 Triton program 是一个向量化逻辑执行单元，不等于一个 CUDA thread；`BLOCK=1024` 不表示启动 1024 个 CUDA 线程，更不意味着每个线程独占一个输出元素。

<!-- source-check: examples/fused_swiglu.py -->
~~~python
@triton.jit
def fused_swiglu_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    """One Triton program owns a contiguous segment of the flattened tensor."""
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # exp(-abs(x)) avoids constructing exp(large positive) for finite x.
    # NaN and infinity still follow IEEE propagation; the wrapper's contract
    # is finite, reasonably scaled input rather than silent special-value use.
    exp_neg_abs = tl.exp(-tl.abs(gate))
    sigmoid = tl.where(
        gate >= 0.0,
        1.0 / (1.0 + exp_neg_abs),
        exp_neg_abs / (1.0 + exp_neg_abs),
    )
    result = (gate * sigmoid) * up
    tl.store(out_ptr + offsets, result, mask=mask)
~~~

host wrapper 的 shape 合同是：两个输入同 shape、同 device、同 dtype，均为 contiguous 2-D，dtype 只允许 FP16、BF16、FP32，输出同 shape/device/dtype。它可以检查 Python 元数据和 `numel` 是否能放入 signed int64，但不应在每次调用前 `torch.isfinite(gate).all()`；那是一次额外的 device reduction 和同步。若训练或 correctness policy 要拒绝特殊值，应该在 preflight 里做一次检查，然后再进入 timed loop。

输出写回只有一次 cast。对 FP16/BF16，不能要求“独立 split 的低精度中间结果”和“fused 中 FP32 保持到最终 store”逐元素严格相等；正确对照是同一次 FP32 reference 后最终 cast，容差按 dtype 选择。FP32 下则可以把 fused 与 split 的中间存储语义做更严格的对比。

## 5. 可融合的边界与函数语义

### 5.1 pointwise、epilogue 与完整 MLP 的边界

`SiLU(G)*U` 是天然的 pointwise fusion：每个 `[t,i]` 只需要 `G[t,i]` 和 `U[t,i]`，没有跨元素归约。`bias+GELU` 也是逐元素 fusion；`residual add` 只要两个输入在相同坐标上相加，也可以和后续 pointwise 合并。RMSNorm/LayerNorm 则带有行归约：必须先得到整行的统计量，再对行内元素归一化。把它和 activation 放进同一个 kernel 可能减少一次物化，却改变寄存器 live range、同步结构和数值顺序。

`GEMM epilogue` 通常指 GEMM 累加器已经得到一个输出元素后，在写回前做 bias、activation 或缩放；它能省掉完整 C 的一次 store/load。但 `XW_gate` 与 `XW_up` 之间有各自的 H 维归约，`ZW_down` 又需要完整的 Z 沿 I 维归约。一个 CTA 不能在没有跨 CTA 同步的情况下直接消费另一个 CTA 尚未完成的输出；把两个 GEMM、SwiGLU 和 down projection 都写进“一个 kernel”并不会自动消除归约依赖。

如果使用 split-K 或多 CTA 生成 partial sum，非线性必须在 partial 合并之后执行。对每个 partial 先 GELU 再相加，得到的是 $\sum_j\operatorname{GELU}(p_j)$，而不是 $\operatorname{GELU}(\sum_jp_j)$；这不是舍入误差，而是函数语义改变。FP32 累加器的 live range、寄存器数量、spill、occupancy（驻留度）和重新计算之间也存在真实 trade-off。

### 5.2 融合顺序会改变函数

`residual + norm` 的合同若定义为

$$
z_{t,h}=x_{t,h}+r_{t,h},\qquad
v_t=\frac{1}{H}\sum_{j=0}^{H-1}z_{t,j}^2,\qquad
y_{t,h}=\frac{z_{t,h}}{\sqrt{v_t+\varepsilon}}w_h.
$$

就不能把 `RMSNorm(x)` 和 `residual` 在 norm 后相加来省一步；那是另一函数。类似地，`bias+GELU` 与 `GELU` 后加 bias 不等价，`SiLU(gate)*up` 与 `SiLU(gate*up)` 也不等价。融合只应重排实现，不应重排数学依赖。

LeetGPU #83 的题面正好把这个边界写成一个可核对的合同：`x/residual[N,C]`、`weight[C]`，先 `z=x+residual`，再做 RMSNorm 和 weight，只有一个 `out[N,C]`，不返回 `z`，也不改原 `residual`。这是 residual-add+norm 的独立题，不等于本章 SwiGLU pointwise kernel。题面见 [83_fused_residual_add_rms_norm](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/83_fused_residual_add_rms_norm) 与 [LeetGPU 题库](https://leetgpu.com/challenges)。

## 6. FLOPs、bytes 和 launch 账本

令 `n=T*I`，每个元素占 `s` bytes。只计算 `G/U -> Z` 这一段，不把两次上投影和下投影藏掉：

| 路径 | 逻辑读写 | 理想元素流量 |
|---|---|---:|
| separate | 读 G、写 S=SiLU(G)；再读 S、读 U、写 Z | `5n`，即 `5ns` bytes |
| fused | 读 G、读 U、写 Z | `3n`，即 `3ns` bytes |

`5n` 的拆法是第一 kernel 的 `read G + write S = 2n`，第二 kernel 的 `read S + read U + write Z = 3n`。因此理想逻辑流量少了 `2ns`，不是宣称时间必然按 `5/3` 缩短。cache 命中、读写合并、launch gap、allocator、stream 排队、CUDA Graph capture 和其他工作都会改变实际时间。

如果 split baseline 写成 `sigmoid(G)`、`G*sigmoid(G)`、`S*U` 三个 kernel，逻辑流量是 `2n + 3n + 3n = 8n`，不是 `5n`。本章的 FP32 benchmark 使用 `torch.ops.aten.silu.out(G, out=S)` 再 `torch.mul(S,U,out=Z)`，明确固定为两 kernel 的 `5ns`。目标 PyTorch 若没有 `aten.silu.out` overload，脚本直接报错，不静默换成三 kernel 链。`5/3` 只是相同有效带宽、忽略计算和 launch 影响时的逻辑流量比，不是全局速度上限；小 shape 还可能因为少一次 launch 而获得超过该比值的链路收益。

以 FP16 `T=4096,I=11008` 为例，`n=45,088,768`，元素大小 `s=2` bytes：

$$
5ns=450,887,680\text{ B},\qquad 3ns=270,532,608\text{ B},
$$
$$
\Delta=180,355,072\text{ B}.
$$

这只计算已经给定的 `G/U -> Z` pointwise 链，不包含生成 `G/U` 的两次 GEMM、读取 `W_gate/W_up`、生成 `Y` 的 down GEMM 或其他框架开销。层级收益还要服从 Amdahl 定律：若 pointwise 只占整层 10%，即使它自身快 2 倍，整层最多是

$$
\frac1{0.9+0.1/2}=1.0526\times,
$$

而不是 2 倍。

FLOPs 也要分层：SiLU 至少包含 sigmoid 的指数/除法与一次乘法，SwiGLU 再有一次乘法；`XW_gate` 和 `XW_up` 各约 `2THI` FLOPs，`ZW_down` 约 `2TIH` FLOPs。只拿 pointwise 的 `n` 元素 bytes 去除整个 MLP 的总时间，会把三个 GEMM 的权重读、输出写和矩阵指令成本完全漏掉。报告时分别列出算法逻辑 bytes、实测 DRAM/L2 traffic、kernel-only 时间和 wrapper/端到端时间。

## 7. 从两个 GEMM 到完整 MLP 的融合粒度

先看一个容易被“合并 GEMM”误导的形状。把两组上投影拼成

$$
W_{gu}=[W_g\mid W_u]\in\mathbb{R}^{H\times 2I},\qquad
[G\mid U]=XW_{gu}.
$$

这确实可以把两个上投影合为一次 GEMM launch；FLOPs 仍然是 `4THI`，因为输出列从 `I` 变成了 `2I`。但普通 GEMM 的 output column tile 往往只覆盖 `[0,I)` 的 G 段或 `[I,2I)` 的 U 段。同一个 feature 的 `G[t,i]` 与 `U[t,i]` 可能由不同 CTA 生成，不能自动在普通 GEMM epilogue 里做 `SiLU(G[t,i])*U[t,i]`。

要在上投影 epilogue 中直接产出 Z，至少需要让 producer co-produce 两个 accumulator tile，或者采用能把同一 feature 的 G/U 配对到同一个 program 的布局和专门 epilogue。常规 `[T,2I]` 的左右半布局地址为 `t*(2I)+i` 与 `t*(2I)+I+i`；交错布局 `[g0,u0,g1,u1,...]` 的地址则为 `t*(2I)+2i` 与 `t*(2I)+2i+1`。这两种布局都合法，但 tile、stride、down GEMM 的读取方式完全不同；不能把交错布局的 `g0,u0` 当成左右半布局的连续前半段。

完整 MLP 的 fusion 还会碰到 down projection 的归约边界。`Y[t,h]` 必须沿全部 `i=0..I-1` 归约。如果一个 CTA 只负责一小块 H 输出，并在现场重算 `G/U/Z`，其他负责不同 H 输出块的 CTA 可能重复计算相同的 `Z[t,i]`；如果一个 CTA 同时负责 `B_T` 个 token 的全部 `H` 个输出，就要保留 `B_T*H` 个累加值。例如 `B_T=16,H=4096` 时，仅 FP32 输出累加值的逻辑大小就是 256 KiB，还没有算 gate/up、地址和临时量。这个数不是编译器实际寄存器报告，但足以提示继续检查资源分配和 spill。三个 GEMM 写进一个 kernel 并不会免费消灭中间张量：必须在“重复计算”与“物化 Z”之间做资源和带宽预算。

排除 `X`、权重和最终 `Y`，只看同一元素字节数 `s` 的纯 forward 中间激活流量，可以分三层记账：

| 组织方式 | 中间激活逻辑读写 | 流量 |
|---|---|---:|
| 分离 projection、SiLU、mul、down | 写 G/U `2n` + SiLU 读写 `2n` + mul 读 S/U 写 Z `3n` + down 读 Z `n` | `8ns` |
| 给定 G/U 的 fused SwiGLU，再 down | 写 G/U `2n` + fused 读 G/U 写 Z `3n` + down 读 Z `n` | `6ns` |
| projection 成对产出 Z 的专门 epilogue，再 down | 写 Z `n` + down 读 Z `n` | `2ns` |

最后一行只有在上投影 producer 真正配对 G/U、执行 activation/mul 并把 Z 交给 down producer 时成立；它不是把两个权重拼成 `[H,2I]` 就自动得到的结果。前面 `5ns→3ns` 的账本只讨论“已经给定 G/U 的 pointwise 链”，不能与这一段完整 MLP 中间激活流量混口径。

存在几种不同的实现粒度：

1. 两次上投影独立 GEMM，单独启动 SwiGLU，再启动 down GEMM。这最容易验证，Z 会完整物化。
2. GEMM 的 epilogue 中加 bias 或做局部 activation，但仍把 G/U 或 Z 写回，省掉一部分中间读。
3. packed 上投影把 gate/up 的权重或输出放在同一布局，再做 pointwise；packed 需要严格区分 `[T,2I]` 的行边界，不能沿用 LeetGPU #54 的全局一维切法。
4. grouped GEMM 用于不同专家或不同 token group；形状和调度由 group 变化，不应把 dense MLP 的固定 `T,H,I` 假设直接搬过去。

更大的 fusion 会延长 `G`、`U`、累加器和输出 tile 的 live range。寄存器不够时可能 spill 到 local memory；local 是线程私有的地址空间，不等于低延迟的物理片上存储。增加 `BLOCK` 或同时保留两路上投影，可能减少 launch 和物化，却降低 occupancy 或产生更多 local traffic。重算一部分 SiLU、牺牲一次 load、增加 CTA 数量，都必须通过 profiler 和多 shape 结果决定，不能从单个 shape 的最好数字推广。

小 batch decode 的 `T` 可能很小，launch latency 和低占用更显著；大 prefill 的 `T` 足以填满 SM，GEMM 复用、Tensor Core 路径和全局流量更重要。同一个 fused kernel 不能只凭 `n` 变小或变大就假定收益方向。至少要把 `T` 小/大、`I` 可整除/不可整除、FP32 与低精度 correctness 分开；FP16/BF16 的性能表还要声明中间是否 cast/store，因为它们的 bytes 和误差语义不同。

## 8. QK Norm 与 RoPE：布局、顺序和可交换性

Attention 前的 QK Norm 不是一个模糊的“把向量归一化”。本章选用 head-dim RMSNorm 作为明确的教学实现：对每个 token、每个 head 的最后一维 `D` 做

$$
\operatorname{RMSNorm}(x)_i=
\gamma_i\frac{x_i}{\sqrt{\frac1D\sum_{j=0}^{D-1}x_j^2+\epsilon}},
\quad i=0,\ldots,D-1.
$$

`gamma` 是 per-dim 可学习缩放，`epsilon` 必须记录数值合同；实现顺序是先算一个 head 的平方和，再按 `head_dim` 广播缩放。无 gamma 是另一个合同。L2 归一化不能与 RMSNorm 混写：常见 `F.normalize` 语义是 `x / max(norm(x), eps)`，这里的 `eps` 不是 `sqrt(sum(x²)+eps)`。LayerNorm 还要减均值并使用方差：

$$
\operatorname{LN}(x)_i=\gamma_i\frac{x_i-\mu}{\sqrt{\frac1D\sum_j(x_j-\mu)^2+\epsilon}}+\beta_i.
$$

原始 QK Norm 论文使用按 head-dim 的 L2 归一和可学习 score scale；本章选的 RMS 版本是另一种具体实现，不能把两者的公式或论文结论互换。现代模型中的 RMS 型实现仍要以对应模型代码为准，例如 Qwen3 modeling_qwen3.py。

### 8.1 split-half RoPE 的合同

RoPE 的输入 cos/sin 表由上游提供；#61 的题面已经给出表，kernel 不应擅自按照自己的频率重算。split-half 的 `rotate_half` 对 `D=8` 是

~~~text
x = [x0,x1,x2,x3 | x4,x5,x6,x7]
rotate_half(x) = [-x4,-x5,-x6,-x7 | x0,x1,x2,x3]
out = x * cos[position] + rotate_half(x) * sin[position]
~~~

对一个二维旋转 pair，真实模型的角度可写成

$$
\begin{bmatrix}x'_j\\x'_{j+D_{rot}/2}\end{bmatrix}
=
\begin{bmatrix}\cos\phi_{p,j}&-\sin\phi_{p,j}\\
\sin\phi_{p,j}&\cos\phi_{p,j}\end{bmatrix}
\begin{bmatrix}x_j\\x_{j+D_{rot}/2}\end{bmatrix},
\qquad
\theta_j=\operatorname{base}^{-2j/D_{rot}},
\qquad
\phi_{p,j}=p\theta_j.
$$

这是说明真实模型如何生成三角表的公式；实际模型仍应读取与 checkpoint/输入布局匹配的 cos/sin 表。#61 的 cos/sin 已经作为输入给出，不能在题面 kernel 内擅自用一个 `base` 重算，也不能把任意随机系数当成满足正交关系的三角表。

CPU reference 的低层“使用已取表行”和上层“按绝对 position 选表”如下，二者不能混成一个未使用 `position` 的参数：

<!-- source-check: examples/rope_qknorm_reference.py -->
~~~python
def rope_from_table_row(x: list[float], cos_row: list[float], sin_row: list[float], drot: int | None = None) -> list[float]:
    """Low-level rotation using one already selected cos/sin row."""
    if drot is None:
        drot = len(x)
    if drot <= 0 or drot > len(x) or drot % 2 or len(cos_row) < drot or len(sin_row) < drot:
        raise ValueError("drot must be a positive even prefix covered by cos/sin")
    prefix = x[:drot]
    rotated_prefix = rotate_half_split(prefix)
    rotated = [prefix[i] * cos_row[i] + rotated_prefix[i] * sin_row[i] for i in range(drot)]
    return rotated + x[drot:]
~~~

<!-- source-check: examples/rope_qknorm_reference.py -->
~~~python
def rope_split_half(x: list[float], cos_table: list[list[float]], sin_table: list[list[float]], position: int, drot: int | None = None) -> list[float]:
    """Select the absolute table row, including any prefix tokens."""
    if position < 0 or position >= len(cos_table) or position >= len(sin_table):
        raise IndexError("absolute position outside cos/sin table")
    return rope_from_table_row(x, cos_table[position], sin_table[position], drot)
~~~

这和 adjacent `(2j,2j+1)` 配对不是一回事；两种布局一旦混用，数值仍可能是有限数，却不是同一个算子。`Drot` 必须是偶数，只旋转 prefix `[0,Drot)`，suffix 原样保留。绝对 position 包括前缀 token；低层函数只接已经选好的 `cos_row/sin_row`，外层 wrapper 才按绝对 position 索引表。若只给 `drot=4` 的表，它必须是对应四维 prefix 的两半布局，不能把完整八维表机械截前四项。

真实三角表每一对共享同一个角度，旋转矩阵才是正交的；#61 的随机测试表虽然重复两半，却不保证 `cos²+sin²=1`。因此“rotation 保范数”只能对真实正交表做性质测试，不能对任意题面输入强行断言。`rope_qknorm_reference.py` 同时检查非零 position、prefix suffix、四维手算和布局往返；position 0 是单位矩阵，不能作为唯一的布局测试。

### 8.2 RMS、RoPE 与 kernel 顺序

标量 RMS 缩放与正交旋转在数学上可交换；无 gamma 或每个旋转 pair 共享同一个 gamma，也可在真实正交 RoPE 下交换。per-dim gamma 一般不可交换，因为旋转会混合一对坐标，而不同 gamma 改变了这两个坐标的相对比例。非零 position 的反例是必要的：在 position 0，旋转是单位矩阵，错误顺序会被掩盖。LayerNorm 更不能默认交换：减均值和逐维 affine 不是旋转不变量，应直接比较 `LN(R(x))` 与 `R(LN(x))`。

融合到一个 kernel 只改变 launch 和中间读写，不改变模型的语义顺序。模型若是 `QKNorm → RoPE`，就先在寄存器中完成 RMS，再用对应绝对位置表旋转；若模型是 `RoPE → QKNorm`，就保留反过来的顺序。K cache 的具体副作用是 `K_norm → RoPE(position) → cache[position]`，后续读取已经是旋转后的表示，不能再次 RoPE；CPU reference 检查原始 K 不变并构造 double-RoPE 反例。Residual Add + RMSNorm 是另一个合同：#83 只写 `out`，不改输入 residual，见本章前面的题面说明。

QK Norm 的关键实现也直接保留在 reference，而非只给文件名：

<!-- source-check: examples/rope_qknorm_reference.py -->
~~~python
def rms_norm(x: list[float], gamma: list[float] | None = None, eps: float = 1e-6) -> list[float]:
    if gamma is not None and len(gamma) != len(x):
        raise ValueError("gamma must be per head-dimension")
    scale = 1.0 / math.sqrt(sum(value * value for value in x) / len(x) + eps)
    return [value * scale * (gamma[i] if gamma is not None else 1.0) for i, value in enumerate(x)]
~~~

### 8.3 题面差异：不要用一个 shape 冒充三个算子

- [LeetGPU #61 RoPE Embedding](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/61_rope_embedding) 的合同是 `Q/cos/sin/output[M,D]`，采用 split-half `rotate_half=[-x2,x1]`，cos/sin 已由输入给出；随机测试系数不保证三角恒等式。
- [LeetGPU #80 Grouped Query Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/80_grouped_query_attention) 是无 batch 的 `Q[Hq,S,D]`、`K/V[Hkv,S,D]`，要求 `Hq%Hkv==0`，`kvhead=qhead//(Hq/Hkv)`，FP32、noncausal。
- #6 是另一个矩形 attention 合同，不能把本章的 QK Norm/RoPE 或 `[B,H,S,D]` wrapper 改名后就声称适配。

这些题目是 free challenge。应从平台总入口按准确题名搜索并核对当前签名，先原样保存平台 `solve`/kernel，再做服务器 wrapper。CPU reference 可以验证数学和索引，但不等于 GPU 题面通过。

## 9. 旧代码、reference 和可执行题面

上一章用户 GEMM 原始实现可以用来解释 accumulator 的生命周期，但不能被写成 MLP 实现。下面是 solutions/cuda/gemm/naive_float.cu 中的原样 kernel；每个线程把一个 C 元素的 `sum` 保存在寄存器，循环结束才写回。这个事实能帮助理解 GEMM epilogue 为什么有一个“最终写回前”的插入点，但它没有 SwiGLU 或真实 MLP 的实测。

<!-- source-check: solutions/cuda/gemm/naive_float.cu -->
~~~cpp
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C,
                                             int M, int N, int K) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;  // K 维度索引
    int m = blockDim.y * blockIdx.y + threadIdx.y;  // M 维度索引
    int idx = m * K + k;

    if ((k < K) && (m < M)) {
        float sum = 0;
        for (int n = 0; n < N; n++) {
            sum += A[m * N + n] * B[n * K + k];
        }
        C[idx] = sum;
    }
}
~~~

reference header 的函数体也只作为 reference 对照，不能归因给用户写过：

<!-- source-check: reference/cuda/include/activations.cuh -->
~~~cpp
__device__ __forceinline__ float gelu(float x) {
  const float sqrt_2_over_pi = 0.7978845608028654f;
  const float coeff = 0.044715f;
  float x3 = x * x * x;
  float inner = sqrt_2_over_pi * (x + coeff * x3);
  return 0.5f * x * (1.0f + tanhf(inner));
}
~~~

上面的 `sigmoid` 直接调用 `expf(-x)`，不具备稳定改写的全部溢出边界。融合还要考虑寄存器压力：限制寄存器可能引起 spill，local memory 不代表低延迟片上存储；`--use_fast_math` 会替换部分运算，可能改变误差和特殊值行为，不能只按运行时间选编译选项。

可核对的公开题面有四个层次：

- [#52 Sigmoid Linear Unit](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/easy/52_silu)：FP32 `input[N]` 到 `output[N]`，`rtol=atol=1e-5`。
- [#54 Swish-Gated Linear Unit](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/easy/54_swiglu)：FP32 一维偶数 `input[N]`，前半 gate、后半 up，输出 `[N/2]`，`solve(input, output, N)`，`rtol=1e-5, atol=1e-4`。
- [#83 Fused Residual Add and RMS Norm](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/83_fused_residual_add_rms_norm)：先 `x+residual` 再 RMSNorm 和 weight，只输出一个 `[N,C]`，不改原 residual，`rtol=atol=1e-5`。
- [#84 SwiGLU MLP Block](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/84_swiglu_mlp_block)：FP32 `x[M,d_model]`、三组 dense 权重和 `output[M,d_model]`，`rtol=atol=1e-4`；完整 MLP 题面不等于本章 pointwise kernel 已覆盖 GEMM。

题面均为 free challenge。先从题面合同实现并保存原始 `solve`/kernel，再进入真实 GPU baseline；题面通过的是对应 challenge 的正确性，不自动赋予完整 `[T,H]`→`[T,I]`→`[T,H]` MLP 的性能结论。

## 模型依赖怎样限制融合：Single-Pass mHC

SwiGLU 的融合保持数学计算不变，只调整中间值的存放位置。DeepSeek-V4.1-Flash 的 Single-Pass mHC 则先改变模型的数据依赖，才能减少遍历；下面比较这两类优化的差别。

### 先找必须等待的归约

mHC（Manifold-Constrained Hyper-Connections，流形约束超连接）维护多条残差流。对一个 token，令第 l 个 block 的输入为 X_l，形状为 n×d：n 是残差流数，d 是每条流的 hidden dimension。三个混合系数分别为 A_l∈R^(1×n)、B_l∈R^(n×n)、C_l∈R^(n×1)，由依赖输入的预测函数 H 产生：

$$
(A_l,B_l,C_l)=\mathcal H(X_l),\qquad
\widehat X_l=A_lX_l,\qquad
X_{l+1}=B_lX_l+C_l\mathcal F_l(\widehat X_l).
$$

这里的 A_l 不是 checkpoint 中一组固定常量。计算它需要读取 X_l 并完成归一化、投影等操作，其中存在跨 hidden dimension 的归约。假设 hidden 被切成多个 tile：第一个 tile 的 X_l 虽然已经在寄存器里，但 A_l 还没有算完，因此不能立刻完成这个 tile 的 A_lX_l。必须保存数据或重新读取，等待全维度统计量就绪。

把几个操作写进同一个函数并不会消除这个依赖。一个 CTA（Cooperative Thread Array，协作线程数组）若能保留全部数据，可以在归约后继续消费，但会受到寄存器和 shared memory 容量限制；多个普通 CTA 之间则不能随意用 block 内 barrier 代替全局协调。实现是否值得融合，首先取决于依赖和资源，不取决于函数名是否带 fused。

### 哪部分能等价搬动，哪部分改变了模型

预测器中的固定归一化权重可以吸收到线性投影权重中。例如行向量 x、固定逐通道 gamma 和矩阵 W 满足

$$
(x\odot\gamma)W=x\bigl(\operatorname{diag}(\gamma)W\bigr).
$$

但 RMS 的分母依赖当前 x，不能离线变成权重常量。在允许的线性部分中可以把标量除法放到投影之后，仍然需要计算这个标量；若中间夹着非线性、量化或舍入，必须重新验证数值顺序。

报告随后把当前 block 使用的输入混合系数从 A_l 改为前一个 block 已产生的 A_(l-1)：

$$
X_{l+1}=B_lX_l+C_l\mathcal F_l(A_{l-1}X_l),\qquad
(A_l,B_l,C_l)=\mathcal H(X_l).
$$

现在 X_l 的一个 tile 到达后，可以同时参与输入混合和下一组系数的统计，不再等待本 block 的 A_l。代价是模型计算发生变化：通常 A_lX_l≠A_(l-1)X_l。这是报告中的架构设计，不能作为对任意旧 checkpoint 都数学等价的编译器变换。

下面的代码只构造这个依赖反例，不实现作者的 mHC 系数预测器：

<!-- source-check: examples/cpu_mhc_case.py -->
~~~python
def mix(coefficients, streams):
    if not streams or len(coefficients) != len(streams):
        raise ValueError("one coefficient per residual stream required")
    dim = len(streams[0])
    if dim == 0 or any(len(row) != dim for row in streams):
        raise ValueError("rectangular nonempty streams required")
    return [sum(a * row[d] for a, row in zip(coefficients, streams)) for d in range(dim)]
~~~

给两条流 [1,0] 与 [0,2]，旧系数 [0.5,0.5] 得到 [0.5,1]；若当前预测器产生不同系数，混合结果通常不同。配套脚本用一个输入相关的 toy predictor 验证这个反例，不能把它当作作者训练后的质量评估。

### 用元素流量核对收益，而不是把 kernel 数当速度

报告比较的是指定残差变换与输入 pre-norm 范围内的激活读写。按它的统计边界，多遍实现为 (4n+4)d 个元素，两遍实现为 (3n+2)d，Single-Pass 为 (2n+2)d。取报告的 n=4、d=5120：

| 实现 | 激活读写元素数 | 含义 |
|---|---:|---|
| 多遍 | 102,400 | 多次消费残差与归一化数据 |
| 两遍 | 71,680 | 合并部分遍历，但仍等待当前输入混合系数 |
| Single-Pass | 51,200 | 使用前一 block 系数消除这条等待依赖 |

这是报告口径下的算法流量，不是整个 Transformer block 的全部 bytes，也不是本机 DRAM 实测。只有各次读写 dtype 一致时，才能统一乘同一个 element size。权重读取、系数工作区、cast、对齐和缓存命中要另算；少一半激活流量也不等于端到端快两倍。

从仓库根目录运行数学与流量核对：

~~~bash
python roadmap/curriculum/operators/04-activation-and-fusion/examples/cpu_mhc_case.py
~~~

实际优化仍按“依赖图 → 可保留的数据 → 编译资源 → profiler”检查：预期减少的是对应激活的遍历和中间存储；如果融合后 spill 增多、occupancy 下降或 launch 已不是瓶颈，时间收益可能小于流量收益。

## 10. 实践：写题与性能对比

### LeetGPU：正确性与代码归档

先核对公开题面中的函数签名、输入输出布局和容差。从 #52 或 #54 开始时，代码必须按题面的一维合同处理；进入 #84 时再把三次 GEMM、gate/up 角色、最终 down projection 和 FP32 参考连接起来。非整除二维 shape、正负零、适度偏置、NaN/Inf 策略和 packed 行边界应在本地 CPU 语义脚本中先覆盖。通过后原样保存平台 `solve`/kernel，单独标注题面版本与本地 wrapper，不把服务器适配文件冒充平台源码。

### 服务器：真实性能

只有对应题面正确性与代码归档完成后，才在 CUDA 环境跑 `examples/validate_fusion.py --benchmark`。正确性先 `isfinite` preflight，再用一次 FP32 reference、最终 cast 和按 dtype 的 `rtol/atol` 做 `assert_close(equal_nan=False)`。benchmark 预分配输入、activation scratch 和 output；FP32 对比 Triton fused 与 `aten.silu.out + torch.mul(out=...)` 的两-kernel split，低精度只做 correctness 或另立中间 cast/bytes 口径。输出必须打印 device、Torch/Python、shape、numel、dtype、BLOCK 和真实边界；CUDA Event 包住的是 Python 发起的循环与 launch gap，不称为纯 kernel duration。纯 kernel 时长、寄存器、spill、occupancy、L2/DRAM traffic 和 launch 时间由 Nsight Systems/Compute 分开核实。

从仓库根目录执行：

~~~bash
python roadmap/curriculum/operators/04-activation-and-fusion/examples/cpu_semantics.py
python roadmap/curriculum/operators/04-activation-and-fusion/examples/rope_qknorm_reference.py
python roadmap/curriculum/operators/04-activation-and-fusion/examples/validate_fusion.py
python roadmap/curriculum/operators/04-activation-and-fusion/examples/validate_fusion.py --benchmark --tokens 4096 --hidden 11008
~~~



默认正确性检查覆盖 FP32、FP16、BF16 和非整除长度，性能对比固定为 FP32。先检查实际计时 shape 的 fused、split 和 reference 三份结果，再预热并计时。共同使用 `3*n*s` 作为 useful GB/s 的分子，便于比较完成相同有效工作的速度；同时另列 split 的 `5*n*s` 与 fused 的 `3*n*s` 算法流量，不能把这两个估算当成 DRAM counter。

## 参考阅读

[CUDA Programming Guide 13.3](../../../../downloads/cuda-programming-guide.pdf) · [DeepSeek-V4.1-Flash 技术报告](../../../../downloads/DeepSeek_V41_Tech_Report.pdf)。

## 章节导航

- [上一章：GEMM](../03-gemm/README.md)
- [算子总览](../README.md)
- [后续主题：Prefill Attention](../README.md#5-prefill-attention)
