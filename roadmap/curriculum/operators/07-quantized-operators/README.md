# 第七章 量化算子：从数值编码到低精度矩阵计算

低位权重需要与 scale、打包布局和计算路径一起设计。反量化若先生成完整 FP16 权重，会增加一次大矩阵写回与读取；在计算 tile 内解包则增加位操作和临时寄存器。

数值误差由校准和量化方法控制，运行效率由布局、指令及数据复用决定。两者使用同一份量化参数对照，才能区分算法收益和实现收益。

## 1. 数值：scale 决定一个整数代表多大的实数

### 舍入、截断与零点

对浮点值 x，仿射整数量化可写为

$$
q=\operatorname{clip}\left(\operatorname{round}(x/s)+z,\ q_{\min},q_{\max}\right),
\qquad \widehat x=s(q-z).
$$

s 是正的 scale，z 是整数 zero point（零点），q 是保存的编码，x̂ 是还原后的近似。z 使实数 0 对应某个整数编码；它不是要给所有还原值额外增加一个任意偏置。

例如 s=0.5、z=0，x=0.75 时 x/s=1.5。采用 round-to-nearest, ties-to-even（就近舍入，正好居中时取偶数），编码为 2，还原成 1.0。x=0.25 则落在 0 和 1 的中间，编码为 0。不能把这种规则换成 C/C++ 整数转换的向零截断，更不能假设所有语言的 round 都采用同一规则。

有限范围带来另一种误差：INT8 编码超出 [-128,127] 时要饱和截断。未饱和时，就近舍入的绝对误差通常不超过 s/2；发生 clipping 后，这个界不再成立。NaN、Inf、零 scale 也要有明确接口约定，下面的标量模型拒绝这些输入。

<!-- source-check: examples/semantics.py -->
~~~python
def quantize_affine(value, scale, zero_point=0, qmin=-128, qmax=127):
    if not math.isfinite(value) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("finite input and positive finite scale required")
    if any(not isinstance(x,int) for x in (zero_point,qmin,qmax)) or qmin > qmax or not qmin <= zero_point <= qmax:
        raise ValueError("invalid quantized range or zero point")
    scaled = min(qmax-zero_point, max(qmin-zero_point, value / scale))
    rounded = round(scaled) + zero_point
    return min(qmax, max(qmin, rounded))
~~~

代码先限制 scaled 的范围，再舍入；由于上下界是整数，这与公式中的最终整数饱和一致，也避免极小 scale 使浮点比值变成无穷后无法转成整数。这是标量语义模型，Python 浮点计算不是 GPU FP32 的逐位模拟。验证 GPU 时，接近舍入边界的输入必须单独测试，不能只用普通随机数掩盖一个 bit 的差异。

### 量化粒度不是一个统一的“粗到细”排列

设激活 X 为 [T,I]、权重 W 为 [I,O]，输出 Y 为 [T,O]。T 是 token 数，I 是输入维度，O 是输出维度。

| 方式 | scale 的一种常见形状 | 每个 scale 服务哪些值 |
|---|---|---|
| per-tensor | [1] | 整个张量 |
| 激活 per-token | [T,1] | 一个 token 的全部输入通道 |
| 权重 per-output-channel | [1,O] | 一个输出通道对应的输入权重 |
| 权重沿 I 分组 | [ceil(I/G),O] | 一个输出通道内连续 G 个输入权重 |
| 二维 block | [ceil(I/GI),ceil(O/GO)] | GI×GO 个权重 |

per-token 与 per-channel 是沿不同轴分组，不能简单说其中一个一定比另一个更细。transpose 后也必须重新解释 scale 的归属：元素移动了，原 scale 不能只按新行号读取。矩阵指令要求的 scale layout 还可能带额外 padding 或 swizzle，它不是数学形状本身。

scale 更密通常有利于局部误差控制，但增加 metadata、转换与寻址成本。量化一个包含极大值和小值的 block 时，极大值可能迫使 scale 变大，让小值舍入为零；clipping、分组和通道变换正是围绕这个矛盾选择。

## 2. 存储：INT4 编码与 scale 寻址

### 沿输入轴分组：先固定 q 和 scale 的形状

权重采用 W[I,O]，连续 G 个输入通道共享一个输出通道的量化步长。第 g 组、输出 o 的步长记为 $\Delta_{g,o}$：

$$
\Delta_{g,o}=\frac{\max_{gG\leq i<\min((g+1)G,I)}|W_{i,o}|}{7},
\quad
q_{i,o}=\operatorname{clip}
\left(\operatorname{round}\frac{W_{i,o}}{\Delta_{\lfloor i/G\rfloor,o}},-7,7\right).
$$

这里选择对称的 15 个编码值 [-7,7]，因此负端的 -8 不参与数值量化；存储时仍使用能表示 [-8,7] 的 signed INT4。两者分别是量化策略与编码格式。全零组使用 Δ=1、q=0，避免除零。

<!-- source-check: examples/quantization_lab.py -->
~~~python
def quantize_int4_group(w: np.ndarray, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Round-to-nearest-even symmetric INT4 per ``[input-group, output]``.

    The returned ``q`` is int8 for convenient teaching/debugging, but its
    values are restricted to [-7, 7].  A packed signed nibble still has the
    representable storage range [-8, 7]; -8 is intentionally never produced
    by this symmetric scale=amax/7 quantizer.
    """
    w = _check_weight(w)
    group_size = _check_group_size(group_size)
    groups = _group_count(w.shape[0], group_size)
    scales = np.ones((groups, w.shape[1]), dtype=np.float64)
    q = np.empty(w.shape, dtype=np.int8)
    for g in range(groups):
        start, end = g * group_size, min((g + 1) * group_size, w.shape[0])
        amax = np.max(np.abs(w[start:end]), axis=0)
        # A finite subnormal can underflow during division.  Keep a positive
        # representable scale; an exactly zero channel still uses identity.
        tiny = np.finfo(np.float64).tiny
        scales[g] = np.where(amax == 0.0, 1.0, np.maximum(amax / _QMAX, tiny))
        # np.rint is ties-to-even, and clipping is explicit rather than an
        # accidental consequence of casting to int8.
        q[start:end] = np.clip(
            np.rint(w[start:end] / scales[g][None, :]), _QMIN, _QMAX
        ).astype(np.int8)
    return q, scales
~~~

例如 I=5、O=2、G=2，q 的形状仍为 [5,2]，scale 为 [3,2]。最后一组只包含输入通道 4，统计最大值时不补入别的行。函数返回 int8 数组只是方便调试，下一步才把这些 4-bit 编码打包。

### 一个字节放两个值

本节约定 signed INT4 使用二进制补码，范围 [-8,7]，先放的元素占低四位。这个约定必须明确；“都是 INT4”不表示不同实现的打包顺序一致。

<!-- source-check: examples/semantics.py -->
~~~python
def pack_int4(values):
    if any(not isinstance(x, int) or not -8 <= x <= 7 for x in values):
        raise ValueError("signed INT4 values must lie in [-8,7]")
    packed = []
    for i in range(0, len(values), 2):
        low = values[i] & 15
        high = (values[i + 1] & 15) if i + 1 < len(values) else 0
        packed.append(low | (high << 4))
    return packed
~~~

[-1,2,-8] 打包得到 [0x2F,0x08]：-1 的低四位是 1111，2 放在高四位；最后一个 -8 的编码是 1000，未使用的高四位补零。读取第 i 个逻辑元素时：

$$
b_i=\mathrm{packed}[\lfloor i/2\rfloor],\quad
u_i=(b_i\gg(4(i\bmod2)))\ \&\ 15,\quad
q_i=\begin{cases}u_i-16&u_i\geq8\\u_i&u_i<8.\end{cases}
$$

解包后的 15 必须还原成 -1，不能直接当作正数 15。逻辑长度为奇数时，最后半字节不属于输入。若按行独立打包，每行字节数是 ceil(I/2)；不能把上一行的尾部 padding 当成下一行的第一个元素。

CUDA 中可用 uint8_t 或更宽的整数载入 packed 数据，再移位、掩码、符号扩展。更宽的 load 要满足对齐，并有独立的尾部处理。内存读得更宽不改变逻辑编码，也不能越过 allocation 边界“多读一点”。

### 量化 block 与执行 block 是两件事

考虑 Y[r,c]=X[r,c]×S[r//G,c//G]。G×G 是共享 scale 的逻辑块，GPU 的 BLOCK 则决定一个 program 处理多少元素，两者没有必须相等的关系。

下面沿输出的线性地址分工。相邻 lane 读取相邻 X，scale 的位置再由行列坐标计算：

<!-- source-check: examples/quant_gpu.py -->
~~~python
@triton.jit
def block_scale_kernel(X, S, Y, M: tl.constexpr, N: tl.constexpr,
                       TILE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = i < M * N
    row, col = i // N, i % N
    scale_cols = tl.cdiv(N, TILE)
    scale_offset = (row // TILE) * scale_cols + col // TILE
    x = tl.load(X + i, mask=valid, other=0.0)
    scale = tl.load(S + scale_offset, mask=valid, other=0.0)
    tl.store(Y + i, x * scale, mask=valid)
~~~

例如 X 是 3×3、G=2，scale 是 2×2。X[2,1] 对应 S[1,0]，X[1,2] 对应 S[0,1]。执行 tile 即使跨过矩阵行边界，也必须分别计算每个元素的 scale，不能整段复用起始元素的 scale。

该 kernel 的 X、S、Y 都是 FP32，只演示 block scale 寻址，并没有解包 INT4。基础题允许一般 FP32 的 S，包括负数；这不改变真实量化通常要求 scale 为正的约定。真实反量化还要在相乘之前加入正确的编码转换。

## 3. 从反量化接到 GEMM

### W8A8：整数点积之后还要恢复尺度

W8A8 表示权重与激活使用 8-bit 编码，不独自规定有无 zero point、scale 粒度或累加方式。若一个点积范围内 s_a、s_w、z_a、z_w 都固定：

$$
y\approx s_as_w\sum_{i=0}^{I-1}(q_{a,i}-z_a)(q_{w,i}-z_w).
$$

展开整数部分：

$$
\sum_iq_{a,i}q_{w,i}
-z_w\sum_iq_{a,i}
-z_a\sum_iq_{w,i}
+I z_a z_w.
$$

因此带 zero point 的计算不是“直接 INT8 dot 再乘两个 scale”。交叉项可以融合或预计算，但不能遗漏。激活 per-token、权重 per-output-channel 且归约轴上 scale 不变时，输出元素乘 s_a[t]s_w[o] 即可；如果 scale 沿 I 分组变化，则必须先对各组 partial 恢复尺度，再合并，不能用一个最终 scale 替代全部分组。

INT8 乘法常用 INT32 累加，仍要检查溢出。带零点的差值可能达到 255；粗略最坏界是 I×255²，超过 INT32 范围时不能继续依赖无限精度的数学公式。输出再量化还涉及除输出 scale、舍入、加输出 zero point 和饱和，每一步都可能影响结果。

### W4A16：少读权重与反量化成本的交换

W4A16 常见路径是读 packed 权重和 scale，在片上解包为计算可用的形式，再与 FP16/BF16 激活计算。**存储位宽、乘法输入格式和累加格式要分别说明。** 文件中是 4 bit，不足以证明机器执行了原生 INT4 MMA。

若把整个权重先写成一个 FP16 中间矩阵，再交给普通 GEMM，至少新增一次大矩阵写回与读取。融合反量化尝试让一个权重 tile 解包后直接被矩阵计算消费，减少物化，但会增加位操作、格式转换和寄存器活跃值；权重 tile 被重复载入多少次，也会重复支付这些成本。

先保持相同量化权重和 scale，比较“独立反量化+GEMM”与“tile 内反量化+GEMM”，才能分辨融合收益。拿它与原始高精度权重比较时，还必须同时报告模型误差，不能只看耗时。

### packed 矩阵怎样参与乘法

为了让同一输出通道的输入权重相邻，这里按 packed[O,ceil(I/2)] 保存。逻辑权重 q[i,o] 的字节偏移为：

$$
p=o\left\lceil I/2\right\rceil+\left\lfloor i/2\right\rfloor,
\qquad
h=4(i\bmod2),\qquad
g=\lfloor i/G\rfloor.
$$

p 决定读哪个字节，h 决定取低四位还是高四位，g 决定读哪个 scale。G 为奇数时，同一个字节里的两个权重可能属于不同量化组。

例如 q[:,o]=[-1,2,-7,7,3]、G=3，打包为 [0x2F,0x79,0x03]。0x79 的低四位是 i=2、属于组 0；高四位是 i=3、属于组 1。如果 $\Delta_{0,o}$=0.5、$\Delta_{1,o}$=2，这两个值分别恢复成 -3.5 与 14，而不是共用同一个步长。

<!-- source-check: examples/quantization_lab.py -->
~~~python
def pack_signed_int4(q: np.ndarray) -> np.ndarray:
    """Pack q[I,O] as packed[O, ceil(I/2)], low input nibble first."""
    q = _validate_storage_q(q)
    input_size, output_size = q.shape
    packed = np.zeros((output_size, (input_size + 1) // 2), dtype=np.uint8)
    for i in range(input_size):
        nibble = (q[i].astype(np.int16) & 0xF).astype(np.uint8)
        if i % 2 == 0:
            packed[:, i // 2] |= nibble
        else:
            packed[:, i // 2] |= nibble << 4
    return packed
~~~

按组写出矩阵计算：

$$
Y_{t,o}=\sum_g
\Delta_{g,o}\sum_{i=gG}^{\min((g+1)G,I)-1}X_{t,i}q_{i,o}.
$$

这解释了为什么 scale 沿归约轴变化时，需要在组内恢复尺度，再合并各组。下面直接读取 packed 字节，每次解码一个输入通道，避免先创建完整的浮点权重矩阵：

<!-- source-check: examples/quantization_lab.py -->
~~~python
def packed_int4_matmul(x: np.ndarray, packed: np.ndarray, scales: np.ndarray, group_size: int) -> np.ndarray:
    """Compute X @ dequant(W) from packed storage without materializing W.

    The intentionally plain loops expose scale addressing at group boundaries:
    each input column contributes to one group's scale row and one output
    channel.  Only a single input-column q vector is decoded at a time.
    """
    x = _as_float_array("X", x, 2)
    packed = np.asarray(packed)
    if packed.ndim != 2 or packed.dtype != np.uint8:
        raise ValueError("packed must be a 2-D uint8 array")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("X must be non-empty")
    input_size = x.shape[1]
    output_size = packed.shape[0]
    expected_bytes = (input_size + 1) // 2
    if packed.shape[1] != expected_bytes:
        raise ValueError("packed byte count does not match X input size")
    scales = _as_float_array("scales", scales, 2)
    group_size = _check_group_size(group_size)
    expected_scales = (_group_count(input_size, group_size), output_size)
    if scales.shape != expected_scales:
        raise ValueError(f"scales must have shape {expected_scales}, got {scales.shape}")
    if np.any(scales <= 0):
        raise ValueError("scales must be positive")
    output = np.zeros((x.shape[0], output_size), dtype=np.float64)
    for group in range(expected_scales[0]):
        start, end = group * group_size, min((group + 1) * group_size, input_size)
        for i in range(start, end):
            nibble = packed[:, i // 2] & (0xF if i % 2 == 0 else 0xF0)
            if i % 2:
                nibble = nibble >> 4
            q_i = _sign_extend_nibble(nibble).astype(np.float64)
            output += x[:, i, None] * (q_i[None, :] * scales[group][None, :])
    return output
~~~

CPU 循环用于核对地址和算术；GPU 实现会把若干输入通道组成 tile。量化分组 G、执行归约块和打包字节宽度各自独立，跨组、跨字节和最后一个不完整块都要覆盖。

### Triton：在计算 tile 内解包

下面的内核读取 FP16 的 X[T,I]、uint8 的 packed[O,ceil(I/2)] 和 FP32 的 scales[ceil(I/G),O]，输出 FP16，累加器为 FP32。每个 program 计算 16×32 个输出，归约块为 32。

<!-- source-check: examples/packed_int4_gpu.py -->
~~~python
if triton is not None:

    @triton.jit
    def plain_kernel(
        x_ptr,
        packed_ptr,
        scales_ptr,
        c_ptr,
        t_size,
        i_size,
        o_size,
        group_size,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """One output tile, direct packed-byte addressing, no dequant buffer."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        # Keep address arithmetic wide: a large flattened packed/scales
        # tensor must not overflow a 32-bit intermediate before pointer add.
        rows = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        cols = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        row_mask = rows < t_size
        col_mask = cols < o_size
        packed_bytes = (i_size + 1) // 2
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # The fixed reduction tile is 32.  Looping over K makes both I tails
        # and arbitrary I/G boundaries explicit in the same baseline.
        for k_block in range(0, tl.cdiv(i_size, BLOCK_K)):
            inputs = (k_block * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)
            input_mask = inputs < i_size
            x_mask = row_mask[:, None] & input_mask[None, :]
            x = tl.load(
                x_ptr + rows[:, None] * i_size + inputs[None, :],
                mask=x_mask,
                other=0.0,
            )

            byte_index = inputs // 2
            packed_mask = col_mask[:, None] & input_mask[None, :]
            packed_byte = tl.load(
                packed_ptr + cols[:, None] * packed_bytes + byte_index[None, :],
                mask=packed_mask,
                other=0,
            )
            shift = ((inputs % 2) * 4).to(tl.int32)
            nibble = (packed_byte >> shift[None, :]) & 0xF
            nibble_i32 = nibble.to(tl.int32)
            signed_q = tl.where(nibble_i32 >= 8, nibble_i32 - 16, nibble_i32)

            scale_index = (inputs // group_size) * o_size + cols[:, None]
            scale = tl.load(scales_ptr + scale_index, mask=packed_mask, other=0.0)
            # Match the reference exactly: q*FP32 scale, cast to FP16, then
            # FP16 x FP16 dot with FP32 accumulation and final FP16 store.
            w_fp16 = (signed_q.to(tl.float32) * scale).to(tl.float16)
            acc += tl.dot(x, tl.trans(w_fp16), out_dtype=tl.float32)

        tl.store(c_ptr + rows[:, None] * o_size + cols[None, :], acc.to(tl.float16), mask=row_mask[:, None] & col_mask[None, :])

else:
    plain_kernel = None
~~~

packed_byte 的逻辑形状是 [BLOCK_N,BLOCK_K]，第一维是输出通道，第二维是输入通道。一个字节经过 shift、mask 和符号扩展得到整数 q。scale_index 再按每个输入通道计算所属量化组，所以 G 不必等于 BLOCK_K。

反量化先以 FP32 计算 q×scale，再转成 FP16；tl.dot 消费 FP16 操作数，保留 FP32 累加。最后 store 转成 FP16。这个顺序也是参考实现的顺序，比较时应将输入量化误差、反量化转换误差和最终输出舍入分开。

tl.trans 调整的是当前权重 tile 的逻辑朝向，让 [T,I] 与 [I,O] 相乘；没有先把完整浮点权重写回 global memory。具体寄存器、shared 或线程间重排仍由编译器决定，可以在读生成代码时检查。

相邻输入通道共用字节，跨 group 的两个半字节仍各取自己的 scale。行、列和归约尾部都由 mask 覆盖，地址乘法使用宽整数。wrapper 检查形状、设备、连续布局及 scale，再在输入张量所属设备上启动。

### FP8、FP4 与微缩放

浮点低位格式不是固定间隔的整数网格。FP8 的指数位和尾数位分配影响范围与分辨率；E4M3 与 E5M2 不能只按“都是 8 bit”互换。FP4 E2M1 的正数可表示集合很稀疏，例如 0、0.5、1、1.5、2、3、4、6，scale 负责覆盖所需幅度。

以每 16 个 FP4 值带一个 8-bit scale 为例，16 个值占 8 字节，scale 占 1 字节，平均为 4.5 bit/值。128 个 INT4 权重配一个 FP16 scale 则是 64+2=66 字节，不是裸 64 字节。还有 per-tensor scale、对齐与分片时，要继续计入。

MXFP4 与 NVFP4 的 scale 粒度和表示不同；具体矩阵指令支持还取决于架构及库版本。DeepSeek-V4.1-Flash 的 main KV 格式采用 E2M1 与 16-channel E4M3 scale，并省略完整 NVFP4 的第二层 global scale；它反量化后执行 Attention，不能据此推断使用了原生 FP4 Attention 矩阵指令。

## 4. 校准、缩放搜索与误差补偿

RTN（Round-to-Nearest，就近舍入）直接将权重投到量化网格；更复杂的方法利用激活分布，改变网格或补偿舍入带来的输出误差。先固定一组 X[T,I]、W[I,O] 和 group size，才能比较这些选择。

PTQ（Post-Training Quantization，训练后量化）在模型训练完成后确定量化参数；QAT（Quantization-Aware Training，量化感知训练）则让参数优化过程感知量化误差。两者的区别在训练过程，最终仍要落到明确的编码、scale、布局与计算路径。下面的 RTN、缩放搜索和 GPTQ 属于训练后处理。

### 校准：收集的是哪一层的输入

线性层的校准数据是进入该层的激活，不是原始 token ID。多条序列可以展平成 token 行，但 padding 和无效位置应从统计中排除。示例采用 token_mask 中 1 表示有效、padding_mask 中 1 表示填充；两者同时给出时取“有效且非填充”的行。

对有效 token 集合 V，常见统计量包括：

$$
a_i^{\mathrm{mean}}=\frac1{|V|}\sum_{t\in V}|X_{t,i}|,
\qquad
a_i^{\max}=\max_{t\in V}|X_{t,i}|,
\qquad
G_X=X_V^TX_V.
$$

平均绝对值、最大绝对值与二阶矩回答不同问题。一个少见的大值可能显著影响最大值，却只轻微改变平均值；相关通道的信息则保存在 Gram 矩阵的非对角元素中。

<!-- source-check: examples/quantization_lab.py -->
~~~python
def calibration_statistics(
    x: np.ndarray,
    token_mask: Optional[np.ndarray] = None,
    padding_mask: Optional[np.ndarray] = None,
) -> CalibrationStats:
    """Compute mean-absolute, max-absolute, and second-order Gram statistics.

    ``token_mask`` is 1=valid, while ``padding_mask`` is 1=padding and is
    excluded.  The Gram matrix is the unnormalised second-order sum, so
    callers can choose its normalization.
    """
    x = _as_float_array("X", x, 2)
    mask = _valid_rows(x, token_mask, padding_mask)
    valid = x[mask]
    try:
        with np.errstate(over="raise", invalid="raise"):
            gram = valid.T @ valid
            meanabs = np.mean(np.abs(valid), axis=0)
            maxabs = np.max(np.abs(valid), axis=0)
    except FloatingPointError as exc:
        raise ValueError("calibration statistics overflowed; reduce input magnitude") from exc
    if not (np.all(np.isfinite(meanabs)) and np.all(np.isfinite(maxabs)) and np.all(np.isfinite(gram))):
        raise ValueError("calibration statistics are non-finite after reduction")
    return CalibrationStats(
        meanabs=meanabs,
        maxabs=maxabs,
        gram=gram,
        valid_tokens=int(valid.shape[0]),
    )
~~~

如果校准集主要是短文本，部署却包含长代码或多轮工具结果，离群通道与激活相关性可能变化。选择 scale、clipping 或参数搜索时只使用校准集，另留一组输入评价泛化误差。把评估集用于挑最好的 scale，会使结果偏乐观。

流式收集统计时，最大值逐批取 max，绝对值和与 Gram 逐批相加，最后按有效 token 总数归一化。不同长度批次的均值不能不加权地平均。浮点统计也可能溢出，代码检查派生结果的有限性。

### SmoothQuant：把激活幅度迁移到权重

用正对角矩阵 $S=\operatorname{diag}(s_i)$ 做通道变换：

$$
XW=(XS^{-1})(SW).
$$

$s_i$ 是输入通道变换比例，$\Delta_{g,o}$ 是量化编码步长；它们属于不同数组。改变 $s_i$ 后，还需要对新的权重重新计算 Δ。

SmoothQuant 根据激活与权重最大值构造比例：

$$
s_i=\frac{(a_i^{\max})^\alpha}
{(\max_o|W_{i,o}|)^{1-\alpha}},
\qquad 0\leq\alpha\leq1.
$$

较大的 $s_i$ 缩小激活第 i 列，同时放大权重第 i 行。激活更容易使用共享量化尺度，代价转移到相应权重。零通道需要有限的尺度约定，具体下限也会影响数值。

<!-- source-check: examples/quantization_lab.py -->
~~~python
def smoothquant_scales(x: np.ndarray, w: np.ndarray, alpha: float) -> np.ndarray:
    """SmoothQuant's separate amax formula, not the AWQ candidate family."""
    x, w = _check_x_w(x, w)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    x_amax = np.max(np.abs(x), axis=0)
    w_amax = np.max(np.abs(w), axis=1)
    # A channel that is zero on both sides has no scale information; identity
    # is the stable convention.  The tiny floor avoids zero or infinity when
    # only one side is zero.
    numerator = np.maximum(x_amax, 1e-12) ** alpha
    denominator = np.maximum(w_amax, 1e-12) ** (1.0 - alpha)
    scales = numerator / denominator
    scales = np.where((x_amax == 0) & (w_amax == 0), 1.0, scales)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("SmoothQuant derived scales are non-finite or non-positive")
    return scales
~~~

变换本身保持浮点函数不变，舍入后的结果则取决于新的网格。下面用激活 per-token、权重 per-output-channel 的对称 INT8 参考路径对照，编码范围取 [-127,127]；CPU 用 int64 保存整数和，真实 kernel 还需核对 INT32 的范围：

<!-- source-check: examples/w8a8_reference.py -->
~~~python
def w8a8_reference(x, w):
    x, w = np.asarray(x, dtype=np.float64), np.asarray(w, dtype=np.float64)
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[0] or min(*x.shape, *w.shape) <= 0:
        raise ValueError("nonempty X[T,I] and W[I,O] required")
    if not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError("finite inputs required")
    sx = np.max(np.abs(x), axis=1, keepdims=True)
    sw = np.max(np.abs(w), axis=0, keepdims=True)
    sx = np.where(sx == 0, 1., np.maximum(sx / 127, np.finfo(np.float64).tiny))
    sw = np.where(sw == 0, 1., np.maximum(sw / 127, np.finfo(np.float64).tiny))
    qx = np.clip(np.rint(x / sx), -127, 127).astype(np.int8)
    qw = np.clip(np.rint(w / sw), -127, 127).astype(np.int8)
    # int64 is the CPU mathematical reference, not a claim about GPU instructions.
    acc = qx.astype(np.int64) @ qw.astype(np.int64)
    y = acc.astype(np.float64) * sx * sw
    return y, qx, qw, sx, sw
~~~

示例先使用一组具有激活离群通道的固定输入，再执行 α=0.5 的通道变换。同一组输入上，W8A8 的输出 MSE 从约 0.394939 降到 0.000173；这是一个数值例子，不是模型质量或加速结论。更换数据、分组和尺度规则后，结果需要重新比较。

### 缩放如何折叠进模型

若前一层输出为 Norm(x)⊙γ+β，可以把 γ、β 除以 s，再把后续线性层的对应输入通道权重乘以 s。前向就不必额外物化 X/s。

如果同一归一化输出同时进入 Q、K、V 三个投影，三个消费者都要使用一致的变换。只改其中一条分支会改变模型。中间若有非线性，也要重新推导，例如一般有 SiLU(x/s)≠SiLU(x)/s。

保存量化模型时，需要同时记录变换后的参数、scale 的轴和格式。加载端若只读 packed 权重，却遗漏相应的输入变换，单独解包测试仍可能通过，整层输出却会错误。

作者的 RMSNorm 配套实现使用两条更新：

~~~python
ln.weight.div_(scales)
fc.weight.mul_(scales.view(1, -1))
~~~

这里 fc.weight 是 PyTorch 的 [O,I] 存储，所以按列缩放；对应本文 W[I,O] 就是按行缩放。前一层的归一化输出同时缩小，二者共同保持浮点函数。共享输入的所有投影都要应用相同的比例。

### AWQ：用校准输出误差选择缩放

AWQ（Activation-aware Weight Quantization，激活感知权重量化）关注低位权重对输出的影响。它利用激活统计搜索通道缩放，而不是简单地按权重绝对值保护几个元素。

这里采用作者实现中的一族候选：令 $a_i$ 为平均绝对激活，取若干 $r$ 值，构造 $s_i\propto a_i^r$，再用一个公共因子规范化尺度范围。对每个候选，量化 SW，再将尺度还原到原始函数中，比较：

$$
E(S)=\frac1{TO}
\left\|X S^{-1}Q(SW)-XW\right\|_F^2.
$$

Q 在这里表示“量化再反量化”的浮点权重，不是裸整数 q。相同候选比较使用相同位宽、分组和舍入规则。

<!-- source-check: examples/quantization_lab.py -->
~~~python
def _awq_scale_candidates(activation_meanabs: np.ndarray) -> list[tuple[Optional[float], np.ndarray]]:
    """AWQ-style teaching subset: explicit identity plus the 20 ratio scales."""
    a = np.asarray(activation_meanabs, dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("activation_meanabs must be a finite non-negative vector")
    candidates: list[tuple[Optional[float], np.ndarray]] = [(None, np.ones_like(a))]
    for ratio in np.linspace(0.0, 0.95, 20):
        powered = np.ones_like(a) if ratio == 0.0 else np.power(a, ratio)
        powered = np.maximum(powered, 1e-4)
        # sqrt-before-multiply avoids an otherwise easy overflow in max*min.
        geometric = math.sqrt(float(np.max(powered))) * math.sqrt(float(np.min(powered)))
        candidates.append((float(ratio), powered / geometric))
    return candidates
~~~

作者搜索代码中的尺度构造为：

~~~python
scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
scales = scales / (scales.max() * scales.min()).sqrt()
~~~

变量名 x_max 容易误导：这个函数中它来自平均绝对激活统计。第二行用公共因子控制尺度范围；输入通道之间的相对比例仍保留。候选误差通过实际模块输出评价，线性层示例只复现其中一条清楚的计算路径。

加入显式 s=1 候选，搜索结果在这批校准输入上的误差就不会比同规则 RTN 更高。这个保证来自候选集合包含基线，不是对新数据质量的保证。

<!-- source-check: examples/quantization_lab.py -->
~~~python
def awq_style_search(
    x: np.ndarray,
    w: np.ndarray,
    group_size: int,
    token_mask: Optional[np.ndarray] = None,
    padding_mask: Optional[np.ndarray] = None,
) -> AWQSearchResult:
    """Select an AWQ-style activation-aware scale using calibration output MSE.

    This is only a linear-layer, symmetric-group-quantization teaching subset:
    it does not claim the complete AWQ algorithm (which can also involve
    module outputs, architecture transformations, clipping, and other model
    details).  There is intentionally no held-out input to leak into the
    selection.  The explicit identity candidate makes the selected calibration
    MSE no worse than this function's RTN baseline by construction.
    """
    x, w = _check_x_w(x, w)
    mask = _valid_rows(x, token_mask, padding_mask)
    x_cal = x[mask]
    y_cal = x_cal @ w
    stats = calibration_statistics(x, token_mask, padding_mask)
    candidates = _awq_scale_candidates(stats.meanabs)
    errors: list[float] = []
    for _, scale in candidates:
        q, q_scales = quantize_int4_group(w * scale[:, None], group_size)
        w_q = dequantize_int4_group(q, q_scales, group_size) / scale[:, None]
        errors.append(float(np.mean((x_cal @ w_q - y_cal) ** 2)))
    best_index = int(np.argmin(errors))
    ratio, scale = candidates[best_index]
    return AWQSearchResult(
        scale=scale.copy(),
        ratio=ratio,
        baseline_mse=errors[0],
        best_mse=errors[best_index],
        candidate_mse=tuple(errors),
        calibration_output=y_cal,
    )
~~~

这段代码展示线性层、对称分组量化的缩放搜索。完整 AWQ 实现还会考虑模型模块的输出、图变换和 clipping。clipping 可以提高大多数小值的分辨率，也会增加被截断值的误差；阈值应通过相同校准目标选择，不能只看权重直方图。

SmoothQuant 和这里的 AWQ 搜索使用不同统计与目标。前者的最大值比例用于平衡激活和权重的量化难度；后者以激活感知的候选缩放比较量化后的输出误差。

### GPTQ：把一次舍入的影响分配给剩余权重

对一个输出通道，令 w 为 I 维权重，δ=ŵ−w。用校准输入定义输出误差：

$$
\mathcal L(\delta)=\frac1T\|X\delta\|_2^2
=\frac12\delta^TH\delta,\qquad
H=\frac2T X^TX.
$$

如果输入通道相关，一个权重的误差可能由其他权重调整部分抵消。GPTQ 利用这个二阶结构逐步量化，而不是要求每个权重都独立选择最近的网格点。

假设先将第 j 个权重固定为 $q_j$，其他仍可调整。令 $e=w_j-q_j$，则约束为 $\delta_j=-e$。对正定 H，拉格朗日条件给出：

$$
H\delta+\mu e_j=0,\qquad
\delta=-\mu H^{-1}e_j,
\qquad
\mu=\frac{e}{[H^{-1}]_{jj}}.
$$

于是最优连续补偿为：

$$
\delta=-\frac{e}{[H^{-1}]_{jj}}[H^{-1}]_{:,j}.
$$

$e_j$ 在这里是单位向量，标量 e 是舍入误差，二者不要混淆。当前权重已受到前面步骤的补偿，每一步量化的不是原始 W 的独立副本。

例如 H=[[2,1],[1,2]]，第一个坐标被迫产生 δ₀=0.4，则第二个连续坐标的最优调整为 δ₁=−0.2。二次型 δᵀHδ 从 0.32 降到 0.24；若采用上面的 1/2 系数，损失对应 0.16 和 0.12。第二个权重之后仍需量化，所以这只是当前约束下的局部最优方向。

### 用 Cholesky 避免每一步重新求逆

为处理病态或校准中未出现的方向，先加入正阻尼：

$$
H_\lambda=\frac2T X^TX+\lambda\mathbf I.
$$

这里的粗体 I 是 I×I 单位阵。示例的 damping 是这个式子中的绝对 λ。作者代码常按 Hessian 对角均值设置相对阻尼，二者需要换算后比较。阻尼太小可能数值不稳定，太大则削弱通道相关性的作用。

令 U 为逆 Hessian 的上三角 Cholesky 因子：

$$
H_\lambda^{-1}=U^TU.
$$

NumPy 默认返回下三角因子，因此代码需要转置。作者源码中名为 Hinv 的变量在完成 Cholesky 后保存的是这个因子，而非仍然保存完整逆矩阵。

H 只由这层输入 X 决定，所以 O 个输出通道可以共享同一份二阶信息。它有 I² 个元素；I=4096 时，一份 float64 矩阵约占 128 MiB，求逆、Cholesky 和工作权重还需要额外空间。示例用小矩阵检查算法；大层量化则需控制统计精度、临时对象和分块更新，不能只按最终 4-bit 权重计算量化过程的峰值内存。

<!-- source-check: examples/quantization_lab.py -->
~~~python
def _prepare_gptq(
    x: np.ndarray,
    w: np.ndarray,
    group_size: int,
    damping: float,
    token_mask: Optional[np.ndarray],
    padding_mask: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate inputs and construct fixed scales plus upper ``chol(inv(H))``."""
    x, w = _check_x_w(x, w)
    group_size = _check_group_size(group_size)
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError("damping must be a finite positive scalar")
    mask = _valid_rows(x, token_mask, padding_mask)
    x_cal = x[mask]
    _, scales = quantize_int4_group(w, group_size)
    t = x_cal.shape[0]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            hessian = (2.0 / t) * (x_cal.T @ x_cal) + float(damping) * np.eye(x.shape[1])
    except FloatingPointError as exc:
        raise ValueError("GPTQ Hessian overflowed; reduce calibration input magnitude") from exc
    if not np.all(np.isfinite(hessian)):
        raise ValueError("GPTQ Hessian is non-finite after reduction")
    try:
        inv_hessian = np.linalg.inv(hessian)
        # np.linalg.cholesky returns lower L for A=L@L.T.  U=L.T is the
        # upper Cholesky factor of inv(H), matching the paper-style update.
        u = np.linalg.cholesky(inv_hessian).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("GPTQ Hessian could not be inverted/Cholesky-factorized") from exc
    if not np.all(np.isfinite(u)) or np.any(np.diag(u) <= 0):
        raise ValueError("GPTQ inverse-Hessian Cholesky is non-finite or non-positive")
    return w, scales, u
~~~

作者实现依次覆盖变量 H：

~~~python
H = torch.linalg.cholesky(H)
H = torch.cholesky_inverse(H)
H = torch.linalg.cholesky(H, upper=True)
Hinv = H
~~~

第一步分解原矩阵，第二步由分解得到逆矩阵，第三步再次分解为上三角因子。阅读后续更新时，Hinv[j,j] 因此是 U 的对角，而不是原逆 Hessian 的对角。

校准中完全为零的输入通道没有提供输出敏感性信息。这里保留原权重轴，并用阻尼保证可逆；通道裁剪、重排和作者实现中的 dead-channel 处理属于额外策略。

对于当前输入通道 j，所有输出通道可并行处理。工作权重临时转成 [O,I]：

$$
v_j=W_{\mathrm{work},:,j},\qquad
e=\frac{v_j-\widehat v_j}{U_{jj}},
\qquad
W_{\mathrm{work},:,j:}\mathrel{-}=e\,U_{j,j:}.
$$

<!-- source-check: examples/quantization_lab.py -->
~~~python
def gptq_quantize(
    x: np.ndarray,
    w: np.ndarray,
    group_size: int,
    damping: float = 0.01,
    token_mask: Optional[np.ndarray] = None,
    padding_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full sequential GPTQ teaching update, with no lazy blocking/act-order.

    ``W_work`` is [O,I].  For ``H=2 X.T@X/T + damping*I`` and upper
    ``U=chol(inv(H))``, each column uses ``e=(w-q)/U[j,j]`` and updates
    ``W_work[:,j:] -= e[:,None]*U[j,j:]``.  Scales are fixed from original W.
    Positive damping keeps a calibration-zero input channel invertible.
    """
    w, scales, u = _prepare_gptq(x, w, group_size, damping, token_mask, padding_mask)
    group_size = _check_group_size(group_size)
    work = w.T.copy()                  # [O, I]
    q = np.empty(w.shape, dtype=np.int8)
    w_hat = np.empty_like(w)
    for j in range(w.shape[0]):
        q_j, quantized_j = _quantize_work_column(work, j, scales, group_size)
        q[j] = q_j
        w_hat[j] = quantized_j
        error = (work[:, j] - quantized_j) / u[j, j]
        work[:, j:] -= error[:, None] * u[j, j:][None, :]
    return q, scales, w_hat
~~~

量化步长固定从原始 W 的组中取得，便于与 RTN 对照。若每量化一部分就重新计算组尺度，量化网格也在改变，不能再把两条路径的差异仅归因于误差补偿。

这里 $v_j$ 是包含所有输出通道的当前权重列，$\widehat v_j$ 是量化后还原的浮点列，对应代码中的 quantized_j；$q_j$ 则是整数编码。代码另有独立的后缀逆矩阵参考：每步重新计算带阻尼 Hessian 后缀的逆，按第一行与对角的比值更新剩余权重。两条路径相互对照，检查 U 的方向和后续状态是否一致。

### 分块延迟更新：把许多小更新合成矩阵乘

逐列实现反复读写很大的剩余权重。选一个计算块 [a,b)，先在局部 W1 中完成该块的量化，把归一化误差列收集到 E，再一次更新块外权重：

$$
W_{\mathrm{work},:,b:}
\mathrel{-}=E\,U_{a:b,b:}.
$$

<!-- source-check: examples/quantization_lab.py -->
~~~python
def gptq_quantize_blocked(
    x: np.ndarray,
    w: np.ndarray,
    group_size: int,
    block_size: int,
    damping: float = 0.01,
    token_mask: Optional[np.ndarray] = None,
    padding_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blocked lazy GPTQ: update W1/Err inside a block, then update its tail.

    For block ``[i1:i2]`` the external update is exactly
    ``work[:,i2:] -= Err @ U[i1:i2,i2:]``.  block_size=1 is the sequential
    route; larger blocks expose the lazy-GPTQ bookkeeping without act-order.
    """
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    w, scales, u = _prepare_gptq(x, w, group_size, damping, token_mask, padding_mask)
    group_size = _check_group_size(group_size)
    block_size = int(block_size)
    work = w.T.copy()
    q = np.empty(w.shape, dtype=np.int8)
    w_hat = np.empty_like(w)
    for i1 in range(0, w.shape[0], block_size):
        i2 = min(i1 + block_size, w.shape[0])
        w1 = work[:, i1:i2].copy()
        err = np.zeros_like(w1)
        for local, j in enumerate(range(i1, i2)):
            q_j, quantized_j = _quantize_work_column(
                w1, local, scales, group_size, scale_column=j
            )
            q[j] = q_j
            w_hat[j] = quantized_j
            e = (w1[:, local] - quantized_j) / u[j, j]
            err[:, local] = e
            w1[:, local:] -= e[:, None] * u[j, j:i2][None, :]
        if i2 < w.shape[0]:
            work[:, i2:] -= err @ u[i1:i2, i2:]
    return q, scales, w_hat
~~~

block_size 是计算更新块宽，G 是量化尺度分组，两者可以不同。固定尺度时，代码可以让计算块跨过量化组边界；取 scale 仍要使用全局输入通道 j，而不是块内 local 下标。

测试比较 block_size=1、2、3 和整个输入维度，与逐列版本及独立后缀参考对齐。实际 GPU 版本还要考虑矩阵乘精度、临时存储与分配开销；数学更新可合并，不等于任何块宽都更快。

### 用独立输入比较误差

随附示例固定 W[5,3]、G=2。校准输入为 [7,5]，其中 5 行有效；独立评估输入为 [6,5]。搜索与 Hessian 只看校准输入。

| 方法 | 校准输出 MSE | 独立评估输出 MSE |
|---|---:|---:|
| RTN | 0.00428730 | 0.01111508 |
| AWQ-style 缩放搜索 | 0.00252992 | 0.00564579 |
| 逐列 GPTQ | 0.00283880 | 0.00697122 |

这组结果用于核对实现和评价流程。模型质量还需用真实校准数据、层输出与任务指标评估；方法之间也不存在由这个小例子确定的普遍排名。

packed 计算与“先还原同一量化权重再矩阵乘”一致，是另一个独立检查。它验证布局和计算路径，而不评价量化后与原始权重相比损失了多少信息。

示例权重仅有 15 个元素，packed q 为 9 bytes，float64 scale 为 72 bytes，总计 81 bytes。它相对于 float64 原权重的 120 bytes 有所减少，但若比较 FP16 部署，scale 也应按实际部署类型计数。小矩阵和很小的 group 会让元数据占比很大，不能只报裸 4-bit 权重的压缩率。

### Mini Decoder：把校准、折叠、打包和评估连成模型闭环

单层 `XW` 误差不能回答“量化后的 decoder 是否仍输出相近 logits”。一个可控的闭环至少要覆盖：多层前向、实际进入各线性层的激活、共享输入折叠、权重量化与 checkpoint 序列化、加载后整模型评估。这里的 NumPy 模型是两层因果 decoder：每层 RMSNorm 的同一个输出并行进入 Q/K/V 和 gate/up，再将 attention 与 MLP 两个支路加回 residual。它刻意包含 Falcon 式 parallel-attention/MLP 的共享输入约束，但不是某个 Falcon checkpoint 的结构复刻。

本例的所有线性权重都使用 `W[I,O]`，即输入通道在第 0 轴、输出通道在第 1 轴；这与本节 `quantize_int4_group`、`dequantize_int4_group` 和 AWQ-style 搜索的约定相同。对 `[I,O]` 按输入轴每 G 行分组，量化 scale 的形状为 `[ceil(I/G),O]`。`pack_signed_int4` 将它转为 `packed[O,ceil(I/2)]`，每字节的低半字节先放较小输入索引，高半字节放下一个输入索引。PyTorch `Linear.weight` 通常保存为 `[O,I]`，接入时必须先转置为 `[I,O]`；不能把存储方向和数学乘法方向混为一谈。

校准集和 heldout 集必须在生成/切分数据时分开。对每个序列同时保留 `token_mask`（1 表示真实 token）和 `padding_mask`（1 表示 padding），有效位置是 `token_mask & ~padding_mask`。校准激活来自完整浮点模型的一次前向过程：每个 calibration batch 只跑原始 FP 模型一遍，在每层的实际计算点同时捕获共享 RMSNorm 输出、attention 输出投影输入、MLP down 输入和 lm_head 输入。深层激活因此属于原始模型轨迹，不是前层已量化后逐层回放得到的轨迹；量化误差的跨层累积最终由 heldout 整模型评估观察。

<!-- source-check: examples/model_quantization_lab.py -->
~~~python
    for start in range(0, data.token_ids.shape[0], batch_size):
        end = min(start + batch_size, data.token_ids.shape[0])
        _, captured = model.forward(
            data.token_ids[start:end], data.token_mask[start:end], data.padding_mask[start:end], capture=True
        )
        for name, x in captured.items():
            rows.setdefault(name, []).append(x)
        token_rows.append(data.token_mask[start:end].reshape(-1))
        padding_rows.append(data.padding_mask[start:end].reshape(-1))
    token_mask = np.concatenate(token_rows)
    padding_mask = np.concatenate(padding_rows)
    if not np.any(token_mask & ~padding_mask):
        raise ValueError("calibration data contains no valid tokens")
~~~

#### 共享输入折叠必须按消费者集合推导

令 RMSNorm 的未缩放输出为 `r`，可学习增益为向量 `γ`，该层送入线性分支的向量为 `z = r ⊙ γ`。给输入通道定义正对角尺度 `S=diag(s)`，令 `γ'=γ/s`，并对每一个消费 `z` 的权重使用 `W'_j = S W_j`。于是：

$$
z'W'_j = \bigl(r\odot\gamma\oslash s\bigr)(S W_j)
= (r\odot\gamma)W_j = zW_j.
$$

这里的 `s` 是输入通道变换尺度，不是 INT4 的分组量化步长。变换后应从新的 `W'_j` 重新搜索/计算量化 scale。若是带 bias 的 LayerNorm，归一化仿射 bias 也要除以 `s`；RMSNorm 通常没有该 bias。若折叠点的输出有多个消费者，所有消费者都必须使用同一组 `s`：本 Mini Decoder 的并行层把 Q、K、V、gate、up 五个 `[I,O_j]` 权重沿输出轴拼接，搜索一份 `s`，然后同时变换五支和共享 norm gamma。这样也直接覆盖 Falcon 旧式 `parallel_attn` 分支中一个 input LayerNorm 同时供 QKV 与 MLP `fc1` 使用的情况；不同 Falcon decoder 架构可能使用分离 norm，不能仅凭模型家族名称推断拓扑。

<!-- source-check: examples/model_quantization_lab.py -->
~~~python
        selection = qlab.awq_style_search(x, joint_w, group_size, token_mask, padding_mask)
        scale = selection.scale
        folded.params[prefix + "norm_gamma"] = model.params[prefix + "norm_gamma"] / scale
        x_folded = x / scale[None, :]
        fold_scales[prefix + "shared_in"] = scale.copy()
        for name in consumer_names:
            folded.params[name] = model.params[name] * scale[:, None]
            clip_choices[name] = search_clipping(
                x_folded, folded.params[name], group_size, token_mask, padding_mask, clip_ratios
            )
~~~

漏掉任一消费者就不再是等价重参数化。例如若先把共享激活除以 `s`，却只对 Q 权重乘 `s`，Q 投影保持不变，但 K、V 和 gate/up 投影会被除以 `s`；attention 的 `QKᵀ` 与 `V` 汇聚都会变化，MLP 也会变化。反例运行中，所有分支一起折叠的输出误差约为 `4.44e-16`，只折 Q 的 attention 输出最大绝对误差为 `3.118`。这不是浮点 roundoff，而是图变换违反共享边约束。

SmoothQuant 作者的 `smooth.py` 把 `fcs` 作为列表处理：对列表中所有下游 Linear 一起统计权重通道范围，LayerNorm 的 weight/bias 除以尺度，列表内每个 FC 的输入列都乘同一尺度；Llama-like RMSNorm 版本同样处理所有 FC，但只改 RMSNorm weight。其 Falcon 分支对 `parallel_attn` 且非 `new_decoder_architecture` 的路径，显式将 QKV 和 MLP `fc1` 放进同一次 `smooth_ln_fcs` 调用；新 decoder architecture 则根据 `ln_attn`/`ln_mlp` 分别处理。

#### clipping 搜索目标与 AWQ 的边界

对每个输入分组 g 和输出通道 o，教学实现用 `c∈{1.00,0.95,0.90,0.85,0.80,0.75}` 搜索截断上限 `c·max(|W[g,o]|)`，再基于截断后的权重重新量化。对 calibration 输入 `X_cal`，每个候选的目标是单线性输出重建误差：

$$
E_{g,o}(c) = \operatorname{MSE}\!\left(X_{cal}Q_c(W),\;X_{cal}W\right),
$$

其中 `Q_c(W)` 表示裁剪、INT4 编码并反量化后的浮点矩阵。候选首项强制为 `c=1`，所以已搜索的这层 calibration MSE 不会高于同一 RTN 网格的未裁剪基线；它不保证 heldout 更好，更不保证下游模型质量更好。共享输入折叠后的 Q/K/V/gate/up 使用 `X_cal/s` 和 `SW` 评估；o_proj、down_proj、lm_head 使用原始浮点轨迹中各自捕获的输入。

这个 per-linear clipping 是用于暴露“局部目标如何进入全模型”的教学策略，不是完整 AWQ 复现。AWQ 作者 `auto_clip.py` 明确跳过名称包含 `q_`、`k_`、`query`、`key` 或 `Wqkv` 的线性层，注释原因是 QK batch matrix multiply 难以精确独立裁剪。当前实验为了形成可观察的全模型对照，仍对每个投影独立做自定义 clipping 搜索，包括 Q/K；因此必须看 heldout 的整模型 logits 和 next-token NLL/PPL，而不能从层级 MSE 推断质量。

#### packed checkpoint 是存储格式，不等于低位推理 kernel

checkpoint 将每个线性权重保存为 signed INT4 packed 字节和 `[ceil(I/G),O]` scale；embedding 与各层/final norm gamma 保留为浮点数组。JSON metadata 写入压缩 NPZ，包含格式名/版本、模型维度、weight orientation、packed orientation、nibble 顺序、scale shape、逻辑矩阵 shape、校准有效 token 数、fold scales 和逐层 clipping 选择。loader 会拒绝未知版本、错误格式标签、张量名或 shape 不匹配、packed dtype/字节数不符、非正或非有限 scale，以及量化范围不符。

模型加载后，教学 loader 会立即解包 INT4 并反量化为 NumPy `float64` `[I,O]` 矩阵，再用普通浮点 `@` 完成前向。因此 packed 表示只证明“模型可序列化、元数据可校验、加载后数值闭环可复现”；这里没有融合反量化 INT4 GEMM kernel，也不代表 INT4 推理加速或实际部署 checkpoint 兼容性。导出采用独占创建，目标文件已存在会直接失败且不会删除或覆盖。默认 demo 把 checkpoint 放在系统临时目录并在退出时清理；`--output` 仅用于一个尚不存在的显式路径。

#### heldout 的整模型误差与 next-token 指标

对 `logits[B,S,V]` 做语言模型 next-token NLL 时，位置 `t` 的 logits 预测 token `t+1`。必须先 shift，再让“当前位置和目标位置都有效”的相邻 token 对参与统计。对 `m_{b,t}` 为有效 mask，参与集合为 `m_{b,t}=1 ∧ m_{b,t+1}=1`；这样 padding 位置既不会作为预测上下文，也不会作为目标 token。token NLL 是所有有效目标的负对数概率均值，perplexity 为 `exp(NLL)`。

<!-- source-check: examples/model_quantization_lab.py -->
~~~python
    valid = data.token_mask & ~data.padding_mask
    predict = valid[:, :-1] & valid[:, 1:]
    if not np.any(predict):
        raise ValueError("next-token evaluation needs at least one adjacent valid token pair")
    shifted_logits = logits[:, :-1, :][predict]
    targets = data.token_ids[:, 1:][predict]
    row_max = np.max(shifted_logits, axis=-1, keepdims=True)
    logsumexp = row_max[:, 0] + np.log(np.sum(np.exp(shifted_logits - row_max), axis=-1))
    target_logits = shifted_logits[np.arange(targets.size), targets]
    nll = float(np.mean(logsumexp - target_logits))
    return {"next_token_count": float(targets.size), "next_token_nll": nll, "perplexity": float(np.exp(nll))}
~~~

可运行的 demo 使用固定随机种子的、未训练的 2 层模型（vocabulary=23、hidden=12、intermediate=20、3 heads、group size=4），16 条 calibration 序列和另行采样的 12 条 shifted heldout 序列；变长序列右侧 padding。共有 76 个有效 calibration token、55 个有效 heldout token；heldout 中仅有 43 个有效 next-token 目标。fold-only 保持 heldout logits，最大绝对误差为 `2.442e-15`。完整量化并从 checkpoint 加载后，heldout 结果为：

| 指标 | 浮点参考 | packed checkpoint 加载后 | 条件 |
|---|---:|---:|---|
| logits MSE | — | 0.0223053 | 只统计 55 个有效输入 token 的词表 logits |
| logits 最大绝对误差 | — | 0.443295 | 与同一随机未训练 FP 模型比较 |
| logits 相对 L2 | — | 0.141773 | `||q-fp||₂ / ||fp||₂` |
| token top-1 一致率 | — | 87.27% | 每个有效位置比较 argmax token |
| next-token NLL | 3.800605 | 3.844157 | 同一 43 个 heldout 目标，越低越好 |
| perplexity | 44.7283 | 46.7193 | `exp(NLL)`，非真实模型质量结果 |

数字用于检查 mask、checkpoint 与评估管线能产生自洽结果；随机初始化、未训练模型上的 NLL/PPL 不是语言建模质量评估，synthetic token 分布也不能代表真实 prompt。部署质量需要对同一个已训练模型、固定 tokenizer 和预处理，使用独立真实 calibration/validation 文本，记录 token 级 NLL/PPL、任务指标及长上下文/特殊输入回归。heldout 只能评估候选，不能回流参与 clipping/scale 选择。单一 seed/单组 shape 也不支持普遍精度结论。

运行完整 CPU 流程与测试：

~~~bash
python roadmap/curriculum/operators/07-quantized-operators/examples/model_quantization_lab.py --demo
python roadmap/curriculum/operators/07-quantized-operators/examples/test_model_quantization_lab.py
~~~

若要保留 checkpoint，可将 `--output` 指向一个新的文件路径；程序不会覆盖已有目标。Mini 模型使用 NumPy 与临时输出，不需要 CUDA、`nvcc` 或模型下载。

#### 可选：适配已在本机的 Hugging Face decoder

不下载权重也可以接入一个已经完整落盘、架构受 Transformers 支持的小模型；官方 `from_pretrained` 接口提供 `local_files_only=True`，且 `trust_remote_code` 默认关闭。自定义模型代码只应在审阅并信任后启用。最小加载入口如下，实际模块名、mask 与 RoPE 路径仍需按该模型实现适配：

~~~python
from transformers import AutoModelForCausalLM, AutoTokenizer

local_model_dir = r"D:\models\small-decoder"
tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    local_model_dir,
    local_files_only=True,
    trust_remote_code=False,
)
~~~

用 forward hook 收集每个实际 Linear 的输入时，要从 attention mask 构造与上文相同的有效 token mask，并分别生成 calibration 和 heldout；不能把 padding 隐式混进激活统计。对 HF `Linear.weight[O,I]`，喂入本目录量化函数前先取转置得到 `[I,O]`，输出的量化/packed 张量加载后再按目标 kernel 的布局转换。融合 QKV 可以在 `[I,O_q+O_k+O_v]` 空间统一缩放；分开的 Q/K/V 也必须共用同一 S。SwiGLU 的 gate 和 up 共用 norm 输入，应一起折叠；若架构是 Falcon parallel-attention/MLP，同一个 LN 输入还同时连接 QKV 与 FC1，需按拓扑一次性扩展消费者集合。残差、旋转位置编码、QK norm、分组查询头、bias 和权重共享会影响等价条件，本 NumPy loader 不会自动识别或处理这些模型特性。

保存前记录模型 config、tokenizer 标识/版本、权重 orientation、分组大小、校准语料与预处理版本、mask 约定、量化格式版本、共享输入映射和所有浮点例外张量；用独立 heldout 评估整模型 logits 与 next-token loss。此路径只说明如何适配校准/误差分析，不会把本实验 NPZ 伪装成 Transformers 或推理引擎可直接加载的 checkpoint。`local_files_only` 与 `trust_remote_code` 的行为见 [Transformers 官方 Auto Classes 文档](https://huggingface.co/docs/transformers/model_doc/auto)。

### 真实预训练 Llama：选择性 W4 overlay 的校准与本地评估

Mini Decoder 只证明数值合同和指标代码可以闭环；真实模型适配还必须按 checkpoint 的实际计算图找归一化点、分支消费者和权重布局。这里明确支持的是标准 Hugging Face `LlamaForCausalLM`：`config.model_type == "llama"`、`model.layers`、每层独立 `input_layernorm` / `post_attention_layernorm`，以及分开的 `q_proj`、`k_proj`、`v_proj`、`gate_proj`、`up_proj`。示例只对这五组投影做选择性 W4；`o_proj`、`down_proj`、`lm_head`、embedding、norm 参数和 bias 保持基座 dtype（norm 增益/偏置只应用等价折叠）。脚本会检查这些模块与维度，不会仅凭模型仓库名称猜结构；融合 QKV、Q/K norm 变体和自定义 Python 架构不在这个适配器的支持合同内。

Llama 每层的输入共享边和折叠操作如下：

| 共享输入 | 消费者 | 张量形状与后续运算 | 折叠动作 |
|---|---|---|---|
| `input_layernorm` 输出 `[B,S,H]` | 独立的 `q_proj`、`k_proj`、`v_proj` | 每个 Linear 的权重存成 `[O,H]`；Q/K 投影后再做 RoPE，V 进入 attention | 三支共用一个 `s∈R^H`；norm 增益除以 s，三个权重的输入列都乘 s |
| `post_attention_layernorm` 输出 `[B,S,H]` | `gate_proj`、`up_proj` | 两支输出 `[B,S,F]`，之后执行 `SiLU(gate) ⊙ up` | 两支共用一个 s；norm 与两个 Linear 同时折叠，乘法和 SiLU 留在原位 |
| attention / MLP branch 输出 | `o_proj`、`down_proj` 及 identity residual | 各 branch 回到 `[B,S,H]` 后与 residual 相加 | 不跨越加法折叠；本例把 o/down 保持为浮点例外 |
| token embedding | `lm_head` | hidden `[B,S,H]` 投影到词表 `[B,S,V]`；权重可能与 embedding 共享 | 两者都不量化、不拆开绑定关系，保留为浮点例外 |

其中 B 是 batch 大小，S 是序列长度，H 是 hidden size，Hq/Hkv 是 Q/K/V 的输出宽度，F 是 MLP intermediate size。GQA 只改变 Hq 与 Hkv 的关系，不改变 Q/K/V 共用同一归一化输入这一事实。SwiGLU 的 gate 与 up 也共用同一输入。折叠的边界是归一化输出到这些 Linear 的输入边，不是跨越 attention、非线性或 residual 的任意图变换。

令 RMSNorm 产生的向量为 `z∈R^H`，分支 j 的权重按数学乘法写作 `W_j∈R^{H×O_j}`。选取正向量 `s∈R^H`，则：

$$
z' = z \oslash s,\qquad W'_j = \operatorname{diag}(s)W_j,\qquad z'W'_j=zW_j.
$$

因此 RMSNorm 的增益改为 `gamma/s`，而同组每个 Linear 的输入列权重都乘以 s。若归一化模块有仿射 bias，也要除以 s；Linear 自身 bias 保持原值，因为线性输出为 `xW+b`，变换只改 x 与 W。Hugging Face `Linear.weight` 的张量形状是 `[O_j,H]`，所以实际乘每列时，要先转成 `[H,O_j]`，不能把 PyTorch 存储方向误当成公式方向。

这个缩放发生在 Q/K/V 线性投影之前，故标准 RoPE 仍作用于原本等价的 Q、K；旋转位置编码本身不用吸收 s。若架构在 Q/K 投影后另有 QK normalization，或使用融合 QKV、自定义 attention 输入变换，就必须重新追踪张量图并写对应的等价变换与测试；当前脚本直接拒绝带 Q/K norm 的变体，而不是悄悄套用标准 Llama 规则。Residual 是 `h + branch(h)` 的并行直连，不能只缩放一边后假定加法仍不变；权重绑定的 embedding 和 `lm_head` 默认作为浮点例外，不独立改写其中一份权重破坏共享关系。

校准与评估数据来自用户已准备的本地 UTF-8 文本文件，每个非空行是一条独立序列。两个文件必须不同且不能包含相同的非空行。模型配置、tokenizer 和 safetensors 权重通过 Transformers `local_files_only=True` 加载，并关闭 `trust_remote_code`；脚本不联网找缺失文件。Tokenizer 对 batch 右侧 padding，forward hook 读取同一批的 `attention_mask`，只收集 mask 为 1 的 `[B,S,H]` 归一化输出：

<!-- source-check: examples/hf_local_llama_quantization.py -->
~~~python
    def _hook(self, name: str):
        def collect(_module, _inputs, output):
            import torch

            if self.mask is None or not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise ValueError(f"{name}: expected [batch,sequence,hidden] output and active attention_mask")
            if tuple(output.shape[:2]) != tuple(self.mask.shape):
                raise ValueError(f"{name}: normalization output and attention_mask shapes differ")
            remain = self.limit - self.counts[name]
            if remain <= 0:
                return
            rows = output.detach().float()[self.mask.bool()]
            if rows.numel() == 0:
                return
            rows = rows[:remain].cpu().numpy()
            if not np.all(np.isfinite(rows)):
                raise ValueError(f"{name}: valid calibration/evaluation activations are non-finite")
            self.parts[name].append(rows)
            self.counts[name] += rows.shape[0]
        return collect
~~~

Forward hook 只做 detached activation 的采样，不建立梯度图。Mask 由当前 batch 设置后才运行模型；句柄在异常路径也必须清除，完整生命周期封装如下：

<!-- source-check: examples/hf_local_llama_quantization.py -->
~~~python
    def close(self):
        for handle in self.handles:
            handle.remove()


def collect_norms(torch, model, tokenizer, texts, groups, batch_size, max_seq_len, device, limit):
    capture = NormCapture(model, (norm for norm, _ in groups), limit)
    try:
        with torch.inference_mode():
            for batch in encoded_batches(tokenizer, texts, batch_size, max_seq_len, device):
                capture.set_mask(batch["attention_mask"])
                model(**batch, use_cache=False)
        return capture.arrays()
    finally:
        capture.close()
~~~

每个 forward 前都要更新与当前 batch 对应的 mask，形状不符时立即报错；`inference_mode` 限定为读前向，`finally` 保证 hook 在异常时也移除。CPU 合同函数采用同一有效 token 约定，并额外拒绝 NaN、负数和大于 1 的 mask：

<!-- source-check: examples/hf_quant_contracts.py -->
~~~python
def valid_token_rows(hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Flatten [batch, seq, hidden] and retain exactly non-padding positions."""
    values = np.asarray(hidden)
    mask = np.asarray(attention_mask)
    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise ValueError("hidden must be [batch,seq,hidden] and mask [batch,seq]")
    if mask.dtype.kind not in "biuf" or not np.all(np.isfinite(mask)) or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("attention_mask must contain only boolean or numeric 0/1 values")
    keep = mask.astype(bool, copy=False)
    if not np.any(keep):
        raise ValueError("attention_mask contains no valid tokens")
    rows = values[keep]
    if not np.all(np.isfinite(rows)):
        raise ValueError("valid hidden states must be finite")
    return rows
~~~

因果语言模型的有效预测对还要同时满足当前位置与下一位置有效：

$$
m_{b,t}^{\mathrm{pred}}=\mathbf{1}[a_{b,t}=1\land a_{b,t+1}=1],\quad
\mathrm{NLL}=-\frac{1}{N}\sum_{(b,t):m_{b,t}^{\mathrm{pred}}=1}
\log p(x_{b,t+1}\mid x_{b,\le t}),\quad \mathrm{PPL}=e^{\mathrm{NLL}}.
$$

`a` 是输入 attention mask，N 是有效 next-token target 数。padding 行既不参与 scale 选择，也不进入 NLL/PPL 的分母。以 heldout 的原始浮点轨迹捕获同一归一化点输入，量化后每个目标 Linear 的局部输出 MSE 才可在 heldout 激活上计算；它用于定位敏感分支，不替代整模型 NLL/PPL。整模型 logits MSE/max error 只对有界抽样的词表向量计算，以免把 `[tokens,vocab]` 全量 logits 留在内存。另可传入独立 JSONL 的 `prompt`/`target` 对，报告 greedy exact match；没有任务文件时，next-token top-1 accuracy 只是 heldout 语言建模指标，不得称为下游任务分数。

校准算法直接复用本章现有 `awq_style_search` 与分组 INT4 helper，不重新实现或宣称 AWQ/GPTQ 作者算法。每个 norm 的校准激活与 QKV 或 gate/up 权重沿输出轴拼接，用同一候选尺度评估量化后输出误差，再对每个 Linear 做对称 groupwise round-to-nearest INT4。候选 scale 只从 calibration rows 中选；代码把同一 norm 的消费者权重拼接，搜索一份 scale，然后才分别编码每个消费者：

<!-- source-check: examples/hf_local_llama_quantization.py -->
~~~python
        joined = np.concatenate([original[name] for name in consumers], axis=1)
        selection = qlab.awq_style_search(x_cal, joined, group_size)
        # Use the exact float32 scale that will be stored and later loaded.
        fold = selection.scale.astype(np.float32).astype(np.float64)
        joined_scaled = joined * fold[:, None]
        _, joined_scales = qlab.quantize_int4_group(joined_scaled, group_size)
        joined_scales = joined_scales.astype(np.float32)
~~~

该教学实现没有 AWQ 作者代码中的完整模型适配、layer 输出策略及全部 clipping 逻辑，也没有 GPTQ 的 Hessian/近似二阶误差补偿、act-order 或 true-sequential 路径。作者仓库的加载器、pseudo/real quantized 分支和 kernel 是另外一套工具链；仅复用同名方法思想不构成论文或作者实现复现。版本也不能混用：Transformers 官方 AWQ 页面注明 AutoAWQ 会把 Transformers 降到 4.47.1；GPTQ 作者仓库记录的旧测试环境是 Transformers 4.21.2 与 PyTorch 1.10.1+cu111。它们分别描述那些工具链的兼容范围，不是本地适配脚本的依赖锁定；脚本依据运行时 Transformers 主版本选择 `dtype`/`torch_dtype` 参数，并将实际 Transformers/PyTorch 版本写入 overlay manifest。

核心布局转换如下。G 是输入维度上的量化分组大小；q 的数学布局为 `[H,O]`，scale 为 `[ceil(H/G),O]`，packed 字节则转为 `[O,ceil(H/2)]`，每字节低半字节先存较小的输入索引：

~~~python
weight_io = linear.weight.detach().float().cpu().numpy().T  # [O,H] -> [H,O]
q, scales = quantize_int4_group(weight_io * scale[:, None], group_size=G)
packed = pack_signed_int4(q)                                 # [O,ceil(H/2)]
restored_io = dequantize_int4_group(q, scales, group_size=G)  # [H,O]
with torch.no_grad():
    linear.weight.copy_(torch.from_numpy(restored_io.T.copy()).to(
        device=linear.weight.device, dtype=linear.weight.dtype
    ))
~~~

最后一行是普通浮点 `Linear` 的教学仿真，并非低位执行：NPZ overlay 在 load 时校验 base 权重与 tokenizer 指纹，解包后反量化回原 dtype，再执行浮点 Linear。Manifest 明确记录 format/version、原始 checkpoint 与 tokenizer 指纹、Transformers/PyTorch 版本、G、逻辑/数学/packed 布局、半字节顺序、每个 fold group 及消费者、校准/heldout 文件指纹、浮点例外和层级/整模型指标。Embedding、`lm_head`、o/down projection、Linear bias、非目标层参数等均作为基座中的浮点例外；被折叠的 norm gamma 则由 overlay 的共享尺度变换。该 overlay 必须与匹配的本地原始 checkpoint 配合，不是独立 checkpoint，也不兼容 vLLM 的模型加载合同。要证明低比特执行，仍需另行接入并检查目标 kernel：packed 文件占用变少只说明存储编码变紧，不能证明 kernel 未先完整反量化，更不能推出时延提升。

加载 NPZ 时，先验证完整 manifest、所有 packed/scales/fold arrays、目标拓扑和参数形状；只有这些检查全部通过，才进入 `no_grad` 修改参数。这样损坏的后续张量不会让前面几层已经折叠一半：

<!-- source-check: examples/hf_local_llama_quantization.py -->
~~~python
def apply_overlay(torch, model, arrays, manifest):
    validate_overlay_for_model(torch, model, arrays, manifest)
    modules = dict(model.named_modules())
    group_size = manifest["quantization"]["group_size"]
    with torch.no_grad():
        for group in manifest["fold_groups"]:
            norm = modules[group["norm"]]
            fold = torch.as_tensor(arrays[group["scale_key"]], device=norm.weight.device, dtype=norm.weight.dtype)
            norm.weight.div_(fold)
            if getattr(norm, "bias", None) is not None:
                norm.bias.div_(fold)
        for record in manifest["tensors"]:
            name = record["name"][:-len(".weight")]
            module = modules.get(name)
            if not isinstance(module, torch.nn.Linear):
                raise ValueError(f"overlay target is not a Linear module: {name}")
            out_features, in_features = record["logical_shape"]
            if list(module.weight.shape) != [out_features, in_features]:
                raise ValueError(f"base model shape mismatch for {name}.weight")
            q = qlab.unpack_signed_int4(arrays[record["packed_key"]], in_features)
            restored_io = qlab.dequantize_int4_group(q, arrays[record["scale_key"]], group_size)
            value = torch.as_tensor(restored_io.T.copy(), device=module.weight.device, dtype=module.weight.dtype)
            with torch.no_grad():
                module.weight.copy_(value)
~~~

完整 NPZ 数组预检包括 uint8 packed dtype、逻辑/packed/scales/fold shape、正且有限的 scale、对称 INT4 码域、奇数输入宽度的 padding nibble、唯一消费者和 exact key set；CPU 测试不需安装 Torch 就能检查这些格式合同。实际模型的层数、norm 宽度与每个 Linear 的 `[O,I]` shape 则在同一预检阶段对照加载模型；`apply_overlay` 完成这一阶段后再解包回浮点权重。

下面先运行无模型、无 Torch 依赖的 NumPy 合同测试，再对一份已在本机的标准 Llama 执行校准和 heldout 评估。命令不会创建或下载数据、权重；校准/heldout 文本应由实验者按任务准备，且需要足够的主存/显存容纳所选本地模型：

~~~bash
python roadmap/curriculum/operators/07-quantized-operators/examples/test_hf_quant_contracts.py
python roadmap/curriculum/operators/07-quantized-operators/examples/hf_local_llama_quantization.py \
  --model-dir /models/local-llama \
  --calibration-file /data/calibration.txt \
  --heldout-file /data/heldout.txt \
  --group-size 128 --device auto \
  --save /data/llama-w4-teaching-overlay.npz
~~~

后续加载时从相同本地基座运行 `--model-dir /models/local-llama --load /data/llama-w4-teaching-overlay.npz --heldout-file /data/heldout.txt`；base safetensors 或 tokenizer 任一指纹不同都会拒绝。可选任务文件为每行一个 JSON 对象，例如 `{"prompt":"2 + 2 =","target":"4"}`，运行时另外传入 `--task-file /data/local-task.jsonl`。导出路径已存在会失败而不覆盖。现场执行结果应分别保存 reference/quantized heldout NLL、PPL、有效 token 数、抽样 logits 差异、逐层 MSE 和可选 exact match；这些指标必须来自真实预训练模型与相应数据，不能用前面的随机 Mini Decoder 数字代替。


<details>
<summary>完整本地 Llama 校准、保存/加载与评估程序</summary>

<!-- source-check: examples/hf_local_llama_quantization.py -->
~~~python
"""Local-only Llama W4 teaching overlay: calibration, heldout evaluation, save/load.

Requires a local safetensors LlamaForCausalLM checkpoint, its local tokenizer,
PyTorch, Transformers and NumPy. It never downloads model files or remote code.
The packed overlay is not a Transformers or serving-engine checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quantization_lab as qlab  # existing teaching implementation; not AWQ/GPTQ author code
from hf_quant_contracts import (
    FORMAT_NAME, FORMAT_VERSION, llama_fold_groups, validate_manifest, validate_overlay_arrays,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted({p.resolve() for p in paths})
    if not files:
        raise ValueError("cannot fingerprint an empty local file set")
    for path in files:
        digest.update(path.relative_to(root.resolve()).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def local_model_fingerprint(model_dir: Path) -> str:
    config = model_dir / "config.json"
    weights = sorted(model_dir.rglob("*.safetensors"))
    if not config.is_file() or not weights:
        raise FileNotFoundError("local config.json and at least one safetensors weight file are required")
    return fingerprint([config, *weights], model_dir)


def local_tokenizer_fingerprint(model_dir: Path) -> str:
    names = {
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "tokenizer.model", "spiece.model",
    }
    paths = [p for p in model_dir.rglob("*") if p.is_file() and p.name in names]
    if not paths:
        raise FileNotFoundError("no local tokenizer files were found beside the model")
    return fingerprint(paths, model_dir)


def file_fingerprint(path: Path) -> str:
    return sha256_file(path.resolve())


def read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"local text file not found: {path}")
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"text file has no non-empty lines: {path}")
    return rows


def check_disjoint(calibration: list[str], heldout: list[str]) -> None:
    overlap = set(calibration) & set(heldout)
    if overlap:
        raise ValueError(f"calibration and heldout contain {len(overlap)} identical non-empty lines")


def load_runtime(model_dir: Path, device_arg: str, dtype_arg: str):
    try:
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install PyTorch, Transformers, NumPy and safetensors in the user's model environment.") from exc

    model_dir = model_dir.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"--model-dir must be an existing local directory: {model_dir}")
    config = AutoConfig.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    if getattr(config, "model_type", None) != "llama":
        raise ValueError(f"expected config.model_type='llama', got {getattr(config, 'model_type', None)!r}")
    if getattr(config, "quantization_config", None):
        raise ValueError("start from an unquantized local Llama checkpoint")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs a local pad_token or eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    device = device_arg
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch reports no CUDA device")
    if dtype_arg == "auto":
        if device.startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_arg]
    major = int(transformers.__version__.split(".", 1)[0])
    dtype_key = "dtype" if major >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        config=config,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        **{dtype_key: dtype},
    )
    if model.__class__.__name__ != "LlamaForCausalLM":
        raise ValueError(f"expected HF LlamaForCausalLM, got {model.__class__.__name__}")
    model.to(device)
    model.eval()
    return torch, transformers, tokenizer, model, device, dtype


def validate_topology(model, torch) -> list[tuple[str, tuple[str, ...]]]:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("expected the standard HF LlamaForCausalLM model.layers topology")
    layers = model.model.layers
    groups = llama_fold_groups(len(layers))
    modules = dict(model.named_modules())
    for norm_name, consumers in groups:
        norm = modules.get(norm_name)
        if norm is None or not isinstance(getattr(norm, "weight", None), torch.Tensor):
            raise ValueError(f"missing expected normalization module {norm_name}")
        if norm.weight.ndim != 1:
            raise ValueError(f"{norm_name}.weight must be one-dimensional")
        if getattr(norm, "bias", None) is not None and norm.bias.shape != norm.weight.shape:
            raise ValueError(f"{norm_name}.bias shape does not match its weight")
        for name in consumers:
            linear = modules.get(name)
            if not isinstance(linear, torch.nn.Linear):
                raise ValueError(f"{name} must be a separate torch.nn.Linear projection")
            if linear.in_features != norm.weight.numel():
                raise ValueError(f"{name}.in_features does not match {norm_name}")
    for layer in layers:
        attn = layer.self_attn
        if any(getattr(attn, name, None) is not None for name in ("q_norm", "k_norm", "q_layernorm", "k_layernorm")):
            raise ValueError("Q/K-normalized Llama variants need an explicit topology adapter")
    return groups


def encoded_batches(tokenizer, texts: list[str], batch_size: int, max_seq_len: int, device):
    import torch

    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        yield {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
        }


class NormCapture:
    def __init__(self, model, group_names: Iterable[str], limit: int):
        self.parts: dict[str, list[np.ndarray]] = {name: [] for name in group_names}
        self.counts = {name: 0 for name in group_names}
        self.limit = limit
        self.mask = None
        modules = dict(model.named_modules())
        self.handles = []
        for name in self.parts:
            self.handles.append(modules[name].register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def collect(_module, _inputs, output):
            import torch

            if self.mask is None or not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise ValueError(f"{name}: expected [batch,sequence,hidden] output and active attention_mask")
            if tuple(output.shape[:2]) != tuple(self.mask.shape):
                raise ValueError(f"{name}: normalization output and attention_mask shapes differ")
            remain = self.limit - self.counts[name]
            if remain <= 0:
                return
            rows = output.detach().float()[self.mask.bool()]
            if rows.numel() == 0:
                return
            rows = rows[:remain].cpu().numpy()
            if not np.all(np.isfinite(rows)):
                raise ValueError(f"{name}: valid calibration/evaluation activations are non-finite")
            self.parts[name].append(rows)
            self.counts[name] += rows.shape[0]
        return collect

    def set_mask(self, mask):
        self.mask = mask

    def arrays(self) -> dict[str, np.ndarray]:
        result = {}
        for name, pieces in self.parts.items():
            if not pieces:
                raise ValueError(f"no valid tokens were captured at {name}")
            result[name] = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()


def collect_norms(torch, model, tokenizer, texts, groups, batch_size, max_seq_len, device, limit):
    capture = NormCapture(model, (norm for norm, _ in groups), limit)
    try:
        with torch.inference_mode():
            for batch in encoded_batches(tokenizer, texts, batch_size, max_seq_len, device):
                capture.set_mask(batch["attention_mask"])
                model(**batch, use_cache=False)
        return capture.arrays()
    finally:
        capture.close()


def read_task_pairs(path: Path | None) -> list[tuple[str, str]]:
    if path is None:
        return []
    pairs = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or not all(isinstance(item.get(key), str) for key in ("prompt", "target")):
            raise ValueError(f"{path}:{line_no}: expected JSON object with string prompt and target")
        pairs.append((item["prompt"], item["target"]))
    if path is not None and not pairs:
        raise ValueError("task file contains no prompt/target pairs")
    return pairs


def evaluate(torch, model, tokenizer, texts, batch_size, max_seq_len, device, groups,
             metric_tokens: int, logit_samples: int, capture_limit: int | None = None):
    import torch.nn.functional as F

    capture = NormCapture(model, (norm for norm, _ in groups), capture_limit) if capture_limit else None
    total_nll = 0.0
    total_targets = 0
    correct = 0
    sample_logits: list[np.ndarray] = []
    sample_targets: list[np.ndarray] = []
    seen_metric_tokens = 0
    try:
        with torch.inference_mode():
            for batch in encoded_batches(tokenizer, texts, batch_size, max_seq_len, device):
                if seen_metric_tokens >= metric_tokens and (
                    capture is None or all(n >= capture.limit for n in capture.counts.values())
                ):
                    break
                mask = batch["attention_mask"].bool()
                if capture is not None:
                    capture.set_mask(batch["attention_mask"])
                logits = model(**batch, use_cache=False).logits.float()
                target_mask = mask[:, :-1] & mask[:, 1:]
                remaining = metric_tokens - seen_metric_tokens
                if remaining <= 0:
                    continue
                positions = target_mask.nonzero(as_tuple=False)
                if positions.numel() == 0:
                    continue
                positions = positions[:remaining]
                rows, cols = positions[:, 0], positions[:, 1]
                predicted = logits[rows, cols]
                labels = batch["input_ids"][rows, cols + 1]
                total_nll += float(F.cross_entropy(predicted, labels, reduction="sum").item())
                correct += int((predicted.argmax(dim=-1) == labels).sum().item())
                total_targets += int(labels.numel())
                seen_metric_tokens += int(labels.numel())
                if sum(x.shape[0] for x in sample_logits) < logit_samples:
                    take = min(logit_samples - sum(x.shape[0] for x in sample_logits), predicted.shape[0])
                    sample_logits.append(predicted[:take].detach().cpu().numpy())
                    sample_targets.append(labels[:take].detach().cpu().numpy())
        if total_targets == 0:
            raise ValueError("heldout data needs at least one adjacent valid next-token pair")
        nll = total_nll / total_targets
        result = {
            "next_token_count": total_targets,
            "next_token_nll": nll,
            "perplexity": math.exp(min(nll, 700.0)),
            "next_token_top1_accuracy": correct / total_targets,
            "metric_token_cap": metric_tokens,
        }
        if capture is not None:
            result["norm_activations"] = capture.arrays()
        if sample_logits:
            result["sample_logits"] = np.concatenate(sample_logits, axis=0)
            result["sample_targets"] = np.concatenate(sample_targets, axis=0)
        return result
    finally:
        if capture is not None:
            capture.close()


def evaluate_exact_match(torch, model, tokenizer, pairs, device, prompt_token_limit, max_new_tokens):
    if not pairs:
        return None
    hits = 0
    for prompt, target in pairs:
        tokens = tokenizer(prompt, add_special_tokens=True, truncation=True,
                           max_length=prompt_token_limit, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items() if key in ("input_ids", "attention_mask")}
        prompt_len = tokens["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(**tokens, do_sample=False, max_new_tokens=max_new_tokens)
        answer = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)
        normalize = lambda s: " ".join(s.casefold().split())
        hits += int(normalize(answer) == normalize(target))
    return {"exact_match": hits / len(pairs), "examples": len(pairs), "max_new_tokens": max_new_tokens}


def make_overlay(torch, transformers, model, groups, calibration_acts,
                 group_size, base_fp, calibration_path, heldout_path, tokenizer_fp):
    modules = dict(model.named_modules())
    packed_arrays: dict[str, np.ndarray] = {}
    fold_records = []
    tensor_records = []
    key_index = 0
    for norm_name, consumers in groups:
        x_cal = calibration_acts[norm_name].astype(np.float64, copy=False)
        original = {
            name: modules[name].weight.detach().float().cpu().numpy().T.astype(np.float64)
            for name in consumers
        }
        joined = np.concatenate([original[name] for name in consumers], axis=1)
        selection = qlab.awq_style_search(x_cal, joined, group_size)
        # Use the exact float32 scale that will be stored and later loaded.
        fold = selection.scale.astype(np.float32).astype(np.float64)
        joined_scaled = joined * fold[:, None]
        _, joined_scales = qlab.quantize_int4_group(joined_scaled, group_size)
        joined_scales = joined_scales.astype(np.float32)
        q_joined = np.empty(joined_scaled.shape, dtype=np.int8)
        for group_index in range(joined_scales.shape[0]):
            start = group_index * group_size
            end = min(start + group_size, joined_scaled.shape[0])
            q_joined[start:end] = np.clip(
                np.rint(joined_scaled[start:end] / joined_scales[group_index][None, :]), -7, 7
            ).astype(np.int8)
        joined_restored = qlab.dequantize_int4_group(q_joined, joined_scales, group_size) / fold[:, None]
        stored_scale_mse = float(np.mean(np.square(x_cal @ joined_restored - x_cal @ joined)))
        fold_key = f"fold_{key_index:04d}"
        packed_arrays[fold_key] = fold.astype(np.float32)
        fold_records.append({"norm": norm_name, "consumers": list(consumers), "scale_key": fold_key,
                             "calibration_mse_before": selection.baseline_mse,
                             "calibration_mse_after": stored_scale_mse,
                             "candidate_ratio": selection.ratio})
        for name in consumers:
            scaled_weight = original[name] * fold[:, None]
            _, scales = qlab.quantize_int4_group(scaled_weight, group_size)
            scales = scales.astype(np.float32)
            q = np.empty(scaled_weight.shape, dtype=np.int8)
            for group_index in range(scales.shape[0]):
                start = group_index * group_size
                end = min(start + group_size, scaled_weight.shape[0])
                q[start:end] = np.clip(
                    np.rint(scaled_weight[start:end] / scales[group_index][None, :]), -7, 7
                ).astype(np.int8)
            packed = qlab.pack_signed_int4(q)
            restored = qlab.dequantize_int4_group(q, scales, group_size)
            module = modules[name]
            weight_key, scale_key = f"packed_{key_index:04d}", f"scales_{key_index:04d}"
            packed_arrays[weight_key] = packed
            packed_arrays[scale_key] = scales.astype(np.float32)
            tensor_records.append({
                "name": name + ".weight",
                "logical_shape": [int(original[name].shape[1]), int(original[name].shape[0])],
                "packed_shape": [int(packed.shape[0]), int(packed.shape[1])],
                "scale_shape": [int(scales.shape[0]), int(scales.shape[1])],
                "source_dtype": str(module.weight.dtype).replace("torch.", ""),
                "fold_group": norm_name,
                "packed_key": weight_key,
                "scale_key": scale_key,
            })
            key_index += 1

    quantized_names = {record["name"] for record in tensor_records}
    folded_names = {record["norm"] + ".weight" for record in fold_records}
    folded_names.update(record["norm"] + ".bias" for record in fold_records
                        if getattr(modules[record["norm"]], "bias", None) is not None)
    float_exceptions = []
    for name, param in model.named_parameters():
        if name not in quantized_names and name not in folded_names:
            float_exceptions.append({"name": name, "shape": list(param.shape),
                                     "dtype": str(param.dtype).replace("torch.", ""),
                                     "storage": "unchanged in the local base checkpoint"})
    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    tied = bool(input_embedding is not None and output_embedding is not None and
                input_embedding.weight.data_ptr() == output_embedding.weight.data_ptr())
    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "artifact_kind": "non-deployable, base-checkpoint-bound teaching overlay",
        "architecture": "LlamaForCausalLM",
        "model_type": model.config.model_type,
        "model_config": {name: getattr(model.config, name) for name in (
            "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "max_position_embeddings", "rope_theta", "tie_word_embeddings",
            "attention_bias", "mlp_bias",
        ) if getattr(model.config, name, None) is not None},
        "base_model_fingerprint": base_fp,
        "tokenizer_fingerprint": tokenizer_fp,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "quantization": {
            "bits": 4,
            "scheme": "symmetric_groupwise_round_to_nearest_even_after_shared_activation-aware fold",
            "group_size": group_size,
            "logical_weight_orientation": "O,I",
            "math_weight_orientation": "I,O",
            "packed_orientation": "O,ceil(I/2)",
            "nibble_order": "low-index-first",
            "scale_orientation": "ceil(I/group_size),O",
            "runtime_execution": "dequantized floating torch.nn.Linear reference",
        },
        "tensors": tensor_records,
        "float_exceptions": float_exceptions,
        "weight_sharing": {"input_embedding_and_lm_head_tied": tied,
                            "lm_head_quantized": False,
                            "reason": "preserve embedding/output tying and keep the output projection as an explicit FP exception"},
        "fold_groups": fold_records,
        "data_fingerprints": {"calibration": file_fingerprint(calibration_path),
                              "heldout": file_fingerprint(heldout_path)},
        "preprocessing": {"format": "one UTF-8 text example per non-empty line",
                          "tokenizer": "local AutoTokenizer; add_special_tokens=True; right padding",
                          "activation_mask": "attention_mask == 1 only",
                          "heldout_layer_mse": "valid normalized rows captured from the original FP model"},
        "metrics": {"calibration_shared_group_search": [
            {"norm": row["norm"], "mse_before": row["calibration_mse_before"],
             "mse_after": row["calibration_mse_after"], "candidate_ratio": row["candidate_ratio"]}
            for row in fold_records
        ]},
        "layout_transform": "W[O,I] -> W_math[I,O] -> packed[O,ceil(I/2)]; dequantized back to W[O,I]",
    }
    return packed_arrays, validate_manifest(manifest)


def load_overlay(path: Path, base_fp: str, tokenizer_fp: str):
    if not path.is_file():
        raise FileNotFoundError(f"overlay not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "manifest" not in archive.files:
            raise ValueError("NPZ has no manifest")
        manifest = validate_manifest(json.loads(str(archive["manifest"].item())))
        if manifest["base_model_fingerprint"] != base_fp:
            raise ValueError("overlay was created from a different local model checkpoint")
        if manifest["tokenizer_fingerprint"] != tokenizer_fp:
            raise ValueError("overlay was created with a different local tokenizer")
        arrays = {name: archive[name].copy() for name in archive.files if name != "manifest"}
    validate_overlay_arrays(arrays, manifest)
    return arrays, manifest


def validate_overlay_for_model(torch, model, arrays, manifest):
    """Validate every array and target before apply_overlay mutates any parameter."""
    validate_overlay_arrays(arrays, manifest)
    modules = dict(model.named_modules())
    seen_norms: set[str] = set()
    for group in manifest["fold_groups"]:
        norm_name = group["norm"]
        if norm_name in seen_norms:
            raise ValueError("duplicate norm or fold-scale key in overlay")
        seen_norms.add(norm_name)
        norm = modules.get(norm_name)
        if norm is None or not isinstance(getattr(norm, "weight", None), torch.Tensor):
            raise ValueError(f"overlay fold target is missing: {norm_name}")
        fold = arrays[group["scale_key"]]
        if tuple(fold.shape) != tuple(norm.weight.shape):
            raise ValueError(f"{norm_name}: fold scale width does not match norm.weight")

    seen_targets: set[str] = set()
    for record in manifest["tensors"]:
        name = record["name"][:-len(".weight")]
        if name in seen_targets:
            raise ValueError(f"duplicate quantized target: {name}")
        seen_targets.add(name)
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(f"overlay target is not a Linear module: {name}")
        out_features, in_features = record["logical_shape"]
        if list(module.weight.shape) != [out_features, in_features]:
            raise ValueError(f"base model shape mismatch for {name}.weight")


def measure_heldout_layer_mse(model, heldout_acts, arrays, manifest):
    """Compare each packed/dequantized Linear to FP on original heldout norm rows."""
    modules = dict(model.named_modules())
    fold_by_norm = {row["norm"]: arrays[row["scale_key"]].astype(np.float64) for row in manifest["fold_groups"]}
    result = {}
    group_size = manifest["quantization"]["group_size"]
    for record in manifest["tensors"]:
        module_name = record["name"][:-len(".weight")]
        norm_name = record["fold_group"]
        x = heldout_acts[norm_name].astype(np.float64, copy=False)
        original = modules[module_name].weight.detach().float().cpu().numpy().T.astype(np.float64)
        q = qlab.unpack_signed_int4(arrays[record["packed_key"]], original.shape[0])
        restored = qlab.dequantize_int4_group(q, arrays[record["scale_key"]], group_size)
        reference_y = x @ original
        candidate_y = (x / fold_by_norm[norm_name][None, :]) @ restored
        bias = modules[module_name].bias
        if bias is not None:
            b = bias.detach().float().cpu().numpy().astype(np.float64)
            reference_y += b[None, :]
            candidate_y += b[None, :]
        result[module_name] = float(np.mean(np.square(candidate_y - reference_y)))
    return result


def apply_overlay(torch, model, arrays, manifest):
    validate_overlay_for_model(torch, model, arrays, manifest)
    modules = dict(model.named_modules())
    group_size = manifest["quantization"]["group_size"]
    with torch.no_grad():
        for group in manifest["fold_groups"]:
            norm = modules[group["norm"]]
            fold = torch.as_tensor(arrays[group["scale_key"]], device=norm.weight.device, dtype=norm.weight.dtype)
            norm.weight.div_(fold)
            if getattr(norm, "bias", None) is not None:
                norm.bias.div_(fold)
        for record in manifest["tensors"]:
            name = record["name"][:-len(".weight")]
            module = modules.get(name)
            if not isinstance(module, torch.nn.Linear):
                raise ValueError(f"overlay target is not a Linear module: {name}")
            out_features, in_features = record["logical_shape"]
            if list(module.weight.shape) != [out_features, in_features]:
                raise ValueError(f"base model shape mismatch for {name}.weight")
            q = qlab.unpack_signed_int4(arrays[record["packed_key"]], in_features)
            restored_io = qlab.dequantize_int4_group(q, arrays[record["scale_key"]], group_size)
            value = torch.as_tensor(restored_io.T.copy(), device=module.weight.device, dtype=module.weight.dtype)
            with torch.no_grad():
                module.weight.copy_(value)


def update_manifest_metrics(manifest, reference, candidate, task_reference, task_candidate):
    ref_logits = reference.get("sample_logits")
    cand_logits = candidate.get("sample_logits")
    compare = {}
    if ref_logits is not None and cand_logits is not None:
        n = min(ref_logits.shape[0], cand_logits.shape[0])
        delta = cand_logits[:n].astype(np.float64) - ref_logits[:n].astype(np.float64)
        denom = float(np.linalg.norm(ref_logits[:n].astype(np.float64)))
        compare = {
            "sampled_heldout_logits_tokens": n,
            "sampled_heldout_logits_mse": float(np.mean(delta * delta)),
            "sampled_heldout_logits_max_abs": float(np.max(np.abs(delta))),
            "sampled_heldout_logits_relative_l2": float(np.linalg.norm(delta) / max(denom, 1e-30)),
            "sampled_heldout_top1_agreement": float(np.mean(
                ref_logits[:n].argmax(axis=-1) == cand_logits[:n].argmax(axis=-1)
            )),
        }
    manifest["metrics"].update({
        "heldout_reference": {k: v for k, v in reference.items() if not k.startswith("sample_") and k != "norm_activations"},
        "heldout_quantized": {k: v for k, v in candidate.items() if not k.startswith("sample_") and k != "norm_activations"},
        "heldout_logit_comparison": compare,
        "optional_task_reference": task_reference,
        "optional_task_quantized": task_candidate,
    })


def save_overlay(path: Path, arrays: dict[str, np.ndarray], manifest: dict[str, Any]):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**arrays, "manifest": np.asarray(json.dumps(manifest, sort_keys=True))}
    with path.open("xb") as stream:
        np.savez_compressed(stream, **payload)
        stream.flush()


def run_task(torch, model, tokenizer, pairs, device, max_seq_len, max_new_tokens):
    model.eval()
    return evaluate_exact_match(torch, model, tokenizer, pairs, device, max_seq_len, max_new_tokens)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="existing local HF Llama checkpoint directory")
    parser.add_argument("--heldout-file", type=Path, required=True, help="UTF-8 file: one independent example per non-empty line")
    parser.add_argument("--calibration-file", type=Path, help="required when creating a new overlay; never used for evaluation")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--calibration-tokens", type=int, default=256, help="valid activation rows captured per norm site")
    parser.add_argument("--layer-metric-tokens", type=int, default=32, help="heldout valid rows used for per-layer MSE")
    parser.add_argument("--metric-tokens", type=int, default=4096, help="maximum heldout next-token targets")
    parser.add_argument("--logit-samples", type=int, default=8, help="bounded heldout vocab-logit rows for MSE comparison")
    parser.add_argument("--task-file", type=Path, help="optional local JSONL, each row has prompt and target for greedy exact match")
    parser.add_argument("--task-max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or an explicit torch device")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--save", type=Path, help="write a new NPZ overlay; existing paths are not overwritten")
    parser.add_argument("--load", type=Path, help="load an overlay against this exact local base model/tokenizer")
    args = parser.parse_args(argv)
    if args.group_size <= 0 or args.batch_size <= 0 or args.max_seq_len < 2:
        parser.error("group-size/batch-size must be positive and max-seq-len at least 2")
    if min(args.calibration_tokens, args.layer_metric_tokens, args.metric_tokens,
           args.logit_samples, args.task_max_new_tokens) <= 0:
        parser.error("token limits and task generation length must be positive")
    if args.save and args.load:
        parser.error("choose --save for a newly calibrated overlay or --load for an existing overlay")
    if not args.load and not args.calibration_file:
        parser.error("--calibration-file is required unless --load is used")

    torch, transformers, tokenizer, model, device, dtype = load_runtime(args.model_dir, args.device, args.dtype)
    groups = validate_topology(model, torch)
    model_dir = args.model_dir.resolve()
    base_fp = local_model_fingerprint(model_dir)
    tokenizer_fp = local_tokenizer_fingerprint(model_dir)
    heldout = read_lines(args.heldout_file)
    heldout_fp = file_fingerprint(args.heldout_file)
    calibration = read_lines(args.calibration_file) if args.calibration_file else None
    calibration_fp = file_fingerprint(args.calibration_file) if args.calibration_file else None
    if calibration is not None:
        check_disjoint(calibration, heldout)
        if calibration_fp == heldout_fp:
            raise ValueError("calibration and heldout files must have different fingerprints")
    task_pairs = read_task_pairs(args.task_file)
    calibration_set = set(calibration or ())
    heldout_set = set(heldout)
    for prompt, target in task_pairs:
        if prompt in heldout_set or prompt in calibration_set or target in calibration_set:
            raise ValueError("task examples must not duplicate calibration/heldout lines")

    max_positions = getattr(model.config, "max_position_embeddings", args.max_seq_len)
    if args.max_seq_len > max_positions:
        raise ValueError(f"max-seq-len {args.max_seq_len} exceeds model limit {max_positions}")
    task_prompt_limit = min(args.max_seq_len, max_positions - args.task_max_new_tokens)
    if task_pairs and task_prompt_limit < 1:
        raise ValueError("task-max-new-tokens leaves no room for a prompt within the model context limit")
    print(json.dumps({"architecture": model.__class__.__name__, "transformers": transformers.__version__,
                      "torch": torch.__version__, "device": str(device), "dtype": str(dtype),
                      "local_files_only": True, "trust_remote_code": False,
                      "base_model_fingerprint": base_fp}, sort_keys=True))

    if args.load:
        arrays, manifest = load_overlay(args.load, base_fp, tokenizer_fp)
        if heldout_fp == manifest["data_fingerprints"]["calibration"]:
            raise ValueError("heldout evaluation file fingerprint matches the overlay's calibration data")
        if args.task_file and file_fingerprint(args.task_file) == manifest["data_fingerprints"]["calibration"]:
            raise ValueError("task evaluation file fingerprint matches the overlay's calibration data")
        old_mse = manifest["metrics"].pop("heldout_layer_mse_current", None)
        old_mse_fp = manifest["metrics"].pop("heldout_layer_mse_data_fingerprint", None)
        if old_mse is None:
            old_mse = manifest["metrics"].pop("heldout_layer_mse", None)
        if old_mse is not None:
            manifest["metrics"]["original_artifact_heldout_layer_mse"] = {
                "data_fingerprint": old_mse_fp or manifest["data_fingerprints"]["heldout"],
                "by_tensor": old_mse,
            }
        validate_overlay_for_model(torch, model, arrays, manifest)

    reference = evaluate(torch, model, tokenizer, heldout, args.batch_size, args.max_seq_len,
                         device, groups, args.metric_tokens, args.logit_samples,
                         capture_limit=args.layer_metric_tokens)
    task_reference = run_task(torch, model, tokenizer, task_pairs, device, task_prompt_limit,
                              args.task_max_new_tokens) if task_pairs else None

    if not args.load:
        calibration_acts = collect_norms(torch, model, tokenizer, calibration, groups,
                                         args.batch_size, args.max_seq_len, device, args.calibration_tokens)
        arrays, manifest = make_overlay(
            torch, transformers, model, groups, calibration_acts,
            args.group_size, base_fp,
            args.calibration_file, args.heldout_file, tokenizer_fp,
        )
        manifest["data_fingerprints"]["task"] = file_fingerprint(args.task_file) if args.task_file else None
        # Preserve base-file fingerprints before mutating the in-memory teaching model.
        manifest["source"] = {"model_dir_label": model_dir.name,
                              "calibration_file_fingerprint": calibration_fp,
                              "heldout_file_fingerprint": heldout_fp}
    validate_overlay_for_model(torch, model, arrays, manifest)
    layer_mse = measure_heldout_layer_mse(
        model, reference["norm_activations"], arrays, manifest
    )

    apply_overlay(torch, model, arrays, manifest)
    candidate = evaluate(torch, model, tokenizer, heldout, args.batch_size, args.max_seq_len,
                         device, groups, args.metric_tokens, args.logit_samples)
    task_candidate = run_task(torch, model, tokenizer, task_pairs, device, task_prompt_limit,
                              args.task_max_new_tokens) if task_pairs else None
    update_manifest_metrics(manifest, reference, candidate, task_reference, task_candidate)
    manifest["metrics"]["heldout_layer_mse_current"] = layer_mse
    manifest["metrics"]["heldout_layer_mse_data_fingerprint"] = heldout_fp
    manifest["metrics"]["evaluation_heldout_file_fingerprint"] = heldout_fp
    manifest["metrics"]["evaluation_task_file_fingerprint"] = file_fingerprint(args.task_file) if args.task_file else None
    validate_manifest(manifest)

    print(json.dumps({"heldout_reference": {k: v for k, v in reference.items() if k != "norm_activations" and not k.startswith("sample_")},
                      "heldout_quantized": {k: v for k, v in candidate.items() if not k.startswith("sample_")},
                      "heldout_layer_mse_current_data": heldout_fp,
                      "heldout_layer_mse_current": layer_mse,
                      "heldout_logit_comparison": manifest["metrics"]["heldout_logit_comparison"],
                      "task_reference_exact_match": task_reference,
                      "task_quantized_exact_match": task_candidate}, sort_keys=True))
    if args.save:
        save_overlay(args.save, arrays, manifest)
        print(f"saved teaching overlay: {args.save.resolve()}")
    if args.load:
        print(f"loaded teaching overlay: {args.load.resolve()}")
    print("Execution used dequantized floating Linear weights; no packed INT4 kernel/performance claim is made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
~~~
</details>

## 5. 性能优化：省下的数据去了哪里

对纯分块乘 scale，至少需要读取 X、写入 Y；若均为 FP32，有效 payload 是 8MN 字节。scale 的最小独立数据量近似为 4ceil(M/G)ceil(N/G)，但实际会被重复请求，也可能命中缓存。有效 GB/s 不能直接当作 DRAM counter。

CUDA/Triton 调参先看数据是否合并访问，再看一个 scale 是否被合理复用。把 program 变大可能减少调度，却增加活跃值和尾部浪费；换成二维 tile 可以对齐 scale block，但不保证权重矩阵指令布局也随之最优。

| 现象 | 有依据的改动方向 | 应观察的结果 |
|---|---|---|
| 反量化写出完整大矩阵 | 融合到消费它的 GEMM tile | 中间 global 写读减少，转换指令和寄存器可能增加 |
| scale 读取或布局转换显著 | 对齐量化块与消费布局，避免重复重排 | scale/cache 事务、转换 kernel、端到端时间变化 |
| 小 M Decode 吃不满矩阵管线 | 比较 GEMV/小 M GEMM、batch、权重复用 | 吞吐、带宽、launch 时间，而不只看 occupancy |
| 更低位反而更慢 | 分离解包、dequant、计算与临时存储成本 | 定位节省带宽是否被额外指令抵消 |

误差也按阶段检查：编码→反量化→单层输出→模型质量。未饱和值可以检查舍入误差界；整体误差可看绝对/相对误差与均方误差；长序列 Attention 或 MoE 路由还要检查最终行为。所谓“INT8 通常无损”不能代替这些验证。

## 6. 实践：从 scale 寻址到量化 GEMM

### LeetGPU：正确性与代码归档

在 [LeetGPU 题库](https://leetgpu.com/challenges)中先做 **Weight Dequantization（#64）**。题面输入 X[M,N] 与二维 S，输出 Y[M,N]，均为 FP32；函数参数包含 M、N、TILE_SIZE。它验证上面的 scale 寻址，不验证 INT4 解包。可先用 3×3、TILE_SIZE=2 手算，再测试非整除形状、负 scale 和零输入。完整逻辑学会后从空白题面写，不把服务器包装当作原始提交版本。

接着做 **INT8 Quantized MatMul（#32）**，区分题目采用的 A[M,K]B[K,N] 与本文的 X[T,I]W[I,O]：题目 K 是本文的 I。它带输入/输出 scale 和 zero point，最终输出 INT8，要求精确匹配。公开参考使用 FP32 matmul、round、缩放、再量化；一个理想 INT32 参考不保证在所有范围和舍入边界与之逐位相同，必须按题面算术顺序验证。

CPU 检查先验证全部 256 组 INT4 双元素组合、奇数尾部、零点修正和 scale 地址：

~~~bash
python roadmap/curriculum/operators/07-quantized-operators/examples/semantics.py
~~~

先在 CPU 上运行数值与布局检查：

~~~bash
python roadmap/curriculum/operators/07-quantized-operators/examples/quantization_lab.py
python roadmap/curriculum/operators/07-quantized-operators/examples/quantization_lab.py --demo
python roadmap/curriculum/operators/07-quantized-operators/examples/w8a8_reference.py
~~~

测试包含 mask、零通道、全部 INT4 编码对、奇数长度、分组尾部、缩放恒等式、独立 GPTQ 参考和分块更新。demo 输出校准与独立评估误差；这些数字不计作 GPU 性能。

### 服务器：真实性能

基础题通过并保存原始代码后，再运行本章 GPU 基线：

~~~bash
python roadmap/curriculum/operators/validate_baselines.py quant
python roadmap/curriculum/operators/validate_baselines.py quant --benchmark
~~~

比较使用同一份输入、同一 scale 和 FP32 输出。正确性包含小矩阵与非整除块；计时选用可整除形状，让 PyTorch 通过 reshape+broadcast 相乘，不先物化完整 scale 矩阵。这样不会把不必要的参考实现开销误算成加速。

脚本报告的是含输出分配、host 包装与 launch 的重复调用时间；纯 kernel 时间另用 profiler 检查。这里的 block-scale kernel 只是一条基础实现，INT4 GEMM、原生低精度指令和模型质量需要各自的实验，不能由这项成绩代替。

#### packed INT4 矩阵乘的现场实验

环境需要 PyTorch、Triton 和兼容的 CUDA GPU。先运行正确性检查，再启用计时：

~~~bash
python roadmap/curriculum/operators/07-quantized-operators/examples/packed_int4_gpu.py
python roadmap/curriculum/operators/07-quantized-operators/examples/packed_int4_gpu.py --benchmark
~~~

正确性输入包括 (T,I,O,G)=(1,1,1,1)、(3,5,7,3)、(17,65,33,16)、(32,128,64,32)，覆盖奇数输入长度、量化组跨字节和输出尾块。参考权重由 NumPy 按相同 scale 精度还原为 FP16，再计算 FP32 参考并转为 FP16 输出。

脚本在对照期间关闭 TF32 和 FP16 reduced-precision reduction，结束后恢复设置。不同归约顺序仍可能产生浮点差异，因此同时检查有限性和明确的误差阈值。

计时使用两种形状：T=1 与 T=128，固定 I=1024、O=768、G=128。每个计时形状先分别验证以下三条路径：

| 路径 | 计入的工作 | 作用 |
|---|---|---|
| packed Triton | tile 内解包、反量化与矩阵乘 | 检查融合计算路径 |
| PyTorch 解包后矩阵乘 | 解包、物化 FP16 权重与矩阵乘 | 同一量化权重的完整参考路径 |
| 预先反量化后 torch.mm | 只计矩阵乘，不计解包 | 观察普通 GEMM 的时间基准 |

三条路径都包含输出分配；输入和 scale 在计时前检查，避免把每次 .item() 同步混进热路径。第三条省略了反量化，其时间不能直接作为同范围加速比的分母。

这组实现固定了布局与精度，便于继续比较输出 tile、归约块和数据复用。更复杂的 packed layout、专门的解包指令或量化算法变更应分别做对照，避免一次同时改变多个因素。


## 参考阅读

- NVIDIA：[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) · [TensorRT 量化格式](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html) · [Transformer Engine 低精度](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- Hugging Face：[Transformers v4.48.0 Auto Classes](https://huggingface.co/docs/transformers/v4.48.0/en/model_doc/auto) · [Transformers v4.48.0 Llama 文档](https://huggingface.co/docs/transformers/v4.48.0/en/model_doc/llama) · [当前 AWQ 加载文档](https://huggingface.co/docs/transformers/quantization/awq) · [量化 API](https://huggingface.co/docs/transformers/main_classes/quantization)
- AWQ：[原论文](https://arxiv.org/abs/2306.00978) · [作者实现](https://github.com/mit-han-lab/llm-awq)
- GPTQ：[原论文](https://arxiv.org/abs/2210.17323) · [作者实现](https://github.com/IST-DASLab/gptq)
- SmoothQuant：[原论文](https://arxiv.org/abs/2211.10438) · [作者实现](https://github.com/mit-han-lab/smoothquant)

## 章节导航

- [上一章：Decode / PagedAttention](../06-decode-paged-attention/README.md)
- [下一章：MoE 算子](../08-moe/README.md)
