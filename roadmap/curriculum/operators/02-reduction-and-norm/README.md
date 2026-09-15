# 第二章 并行归约、Softmax 与归一化

一行有 1000 个数，分给 256 个线程后，每个线程先得到局部统计量，再由 warp 和线程块合并。Softmax、RMSNorm 和 LayerNorm 随后把这些统计量用于整行输出。

对 [R,D] 张量，归约沿 D 进行。尾部补值、同步位置、累加精度和分母计数共同决定结果；选择一行一个 program 或多阶段归约，则影响并行度与数据复用。

## 1. 先固定数学语义：`[R, D]`、归约轴和广播

设输入是二维 row-major 张量 $X\in\mathbb{R}^{R\times D}$。行索引为 $r\in[0,R)$，列索引为 $c\in[0,D)$，物理地址（以元素为单位）是

$$
\operatorname{offset}(r,c)=r\times D+c.
$$

这里的 `row` 不是线程，也不是一个 CUDA block；它只是逻辑张量的第一个轴。对最后一维做归约，写作

$$
s_r=\sum_{c=0}^{D-1}X_{r,c},
$$

输出形状是 `[R]`。如果对第 0 维归约，则是 $t_c=\sum_{r=0}^{R-1}X_{r,c}$，输出形状是 `[D]`，线程划分、访问连续性和跨 CTA 合并都随之改变。本章默认的是 **row-wise reduction**：每一行独立，归约轴是列轴，输出或者统计量按行产生。

广播不是“自动复制一份数组”这么简单。若 `mean` 是 `[R]`，在 LayerNorm 中它要解释为 `mean[r]`，并广播到所有 `c`；若 `gamma` 是 `[D]`，它解释为 `gamma[c]`，广播到所有 `r`。例如

$$
Y_{r,c}=\gamma_c\frac{X_{r,c}-\mu_r}{\sqrt{\sigma_r^2+\varepsilon}}+\beta_c.
$$

一个 kernel 里，`mu_r` 可能是某个线程或某个 CTA 先算出的标量，`gamma_c` 则是每个有效列各自加载的向量。两者的“广播方向”完全不同。把 `[R]` 当成 `[D]`，代码仍可能编译，结果却只在 `R=D` 的偶然测试上看不出来。

### 1.1 归约轴决定地址函数

对 contiguous `[R,D]` 做 row-wise reduction，一个线程处理列方向的跨步位置：

~~~text
row r:  X[r, 0], X[r, 1], ... X[r, D-1]
thread tid: c = tid, tid + BLOCK, tid + 2*BLOCK, ...
~~~

同一个 warp 的相邻 lane 访问相邻列，因而天然适合合并读取。若做 column-wise reduction 且把相邻 lane 映射到不同的行，同一列的地址才会相隔 `D` 个元素；若仍按列连续映射 lane，访问行为可能完全不同。不能把归约轴单独等同于某一种 coalescing 结果，必须写出 lane 到坐标的映射。

`D` 不一定等于线程数。以 `D=1000, BLOCK_SIZE=256` 为例，线程 `tid=0..231` 分别读取列 `tid、tid+256、tid+512、tid+768`，每个线程 4 个有效元素；`tid=232..255` 读取 `tid、tid+256、tid+512`，只有 3 个有效元素。不能写成“所有线程固定读 4 项”然后忘记最后 24 个线程的第四次访问已经超出 `D`。正确实现要么在第四次加载上做有效性判断，要么把逻辑迭代上界 round up 后让尾部变成加 0。

### 1.2 四个执行层次第一次同时出现

- **thread（线程）**：拥有自己的寄存器和局部累加器，负责一组列。
- **warp（线程束）**：通常由 32 个线程组成，以 SIMT 指令组执行；shuffle 能在同一 warp 的寄存器之间交换值。
- **CTA（Cooperative Thread Array，协作线程阵列）**：CUDA 编程模型里的 block；一个 CTA 内的线程可以通过 shared memory 和 block barrier 协作。
- **grid（网格）**：所有 CTA 的集合。不同 CTA 默认没有隐式 barrier；跨 CTA 的归约需要第二个 kernel、原子操作或显式的多阶段协议。

因此，“一个 row 一个 block”是一种映射，不是数学定理；“一个 row 一个 warp”也是映射。先决定一行的统计量需要多少协作，再决定它由 thread、warp 还是 CTA 承担。

## 2. CUDA 归约的两级结构：寄存器、shuffle、shared

对一行由一个 CTA 负责的设计，归约按以下顺序落地：

$$
\text{thread-private sum}
\rightarrow
\text{warp reduction by shuffle}
\rightarrow
\text{one value per warp in shared}
\rightarrow
\text{warp 0 reduction}.
$$

第一步每个线程只写自己的寄存器 `local`，没有线程间通信；第二步 32 个 lane 交换寄存器值；第三步每个 warp 的 lane 0 把结果写入 shared；第四步由 warp 0 读取这些少量结果并再次用 shuffle 归约。对于 128 线程 CTA，有 4 个 warp；对于 256 线程 CTA，有 8 个 warp。shared 只保存 4 或 8 个 float，而不是保存整行。

### 2.1 `__shfl_down_sync` 的两个条件

`__shfl_down_sync(mask, value, delta)` 让一个 lane 读取同一 warp 内另一个 lane 的寄存器值。它不是 shared load，也不是 barrier。安全使用 `0xffffffffu`（FULL mask）至少需要两个事实同时成立：

1. 这条 shuffle 指令所在的控制流由 warp 内所有 32 个 lane 执行；
2. 被 mask 声明的源 lane 也确实执行了同一条指令，并且提供了有效寄存器值。

本章的 `BLOCK_SIZE=128/256` 都是完整 warp 数，第一层归约的每个 warp 32 个 lane 都执行循环；第二层只放在 `warp == 0` 分支内，但 warp 0 本身仍是完整的 32 个 lane。因此 FULL mask 合法。若只让 `lane < active_lanes` 执行 shuffle，却仍写 FULL mask，或者因为尾部线程提前 return，行为就不再有这个保证。`width` 可以把一个 warp 切成独立子组，但不等于为任意活跃线程集合创建了同步协议。

### 2.2 shuffle 和 barrier 不能互换

同一 warp 的 shuffle 用于寄存器交换；它不为其他 warp 的 shared 写入提供可见性，也不等待 CTA 中的其他 warp。于是典型顺序必须是：每个 warp 先完成自己的 shuffle，lane 0 写 `warp_sum[warp]`，然后所有 CTA 线程执行 `__syncthreads()`，最后 warp 0 才读取 shared。CUDA 手册的 §5.4.4.1 明确了 `__syncthreads()` 的等待和排序语义；PDF 第 577 页是本地版本的精确页码。

不能用“只有 warp 0 读 shared”来删掉 barrier。warp 0 读的是其他 warp 的写入，不是自己的寄存器；没有 barrier，代码缺少跨 warp 的 happens-before。反过来，不能把 shuffle 当成全 CTA barrier：它不能让 warp 1 等待 warp 2，也不能排序 shared/global memory。

### 2.3 为什么本章不使用 `__reduce_add_sync`

CUDA Programming Guide §5.4.6.4（PDF 第 599–600 页）给出的 `__reduce_add_sync` 支持范围包括整数类型，并要求 Compute Capability 8.x 或更高版本；它不是一个可以泛化替代 FP32 手写归约的接口。这里要处理的是 FP32 row sum，同时教学目标是把两级数据流和同步边界显式展开，所以使用 `__shfl_down_sync`。即使某一目标架构有其他 reduction 指令，也应先核对生成代码、支持的类型和误差语义，不能把 API 名字直接等同于“更快的 FP32 reduction”。

## 3. 完整 FP32 row sum：从 `D=1000` 跑一遍控制流

下面的 row_sum.cu 固定使用 `grid.x=rows`、一个 CTA 负责一行；`cols` 通过 `long long` round-up 形成固定迭代边界，越过真实列数的访问贡献 `0.0f`。launcher 拒绝空 shape、空指针、`rows*cols` 超过有符号 int32 索引范围和无效 block size，并以 `cudaError_t` 返回 launch 错误。具备 CUDA 工具链时，可用 `nvcc` 编译后再运行小 shape 和 sanitizer 验证。

<!-- source-check: examples/row_sum.cu -->
~~~cpp
template <int BLOCK_SIZE>
__global__ void row_sum_fp32(const float* x, float* y, int rows, int cols) {
    static_assert(BLOCK_SIZE == 128 || BLOCK_SIZE == 256, "use four or eight warps");
    __shared__ float warp_sum[BLOCK_SIZE / 32];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = blockIdx.x;
    const long long rounded_cols =
        ((static_cast<long long>(cols) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    float local = 0.0f;

    // The conditional load is the mask: tail elements contribute zero.
    // Do not return before the barriers; all BLOCK_SIZE threads must arrive.
    for (long long col = tid; col < rounded_cols; col += BLOCK_SIZE) {
        const bool valid = row < rows && col < cols;
        local += valid ? x[row * cols + col] : 0.0f;
    }

    // All warps are complete for BLOCK_SIZE 128 or 256. The FULL mask is
    // therefore valid here; shuffle itself does not provide a memory barrier.
    for (int delta = 16; delta > 0; delta >>= 1)
        local += __shfl_down_sync(0xffffffffu, local, delta);

    if (lane == 0) warp_sum[warp] = local;
    __syncthreads();

    // Warp zero reduces one value from each participating warp. Lanes beyond
    // the warp count contribute zero but still execute every shuffle.
    if (warp == 0) {
        float block_total = lane < BLOCK_SIZE / 32 ? warp_sum[lane] : 0.0f;
        for (int delta = 16; delta > 0; delta >>= 1)
            block_total += __shfl_down_sync(0xffffffffu, block_total, delta);
        if (lane == 0 && row < rows) y[row] = block_total;
    }
}
~~~

必要的 launcher 也作为原样片段保留，便于核对错误路径和索引范围：

<!-- source-check: examples/row_sum.cu -->
~~~cpp
extern "C" cudaError_t launch_row_sum(const float* x, float* y, int rows, int cols,
                                       int block_size) {
    if (rows <= 0 || cols <= 0 || x == nullptr || y == nullptr)
        return cudaErrorInvalidValue;
    if (static_cast<long long>(rows) * cols > 0x7fffffffLL)
        return cudaErrorInvalidValue;
    const dim3 grid(rows);
    if (block_size == 128) {
        row_sum_fp32<128><<<grid, 128>>>(x, y, rows, cols);
    } else if (block_size == 256) {
        row_sum_fp32<256><<<grid, 256>>>(x, y, rows, cols);
    } else {
        return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
~~~

对 `D=1000` 和 `BLOCK_SIZE=256`，`rounded_cols=1024`。因此 tid 0–231 的循环列为 `tid、tid+256、tid+512、tid+768`，第四项最大是 `999`；tid 232–255 的第四项分别是 `1000..1023`，条件为 false，累加 0。warp 0–6 各自先得到一个局部和，warp 7 也参与同样的第一层流程，只是它的 24 个尾部线程在第四次取 0；8 个 warp 的 lane 0 把结果写入 `warp_sum[0..7]`。barrier 后 warp 0 的 lane 0–7 读取这 8 项，lane 8–31 读 0，最终只有 warp 0 的 lane 0，也就是 CTA 的 thread 0，把结果写入 `y[row]`。

这个“全 warp 参与、无提前退出”设计有一个实际代价：尾部线程会执行指令，但不会读取越界地址。对于 row-wise reduction，尾部浪费通常比一个错误的 barrier 更容易接受；只有在测量确认尾部占主导时，才考虑改变映射。代码中只有一次 CTA 级 `__syncthreads()`：它等待所有 warp 把 partial 写入 shared。两轮 shuffle 都在完整 warp 内进行，第二轮只操作 warp 0 的寄存器，因此不需要第二次 block barrier。shuffle 本身不是 memory barrier；这里的跨 warp 可见性由那一次 `__syncthreads()` 提供。

## 4. 从已有 1D Softmax 归档代码理解 partial 合并

已有 solutions/triton/fused_softmax.py 是 LeetGPU Softmax 的 1D、三阶段归档版本。它不是本章 row-wise kernel，也不能用“已经有 Softmax 通过记录”替代 `[R,D]` row-wise 的正确性或性能验证。下面保留三个关键函数的连续原样片段；构建脚本会用 `source-check` 将正文片段与文件内容逐字核对。

第一阶段把一维输入分成若干连续 block，每个 block 输出自己的最大值和相对于该最大值的指数和：

<!-- source-check: solutions/triton/fused_softmax.py -->
~~~python
@triton.jit
def softmax_partial(
    input: torch.Tensor,
    partial_max: torch.Tensor,
    partial_sum: torch.Tensor,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    input = input.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(input + offset, mask=mask, other=-float("inf"))
    tmpMax = tl.max(x, axis=0)
    tmpSum = tl.sum(tl.exp(x - tmpMax), axis=0)
    tl.store(partial_max + pid, tmpMax)
    tl.store(partial_sum + pid, tmpSum)
~~~

这里 `other=-inf` 让尾部不改变 max，`exp(-inf)=0` 让尾部不改变 sum。注意这是对每个 block 内部的稳定化，不是最终的全局归一化；`tmpSum` 的参考点是 `tmpMax`。

第二阶段先合并 block max，再把各 block 的局部 sum 重新标度到全局 max：

<!-- source-check: solutions/triton/fused_softmax.py -->
~~~python
@triton.jit
def softmax_reduce(
    global_max: torch.Tensor,
    global_sum: torch.Tensor,
    partial_max: torch.Tensor,
    partial_sum: torch.Tensor,
    num_blocks: tl.constexpr,
    REDUCE_SIZE: tl.constexpr,
):
    offset = tl.arange(0, REDUCE_SIZE)
    mask = offset < num_blocks
    local_sum = tl.load(partial_sum + offset, mask=mask, other=float(0))
    local_max = tl.load(partial_max + offset, mask=mask, other=-float("inf"))
    tmp_max = tl.max(local_max, axis=0)
    local_sum *= tl.exp(local_max - tmp_max)
    tmp_sum = tl.sum(local_sum, axis=0)
    tl.store(global_max, tmp_max)
    tl.store(global_sum, tmp_sum)
~~~

数学上，若第 $b$ 个块的局部统计是

$$
m_b=\max_{i\in b}x_i,
\qquad q_b=\sum_{i\in b}e^{x_i-m_b},
$$

全局最大值 $M=\max_bm_b$，则

$$
\sum_i e^{x_i-M}=\sum_b q_b e^{m_b-M}.
$$

这就是代码中的 `local_sum *= exp(local_max - tmp_max)`。例如两个 block 的局部统计为 $(m_0,q_0)=(3,2)$、$(m_1,q_1)=(1,4)$，全局最大值是 $M=3$，合并和为

$$
2e^{3-3}+4e^{1-3}=2+4e^{-2}\approx2.541341.
$$

若只把 `2+4` 相加，等价于错误地假设两个局部 sum 使用同一个参考点。partial sum 不是原始指数和，不能脱离 partial max 单独合并。

第三阶段重新读取输入，用全局 max 和全局 sum 生成输出：

<!-- source-check: solutions/triton/fused_softmax.py -->
~~~python
@triton.jit
def softmax_sum(
    global_max,
    global_sum,
    input,
    output,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    input = input.to(tl.pointer_type(tl.float32))
    output = output.to(tl.pointer_type(tl.float32))
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(input + offset, mask=mask, other=-float("inf"))
    gm = tl.load(global_max)
    gs = tl.load(global_sum)
    tmp_res = tl.exp(x - gm) / gs
    tl.store(output + offset, tmp_res, mask=mask)
~~~

这份 Triton 代码的结构是“GPU partial 统计 → GPU merge → GPU normalize”。三个 kernel 都在 device 上执行：`softmax_partial` 写 partial 数组，`softmax_reduce` 读取 partial 并写 global scalar，`softmax_sum` 重新读取输入完成归一化。它是 1D global Softmax：所有 program 在同一个线性 `N` 上协作；本章后面的 row-wise Softmax 是每个 program 独立处理一行，统计量不会跨行混合。两者都使用 max subtraction，但并行域、scratch 形状和归约位置不同。

原始归档旁边的 CUDA 历史文件可以用来理解另一路径，但 `softmax_online.cu` 的 host `solve` 中有一处笔误：`blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerGrid` 使用了未定义的 `threadsPerGrid`。因此本章只引用它的核函数结构和 host merge 公式，不把它说成可直接编译或已验证，也不修改旧文件。那条旧 CUDA 路径是 GPU partial → host D2H merge → normalize kernel；host 计算出的两个 scalar 作为 kernel 参数传回，不等于把 partial 数组显式 H2D 搬回。无论如何，分块统计后重新 normalize 都意味着输入至少被读取两遍、输出写一遍，即理想 3N 个元素，再叠加 partial scratch、host 往返和 kernel launch 成本；不能把这种路径写成天然只有 2N 元素流量。

## 5. 把 1D 经验改成 row-wise Triton Softmax

本章 rowwise_softmax.py 选择最容易审计的映射：一个 Triton program 负责一行，`tl.program_id(0)` 对应 `row`，`tl.arange(0, BLOCK)` 对应列向量。`BLOCK` 取不小于 `D` 的 2 的幂，超过真实列数的 lane 由 `mask` 排除。这里的 mask 是**逻辑元素访问 mask**，决定哪些 `tl.load`/`tl.store` 位置有效；它不是 CUDA shuffle 的参与线程 mask。

这两个概念必须分开。Triton 的 `mask=offsets < cols` 可以让一个向量化 load 的无效位置返回 `-inf`，也可以让 store 不写 padding；它不意味着一个 CUDA thread 从控制流中退出。CUDA reduction 中的尾部线程仍须经过 shuffle 和 barrier，只能让它们的贡献为 0 或 `-inf`。如果把“逻辑元素无效”实现成 `return`，恰好会破坏后续 `__syncthreads()` 对全 CTA 的等待合同。

<!-- source-check: examples/rowwise_softmax.py -->
~~~python
@triton.jit
def softmax_row_kernel(x, y, rows, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < cols
    values = tl.load(x + row * cols + offsets, mask=mask, other=-float("inf"))
    values = values.to(tl.float32)
    row_max = tl.max(values, axis=0)
    numerators = tl.exp(values - row_max)
    denom = tl.sum(numerators, axis=0)
    result = numerators / denom
    tl.store(y + row * cols + offsets, result, mask=mask)
~~~

`tl.max(values, axis=0)` 是对这一个 program 的列向量做归约，得到一个 row scalar；它不是对每个元素执行 `tl.maximum`。后者只比较两个同形状向量，是逐元素运算，不会消除列轴。已有 Softmax 讨论里“max 到底在哪个轴上”的混淆，放到这里就能用 shape 检查解决：输入是 `[R,D]`，一次 program 内的 `values` 是 `[BLOCK]`，因此 `axis=0` 才是行内列归约。

max subtraction 的顺序是先求 $m_r=\max_cx_{r,c}$，再求 $q_r=\sum_ce^{x_{r,c}-m_r}$，最后输出 $e^{x_{r,c}-m_r}/q_r$。由于输入在 load 后立即转成 FP32，max、exp 输入和 sum 都以 FP32 语义进行；最终 store 到 `y` 时保持输入 dtype。对于有限输入，减去行内最大值后指数的最大输入为 0，避免直接 `exp(x)` 的上溢。

wrapper 的默认合同是 contiguous、CUDA、浮点 16/32 或 bfloat16、正的有限 `rows/cols`，并限制 `rows*cols <= INT32_MAX` 以保持指针索引简单。它默认不对每次调用做 `torch.isfinite(x).all()`，因为那会额外发起一次 GPU 归约并同步，污染 benchmark。正确做法是在 correctness preflight 里显式调用 `softmax(x, check_finite=True)` 一次，计时路径使用默认参数。对于 all-`-inf`、NaN 或空行，普通 max subtraction 的语义需要额外约定；本例拒绝非有限输入和空 shape，而不是静默产出 NaN。

## 6. RMSNorm 与 LayerNorm：同一行里统计量不同，mask 也不同

RMSNorm（Root Mean Square Normalization，均方根归一化）不减均值。对一行 $x\in\mathbb{R}^D$，带缩放参数 $\gamma$ 的 forward 是

$$
\operatorname{rms}(x)=\sqrt{\frac1D\sum_{c=0}^{D-1}x_c^2+\varepsilon},
\qquad
y_c=\gamma_c\frac{x_c}{\operatorname{rms}(x)}.
$$

LayerNorm（Layer Normalization，层归一化）先求均值，再求带偏估计的方差：

$$
\mu=\frac1D\sum_cx_c,
\qquad
\sigma^2=\frac1D\sum_c(x_c-\mu)^2,
$$

$$
y_c=\gamma_c\frac{x_c-\mu}{\sqrt{\sigma^2+\varepsilon}}+\beta_c.
$$

这里的方差分母是 `D`，即 `unbiased=False` 的 biased variance，不是统计学样本方差的 `D-1`。`eps` 放在平方根里的方差之后：`rsqrt(variance + eps)`。把它写成 `rsqrt(variance) + eps` 或除法外加 eps 都是不同的数值函数。RMSNorm 通常只有 gamma，没有 beta；LayerNorm 同时有 gamma 和 beta，二者都是 `[D]`，沿行轴广播。

### 6.1 padding 对 RMSNorm 和 LayerNorm 的影响不同

当 `D=257` 而 program 的 `BLOCK=512` 时，RMSNorm 可以把无效位置的 $x$ 补 0，因为 $0^2=0$，平方和仍等于有效 257 个元素的平方和。但 LayerNorm 不能在求均值后直接对整个 padded 向量做 `sum((values - mean)^2)`：无效位置的 `values` 已经是 0，于是这些位置贡献 $(-\mu)^2=\mu^2$，污染的是方差平方和分子；分母仍必须是逻辑列数 `D`，不能把问题描述成改用 512 作分母。以最小例子 `x=[3,4,5]`、补 0 到 `BLOCK=4` 为例，mean=4，正确平方和为 $1+0+1=2$，正确 biased variance 为 $2/3$；错误写法把 padding 的 $(0-4)^2=16$ 加进分子，得到 `(1+0+1+16)/3=6`。恒定行 `x=[3,3,3]` 的有效 centered 值全是 0，最终输出可能仍然等于 beta，所以单测常数输出抓不住这个 bug；非恒定偏置行才是必要覆盖。

正确写法是先得到 padded load 的 `values`，再构造

~~~python
centered = tl.where(mask, values - mean, 0.0)
variance = tl.sum(centered * centered, axis=0) / cols
~~~

这样只有真实列参与 centered square，分母也仍是逻辑列数 `cols`。`gamma`、`beta` 的 load 和最终 store 同样使用 `mask`。常数行是必测边界：真实方差应为 0，输出在 gamma/beta 约定下应接近 beta；如果把 padding 的 mean² 计入，`D=257` 或 `D=1000` 很快会暴露错误。

数值抵消还可以用带大偏置的小波动行观察。对 `[10000, 10001, 9999, 10002]`，均值为 `10000.5`，正确的 biased variance 是 `1.25`。如果把 `E[x²]-E[x]²` 用有限精度直接相减，两个约为 $10^8$ 的量相减可能把结果压成 0 或引入明显误差；centered square 或 Welford 能避免把这个误差藏在“方差看起来很小”的输出里。测试不能只放零均值随机数。

### 6.2 教学 forward 实现与稳定统计

下面是 rowwise_norms.py 中两个 forward kernel 的原样片段；该文件的 wrapper 是 forward-only：没有实现 backward，也不把前向正确性写成训练路径已经覆盖。RMSNorm kernel 直接对补零后的有效值做平方和，gamma 的加载和输出 store 都有列 mask：

<!-- source-check: examples/rowwise_norms.py -->
~~~python
@triton.jit
def rms_norm_kernel(x, weight, y, rows, cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < cols
    values = tl.load(x + row * cols + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(values * values, axis=0) / cols
    inv_rms = tl.rsqrt(mean_square + eps)
    result = values * inv_rms * gamma
    tl.store(y + row * cols + offsets, result, mask=mask)
~~~

<!-- source-check: examples/rowwise_norms.py -->
~~~python
@triton.jit
def layer_norm_kernel(x, weight, bias, y, rows, cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < cols
    values = tl.load(x + row * cols + offsets, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(bias + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(values, axis=0) / cols
    centered = tl.where(mask, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / cols
    normalized = centered * tl.rsqrt(variance + eps)
    result = normalized * gamma + beta
    tl.store(y + row * cols + offsets, result, mask=mask)
~~~

这份实现采用“两遍统计”的直观写法：第一遍逻辑上得到 mean，第二遍用 centered 值得到 variance，然后变换并写回。`E[x^2]-E[x]^2` 只需要一次平方和，但当 $x$ 的均值较大而方差很小时，两个接近的大数相减会发生严重数值抵消；FP32 也不是无限精度。教学版本使用 centered square，代价是需要保留或重新获得 mean 后的值，但数值语义更清楚。

更一般的并行实现可以用 Welford 状态 $(n,\mu,M_2)$：$n$ 是段内样本数，$\mu$ 是这段样本的均值，$M_2=\sum_i(x_i-\mu)^2$ 是以该段均值为中心的平方和。一个元素初始化为 $(1,x,0)$。两段状态 A、B 合并时，必须先处理空状态：若 $n_A=0$，直接返回 B；若 $n_B=0$，直接返回 A。只有两段都非空时，才令 $\delta=\mu_B-\mu_A$、$n=n_A+n_B$ 并计算：

$$
\mu=\mu_A+\delta\frac{n_B}{n},
\qquad
M_2=M_{2A}+M_{2B}+\delta^2\frac{n_A n_B}{n}.
$$

若两个输入状态都为空，空状态仍表示为 $(0,0,0)$，不能执行除以 $n$，也不能先计算 `delta * delta * nA * nB / n` 再期待 `0*inf` 自动安全。最终 biased variance 是 $M_2/D$，而不是 $M_2/(D-1)$。在 block reduction 中，padding lane 不生成状态，或者生成 `n=0` 的空状态；这比把补零当作真实样本再事后修正更不容易出错。

合并式中的修正项来自同一个新均值。对 A 段，把每个样本相对新均值 $\mu$ 展开：

$$
\sum_{i\in A}(x_i-\mu)^2
=M_{2A}+n_A(\mu_A-\mu)^2,
$$

因为 $\sum_{i\in A}(x_i-\mu_A)=0$，展开后的交叉项消失。B 段同理；再代入 $\mu_A-\mu=-\delta n_B/n$、$\mu_B-\mu=\delta n_A/n$，两个均值偏移项相加为

$$
n_A\frac{\delta^2n_B^2}{n^2}+n_B\frac{\delta^2n_A^2}{n^2}
=\delta^2\frac{n_An_B}{n},
$$

这就得到两段合并公式中的第三项。

forward 之外，LayerNorm 的反向也沿列轴归约，但归约对象不同。对 $X\in\mathbb{R}^{R\times D}$，设 $g_{r,c}=\partial L/\partial Y_{r,c}$、$u_{r,c}=g_{r,c}\gamma_c$、$\hat{x}_{r,c}=(x_{r,c}-\mu_r)\operatorname{invstd}_r$，则每一行的输入梯度为

$$
\frac{\partial L}{\partial x_{r,c}}
=\operatorname{invstd}_r
\left(u_{r,c}-\operatorname{mean}_{j}(u_{r,j})
-\hat{x}_{r,c}\operatorname{mean}_{j}(u_{r,j}\hat{x}_{r,j})\right).
$$

这里两个 `mean` 都沿当前 row 的列 $j\in[0,D)$ 计算。参数梯度则沿 row 轴归约，输出仍是 `[D]`：

$$
\frac{\partial L}{\partial\gamma_c}=\sum_{r=0}^{R-1}g_{r,c}\hat{x}_{r,c},
\qquad
\frac{\partial L}{\partial\beta_c}=\sum_{r=0}^{R-1}g_{r,c}.
$$

RMSNorm backward 同样需要一行内的点积统计。为避免与输入行索引 $r$ 混淆，令 $\rho=(D^{-1}\sum_cx_c^2+\varepsilon)^{-1/2}$，$h_c=g_c\gamma_c$，则

$$
\frac{\partial L}{\partial x_c}
=\rho\left(h_c-x_c \rho^2\frac1D\sum_jh_jx_j\right).
$$

这说明 backward 不是把 forward 的 store 再跑一遍：`dx` 的两个统计量沿每个 row 的列归约，而 `dgamma/dbeta` 沿所有 row 归约到 `[D]`。当前示例只覆盖 forward，任何训练 benchmark 都必须另行实现和验证。

## 7. 形状决定映射：短行、多行、中行和大行

没有一个 `num_warps` 能对所有 reduction shape 都正确优化。可以先按行长和行数建立粗粒度决策，再用 metadata、计时和 profiler 修正。

### 7.1 很短的行：一个 warp 包多行

若 `D<=32`，让一个 warp 处理一行会有大量 lane 空闲。可以在 `R` 足够多、每行很短的前提下，让一个 CTA 中的不同 warp 或 sub-group 同时处理多行，摊薄 CTA 固定开销并提高一个 CTA 内的 lane 利用率。它不会解决 `R` 很小造成的总工作不足；如果本来只有很少几行，把多行塞进同一个 CTA 反而会减少 grid 中的 CTA 数。代价是 lane-to-row 映射、mask 和结果写回更复杂。

### 7.2 中等行：一行一个 CTA

`D` 从几十到几千时，一行一个 CTA 是直接的基线。每个 thread 负责若干列，warp shuffle 做局部归约，shared 只存 warp partial。当前教学 kernel 要求 `BLOCK>=D`，所以 `D=257` 只能与 `BLOCK=512` 对照；`D=257,BLOCK=256` 会漏掉第 257 列，不能作为同一 kernel 的合法配置。padding 提供规则的编译布局，却增加无效 lane 和可能的寄存器活跃值，不能只看 `next_power_of_2` 就断言更快。

### 7.3 很大的行：split 与 combine

当一行长到单个 CTA 的线程、寄存器或执行时间都不合适，可以让多个 CTA 分担同一行，分别输出 partial max/sum 或 partial norm state，再由第二阶段 combine。Softmax 的 `(m,q)` 合并公式已经展示了这个协议；LayerNorm 可以合并 Welford 状态。它增加 scratch、第二个 kernel 或原子协议，以及可能的输入重读。只有当单 CTA 无法提供足够并行度或延迟成为问题时，这笔成本才可能值得。

### 7.4 三个容易误判的旋钮

- `R` 很小：即使单行 kernel 内部做得好，整个 grid 也可能只占少量 SM；把几行塞进一个 CTA 只会进一步减少 CTA 数，测得的吞吐低首先说明总工作量和并行度不足，不代表归约指令本身低效。
- `D` 很大：一个线程持有多个中间值会拉长寄存器活跃区间；寄存器不够可能 spill 到 local memory，反而增加 global memory 流量。
- `num_warps`：它决定 program 的并行组织候选，不是寄存器数、occupancy 或速度的同义词。`BLOCK/(32*num_warps)` 只能估算每个线程平均负责的逻辑元素份额；编译器可能重排向量、复制临时值或延长 live range，实际 registers/thread 必须读编译 metadata 或 Nsight Compute 结果确认。

因此，`D=1000, BLOCK=1024, num_warps=8` 的 `1024/(32*8)=4` 是每线程分到的平均逻辑位置上限，其中 24 个位置是 padding；它不等于实际寄存器数。对 `BLOCK=256`，线程按跨步循环访问 `D=1000`，tid 0–231 读 4 个有效位置，tid 232–255 读 3 个有效位置，不能用 `1000/256=3.90625` 代替这个离散分布。逻辑 tile、线程布局、编译器寄存器分配和硬件驻留是四层不同事实。

## 8. 算法 bytes、缓存流量和 roofline 不是一回事

对 contiguous `[R,D]` 的 FP32 row-wise Softmax，理想的最小逻辑流量是一次读输入、一次写输出：

$$
\text{bytes}_{\text{ideal}}=2RD\times4.
$$

这只是算法级有效读写。已有 Triton 三阶段 1D 代码先由 GPU 读输入生成 partial，再由 GPU merge，最后重新读输入 normalize，因此输入是两次 read、输出一次 write，至少是 $3N\times4$ bytes；此外还有 partial max/sum scratch 和三个 kernel 的 launch overhead。它没有把 partial 数组送到 host。旧 CUDA `softmax_online.cu` 是另一条 GPU partial → host merge → normalize 路径，才额外涉及 partial 数组 D2H；host 计算出的 global max/sum 作为两个 scalar 参数传回 normalize kernel，不应和 Triton GPU merge 混写。即使某个实现被称作“fused”，只要仍然跨 kernel 物化统计量，就不能按单 kernel 的 2N 流量记账。

Norm 还要读 gamma，LayerNorm 还要读 beta。若 gamma/beta 很小并被 cache 复用，逻辑上确实发生了 load，但它们的 DRAM 流量可能与输入不同；报告时要把“算法逻辑读”与“观测到的 DRAM/L2 traffic”分开。对大 `R`，参数 cache 复用和输入 streaming 可能同时存在；对小 `R`，launch 和固定开销可能比数据流量更显著。

roofline 也不能只写成“带宽受限”。Softmax 有 max、exp、sum、除法；Norm 有平方、加法、rsqrt、乘法和 affine。exp/rsqrt 的吞吐、归约依赖链、warp shuffle、shared barrier、低 occupancy 都可能限制实际速度。若输入很小，launch latency；若 `D` 很大，寄存器/occupancy；若多 CTA split，combine 和 scratch；若算术密集，特殊函数管线，都可能成为主因。有效带宽只是把时间换算成一个数字，不能替代因果分析。

## 9. 实验设计：先固定语义，再观察资源和流量

一次有意义的实验只改变一个主要旋钮，并保留 shape、stride、dtype、eps、输入范围、输出布局、预分配方式和计时边界。下面的表是实验计划，不是预填结果；“预测可观测指标”表示要去 profiler 或编译 metadata 查什么。

| 实验旋钮 | shape / 对照 | 预测可观测指标 | 需要避免的结论 |
|---|---|---|---|
| `BLOCK` | `D=256/257/511/512`，每个 shape 分别比较合法配置 | 无效 lane 比例、寄存器、kernel time、load/store sector | `D=257,BLOCK=256` 会漏元素，不能当候选 |
| `num_warps` | `D=1000`：2/4/8 | registers/thread、active warps、eligible warps、shuffle/同步占比 | 不能从 logical elements 推导 registers |
| rows 数 | `R=1/8/4096`、同一 D | grid 覆盖、SM 活跃度、launch 占比 | 小 R 的低吞吐不等于算子全局低效 |
| 归约方式 | one-CTA vs split-row | scratch bytes、第二 kernel、L2/DRAM traffic、combine time | split 只增加并行度，不保证端到端收益 |
| 精度 | FP16/BF16 输入、FP32 accumulation | 输出误差、特殊函数/转换指令、耗时 | 不把不同 dtype 的速度放同一结论 |
| 输入分布 | constant、negative、scaled、随机 | max/variance 数值、误差、异常值 | 有限输入也可能平方溢出 FP32 |

计时流程应先完成一次 compile/JIT 和 correctness preflight，再 warmup 若干次；正式计时使用 CUDA event 或等价的 device-side timing，排除首次编译和内存分配。所有候选实现使用相同 dtype、同一输入、同一输出预分配方式和同一同步边界。至少覆盖 `D=1, 31, 32, 33, 257, 1000`，再覆盖 `R=1` 和足够大的 R；误差同时报告 max absolute error、max relative error、行和误差以及非有限计数。

数值边界要单独写进测试合同：有限输入、all-`-inf`、NaN、空 shape、常数行、全负输入和缩放输入分别处理。Softmax 本例把非有限输入和空 shape 拒绝；Norm 的有限输入也不是“任意幅度安全”，因为 FP32 中 `x*x` 可能溢出，即使 `x` 本身是 finite。测试应包含常数、负值和适度 scaled 值，并在需要覆盖极大幅度时明确采用更高精度平方或范围限制；不能把有限输入自动宣传成任意数值范围都安全。

本地 CPU 语义测试位于 cpu_semantics.py。它不调用 CUDA，独立检查 row sum 的尾部覆盖、Softmax 分块 `(m,q)` 合并、LayerNorm centered mask、Welford 合并（含 `n=0`）以及特殊值拒绝逻辑。它是数学和索引回归，不是 `nvcc` 编译、GPU 正确性、真实带宽或 profiler 证据。

## LeetGPU：正确性与代码归档

LeetGPU 题面以平台当前页面为准；公开题面源码也已核对：[Reduction #4](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/4_reduction)、[RMS Normalization #50](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/50_rms_normalization)、[Layer Normalization #113](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/113_layer_normalization)。平台总入口是 [LeetGPU Challenges](https://leetgpu.com/challenges)，页面搜索时使用准确题名；页面是 SPA，能取得 HTTP 响应不等于已登录并点击执行过题目。

三道题和本章教学实现的对应关系要写清楚：

- **#4 Reduction**：FP32 一维 `input[N] -> output`，约束 `N<=1e8`，性能 shape 为 `N=4194304`。它检验的是单向量求和，不是 `[R,D]` 多行 kernel 的平台通过证明。
- **#50 RMS Normalization**：`input[N]`，`gamma` 和 `beta` 都是 scalar float，`eps=1e-5`，输出 `y[N]`，约束 `N<=100000`、性能 shape 为 `100000`；公式为 `y=gamma*x/sqrt(mean(x²)+eps)+beta`。它和本章教学的无 bias、`gamma[D]` 的二维 RMSNorm 不同，不能直接拿教学 wrapper 当 #50 提交版。
- **#113 Layer Normalization**：`input[N,C]`、`weight[C]`、`bias[C]`、`output[N,C]`、`eps`，FP32，方差是 biased `1/C`，容差为 `rtol/atol=1e-4`。它与本章的 `[R,D]` row-wise LayerNorm 结构相同，只是题面参数名使用 `N,C,weight,bias`；仍需从题面确认实际 `solve` 签名和平台模板。

本章的 Softmax 题目入口是已核实的 [Softmax](https://leetgpu.com/challenges/softmax)，平台题意为 1D `input/output/N`；已有 `solutions/triton/fused_softmax.py` 是该 1D 归档版本，不能把它当作本章 `[R,D]` row-wise Softmax 已通过。三个 Norm/Reduction 题目都应分别按当前题面提交和归档，不能把某一个题的成绩外推到其他 shape 或算子。

公开题面源码用于核对约束和公式，不需要把外部 reference 全部复制进课程。提交前以平台当前题面为最终合同；若平台页面无法执行或需要登录，使用官方公开题面和本章 CPU reference 做语义检查，不能把语义检查写成平台通过。平台代码、服务器包装和本章教学实现保持三份边界，不互相覆盖。

## 服务器：真实性能

服务器实验的前置是 LeetGPU 题面正确性与原始 `solve`/kernel 归档，再有独立 reference 和多 shape 正确性结果。下面的验证脚本提供可复用的 Triton/PyTorch forward 检查与 kernel-only 计时；计时排除 JIT compile、首次分配、finite preflight 和 wrapper 参数检查。

从仓库根目录，在具备 CUDA、PyTorch 和 Triton 的环境中执行：

~~~bash
python roadmap/curriculum/operators/02-reduction-and-norm/examples/validate_forward.py
python roadmap/curriculum/operators/02-reduction-and-norm/examples/validate_forward.py --benchmark --dtype float32
~~~

脚本的 correctness 先用 FP32 计算 reference，再 cast 到被测 dtype，覆盖 `R=1/2/3`、`D=1/31/32/33/257/1000`、常数行、全负行和 `[10000,10001,9999,10002]` 这类非恒定偏置行。correctness 直接调用 row-wise Softmax、RMSNorm、LayerNorm kernel；`--benchmark` 只对预分配的 Softmax kernel 与同样预分配的 `torch.softmax(..., out=...)` 计时，当前脚本不报告 Norm 性能数字。Norm 若后续加入 kernel-only 计时，数字必须不包含 Python wrapper 的 contiguous/CUDA/device/dtype/eps 检查和 output allocation；若要报告 wrapper 端到端延迟，必须另列计时边界，不能与 kernel-only 数字混在一起。脚本使用 CUDA Event 包住 Python 发起的循环，平均值包含 kernel launch 的 host gap，不能当作 profiler 的纯 kernel duration；纯 kernel 时长需由 Nsight Systems/Compute 或等价工具单独获取。

## 参考资料

实现时注意同步与归约的不同约束：`__syncthreads()` 等待 block 内未退出线程并提供排序；`__reduce_add_sync` 的类型与架构支持有限制，不能直接用于 float；shuffle 支持 float，但源 lane 必须参与，且它本身不提供共享内存的排序保证。这些条件分别落实在本章的实现中。

## 参考阅读

[CUDA Programming Guide 13.3](../../../../downloads/cuda-programming-guide.pdf)。

## 章节导航

[上一章：访存与布局算子](../01-memory-and-layout/README.md) · [下一章：GEMM：从分块实现到性能优化](../03-gemm/README.md) · [返回完整算子体系](../README.md)
