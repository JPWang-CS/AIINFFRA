# 第一章 访存与布局算子：从数据搬运到高效转置

copy（复制）、transpose（转置）、permute（轴置换）、gather（聚集读取）、scatter（分散写入）和 embedding（嵌入查表）的主要工作是移动数据。逻辑坐标决定值的来源与去向，stride（步长）决定地址，线程分工决定一次内存指令覆盖哪些位置。

view 只改变张量的解释方式，物化转置则产生新的存储布局。两者的流量、别名关系和计时范围不同；对齐、尾部与共享内存转置将在具体实现中分别处理。

## 1. copy、view、transpose 和 permute

### 1.1 copy 复制逻辑元素，布局合同需要另外声明

给定源张量 src 和目标张量 dst，copy 的逻辑合同是：对每个合法逻辑坐标 i，dst[i] = src[i]，两边逻辑形状相同，但物理stride不一定相同。通用copy可以把非连续源复制到连续目标；下面的最小实现单独限定两端都是contiguous。若源和目标都是 contiguous，N 个元素的 copy 只读 N 个元素、写 N 个元素，没有有效 FLOPs。

copy 可以是设备内的 global-to-global 搬运，也可以是 host-to-device、device-to-host 或 peer-to-peer。不要把 host 传输时间和 device kernel 时间混成一个数字：它们经过的链路、同步方式和带宽上限不同。本章的 transpose 实验只测设备上的 out-of-place kernel，输入和输出先分配在同一张 GPU 上。

### 1.2 view 只改解释，不搬数据

view 通过修改 shape、stride 或 storage offset 解释同一块 storage。例如二维 A 的转置 view 可以把 shape 从 [M, N] 改成 [N, M]，把 stride 从 [N, 1] 改成 [1, N]，但它不产生新的 N×M 数据。创建 view 的时间通常只是元数据操作，不能拿它与真实 transpose 的 kernel 时间比较。

~~~python {1-2}
y = x.transpose(0, 1)                 # view：通常只改变 shape/stride
z = y.contiguous()                    # materialize：真正读写一份转置后的 storage
assert z.shape == (x.shape[1], x.shape[0])
~~~

下游算子是否接受非 contiguous stride 决定 materialize 的位置。如果下游 kernel 能直接按 stride 读取，立即 materialize 可能是多余搬运；如果下游只接受 contiguous，view 的成本可能延迟到后面的 contiguous。

### 1.3 transpose 是二维轴交换，真实 transpose 是布局重排

对 row-major 的 A[M, N]，二维 transpose 的数学定义是

$$
B[c,r] = A[r,c],
\qquad 0 \le r < M,\quad 0 \le c < N.
$$

如果 B 是一个真实、contiguous 的新张量，它按 B[c, r] 的 row-major 顺序存储，线性地址是 c×M + r。因此每个元素的物理地址都重新排列；这与共享原始storage、stride为[1,N]的转置view不同。

“真实 transpose”在本章特指 out-of-place materialization：源和目标 storage 不重叠，目标拥有 N×M 个元素，并且 benchmark 覆盖生成目标数据的读写时间。可能有 overlap 的 in-place 变体需要另立 alias 合同和依赖分析，不能套用本章的无 overlap 假设。

### 1.4 permute 是任意轴排列

对 X[D0, D1, ..., Dk-1]，permute(p) 把输出轴 j 映射到输入轴 p[j]。二维 permute([1, 0]) 是 transpose；三维 permute([0, 2, 1]) 只交换后两个轴。若只是返回 stride view，它仍然不搬数据；若目标要求 contiguous，就要执行多维真实重排。高维 permute 的难点是最后一维是否仍连续、每个 program 的线性化方式，以及小维度带来的尾部浪费。

## 2. 形状、步长、元素大小与字节地址

### 2.1 shape 说元素数，stride 说跳多少

对 rank-R 张量，shape[d] 是第 d 维长度，stride[d] 通常以 element 为单位，表示该维下标增加一格时 storage offset 增加多少个元素。给定逻辑下标 i，element offset 是

$$
o(\mathbf{i}) = \text{storage\_offset} + \sum_{d=0}^{R-1} i_d \cdot \text{stride}_d.
$$

真正的 byte address 还要乘上 elementbyte：

$$
\text{byte\_address}(\mathbf{i}) =
\text{base} + o(\mathbf{i}) \times \text{elementbyte}.
$$

unit 必须说清楚：PyTorch 的 stride 以元素为单位，NumPy 的 strides 以字节为单位；CUDA 指针加法通常以所指向的 C++ 类型为单位；手算 global transaction 时最终必须回到 byte address。把 stride=1 直接说成“1 byte”是常见错误。

上式的 `base` 指底层存储的起始地址。如果使用 `tensor.data_ptr()`，得到的已经是该张量第一个逻辑元素的地址，包含切片带来的 storage offset。此时只应再加各维的 stride 偏移，不能重复加 `storage_offset`。

例如，`x` 是连续的 FP32 `[4,6]` 矩阵，`v=x[1:3,1:5:2]` 的形状为 `[2,2]`，stride 为 `(6,2)`，storage offset 为 7。计算 `v[1,1]` 时有两种等价写法：

$$
\begin{aligned}
\text{address}&=\text{storage base}+4(7+1\times6+1\times2),\\
&=v.\operatorname{data\_ptr}()+4(1\times6+1\times2).
\end{aligned}
$$

两者都指向底层第 15 个元素。把 PyTorch 张量传给 Triton 时，传入指针对应该张量的起始位置；kernel 按传入的 stride 寻址即可。这个例子同时说明，形状相同的张量不一定有相同的地址步长。

### 2.2 contiguous 不是唯一合法布局

row-major contiguous 的二维 [M, N] 张量通常是 stride=[N, 1]；列方向相邻元素的 byte 间距是 elementbyte，行方向相邻元素的 byte 间距是 N×elementbyte。转置 view 的 stride 变成 [1, N]，逻辑上合法，但若沿输出最后一维遍历，物理访问可能是大步长。

每个 kernel 都要先声明支持哪一种合同：

- **contiguous-only**：输入或输出必须满足标准 contiguous stride，地址计算可以简化，向量化前提更容易证明。
- **arbitrary-stride**：每个维度显式传入 stride，功能覆盖更广，但索引整数、访存模式和向量化条件都要重新证明。
- **same-storage view**：只返回 shape/stride 元数据，不进入本章的真实搬运 benchmark。

如果一个 kernel 在 contiguous tensor 上正确，却在 x[:, ::2]、转置 view 或带 offset 的 slice 上错误，说明 shape/stride 合同没有实现。

### 2.3 元素大小决定跨度与流量账本

`float32` 的每元素大小为 4 bytes，`float16`/`bfloat16` 为 2，`int8` 为 1。相同的 element stride 在不同 dtype 下对应不同 byte stride；相同 shape 在不同 dtype 下也有不同的有效流量。一个 `float4` 变量包含四个 `float32`，逻辑上仍是 4 个元素、16 bytes，不能因为 C++ 类型变宽就把有效工作量少算四倍。

对于 N 个元素、每元素 E bytes 的 out-of-place copy 或 transpose，理想有效数据量是

$$
\text{bytes}_{\text{effective}} =
N \times E\ \text{(read)} + N \times E\ \text{(write)} = 2NE.
$$

若 kernel 时间为 t 秒，十进制有效带宽是

$$
BW_{\text{effective}} = \frac{2NE}{t \times 10^9}\ \text{GB/s}.
$$

这只是“有用读写字节”除以时间，不是硬件真正从 DRAM 发出的字节数；不合并访问、尾部 sector、缓存回写和额外 staging 都可能让实际传输更多。

这个计算描述完成任务所需的有效读写。实现层还需要根据各 lane 的地址判断请求能否合并，并处理尾部的无效位置；实际经过 L2 或 DRAM 的流量则由缓存行为共同决定。因此，比较带宽时应分别列出算法有效字节数和性能分析器测得的实际传输量。

## 3. CUDA 与 Triton 的最小 copy

### 3.1 CUDA：线程索引到 byte address

最小 contiguous copy 是把线性 index 映射到 global memory。工程版本要有边界判断、错误检查和明确 alias 合同；下面只保留关键片段，完整例程按题面重新组织。

~~~cpp {1,3-4}
__global__ void copy_f32(const float* __restrict__ src,
                         float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
~~~

`__restrict__` 是编译器可依赖的 aliasing contract：在这个调用合同中，src 指向的对象不会通过 dst 可达的别名被修改，反之亦然。它不是同步原语，也不会让任意重叠指针变得安全。只有 API、调用者和生命周期真的保证不重叠时才能写；若允许 src == dst 或部分 overlap，就删除这个限定并为重叠语义单独设计。

CUDA Programming Guide 13.3 的 §2.3.3.3 还提醒：kernel 内没有地址空间限定的自动变量会尽量进寄存器，必要时进入 local memory；local memory 在物理上属于 device memory 路径，不要把“线程私有”误读成“片上”。为了索引添加的局部数组，可能最后带来额外 global/local load/store。

### 3.2 Triton：program 负责一段线性 block

Triton 通常让一个 program 处理 BLOCK 个元素，用 tl.arange 生成偏移，用 mask 保护尾部。tl.multiple_of 是对编译器的对齐/倍数提示，不是运行时边界检查；给错提示可能改变生成代码并造成未定义行为。

~~~python {4-6}
@triton.jit
def copy_kernel(src, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(src + offs, mask=offs < n, other=0.0)
    tl.store(dst + offs, x, mask=offs < n)
~~~

CUDA的一线程索引与Triton的一整块逻辑偏移，承担相似的数据划分职责，但不是逐线程的一一映射。Triton如何把BLOCK个值分到thread/warp由编译器布局与配置决定；mask控制的是逻辑tile中的有效访问。两者的性能问题仍是同一个：相邻 lane 是否落在少量 32B segment，尾部是否引入大量无效 lane，目标指针对齐是否满足向量化条件。

### 3.3 scalar、float4 和 tail

float4 向量化常用来减少指令条数、改善每条 load/store 的粒度，典型前提是地址满足 16B 对齐且主循环有至少四个连续 float。逻辑上可以把 N 分成 N/4 个 vector 和 N%4 个 scalar tail：

下面先用串行伪代码说明向量主体与尾部划分；它不是可以让所有CUDA线程原样重复执行的kernel。

~~~cpp {2-5}
std::size_t vec_n = n / 4;
auto* src4 = reinterpret_cast<const float4*>(src);
auto* dst4 = reinterpret_cast<float4*>(dst);
for (std::size_t j = 0; j < vec_n; ++j) dst4[j] = src4[j];
for (std::size_t i = vec_n * 4; i < n; ++i) dst[i] = src[i];
~~~

这段片段只有在 src、dst 均 16B 对齐、区域足够大且不重叠时才符合合同。若 base 地址、slice offset 或每行 stride 不是 16B 对齐，不能只看 N%4==0 就强转 float4*。对二维 copy，每行起始地址都要重新检查；第一行对齐不代表下一行 stride×elementbyte 仍对齐。

> [!WARNING] 边界条件
> 向量化不自动减少 DRAM 流量。对 N 个 float32 的 copy，理想有效流量仍是 2N×4 bytes；如果原来已经完美 coalesced，float4 可能只减少指令和地址计算，实际 sector 数几乎不变。错位的 float4 可能跨更多 32B segment、触发未对齐访问或使 tail 处理变复杂，结果反而变慢。是否收益必须用同一 shape、同一计时范围和 transaction/sector 证据验证。

## 4. coalescing：从 lane 地址推导 transaction

CUDA Programming Guide 13.3 的 §2.3.4.1 描述 global memory 的核心模型：一个 warp 的请求会合并为满足这些地址所需的尽可能少的 32-byte memory transactions。对 float32，若 32 个 lane 访问连续 128 bytes，理想需要四个 32B transaction；若相邻 lane 的地址相隔至少 32 bytes，可能需要 32 个 transaction，而有效使用的仍只有 128 bytes，事务利用率只有 12.5%。官方 Best Practices §10.2.1 用 maximize bytes used / bytes transferred 表达同一原则。

### 4.1 copy 的连续访问

一维 contiguous copy 中，lane l 访问 base + (pid×BLOCK+l)×4。当一个 warp 内的 l=0..31 连续时，它覆盖四个 32B segment；边界 warp 可能只使用其中一部分 word，但 segment 仍可能被取回。所谓合并不是要求 lane 按严格顺序执行，而是要求它们请求落在尽量少的 segment 中；在同一 segment 内做合法 permutation 通常不增加 transaction。

### 4.2 地址重复、请求合并与显式复用

判断“不同 thread 重复读一行”的成本，需要区分三个场景：

1. **32 个 thread 读同一个地址**：逻辑上只有一个 word 的有效数据，但 global memory 的最小服务粒度仍受 segment/缓存系统影响；同一 warp 的请求落在同一 32B segment，通常不需要为每个 lane 各发一个 segment。不能进一步保证永远只访问 DRAM 一次，因为 L1/L2 命中、其他 warp 流量和请求合并由硬件与时序决定。
2. **32 个 thread 读同一行的连续 32 个元素**：这是 128B 连续访问，理想是四个 32B transaction。它们读的是同一行，但不是重复同一地址；每个 lane 使用不同 word。
3. **32 个 thread 读同一行的同一列**：若它们实际落在同一地址，属于第一个场景；若是不同 row 的同一 column，则地址间距通常是 leading dimension，可能完全不合并。代码里“行相同”或“列相同”不是判断依据，必须写出 lane 到 byte address 的函数。

重复访问有时可以被 cache 或请求合并隐藏，有时会制造大量无效 transaction；不能仅凭重复访问就断言应该使用 shared。shared staging 是程序显式控制的 block 级复用，cache 是硬件对地址流、容量和替换的结果。只有复用在同一 block 内可证明、工作集和同步成本合适时，搬到 shared 才有明确理由。
## 5. 从一份可运行 copy 观察向量化的收益和边界

完整程序在 copy_alignment.cu。它比较标量copy与“对齐时使用float4，否则回退标量”的实现，分别检查起始偏移0和1个float，以及1、3、4、5、257和较大输入。两个实现使用同一grid，标量版本通过grid-stride循环覆盖剩余元素，避免把grid数量变化直接混成向量化收益。

下面是关键的设备代码。tid和step分别表示全局线程编号与整个grid的线程数：

~~~cpp {1-4,9-10}
const bool aligned = ((reinterpret_cast<std::uintptr_t>(src) |
                       reinterpret_cast<std::uintptr_t>(dst)) & 15u) == 0;
if (!aligned) {
    for (std::size_t i = tid; i < n; i += step) dst[i] = src[i];
    return;
}
const std::size_t vectors = n / 4;
const float4* src4 = reinterpret_cast<const float4*>(src);
float4* dst4 = reinterpret_cast<float4*>(dst);
for (std::size_t v = tid; v < vectors; v += step) dst4[v] = src4[v];
for (std::size_t i = vectors * 4 + tid; i < n; i += step) dst[i] = src[i];
~~~

这里的对齐分支在一次调用中对所有线程相同，因为所有线程收到相同的src、dst基指针。没有16B对齐就不解引用float4指针。尾部与向量主体写入不相交的区域：主体覆盖0到4×vectors−1，尾部从4×vectors开始；不同线程又通过tid与step分开，因而不会重复写同一元素。

以N=7为例，vectors=1：线程0处理包含元素0、1、2、3的一个float4，标量尾部由线程0、1、2分别处理元素4、5、6。以N=3为例，vectors=0，全部由标量尾部处理。对src+1这样的slice，即使原始分配对齐，当前地址偏移4B，也会回退到标量路径。

restrict的前提独立于对齐。两个不同、都16B对齐的地址仍可能指向重叠区域；删除限定也不能让并发部分重叠copy自动拥有memmove的语义。本程序通过两次独立分配保证src与dst不重叠。

> [!IMPORTANT] 检查完整条件，不只检查 N 是否整除
> **向量化需要访问宽度、源地址、目的地址、连续元素和边界同时成立。** 元素数量能被4整除，不能证明地址16B对齐；第一行对齐，也不能证明后续每一行都对齐。

示例先初始化输入和输出、执行正确性比较，再预热与计时。复制不改变数值，有限测试输入应逐元素相等；未写入位置预填为非有限位模式，避免遗漏写入恰好被零值掩盖。对于要完整覆盖特殊浮点表示的测试，还可按位检查NaN payload与正负零，而不是用普通浮点相等判断。

这个程序重复使用同一对buffer，因此给出的是热工作集条件下的有效copy带宽。它不是“显存峰值测量工具”，也不意味着float4版本一定更快。向量化减少的指令是否足以抵消对齐检查、循环和尾处理，需要结合生成指令及目标形状判断。

**实践绑定：copy 的实验合同。** `copy_alignment.cu` 只研究 contiguous FP32、非重叠输入输出、起始偏移 0/1 和 `N=1,3,4,5,257` 等尾部/对齐边界；它是转置题之前的访存预实验，不是 Matrix Transpose 的平台提交。报告时保留 `N`、起始偏移、预分配 buffer、warmup/iterations 和有效字节 `2N×4`，不要把向量化 copy 的结果外推到任意 stride。

## 6. 真实转置：读和写为什么不能沿用同一组线程坐标

### 6.1 一个输出布局要求怎样改变地址

设A为3×5矩阵，连续存储。原数据的线性编号为：

~~~text
A:
 0  1  2  3  4
 5  6  7  8  9
10 11 12 13 14

B = A 的真实转置，shape=5×3:
 0  5 10
 1  6 11
 2  7 12
 3  8 13
 4  9 14
~~~

A[1,4]在线性位置9，B[4,1]在线性位置4×3+1=13。真实转置把值从输入位置9写到输出位置13；view只改变解释方式，不完成这次实际搬运。

对naive kernel，如果lane沿输入列连续，每条加载可访问相邻元素。但同样的lane直接写B[col*M+row]时，输出位置相隔M个元素。**输入合并，不保证输出也合并。** 这正是共享内存转置要解决的问题，而不是为了“所有global访问都先搬到shared”。

### 6.2 shared tile 分开输入阶段与输出阶段

共享 tile 转置把输入阶段和输出阶段分成两个地址合同。输入阶段让 `threadIdx.x` 沿输入列变化，输出阶段让它沿输出列变化：

~~~cpp {4,8-11}
const int in_row = blockIdx.y * 32 + threadIdx.y;
const int in_col = blockIdx.x * 32 + threadIdx.x;
tile[threadIdx.y][threadIdx.x] =
    (in_row < M && in_col < N) ? input[in_row * N + in_col] : 0.0f;
__syncthreads();

const int out_row = blockIdx.x * 32 + threadIdx.y;
const int out_col = blockIdx.y * 32 + threadIdx.x;
if (out_row < N && out_col < M) {
    output[out_row * M + out_col] = tile[threadIdx.x][threadIdx.y];
}
~~~

关键不只是读shared时把两个下标交换。输出block坐标和线程坐标也要配套改变。只交换tile索引却保留原来的全局输出坐标，会把正确值写到错误位置。

矩形尾块还要求分别判断输入和输出。前面的3×5例子中，ty=3、tx=0没有合法输入行，却需要输出B[3,0]=A[0,3]。因此在输入阶段按in_row越界直接return，会漏掉后面的合法输出。所有线程经过装载与同步阶段，再按输出坐标判断store，证明过程最清楚。

### 6.3 padding 改了什么，没有改什么

对float tile[32][32]，列访问tile[lane][0]的word偏移为32×lane，bank编号为0，32个线程竞争同一个bank的不同word。改成tile[32][33]后，word偏移为33×lane，bank编号变成lane，列访问分散到32个bank。

逻辑有效数据仍是32×32，只是物理行步长增加一个float。每块shared容量从4096B变成4224B，增加128B。这是用少量空间改变访问映射，不是给输出矩阵增加一列，也不是修改数学计算。

无padding与有padding版本使用相同线程布局，可以较清楚地观察bank布局的影响。naive版本与tiled版本还同时改变了线程组织和shared staging，二者的整体时间差不能全部归因于padding。后续优化可以研究32×8线程块协作处理32×32 tile、每线程搬多个元素，但要分别分析寄存器、循环和同步变化，不能一次改完再猜原因。

**实践绑定：transpose 的题面与形状。** 平台入口是 [Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose)；平台 `solve/kernel` 的参数、矩形限制和容差以题面为准。本章至少用 `1×1`、`3×5`、`32×32`、`37×65` 检查地址与尾块：其中 `3×5` 直接对应上面的 `A[1,4]→B[4,1]`，`37×65` 检查两个轴的独立 mask。共享 tile 的 CUDA 程序只作为服务器实验实现，不能当作平台原始提交。

## 7. 从转置推广到 gather、scatter、embedding 和数据打包

这些算子的共同基础是索引映射，而不是某个固定的 tile（分块）尺寸。先确定逻辑坐标如何映射到地址，再讨论线程布局和缓存复用；否则即使 tile 看起来规则，也可能把重复读取或冲突写入隐藏起来。

| 操作 | 数学/地址合同 | 首先要解决的性能或正确性问题 |
|---|---|---|
| Gather | out[i]=input[index[i]] | index合法性、读取是否聚集、相同输入的重复使用 |
| Scatter | output[index[i]]=input[i] | index重复时的写入语义，是否需要归约或唯一写者 |
| Embedding lookup | out[token,d]=weight[token_id[token],d] | 一行特征如何分工，相同token的缓存复用，权重工作集 |
| KV copy | 按token/head/page索引搬运K/V数据 | page边界、源/目标布局、正在使用的数据何时可覆盖 |
| Pack/unpack | 把数值编码为约定的位布局，再按相同规则还原 | 位宽、符号扩展、字节顺序、尾元素及scale对应关系 |

例如gather的index=[2,2,0]，输出应是[input[2],input[2],input[0]]，重复读取在数学上完全合理。若把同样的index用于scatter，两个线程都要写output[2]，就必须先定义最终值：是否保证唯一index、按某种顺序覆盖，还是执行求和？如果需要sum，应使用有正确原子/分层归约语义的实现；简单非原子并发写不能替代它。

Embedding通常让相邻lane处理同一个token的一行连续特征，利用行内合并；不同token行之间的局部性则由索引分布决定。某个重复token很多的数据集可能产生很高缓存命中，不能代表随机token的大权重工作集。

Pack也不只是在内存中把元素排紧。以两个4-bit编码合成一个byte为例：

~~~cpp {1-2}
unsigned packed = (q0 & 0xFu) | ((q1 & 0xFu) << 4);
unsigned low_code = packed & 0xFu;
unsigned high_code = (packed >> 4) & 0xFu;
~~~

这里提取的是无符号bit code。如果量化格式规定signed INT4，还要按它的编码约定解释符号；scale和zero-point也不能从这个byte中凭空恢复。该例只解释位布局，不代替量化算法。后续Quantized GEMM需要把packing合同一直追踪到矩阵指令的输入布局。

## 8. benchmark：测到的是数据搬运，还是一个元数据操作

### 8.1 基线必须真的完成相同工作

对连续copy，可以比较预分配输出上的copy操作；对真实转置，必须要求目标结果实际物化并具有相同布局。只计x.transpose的时间是在计view，不是转置kernel。

PyTorch的例子应区分两种测量：

~~~python {2,6}
# 端到端物化路径：含输出分配等影响。
result = x.transpose(0, 1).contiguous()

# 预分配路径：研究固定输出buffer的实际复制。
out = torch.empty((N, M), device=x.device, dtype=x.dtype)
out.copy_(x.transpose(0, 1))
~~~

如果x有长度为1的维度，转置后仍可能满足contiguous条件，contiguous()不一定触发新的复制。用于真实搬运的基准应确认输出是否实际分配/写入，或使用预分配out.copy_路径。完整baseline还要记录输出是否预分配、是否允许覆盖、是否计入allocator和框架dispatch。

对于任意stride输入，要么实现相同stride合同，要么在接口明确拒绝。不能让自己的kernel只处理连续张量，却把PyTorch处理非连续输入的额外成本算成优化收益。

### 8.2 工作集与缓存条件必须写出来

以4096×4096的FP32转置为例，输入大小64 MiB，输出64 MiB，算法有效流量是128 MiB，即134,217,728B。若一次操作耗时0.20ms，则有效带宽为：

$$
BW=\frac{134217728}{0.20\times10^{-3}\times10^9}
=671.08864\ \mathrm{GB/s}.
$$

0.20ms只是算术示例，不是本章实测。也不能因为得到671GB/s就认定DRAM在提供这个吞吐：重复使用同一buffer时，缓存、写回、访问模式都会参与。

至少区分两种条件：反复使用相同buffer的稳态热工作集，以及轮换多个buffer或增大工作集以研究缓存容量之外的行为。后一种也不是简单执行一次“清缓存”就能宣布完全冷缓存，需要明确方法和实际counter。记录实际DRAM/L2流量，比把算法字节数直接当成总线流量更可靠。

### 8.3 比较结果时保留整个条件

一次可复算的报告包含：GPU与CC、工具/框架版本、shape、stride、dtype、元素起始偏移、alias条件、输入与输出是否预分配、grid/block、warmup、重复次数、计时范围、误差和有效字节口径。

在此基础上再解释变化：

- 标量变向量：是否减少了访存指令和地址计算？有效DRAM流量不一定变化。
- naive变tiled：读写合并、线程布局与shared staging都可能变化。
- tile32变tile33：逻辑工作不变，主要观察bank布局与shared容量变化。
- 同形状变不同stride：这是工作负载变化，不是同一配置的纯速度对比。

## 9. 已有 Vector Add 测量：把结果放回访存模型

已保存的 Vector Add 测量使用 N=2^25、BLOCK_SIZE=256、RTX3090：Triton有效带宽840.1GB/s，torch.add为843.0GB/s。它说明在当时条件下两个实现很接近，不能单凭接近torch就证明DRAM饱和，更不能把这个数字当成不同GPU的目标。

**已有实验的代码与证据：** 下面的 kernel 是 `solutions/triton/vector_add.py` 中的原样片段；该文件明确是本地验证/benchmark wrapper，不是单独归档的 LeetGPU 原始 `solve`。

<!-- source-check: solutions/triton/vector_add.py -->
~~~python
@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 每个 program（block）负责一段连续元素
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 尾块越界保护：越界位置不参与计算
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)
~~~

复算入口是 `python solutions/triton/vector_add.py`；GPU 段用 CUDA tensor 和 `triton.testing.do_bench`（含预热与同步计时），正确性覆盖 `N=1/256/257/1000/2^20`。weekly 记录的 GPU 测量为 [2026-08-24](../../../../weekly/2026-08-24-triton-vector-add.md)：AutoDL RTX 3090、`N=2^25`、FP32、`BLOCK_SIZE=256`，Triton `0.479 ms / 840.1 GB/s`，`torch.add` `0.478 ms / 843.0 GB/s`，按 `3N×4`（两次输入读取加一次输出写入）计算有效带宽。它只描述这台 RTX 3090、这个 shape 和这个计时口径，不能外推到其他 GPU、任意 stride 或证明 DRAM 已达峰值；它也不替代 Matrix Transpose 的平台题面实现。

Vector Add每元素需要两个输入读取和一个输出写入，FP32算法流量为12N；纯copy与transpose是8N。比较两类算子时先统一流量口径，不能直接把同样的GB/s看成相同总工作量。

“不同thread会重复读一行”的讨论在这里继续变得具体：先展开同一条warp指令的地址，再区分请求合并、cache复用和显式shared复用。已有 MatMul平台源码 中的广播地址表达式给出逻辑tile，不直接规定每个thread读哪些位置；进入性能分析还需要看编译布局和生成指令。

> [!TIP] 保留问题，也保留判断依据
> **同一行、同一地址、同一次warp请求、同一份缓存数据，是四个不同层次。** 先问清“重复”发生在哪一层，再决定是否需要shared staging、重新分块或改变布局。

## 排错场景：输出正确，为什么仍可能是错误的布局实现

1. 如果一个转置结果逐元素正确，但下游把它当 contiguous 读时变慢，先检查返回的是 stride view 还是已物化 storage；不要用一次 `x.transpose()` 的时间替代真实搬运。
2. 如果 `x[:, ::2]` 传入后只有部分列错位，打印“传入指针对应的第一个逻辑元素地址”和各维 stride，确认没有把 `data_ptr()` 与底层 storage base 的 offset 重复相加。
3. 如果 `float4` 版本只在大输入变快，先用起始偏移 0/1、每行 stride 和尾部 N=3/5/257 重现；N 可被 4 整除并不能证明每行地址满足 16B 对齐。

## LeetGPU：正确性与代码归档

题目入口：[Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose)。

读懂本章后，从平台题面实现其要求的solve/kernel。确认题面约定的矩阵形状、输入输出存储与接口，再写地址与mask；本章的独立CUDA程序和GPU基础篇转置例程用于理解与对照，不是可直接当作平台原始提交的记录。

通过后原样保存平台版本，另行保留服务器的主机代码与benchmark包装。基础case至少覆盖1×1、3×5、32×32和37×65；若平台题面限制方阵，平台按题面验证，矩形能力在服务器独立测试，不能伪称平台也验证过。

已有Vector Add和MatMul无需重新作为入门题提交。它们在本章承担源码复盘和访存比较的作用。

## 服务器：真实性能

从仓库根目录执行：

~~~bash
nvcc -O3 -std=c++17 -lineinfo --resource-usage \
  roadmap/curriculum/operators/01-memory-and-layout/examples/copy_alignment.cu \
  -o /tmp/copy-alignment
/tmp/copy-alignment
compute-sanitizer --tool memcheck --error-exitcode=1 /tmp/copy-alignment

nvcc -O3 -std=c++17 -lineinfo --resource-usage \
  roadmap/curriculum/gpu/03-registers-and-memory-system/examples/transpose_tiled.cu \
  -o /tmp/transpose-layout
/tmp/transpose-layout 3 5 5
/tmp/transpose-layout 37 65 5
/tmp/transpose-layout 4096 4096 100
compute-sanitizer --tool racecheck --error-exitcode=1 /tmp/transpose-layout 37 65 5
~~~

先检查输出与参考，再看时间；小case用于覆盖地址与尾部，不用于代表大输入吞吐。copy_alignment固定同一对buffer重复计时，并报告起始偏移，正好用于观察对齐分支与热工作集；研究显存持续带宽时应另外设计工作集与计数器实验。

查看编译资源与机器指令后，才判断float4是否形成相应访存指令、padding是否改变shared请求。若Nsight Compute权限不可用，保留Nsys时间线、编译资源、源码配置和正确性结果，并把缺少counter导致无法证明的因果留在分析中，不补造数据。

服务器记录表只预留同条件结果，不预填目标数字：

| 实验 | 固定合同 | GPU / CC | correctness | mean ms | effective GB/s | Nsys/NCU 资源与流量 |
|---|---|---|---|---:|---:|---|
| contiguous copy，scalar vs `float4` | FP32；非重叠；`N=`；offset=`0/1` |  |  |  |  |  |
| tiled transpose，naive vs shared | FP32；`M×N=`；`32×32` tile；padding=`0/1` |  |  |  |  |  |

[返回完整算子体系](../README.md) · [下一章：并行归约、Softmax 与归一化](../02-reduction-and-norm/README.md)

## 参考阅读

[CUDA Programming Guide 13.3](../../../../downloads/cuda-programming-guide.pdf)。
