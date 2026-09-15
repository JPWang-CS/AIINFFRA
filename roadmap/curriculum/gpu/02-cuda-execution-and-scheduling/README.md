# 第 2 章：CUDA 执行模型与指令调度

一个线程块先驻留到 SM（Streaming Multiprocessor，流式多处理器），其中的 warp 才能等待操作数、进入就绪状态，并由调度器发射指令。驻留、就绪和发射分别受到资源容量、依赖和执行管线的限制。

依赖链与多个独立累加器、分支与 predication 会产生不同的执行行为。源码给出工作分配，编译产物说明指令选择，profiler 则显示这些指令实际怎样推进。

## 1. 从 thread 到 warp：线性编号决定了第一层映射

CUDA 允许一维、二维、三维的 grid 和 block，但硬件最终把一个 block 按线性 thread ID 切成连续的 32-thread warp。对一个 block，线性 thread ID 可以写成：

$$
tid = threadIdx.x + blockDim.x\,threadIdx.y + blockDim.x\,blockDim.y\,threadIdx.z
$$

第 $w$ 个 warp 中的 lane 为：

$$
warp\_id = \left\lfloor\frac{tid}{32}\right\rfloor,\qquad lane = tid\bmod32
$$

因此，一个 `dim3 block(32, 8)` 的前 32 个 thread 是 `threadIdx.y=0, threadIdx.x=0..31`，第二个 warp 是 `threadIdx.y=1` 的 32 个 thread；而 `dim3 block(16, 16)` 的第一个 warp 包含 `y=0` 与 `y=1` 两行的 32 个线程，第二个 warp 包含 `y=2` 与 `y=3` 两行。它们不是“每一行一个 warp”。这决定了 global coalescing、shared bank、分支分歧和 warp-level primitive 的行为。

```cpp
__global__ void print_mapping(int* warp_ids, int* lanes) {
    int tid = threadIdx.x + blockDim.x * threadIdx.y;
    warp_ids[tid] = tid / 32;
    lanes[tid] = tid % 32;
}
```

如果矩阵的逻辑列希望映射到连续 lane，常见的二维布局是 `threadIdx.x -> column`；但这只是设计选择，不是 CUDA 自动保证的矩阵语义。对 `dim3 block(16, 16)`，线性 thread 0–15 来自 `threadIdx.y=0`，thread 16–31 来自 `threadIdx.y=1`，所以 warp 0 横跨两个逻辑行；只有 `dim3 block(32, 8)` 才让第一个 warp 完整对应 `threadIdx.y=0` 的一行。

下面保留一个真实 naive GEMM 核函数及其启动配置，用于具体分析 thread 到输出元素的映射：

<!-- source-check: solutions/cuda/gemm/naive_float.cu -->
```cpp
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

extern "C" void solve(const float* A, const float* B, float* C,
                      int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + 15) / 16, (M + 15) / 16);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```

这里的接口合同是 `A[M,N] × B[N,K] => C[M,K]`，所以每个有效 thread 负责一个 `C[m,k]`，并沿 `n` 做归约。`k` 是输出列，来自 `blockIdx.x` 和 `threadIdx.x`；`m` 是输出行，来自 `blockIdx.y` 和 `threadIdx.y`；`idx = m*K+k` 是 row-major 的 C 线性地址。`sum = 0` 明确从零开始，因而写回必须用 `C[idx] = sum`，不能用依赖旧 C 内容的 `+=`。`if (k < K && m < M)` 是输出边界 mask：尾部 block 的无效线程既不读 A/B，也不写 C。

对 `dim3 block(16,16)`，一个 warp 的 32 个 lane 横跨两行：前 16 个 lane 是同一 `m` 的 `k=...+0..15`，后 16 个 lane 是下一 `m` 的 `k=...+0..15`。因此在循环的某个固定 `n` 上，warp中两个半warp访问的是同一段连续的B元素，而非两个不同的B行；同一段请求可以合并，跨warp或跨调用的实际流量还要看缓存；不要把“单线程沿 n 迭代时 B 列方向 stride 为 K”误说成“warp 不合并”。A 的 `A[m*N+n]` 在每一半 warp 内对固定 m 相同，属于同地址复用（广播/缓存行为仍须以实际生成代码和 profiler 为准）。C 的写也按 k 连续成两段。

小手算：令 `M=3,N=2,K=5`，thread `(m=1,k=2)` 的 `idx=1*5+2=7`，归约读取 `A[1*2+0]=A[2]`、`B[0*5+2]=B[2]`，再读取 `A[1*2+1]=A[3]`、`B[1*5+2]=B[7]`，最后写 `C[7] = A[2]B[2] + A[3]B[7]`。若 thread 落在 `m=3` 或 `k=5`，mask 使其不执行这些访问和写回。

Triton MatMul 也可用 `pid_m`、`pid_k` 表达 program tile；它与 CUDA block 的概念相近，但具体 lane 到 tile 元素的映射由 Triton compiler 和 layout 决定，不能逐字符等同。

## 2. resident、eligible、issue：一个 warp 如何真的获得执行机会

### 2.1 Resident 是资源状态

kernel launch 后，block 被调度到 SM。一个 block 能否 resident，取决于它的 thread 数、warp 数、寄存器、shared memory，以及设备对 block/warp/thread 的上限。resident 只说明执行上下文和资源已经放在某个 SM 上，并不表示它正在执行，也不表示它的下一条指令马上会发出。

可以先用一个未取整的资源下界估算：

$$
R_{block}\approx T_{block}\times R_{thread},\qquad
W_{block}=\left\lceil\frac{T_{block}}{32}\right\rceil
$$

再与每个 SM 的 `regsPerMultiprocessor`、`sharedMemPerMultiprocessor`、`maxThreadsPerMultiProcessor`、`maxBlocksPerMultiProcessor` 和最大 active warps 比较。真正分配通常还受架构粒度影响，所以 `R_block` 只是解释方向。手册打印页 70–72（PDF 页 86–88）给出了 resident block、occupancy 和资源限制的完整关系，并提醒 `--maxrregcount` 可能以 spill 换取更多 resident block。

### 2.2 Eligible 是“下一条指令准备好了”

> [!IMPORTANT] 驻留不等于正在执行
> Resident 说明资源已经分配，eligible 说明下一条指令具备发射条件，issue 才是一次实际发射。**增加驻留 warp，只有在增加了可用的就绪工作时，才可能帮助隐藏延迟。**

resident warp 可能在等待 global load、shared load、依赖链上的前一条结果、barrier，或者因为其路径当前 inactive。只有当 warp 的 active threads 有下一条可执行指令、操作数和必要同步条件已满足时，它才是 eligible warp。调度器在每个周期从 eligible warps 中选择一个或多个候选，具体 issue 宽度、调度器数量和指令吞吐率由目标架构决定。

因此，下面两个 kernel 的 occupancy 可能相同，但 eligible/issue 行为不同：

```cpp
// 依赖链：每条 FMA 都等待上一个 accumulator 的结果
float x = seed;
for (int k = 0; k < K; ++k) {
    x = fmaf(a[k], b[k], x);
}
```

```cpp
// 四条独立链：调度器有机会在等待一条链时发射另一条链
// 假设 K 是 4 的倍数；一般输入需另加尾部处理。
float x0 = seed, x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
for (int k = 0; k < K; k += 4) {
    x0 = fmaf(a[k + 0], b[k + 0], x0);
    x1 = fmaf(a[k + 1], b[k + 1], x1);
    x2 = fmaf(a[k + 2], b[k + 2], x2);
    x3 = fmaf(a[k + 3], b[k + 3], x3);
}
float x = x0 + x1 + x2 + x3;
```

第二段保持数学求和目标不变，但改变了浮点加法次序，因此对照时需要检查误差容限，而不是要求逐位相同。它提高 Instruction-Level Parallelism（指令级并行，ILP）的机会，但会增加 accumulator 和 load 地址的活跃值。若寄存器压力让 active warps 下降，ILP 的收益可能被反噬。正确的性能问题不是“occupancy 够不够”，而是“在这个 shape 和数据路径下，eligible warps 是否足够填补当前指令的 latency，增加 ILP 的寄存器成本是否值得”。

### 2.3 Issue 是一次具体的指令发射

issue 不等于完成。一个 warp 发出 global load 后，可能要等待若干周期，期间调度器切换到其他 eligible warp；warp context 保存在片上，切换不需要像 CPU 线程那样保存完整上下文。若只有一个 warp resident，或者所有 warp 都在等同一依赖，SM 会出现 issue slot 空洞。若有多个独立 warp 或同一 warp 内有足够 ILP，空洞可能被填上。

**工程读数的正确顺序**是：先看 kernel 是否被资源限制为很少的 resident warp，再看 eligible/issue stall 的分类，最后判断是增加 block 数、缩短依赖链、改变 tile、改内存路径还是接受当前瓶颈。不要把一个 occupancy 百分比直接翻译成吞吐百分比。

**反例：`100% occupancy` 等于 `100% pipeline utilization`。** 100% occupancy 只表示 active warp 数达到该架构定义的上限；如果所有 warp 都在等待同一条长依赖或同一批内存请求，issue 仍可能空闲。相反，一个寄存器较多、occupancy 较低的 kernel 可能凭借更高 ILP 和更少的冗余同步获得更高吞吐。

**带答案的追问。**

问：把 block 从 128 threads 改成 256 threads，一定会让 kernel 更快吗？

答：不一定。它可能减少 block 数、改善每 block 的协作效率，也可能提高每 block 的寄存器/shared 消耗，降低 resident blocks；同时 warp 的尾部和分支分组会变化。需要同时测 registers/thread、active warps、eligible/stall 原因和 kernel time。

## 3. SIMT：同一条指令不等于同一条数据路径

SIMT（Single Instruction, Multiple Threads，单指令多线程）让程序员以 thread 标量语义写 kernel，同时由硬件以 warp 为单位取得高吞吐。一个 warp 的线程通常从相同 program address 开始，但每个线程有自己的寄存器状态和控制流语义。若条件分支在 warp 内产生两组路径，硬件会依次执行被至少一个 active thread 选中的路径，并关闭不在该路径上的线程；这就是 divergence（分支分歧）。

```cpp
if ((global_id & 1) == 0) {
    output[global_id] = expensive_even(input[global_id]);
} else {
    output[global_id] = expensive_odd(input[global_id]);
}
```

如果一个 warp 的 16 个 lane 走 even、16 个 lane 走 odd，两个分支都要执行，算术吞吐的有效利用率会下降。不同 warp 走不同分支没有同一 warp 内的串行化问题；分歧是 warp 内现象，不是“整个 grid 只能走一个分支”。

### 3.1 Predication 何时更好

对非常短的分支，编译器可能使用 predicate，让两条路径变成带条件的指令，而不是显式 branch。你也可以用选择表达式写出相同意图：

```cpp
float even_value = input[i] * 2.0f;
float odd_value = input[i] + 3.0f;
output[i] = ((i & 1) == 0) ? even_value : odd_value;
```

但 predication 不是免费消除分歧：如果两边都有昂贵计算，两个值可能都被计算，只是最终只写一个。短、均衡、低成本的选择适合 predication；长、稀疏或一边很昂贵的路径可能更适合 branch。应在 PTX/SASS 中确认编译器选择，并用相同输入测量。

### 3.2 Independent Thread Scheduling 不等于 warp 无关

Compute Capability（计算能力，简称 CC）7.0 及以后支持 Independent Thread Scheduling（独立线程调度，ITS）：硬件维护更细粒度的线程执行状态，允许 divergent/reconvergent 的调度灵活度提高。它不意味着一个 warp 的线程突然变成完全独立的标量处理器，也不授权省略所有同步。依赖 warp 内数据交换、共享状态或隐式 lockstep 的代码，仍应使用明确的 `__syncwarp(mask)` 或更高层同步原语。

```cpp
// 前置条件：完整 warp 的 32 个线程均到达这里，lane 0 提供有效 value。
const unsigned mask = 0xffffffffu;
int x = __shfl_sync(mask, value, 0);
```

较早架构的 warp 共享 program counter 和 active mask；支持 Independent Thread Scheduling 的架构可以更独立地推进线程。跨 lane 交换数据时不能依赖隐式锁步，应使用与参与范围一致的同步操作，例如 `__syncwarp()`。

**反例：用 `__syncthreads()` 替代所有同步。** `__syncthreads()` 是 block 级 barrier，成本和语义都比 warp 级同步更强；但它也不能修复 data race、错误的 active mask 或 barrier 前提前 return。应根据参与者范围选择 warp、block 或更大范围同步。

## 4. Wave 与尾效应：最后一批 block 的并行度会掉下来

假设 kernel 在每个 SM 上最多同时驻留 $B$ 个 block，设备有 $S$ 个 SM，grid 有 $G$ 个 block。粗略的 wave 数是：

$$
waves=\left\lceil\frac{G}{S\times B}\right\rceil
$$

前面的 wave 可以让大多数 SM 都有足够 block；最后一个 wave 只剩余：

$$
G\bmod(S\times B)
$$

个 block。若余数很小，很多 SM 在尾部没有工作，这就是 wave tail effect（波次尾效应）。即便 grid 数量刚好填满资源，block 内部也可能有 thread tail：最后一个 warp 的 active threads 少于 32，边界 mask 会让 lane inactive。

**完整算例。** 假设 `S=4`，资源允许每个 SM 驻留 `B=2` 个 block，grid 有 `G=10` 个 block。每波容量是 `S×B=8`，所以前一波使用 8 个 block，第二波只剩 2 个 block；最后一波只使用 8 个 block 槽位中的 2 个，即槽位填充率 25%；若分布到两个不同 SM，则 SM 覆盖率为 50%。这两个百分比不是同一指标，也不是实际指令利用率。若 `G=8`，`G mod (S×B)=0`，它是一个完整波，不存在 grid-level remainder；若 `G=12`，仍是两波但第二波有 4 个 block。不能仅据此断言 8 一定比 10 快，因为 block 的工作量、调度顺序和内存请求也会影响结果。

对矩阵 kernel，尾效应来自两层：M/N/K 的 tile 余数造成边界 block 中部分 lane inactive，grid 的 block 数与 SM×resident-block 容量不匹配造成最后 wave 不满。性能测试至少应包含整除 shape 和非整除 shape，否则会把一个有利的尾部形状误当成通用优化。

**带答案的追问。**

问：增加 grid block 数一定能消除尾效应吗？

答：不一定。增加后可能从一个很短的尾 wave 变成新的尾 wave，也可能带来更多边界 block 和内存流量；只有把 block 工作量、resident 上限和实际 launch shape 一起计算，才能决定是否值得。

## 5. 计算管线：普通 CUDA Core、FMA 与 Tensor Core

“计算管线”至少要分成普通标量/向量算术路径和矩阵乘加路径。普通 CUDA kernel 的 `fmaf` 通常表达一个 FP32 fused multiply-add（融合乘加，FMA）：

```cpp
float acc = 0.0f;
for (int k = 0; k < K; ++k) {
    acc = fmaf(a[k], b[k], acc);
}
```

Tensor Core 是专门的矩阵乘加数据路径，典型操作是 Matrix Multiply-Accumulate（矩阵乘加，MMA）。它要求特定 tile、输入类型、累加类型和布局；“用了矩阵乘”不等于编译器一定选择 Tensor Core。普通 CUDA Core FMA 和 Tensor Core MMA 的吞吐、占用资源和精度规则不同，必须从生成代码或 profiler 证据确认。

### 5.1 乘法输入、累加器和输出不是一个 dtype

一个常见的混合精度路径是：输入为 FP16（半精度浮点）或 BF16（脑浮点 16 位），乘法在低精度输入下进行，累加器为 FP32，最终根据接口要求把结果转换为 FP16/BF16 或保留 FP32。可以用公式表达为：

$$
acc_{fp32}\leftarrow acc_{fp32}+convert(a_{in})\times convert(b_{in}),
$$

然后：

$$
C_{out}=convert_{out}(acc_{fp32} \text{ 或完成缩放后的值})
$$

示例：

```cpp
#include <cuda_fp16.h>
__global__ void half_input_float_accum(const __half* a, const __half* b,
                                       float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float af = __half2float(a[i]);
    float bf = __half2float(b[i]);
    c[i] = fmaf(af, bf, 0.0f);
}
```

这里输出是 FP32，但它并不代表输入存储、带宽或整个算子使用了 FP32 Tensor Core 路径。反过来，输出 FP16 也不必然说明累加器只有 FP16；接口 dtype、内部 accumulator dtype 和硬件 MMA 类型必须分开记录。

FP16 tiled GEMM 使用另一套合同：`A[M,K] × B[K,N] => C[M,N]`，沿 `k` 归约；`threadIdx.x` 映射输出 `m`，`threadIdx.y` 映射输出 `n`。它不是 naive GEMM 的变量重命名，也不能直接把 FP16 tiled 与 FP32 naive 称为同一精度优化。shared memory 中的 `A_s/B_s` 元素为 `half`，因此这里不套用 float32 tile 的 bank-conflict 数字；实际 bank 行为应按 half 的地址访问和目标架构验证。

<!-- source-check: solutions/cuda/gemm/tiled_fp16.cu -->
~~~cpp {8-10,16-27,29-35,39-42}
constexpr int tileLen = 32;

__global__ void kernel(const half* A, const half* B, half* C,
                       int M, int N, int K, float alpha, float beta) {
    int m = blockIdx.x * tileLen + threadIdx.x;
    int n = blockIdx.y * tileLen + threadIdx.y;
    float sum = 0.0f;

    // 每个线程先搬运暂存
    __shared__ half As[tileLen][tileLen];
    __shared__ half Bs[tileLen][tileLen];

    // 遍历 K
    for (int t = 0; t < (K + tileLen - 1) / tileLen; ++t) {
        // 每个线程搬运自己负责的那块
        // A 矩阵按列步进
        int aK = t * tileLen + threadIdx.y;
        As[threadIdx.x][threadIdx.y] = (m < M && aK < K)
            ? A[m * K + aK] : __float2half_rn(0.0f);

        // B 矩阵按行步进
        int bK = t * tileLen + threadIdx.x;
        Bs[threadIdx.x][threadIdx.y] = (n < N && bK < K)
            ? B[bK * N + n] : __float2half_rn(0.0f);

        // 同步
        __syncthreads();

        // 计算
        for (int k = 0; k < tileLen; k++) {
            sum += __half2float(As[threadIdx.x][k]) *
                   __half2float(Bs[k][threadIdx.y]);
        }

        // 同步
        __syncthreads();
    }

    if (m < M && n < N) {
        C[m * N + n] = __float2half_rn(
            alpha * sum + beta * __half2float(C[m * N + n]));
    }
}
~~~

这段代码说明“输入、累加器、输出”是三个独立问题：`half` 省存储和带宽，FP32 `sum` 保留更宽的累加表示，最终转换回 half；`alpha/beta` 还决定了旧 C 是否属于数值合同。

注意原文件最后无条件计算 `beta * __half2float(C[m * N + n])`，所以即使调用者传 `beta=0`，C 也必须已经初始化；IEEE 浮点下 `0 * NaN` 仍然是 NaN，不能靠零系数消除未初始化读取。若新版本的合同明确允许 `beta==0`，可在 kernel 中为 `beta==0` 单独分支，直接写 `__float2half_rn(alpha * sum)`，从而避免读取旧 C；这里只记录改进方向，不修改作为出处的原文件。源码头部关于“HBM 减少 K/TILE 倍”等性能推断不作为本章事实；能否减少真实 DRAM 流量必须结合 cache、shape、设备和 profiler 测量。

### 5.2 Triton `tl.dot` 的 `input_precision`

> [!WARNING] FP32 累加器不是全部数值合同
> **输入存储类型、乘法允许的输入精度、累加类型和输出类型要分别确认。** 只把 acc 或输出 tensor 声明为 FP32，不能排除 FP32 输入的 dot 使用 TF32 路径。

对 FP32 输入，Triton 的 `tl.dot` 需要明确理解 input precision。一个容易出错的对照是：PyTorch reference 关闭 TF32，而 Triton `tl.dot` 使用默认输入精度，编译器可能选择允许 Tensor Core/TF32 风格的内部路径；大归约维度下，数值误差就会超过严格 FP32 比较阈值。严格 IEEE FP32 对照应写成：

```python
acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
acc += tl.dot(tile_a, tile_b, input_precision="ieee")
```

同时在 PyTorch reference 侧固定：

```python
torch.backends.cuda.matmul.allow_tf32 = False
```

`input_precision="ieee"` 约束的是 dot 输入计算口径，不等于“所有中间张量都自动变成 FP32”，也不等于“输出 dtype 已经确定”。必须另外检查 `tile_a`、`tile_b`、`acc` 和 `tl.store` 的 dtype。小 shape 通过而长归约出现误差时，先检查 pointer 与 mask 的索引范围，再用同 shape、dtype 和容差分别比较 IEEE 与 TF32；把精度策略显式化，才能把数值路径差异与寻址错误分开。

**反例：看到 `dtype=tl.float32` 就断言用了 IEEE FP32。** accumulator dtype 只描述累加器的表示；输入 dot 的实现路径、Tensor Core/TF32 允许与否、转换和最终 store 仍可能不同。数值语义要用参数、框架开关、误差统计和生成代码共同定义。

**带答案的追问。**

问：把 Triton 的输出 tensor 声明为 `torch.float32`，是否自动修复了 `tl.dot` 的 FP32 误差？

答：不一定。输出存储 dtype 只决定写回表示；误差可能已经在输入乘法和归约中产生。应显式设置 `input_precision="ieee"`，关闭 reference 的 TF32，固定 shape/seed，再比较最大绝对误差和最大相对误差。

## 6. 依赖链、ILP 与一个可运行的执行对照

Linux服务器可从仓库根目录执行下面的命令。先跑小输入检查边界，再扩大规模观察指令与资源；后面的PowerShell命令则在本章目录执行。

```bash
cd roadmap/curriculum/gpu/02-cuda-execution-and-scheduling
nvcc -O3 -std=c++17 -lineinfo --resource-usage \
  examples/execution_and_scheduling.cu -o /tmp/cuda-execution
/tmp/cuda-execution 257 20 5
compute-sanitizer --tool memcheck --error-exitcode=1 /tmp/cuda-execution 257 20 5
/tmp/cuda-execution 1048576 200 50
```

程序将每个输出与对应CPU参考比较，非有限值或超限误差返回失败。`source_FMA_GFLOP_s` 只按源码写出的FMA工作量计数，每次FMA计2 FLOP；检查编译器是否保留这些指令后，才能把归一化吞吐用于分析硬件。

本章的 execution_and_scheduling.cu 包含四个 kernel：

- `dependent_chain_kernel`：一个 accumulator 的长依赖链；
- `independent_accumulators_kernel`：多个 accumulator 展开链；
- `divergent_branch_kernel`：同一 warp 的偶数/奇数 lane 走两条路径；
- `predicated_select_kernel`：用选择表达式形成短选择。

程序用相同输入、四个各自对应的 CPU reference 和 CUDA events 检查结果。`independent_accumulators_kernel` 每个元素每轮执行四个 FMA，算术工作量不等于 `dependent_chain_kernel` 的每元素一个 FMA；比较时应按实际 FMA 数分别正规化吞吐，并同时记录误差、寄存器和 kernel time。使用它的顺序是先确认每个输出对应自己的 CPU reference，再用 Nsight Compute 或编译资源报告观察寄存器和分支，再改变 `n`、block size 和迭代次数。

```powershell
nvcc -O3 -std=c++17 -lineinfo --resource-usage `
  examples/execution_and_scheduling.cu -o execution_and_scheduling.exe

./execution_and_scheduling.exe 1048576 200 50
```

如果输出正确但 dependent kernel 更快，不能立即否定 ILP 分析：可能是编译器融合、数学强度、内存访问或计时粒度主导。若 independent 版本寄存器显著增加而 active warps 下降，ILP 也可能不值得。该例的价值是把“依赖链”和“分歧”变成可复现的代码，而不是预先假定某个 GPU 一定有固定比例提升。

## 7. 从 CUDA 源码到 PTX 与 SASS

CUDA 编译路径需要同时看三个身份：

1. CUDA C++ 源码表达 thread、block、barrier、dtype 和地址。
2. PTX（Parallel Thread Execution，并行线程执行）是面向虚拟 ISA 的中间表示，适合检查 predicate、branch、address space、FMA/MMA 候选和寄存器临时值。
3. SASS 是面向具体 CC 目标的机器指令，适合确认最终的 branch/predicate、global/shared/local load/store、barrier 和架构特定 MMA 指令。

命令如下：

```powershell
nvcc -O3 -std=c++17 -lineinfo --resource-usage `
  examples/execution_and_scheduling.cu -o execution_and_scheduling.exe

nvcc -O3 -std=c++17 -ptx -arch=sm_XX `
  examples/execution_and_scheduling.cu -o execution_and_scheduling.ptx

nvcc -O3 -std=c++17 -cubin -arch=sm_XX `
  examples/execution_and_scheduling.cu -o execution_and_scheduling.cubin

cuobjdump --dump-ptx execution_and_scheduling.exe > execution_and_scheduling.ptx.txt
cuobjdump --dump-sass execution_and_scheduling.exe > execution_and_scheduling.sass.txt
nvdisasm --print-code --print-line-info execution_and_scheduling.cubin
```

`-arch=sm_XX` 应替换为当前测量设备的 Compute Capability；不要用另一台设备的 SASS 代替目标设备证据。`--resource-usage` 和 `-Xptxas=-v` 适合确认寄存器、shared、local；`cuobjdump`/`nvdisasm` 适合确认最后的指令形态。调试时先缩到小 shape，launch 后立即检查 launch/configuration error，再在对应 stream/device 同步处观察异步执行错误；`CUDA_LAUNCH_BLOCKING=1` 可用于临时定位错误归属，`compute-sanitizer` 可辅助发现非法内存访问。两者都会改变执行/观测条件，不能用于性能计时。

若要检查 Triton MatMul 的低层结果，先运行实际 kernel，再从当前环境的 Triton cache 定位对应 cubin/PTX；缓存布局随 Triton 版本和环境变化，不应假定固定路径。对照时固定 precision、shape 与误差容限，再使用 `cuobjdump`/`nvdisasm` 和 profiler 查看目标环境的编译产物。

## 8. 实践复盘：从 GEMM、Softmax 和 Triton 代码提炼的机制

### 8.1 从真实 naive GEMM 读懂调度

naive FP32 GEMM 把一个 `C[m,k]` 分给一个 thread。16×16 block 让 warp 横跨两行，但每一半 warp 的 `k` 连续，因此固定归约 `n` 的 B 访问和 C 写回仍是两段连续地址；A 则在每一半 warp 内由同一行线程复用同一地址。调度分析必须沿“一条 warp 指令、固定一次 n”展开，不能把同一线程不同 n 之间的列 stride 误当成 warp 访问不合并。

### 8.2 从真实 tiled GEMM 读懂生产者—消费者

FP16 tiled GEMM 让 block 中线程先把 global memory 的 A/B 子块写入 shared memory，再由所有线程消费这两个 tile 做 32 次局部乘加。第一处 `__syncthreads()` 保证 producer 阶段完成后 consumer 才读取；第二处保证所有线程读完旧 tile 后，下一轮 producer 才能覆写 shared memory。tile 尺寸、寄存器与 barrier 都会改变 resident/eligible/issue 行为，所以“更多线程”或“更大 tile”不是自动加速理由。

### 8.3 复盘的证据边界

旧代码提供地址、线程、dtype 和同步的真实证据；性能结论只能连同原实验的设备、shape、精度和计时范围一起引用。新的判断应按“硬件机制 → 代码旋钮 → 预期 counter → 实测”闭环验证，不把未在当前环境测量的数字写成结论。

## 9. CUDA Tile：把线程分工交给编译器之后，还要控制什么

写 CUDA SIMT kernel 时，通常先确定一个 thread 负责哪些元素，再安排 warp 和 block 的协作。CUDA Tile 换了表达粒度：一个 tile block 的代码描述一组元素的计算，编译器安排这些元素如何分布到硬件线程、使用哪些加载和计算指令。这里的 SIMT 是 Single Instruction, Multiple Threads，单指令多线程；两种写法改变的是编程接口，不是 GPU 突然换了一套没有线程的执行硬件。

例如向量加法，SIMT 写法计算 `blockIdx.x * blockDim.x + threadIdx.x`；tile 写法取第 `bid` 个长度为 128 的数据块，相加后存回。后者少写了线程索引，但仍要决定 tile 大小、访问模式、数据类型、边界和 grid。把这些决定交给一个更粗的接口，并不意味着性能优化已经完成。

CUDA Tile 有 Python 和 C++ 两个入口：Python 使用 `cuda.tile`，C++ 使用 `cuda_tile.h` 和 `cuda::tiles`。本节 C++ 语法以 CUDA Toolkit 13.3 为基准；旧 Toolkit 不因为支持普通 CUDA C++ 就同时支持 `__tile_global__`。Python 包、驱动、编译器后端与目标 GPU 也必须互相兼容。Triton 是另一套编译系统，不能把两者相似的 API 名称当成二进制或语言兼容承诺。

### 9.1 Array、tile 和硬件缓冲区是三个层次

Array 表示设备内存中的数组，包含 shape、dtype、stride 等信息。Tile 是 kernel 内具有编译期 shape 和 dtype 的值。假设从 `A[10,16]` 取出一个 `2×4` tile：源数组还在 global memory；加载产生的是八个逻辑数值。编译器可能让它们位于寄存器、通过 shared memory 中转，或采用其他合法的数据通路。

**Tile 不是 shared memory 的别名，也不是“一个 thread 的数组”。** `tile.shape=(2,4)` 只说明八个元素怎样组织，不能据此判断八个线程、八个寄存器或某个固定的物理布局。寄存器占用、shared 用量和实际指令，要到编译产物与 profiler 中确认。

Tile 具有值语义。写 `y = x + 1` 是产生新值，不会自动把源 Array 加一；只有 store 才把结果写回 Array。编译器可以消除不必要的复制，但程序不能依赖两个 tile 变量作为可互相修改的存储别名。Tile 的各维大小要求为 2 的幂，Array 的大小则不需要，例如长度 1025 的数组仍能用长度 128 的 tile 处理。

把已经熟悉的 Triton 语句放在这个层次中，就容易分清地址和值：

~~~python
offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
ptrs = a + offsets                         # 指针组成的 tile，还没有读取 a
x = tl.load(ptrs, offsets < N, other=0.0)   # 数据组成的 tile
y = x + 1.0                               # 新的逻辑值
tl.store(c + offsets, y, offsets < N)       # 写入 c 的有效元素
~~~

`ptrs` 中每个位置对应一个地址，这与你之前对 MatMul 地址块的理解相同。至于由哪些 lane 发出多少条内存指令，不是 Python 中“一个位置”与 CUDA 中“一个 thread”的一一对应关系。

### 9.2 Tile 坐标和元素坐标不能混用

设 Array 的 shape 为 `(M,N)`，tile 的 shape 为 `(TM,TN)`。tile 坐标 `(bm,bn)` 中的局部元素 `(i,j)` 对应：

$$
r=b_mT_M+i,\qquad c=b_nT_N+j.
$$

对于 row-major 连续数组，它的元素偏移是：

$$
\operatorname{offset}(i,j)=(b_mT_M+i)N+(b_nT_N+j).
$$

取 `A[10,16]`、tile `(2,4)`、tile 坐标 `(1,2)`，起点是元素 `(2,8)`，不是元素 `(1,2)`。这个 tile 的八个元素如下：

| tile 内局部行 | 第 0 列 | 第 1 列 | 第 2 列 | 第 3 列 |
|---|---|---|---|---|
| 0 | A[2,8]，偏移 40 | A[2,9]，41 | A[2,10]，42 | A[2,11]，43 |
| 1 | A[3,8]，偏移 56 | A[3,9]，57 | A[3,10]，58 | A[3,11]，59 |

规则 tile-space load 接收的是 `(1,2)` 这种块坐标；gather 接收的是具体元素的索引。Triton 常见指针写法则直接构造这八个地址。三种表达可以描述同一片数据，但传入的数字含义不同。

规则访问为编译器提供了数组与分块结构，受支持的设备上可能采用 Tensor Memory Accelerator（TMA，张量内存加速器）搬运。Gather 更适合查表、置换和数据相关索引。不要为了表面统一，把天然规则的二维 tile 全部改写成任意指针访问；也不要因为用了规则 view，就认定最后一定发出了 TMA 指令。

### 9.3 同一个向量加法，三种表达

普通 CUDA C++ 显式写一个线程的元素索引：

~~~cpp
__global__ void add_simt(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
// n > 0；a、b、c 指向至少 n 个 float，c 不与输入重叠。
add_simt<<<(n + 127) / 128, 128, 0, stream>>>(a, b, c, n);
~~~

CUDA Tile C++ 则为指针附上数组 shape，再定义分块：

~~~cpp
#include "cuda_tile.h"

__tile_global__ void add_tile(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ c, int n) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    a = ct::assume_aligned(a, 16_ic);
    b = ct::assume_aligned(b, 16_ic);
    c = ct::assume_aligned(c, 16_ic);
    auto av = ct::partition_view{ct::tensor_span{a, ct::extents{n}},
                                 ct::shape{128_ic}};
    auto bv = ct::partition_view{ct::tensor_span{b, ct::extents{n}},
                                 ct::shape{128_ic}};
    auto cv = ct::partition_view{ct::tensor_span{c, ct::extents{n}},
                                 ct::shape{128_ic}};
    int block = ct::bid().x;
    auto x = av.load_masked(block);
    auto y = bv.load_masked(block);
    cv.store_masked(x + y, block);
}
// CUDA Tile C++ 的第二个 launch 参数必须为 1，不是指定一个物理线程。
// n > 0；独立 cudaMalloc 分配满足此处的对齐与不重叠合同。
add_tile<<<(n + 127) / 128, 1, 0, stream>>>(a, b, c, n);
~~~

这里 `128_ic` 是编译期常量，`n` 是运行时数组长度。`load_masked` 处理最后一个不完整 tile；普通 `load` 要求整个 tile 在界内。`assume_aligned` 是程序向编译器作出的保证，不会把未对齐指针修正为对齐指针。假如把 `a+1` 传入，FP32 地址通常只移动四字节，就不能继续作出同样的 16 字节保证。

Python cuTile 的完整向量加法如下。设备数组由 PyTorch 管理，计算由 cuTile kernel 完成；检查值选用可精确表示的小整数，避免把舍入误差和地址错误混在一起。输出分配额外七个元素作为哨兵区，传给 kernel 的数组 view 则严格限制为前 n 个元素。

<!-- source-check: examples/tile_vector_add.py -->
~~~python
#!/usr/bin/env python3
"""Minimal cuTile vector add with explicit tail-storage validation."""

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a cuTile FP32 vector add for several tail sizes."
    )
    return parser.parse_args()


def run() -> int:
    # Keep optional GPU-stack imports after argparse so --help works on a CPU-only
    # machine and does not require torch or cuda.tile to be importable.
    try:
        import torch
        import cuda.tile as ct
    except ImportError as error:
        # A missing top-level module is an expected local skip.  Broken imports
        # inside an installed package remain errors and must not be masked.
        missing = getattr(error, "name", "")
        if isinstance(error, ModuleNotFoundError) and missing in (
            "torch",
            "cuda",
            "cuda.tile",
        ):
            print(f"SKIP: cuTile/torch stack unavailable: {error}")
            return 77
        raise

    if not torch.cuda.is_available():
        print("SKIP: no usable CUDA device")
        return 77

    from importlib.metadata import PackageNotFoundError, version

    try:
        tile_version = version("cuda-tile")
    except PackageNotFoundError:
        tile_version = "unknown"
    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"cuda.tile={tile_version} gpu={torch.cuda.get_device_name(0)}"
    )

    @ct.kernel
    def vec_add(a, b, c, TILE: ct.Constant[int]):
        bid = ct.bid(0)
        x = ct.load(
            a,
            index=(bid,),
            shape=(TILE,),
            padding_mode=ct.PaddingMode.ZERO,
        )
        y = ct.load(
            b,
            index=(bid,),
            shape=(TILE,),
            padding_mode=ct.PaddingMode.ZERO,
        )
        ct.store(c, index=(bid,), tile=x + y)

    device = torch.device("cuda")
    sentinel = -12345.25

    def run_case(n: int) -> None:
        # Integer-like FP32 values keep the reference comparison exact.
        indices = torch.arange(n, dtype=torch.float32, device=device)
        a = indices - 7.0
        b = torch.remainder(indices, 13.0) + 2.0
        output_storage = torch.full(
            (n + 7,), sentinel, dtype=torch.float32, device=device
        )
        c = output_storage[:n]

        if n == 0:
            # Never submit a zero-sized grid; the empty case is host-side only.
            print("case n=0: bypassed launch (no zero grid)")
            return

        if not (a.is_contiguous() and b.is_contiguous() and c.is_contiguous()):
            raise AssertionError(f"case n={n} did not produce contiguous arrays")
        pointers = (a.data_ptr(), b.data_ptr(), output_storage.data_ptr())
        if len(set(pointers)) != len(pointers):
            raise AssertionError(f"case n={n} unexpectedly aliases input/output")

        grid = ((n + 127) // 128,)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            vec_add,
            (a, b, c, 128),
        )
        torch.cuda.current_stream().synchronize()

        expected = a + b
        torch.testing.assert_close(c, expected, rtol=0, atol=0)
        if not torch.all(output_storage[n:] == sentinel).item():
            raise AssertionError(f"case n={n} overwrote the output tail")
        print(f"case n={n}: PASS")

    for n in (0, 1, 127, 128, 129, 1025):
        run_case(n)
    return 0


def main() -> int:
    parse_args()
    return run()


if __name__ == "__main__":
    sys.exit(main())
~~~

`ct.load` 从传入的数组获得长度，`index` 仍然是 tile 坐标。Python 的 tile-space store 丢弃部分尾块的越界写入；load 则在这里显式选择零填充。`ct.Constant[int]` 固定的是 tile 大小，改变这个值可能触发新的编译特化，而不是给同一份机器码换一个普通标量参数。

### 9.4 尾块：哪些元素需要有定义

长度 129、tile 长度 128 时，grid 有两个 tile block。第二个 block 只有第一个元素有效，其余 127 个加载位置填零，输出只存第一个位置。长度为零时在 host 直接返回，不发起零大小 grid，也不读取一个完全处于 Array 外的 tile。

本节的部分尾块处理不应推广成“任意越界 tile 都安全”：tile-space 接口要求正确计算 grid，完全在数组外的 tile 不属于部分边界块的合同。Gather/scatter 的边界合同另外定义。

填充值还取决于接下来的计算。加法或 GEMM 归约的尾项用零；求最大值通常需要负无穷，否则全为负数的有效输入会被填充的零改变。Softmax 在减最大值和指数运算之前要排除无效位置，让无效位置的指数贡献为零。不能因为一个向量加法例子用了 `ZERO`，就把所有算子统一零填充。

同样，元素级选择 `ct.where(mask, x, y)` 不等于阻止此前的非法 load。如果地址可能越界，应先用正确的加载接口或 mask 保护访问，然后选择计算结果。选择表达式处理的是值，不会撤回已经表达的内存读取。

### 9.5 GEMM 的循环与 accumulator 并没有消失

沿用本课程的命名，`A[M,N] @ B[N,K] = C[M,K]`，N 为归约维。一个 tile block 负责输出的 `BM×BK` 块，每次沿 N 读取 `BM×BN` 和 `BN×BK`。Python cuTile 可以直接表达这个算法：

~~~python
@ct.kernel
def matmul_tiles(A, B, C, BM: ct.Constant[int],
                 BN: ct.Constant[int], BK: ct.Constant[int]):
    bm, bk = ct.bid(0), ct.bid(1)
    steps = ct.num_tiles(A, axis=1, shape=(BM, BN))
    acc = ct.full((BM, BK), 0, dtype=ct.float32)
    for bn in range(steps):
        a = ct.load(A, index=(bm, bn), shape=(BM, BN),
                    padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(bn, bk), shape=(BN, BK),
                    padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a, b, acc)
    ct.store(C, index=(bm, bk), tile=acc.astype(C.dtype))
~~~

这段 kernel 以连续二维 FP16 输入、FP32 累加为例，launch grid 是 `(ceil(M/BM), ceil(K/BK))`；tile 大小需要符合目标实现的矩阵运算约束。`ct.mma` 返回更新后的 accumulator，不能只调用而丢弃返回值。输出转换放在归约结束后，避免每一小块都提前舍入到低精度。

它与熟悉的 `acc += tl.dot(tile_a, tile_b)` 有相同的分块数学结构，但不是逐条相同的指令合同。数据复用次数、矩阵指令布局、流水深度和资源分配仍由源代码与编译共同决定。把 MatMul 从 Triton 改写为 cuTile，并不能仅凭代码变短就证明更快。

以输入元素占 s 字节计算，一轮理想 tile 加载需要：

$$
B_{\mathrm{load}}=s(B_MB_N+B_NB_K),\qquad
F_{\mathrm{tile}}=2B_MB_NB_K.
$$

增大输出块可提高这部分计算/读取比，同时把 accumulator 从 `BM×BK` 个 FP32 值继续扩大。无论采用哪种语言，这个复用与资源压力的矛盾都存在；最终还要计算输出写回、多级缓存和并发 CTA 的影响。

### 9.6 编译器提示和性能证据

CUDA Tile 的 hints（优化提示）影响编译选择。例如 occupancy hint 表示希望每个 SM 驻留的 CTA 数，不是传统 occupancy 百分比；latency hint 是访存强度的等级提示，不是“这次访问需要几纳秒”；允许 TMA 只表示允许选择该通路，不保证选择成功。

这些提示也不是 Triton `num_warps`、`num_stages` 的替代拼写。调整一个提示后，应先检查编译产物有没有变化，再比较 registers、shared、active warps、内存指令和 kernel time。若编译器生成相同代码，多测一个不同提示的配置并没有构成新的优化方案。

普通 SIMT kernel 与 tile kernel 可以在同一 host 程序中使用相同 global buffer，按 stream 顺序先后调用。C++ tile 函数与普通 `__device__` 函数则不能直接互相调用。不要把“共享设备内存和 launch 体系”误解为“kernel 内任意混用两种语言”。

## 10. Driver API、Runtime API、context 与 module

CUDA Runtime API 与 CUDA Driver API 最终都驱动同一套 GPU 执行机制，但把控制权放在不同层。Runtime API（`cuda*`）通常由编译器生成的 host stub 管理 kernel 参数、当前设备的 primary context、模块装载与常见资源；它适合大多数单应用 CUDA C++ 程序。Driver API（`cu*`）显式操作 `CUdevice`、`CUcontext`、`CUmodule`、`CUfunction`、`CUstream` 和 `CUdeviceptr`，可以从 PTX/CUBIN 装载模块并做 JIT，适合运行时编译、插件、框架后端和需要精细控制 context 的系统代码。

`CUcontext` 是一组设备地址空间、模块和相关状态的执行上下文，不等于 CPU 线程。Driver API 的许多调用作用于调用线程当前的 context；应用需要明确 retain/create、set-current，并在资源销毁后 release/destroy。Runtime 默认使用设备的 primary context。需要 Runtime/Driver 混用时，常见做法是 Driver 侧 `cuDevicePrimaryCtxRetain` 后 `cuCtxSetCurrent`，让两边指向同一个 primary context；不要在 Runtime 已使用 primary context 时再随意创建另一个 context 并混着传指针。跨 context 的 `CUdeviceptr`、stream、module/function 句柄不能仅因数值看起来相同就互换。

模块（module）是设备代码和相关符号的装载单元；函数句柄（`CUfunction`）由模块内符号名查得。下面的完整最小程序把 PTX 作为字符串嵌入可执行文件，显式建立 primary context、装载 PTX、查找 kernel、分配显存、传参启动并复制结果。PTX 中每个参数的宽度和 host 侧 `kernel_params` 指向的对象必须逐项匹配：三个地址是 64 位 `CUdeviceptr`，`n` 是 32 位 `int`。

```cpp
.visible .entry vector_add(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_c,
    .param .u32 param_n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<7>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [param_a];
    ld.param.u64 %rd2, [param_b];
    ld.param.u64 %rd3, [param_c];
    ld.param.u32 %r2, [param_n];
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r0, %r1, %r0, %r3;
    setp.ge.u32 %p1, %r0, %r2;
    @%p1 bra DONE;

    cvt.u64.u32 %rd4, %r0;
    shl.b64 %rd4, %rd4, 2;
    add.u64 %rd5, %rd1, %rd4;
    add.u64 %rd6, %rd2, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    add.u64 %rd5, %rd3, %rd4;
    st.global.f32 [%rd5], %f3;

DONE:
    ret;
}
```

context 复用、PTX module/function 的装载与查找、Driver kernel launch 的参数数组如下。每个 `kernel_params[i]` 是“指向 host 参数存储”的指针，不是把设备地址直接当成 host 指针传入。这个例子用同步 HtoD/DtoH 接受普通 `std::vector` 页内存；若改成 `cuMemcpy*Async`，host buffer 必须满足对应的页锁定/注册条件，并且其生命周期要覆盖异步传输。

<!-- source-check: examples/driver_api_vector_add.cu -->
```cpp
    CU_CHECK(cuDevicePrimaryCtxRetain(&context, device));
    CU_CHECK(cuCtxSetCurrent(context));
    CU_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    CU_CHECK(cuModuleLoadDataEx(&module, kVectorAddPtx, 0, nullptr, nullptr));
    CU_CHECK(cuModuleGetFunction(&function, module, "vector_add"));

    CU_CHECK(cuMemAlloc(&d_a, bytes));
    CU_CHECK(cuMemAlloc(&d_b, bytes));
    CU_CHECK(cuMemAlloc(&d_c, bytes));
    // Synchronous copies accept these std::vector pageable host buffers.
    CU_CHECK(cuMemcpyHtoD(d_a, host_a.data(), bytes));
    CU_CHECK(cuMemcpyHtoD(d_b, host_b.data(), bytes));

    CU_CHECK(cuLaunchKernel(function,
                            blocks, 1, 1,
                            threads, 1, 1,
                            0, stream, kernel_params, nullptr));
    CU_CHECK(cuStreamSynchronize(stream));
    CU_CHECK(cuMemcpyDtoH(host_c.data(), d_c, bytes));
```

`cuModuleLoadDataEx` 的 Driver API 原型包含 `numOptions`、`options` 两个 JIT 参数；即使没有 JIT 选项，也要传 `0, nullptr`。kernel launch 是异步入队；这里先等 stream 完成再读回，避免把 host 校验和设备执行的完成边界混在一起。错误清理按“停止/等待使用者 → 销毁 stream → 释放设备内存 → unload module → release primary context”的顺序进行。主 context 由 retain/release 计数管理，不能对它调用 `cuCtxDestroy`。

```bash
nvcc -std=c++17 -O2 roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/driver_api_vector_add.cu -lcuda -o driver_api_vector_add
./driver_api_vector_add --help
./driver_api_vector_add 257
```

程序以无设备退出码 77 表示跳过，其他 Driver 错误仍是失败。PTX 使用 `.target sm_70` 和 `.version 7.0`，driver 会在 `cuModuleLoadDataEx` 时为实际设备 JIT；本例不依赖 NVCC 生成设备代码，但仍需 CUDA Toolkit 头文件、NVCC host 编译器与兼容的 NVIDIA Driver。

### 10.1 Runtime 初始化、lazy loading 与错误定位

Runtime 初始化和 device module 装载是两个不同边界。第一次 Runtime 调用可能建立线程关联的 Runtime 状态并初始化/选择 primary context；这不代表程序中每个 kernel 的设备代码都已经装入。CUDA 11.7 引入 lazy loading；CUDA 12.2 起 Linux 默认启用，12.3 起 Windows 默认启用，当前文档将默认行为描述为 lazy。`CUDA_MODULE_LOADING=LAZY|EAGER` 要在进程启动前设置：lazy 把具体 kernel 装载推迟到首次需要时，eager 把装载开销前移。运行时可用 Driver API `cuModuleGetLoadingMode` 查询模式；`cudaFuncGetAttributes` 或 `cuModuleGetFunction` 可显式触发指定函数装载。仅调用 `cudaFree(nullptr)` 可以帮助把 Runtime/context 初始化从后续阶段分离，不能据此声称 kernel 已预载。

下面示例把第一次调用 `add_one` 作为可能发生模块装载的位置；加 `--preload` 则用 `cudaFuncGetAttributes` 提前装入该函数。launch 后立即检查 `cudaGetLastError()`，再在 stream/device 同步点观察异步执行错误；调试时可以把 `CUDA_LAUNCH_BLOCKING=1` 作为定位辅助手段，但不能用它测性能，也不能用逐 kernel 同步的调试行为代表正式异步路径。

<!-- source-check: examples/runtime_lazy_loading.cu -->
```cpp
    // This explicit no-op allocation forces Runtime/context initialization;
    // it does not, by itself, prove every kernel module has been loaded.
    CUDA_CHECK(cudaFree(nullptr));
    CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(int),
                          cudaMemcpyHostToDevice));

    if (preload) {
        cudaFuncAttributes attributes{};
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, add_one));
        std::printf("kernel preloaded: registers=%d\n", attributes.numRegs);
    } else {
        std::printf("first kernel launch is the first use of add_one\n");
    }

    add_one<<<blocks, threads>>>(device_input, device_output, n);
    // Immediate check reports launch/configuration errors.  Execution faults
    // are observed at the stream synchronization boundary below.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
```

```bash
nvcc -std=c++17 -O2 roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/runtime_lazy_loading.cu -o runtime_lazy_loading
CUDA_MODULE_LOADING=LAZY ./runtime_lazy_loading
CUDA_MODULE_LOADING=EAGER ./runtime_lazy_loading --preload
CUDA_LAUNCH_BLOCKING=1 ./runtime_lazy_loading
```

比较首次调用延迟时，分别记录 Runtime/context 初始化、函数预载、首次 launch 和稳定重复调用；否则 lazy module load、JIT、allocator 初始化和 kernel 本身会混成一个数字。若错误只在同步时出现，应沿该 stream 上此前入队的操作逆向定位，不能简单把同步 API 当作出错 API 的来源。

## 11. CUDA Python：控制 API 与 kernel DSL 是两层

Python 只是 CUDA 的宿主语言入口，不自动意味着“Python 函数会在 GPU 执行”。先区分控制面与 kernel 面：`cuda.bindings` 把 Driver/Runtime C API 暴露给 Python，负责初始化、context、stream、内存、模块与版本查询；`cuda.core` 提供更 Pythonic 的控制接口；`cuda.lang`（SIMT）和 `cuda.tile`（Tile）才是写设备函数/kernel 的 DSL。Numba-CUDA 的 `@cuda.jit` 是另一种 SIMT kernel DSL。CuPy、PyTorch 等是数组/框架层，替你管理一部分设备内存和当前 stream；它们不是 `cuda.bindings`，也不把普通 Python 函数变成 kernel。

这一区分决定所有权和地址解释。CPU `numpy.ndarray` 的 `data` 是 host 虚拟地址，不能当作 `CUdeviceptr` 解引用；CUDA 设备指针只在所属 device/context 的设备地址空间有效。对连续 FP32 向量，元素 `i` 的字节偏移为 `4i`，而 `ptr + i` 的 C/C++ 指针算术是 `4i` 字节；Driver API 的 `CUdeviceptr` 则是字节地址，传给 kernel 前要遵从 kernel 参数 ABI。Python binding 返回的 `CUdeviceptr` 不是可由 Python 直接索引的 NumPy 数组。

下面的程序只用 `cuda.bindings.driver` 建立/选中 primary context、创建非默认 stream、分配设备地址、在该 stream 排入一个异步清零，再 D2H 回 NumPy 缓冲区检查哨兵值。它刻意不伪装成 kernel DSL：清零是 Driver API 操作。primary context retain 会返回句柄但不会自动将它设为当前；可复用函数还要先保存调用线程原来的 current context，结束时恢复它。分配、排队操作、同步、读回、释放与 context release 的次序构成生命周期合同。

<details>
<summary>完整 Python Driver bindings 程序</summary>

<!-- source-check: examples/cuda_python_bindings_lifecycle.py -->
```python
#!/usr/bin/env python3
"""Show explicit CUDA Driver bindings, context, stream, and device-pointer lifetime."""

import sys

import numpy as np


def main() -> int:
    try:
        from cuda.bindings import driver as cu
    except ImportError as error:
        print(f"SKIP: cuda.bindings is unavailable: {error}")
        return 77

    success = cu.CUresult.CUDA_SUCCESS

    def checked(call, where):
        result = call()
        status, *values = result
        if status != success:
            _, name = cu.cuGetErrorName(status)
            _, message = cu.cuGetErrorString(status)
            raise RuntimeError(f"{where}: {name}: {message}")
        return values[0] if values else None

    def cleanup(call, where):
        try:
            checked(call, where)
        except Exception as error:
            print(f"cleanup warning: {error}", file=sys.stderr)

    checked(lambda: cu.cuInit(0), "cuInit")
    previous_context = checked(cu.cuCtxGetCurrent, "cuCtxGetCurrent")
    driver_version = checked(cu.cuDriverGetVersion, "cuDriverGetVersion")
    device_count = checked(cu.cuDeviceGetCount, "cuDeviceGetCount")
    print(f"driver API version={driver_version}; visible devices={device_count}")
    if device_count == 0:
        print("SKIP: no CUDA device is visible")
        return 77

    device = checked(lambda: cu.cuDeviceGet(0), "cuDeviceGet")
    context = None
    stream = None
    device_ptr = None
    context_is_current = False
    host_word = np.full(1, 0xA5A5A5A5, dtype=np.uint32)
    try:
        context = checked(lambda: cu.cuDevicePrimaryCtxRetain(device),
                          "cuDevicePrimaryCtxRetain")
        checked(lambda: cu.cuCtxSetCurrent(context), "cuCtxSetCurrent")
        context_is_current = True
        stream = checked(lambda: cu.cuStreamCreate(0), "cuStreamCreate")
        device_ptr = checked(lambda: cu.cuMemAlloc(4), "cuMemAlloc")
        print(f"device pointer handle={device_ptr}; allocated bytes=4")

        # The 32-bit pattern is written asynchronously in this stream.
        checked(lambda: cu.cuMemsetD32Async(device_ptr, 0, 1, stream),
                "cuMemsetD32Async")
        checked(lambda: cu.cuStreamSynchronize(stream), "cuStreamSynchronize")
        checked(lambda: cu.cuMemcpyDtoH(host_word, device_ptr, host_word.nbytes),
                "cuMemcpyDtoH")
        np.testing.assert_array_equal(host_word, np.array([0], dtype=np.uint32))
        print(f"D2H verification passed: {host_word[0]}")
    finally:
        # Never release storage while work queued in its stream can still use it.
        if stream is not None:
            cleanup(lambda: cu.cuStreamSynchronize(stream), "final stream sync")
        if device_ptr is not None:
            cleanup(lambda: cu.cuMemFree(device_ptr), "cuMemFree")
        if stream is not None:
            cleanup(lambda: cu.cuStreamDestroy(stream), "cuStreamDestroy")
        if context_is_current:
            cleanup(lambda: cu.cuCtxSetCurrent(previous_context),
                    "restore previous current context")
        if context is not None:
            cleanup(lambda: cu.cuDevicePrimaryCtxRelease(device),
                    "cuDevicePrimaryCtxRelease")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

</details>

`checked` 解包 Python binding 的 `(CUresult, outputs...)` 返回值；设备句柄、context、stream 和 device pointer 都是 opaque handle，不是 host 数组。`cuDevicePrimaryCtxRetain` 的返回值要和 `cuDevicePrimaryCtxRelease(device)` 配对，`cuCtxSetCurrent` 负责把 context 绑定到调用线程；primary context 与 Runtime API 共享，避免在同一进程里无意创建第二个互不兼容的地址空间。代码先以 `cuCtxGetCurrent()` 保存旧 context，最终恢复而不是把调用者的 current context 清空。`cuMemsetD32Async` 按 stream 排队，host 函数返回不表示设备操作已完成；同步后用阻塞式 `cuMemcpyDtoH` 读回普通 NumPy host 数组。哨兵从 `0xA5A5A5A5` 变为 `0` 且断言通过，才证明写入和 D2H 地址/字节数都可观察；若换成 `cuMemcpyDtoHAsync`，还必须处理目标 host buffer 的页锁定要求及其生命周期。数组框架拥有的内存也不能在异步 kernel 使用前被 Python 引用计数释放。

设备代码应写在 DSL/kernel 里。例如既有 cuTile 向量加法的 `@ct.kernel` 函数体只使用 tile 运算；host 侧则把框架分配的数组和 stream 交给 `ct.launch`。Numba-CUDA 的 `@cuda.jit` 则表达 SIMT thread：每个 thread 算 `i=cuda.grid(1)`，自己验证索引有效后才读写数组。它不等同于 `cuda.bindings.driver`。下面的完整小程序用不整除 block 的 257 个 FP32 元素，最后一个 block 有效 lane 与无效 lane 同时存在；kernel 内的 `if i < n` 是内存安全条件，不是由 CuPy/Numba 自动补上的检查。

<details>
<summary>完整 Numba-CUDA SIMT 向量加法</summary>

<!-- source-check: examples/numba_simt_vector_add.py -->
```python
#!/usr/bin/env python3
"""A bounded SIMT vector add with explicit dtype, stream, and tail guard."""

import sys

try:
    import numpy as np
    from numba import cuda
except ImportError as error:
    print(f"SKIP: NumPy or Numba-CUDA is unavailable: {error}")
    raise SystemExit(77)


@cuda.jit
def vector_add(a, b, out, n):
    i = cuda.grid(1)
    if i < n:
        out[i] = a[i] + b[i]


def run_case(n: int) -> None:
    sentinel = np.float32(-12345.25)
    host_a = np.arange(n, dtype=np.float32)
    host_b = np.full(n, 2.0, dtype=np.float32)
    host_out = np.full(n + 1, sentinel, dtype=np.float32)

    if n == 0:
        np.testing.assert_array_equal(host_out, np.array([sentinel], dtype=np.float32))
        print("n=0: host-only empty case; no zero-sized grid was launched")
        return

    threads = 128
    blocks = (n + threads - 1) // threads
    with cuda.gpus[0]:
        stream = cuda.stream()
        device_a = cuda.to_device(host_a, stream=stream)
        device_b = cuda.to_device(host_b, stream=stream)
        device_out = cuda.to_device(host_out, stream=stream)
        vector_add[blocks, threads, stream](device_a, device_b, device_out, n)
        result = device_out.copy_to_host(stream=stream)
        # Keep arrays, stream, and host buffers alive until all queued work ends.
        stream.synchronize()

    expected = np.full(n + 1, sentinel, dtype=np.float32)
    expected[:n] = host_a + host_b
    np.testing.assert_array_equal(result, expected)
    print(f"n={n}: PASS; tail sentinel={result[n]}")


def main() -> int:
    if not cuda.is_available():
        print("SKIP: no usable CUDA device")
        return 77
    for n in (0, 1, 127, 128, 129, 257):
        run_case(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

</details>

`blocks=ceil(n/threads)` 可由整数式 `(n+threads-1)//threads` 得到。取 `n=257`、`threads=128` 时 `blocks=3`，全 grid 有 384 个 thread，只有索引 `0..256` 有效；`i=257..383` 必须在解引用前退出。输出多留一个 FP32 哨兵，因此可直接检查 `out[n]` 未被尾块越界写覆盖。所有 host/device 数组都用明确 `np.float32`，避免平台默认 `float` dtype 改变 kernel 参数类型或带宽账本。

手册示例需要按库的真实语义修正：Numba-CUDA 对设备数组越界访问默认不会像 NumPy 一样自动抛 `IndexError`，普通 kernel 必须显式判断 `i < n`；CuPy `zeros` 的默认 `dtype` 是 Python `float`，不是 FP32，算子实验应写 `cp.zeros(shape, dtype=np.float32)`；CuPy 同步对象是 `cp.cuda.Stream` / `cp.cuda.Device` 等，针对单个 stream 用 `stream.synchronize()`，不要调用不存在的顶层 `cp.synchronize()`。Numba 的设备级同步为 `cuda.synchronize()`，但优先同步实际提交工作的 stream。

Numba kernel 若要使用 CuPy 当前 stream，可让 Numba 包装外部 stream：`cp_stream = cp.cuda.get_current_stream()`，`numba_stream = cuda.external_stream(cp_stream.ptr)`，再以 `kernel[grid, block, numba_stream](...)` launch。两边必须指向同一 GPU/context；CuPy stream 所有者和它必须在用到 wrapper 的期间存活，Numba 不负责销毁外部 stream。跨库传数组时还要遵循 CUDA Array Interface 的 stream 同步合同，不能只传一个裸地址就假设生产者写入已经完成。

小手算：分配 4 bytes 得到设备地址 `P`，`cuMemsetD32Async(P, 0, 1, S)` 写一个 32-bit word；不是写一个 float 元素的 Python 索引，也不是“提交即完成”。回读哨兵并断言数值，可以区分“只成功入队”与“设备确已写入”。若删掉 `cuStreamSynchronize(S)` 并立即 `cuMemFree(P)`，host 代码就丢失了“异步读写已结束”的生命周期边界。运行时用 CUDA 13.x `cuda-bindings` 与兼容驱动；CPU-only 环境可运行纯 Python 语法/形状检查，但这两条 GPU 路径须在具备相应库和设备的环境现场运行。

## 12. cuTile 的 view、shape 变换与原子更新

cuTile 的 `Array` 是设备上的多维数组，`Tile` 是 kernel 内按值处理的固定形状数据；`TiledView` 是把数组坐标划分为 tile-space 的逻辑视图。创建 view 不会复制数据，也不决定 thread/lane 到元素的物理映射。程序员给出逻辑 tile 形状和访问步长，编译器再把 tile 操作映射到 CTA 内部。不能从 `tile_shape=(2,4)` 推导出“两个 warp”或“一行一个 warp”。

对连续 row-major 的二维数组 `A[R,C]`，元素 `(i,j)` 的字节地址为：

$$
addr(i,j)=base+(iC+j)\,sizeof(T).
$$

若 tile 形状为 `(r,c)`、相邻 tile 起点步长为 `(u,v)`，tile-space 坐标 `(p,q)` 的逻辑起点是 `(pu,qv)`。默认步长 `(r,c)` 让 tile 相邻且无重叠；若 `u<r`，相邻 tile 重叠；若 `u>r`，两块之间有空隙。view 坐标不是元素坐标：`view.load((1,0))` 取第二个 tile，而不是直接取 array 元素 `(1,0)`。

以 `A=arange(16).reshape(4,4)` 为例，底层 int32 行跨度为 `4×4=16` bytes。`tiled_view((2,4), traversal_steps=(1,4))` 的 tile `(1,0)` 从元素行 1 开始，覆盖 `[[4,5,6,7],[8,9,10,11]]`，与 tile `(0,0)` 的第二行重叠。这个公式足以在 CPU 上手算逻辑地址，但不告诉你编译器把元素分到哪些 lane 或寄存器。

`Tile.reshape(new_shape)` 只改变 tile 的逻辑维度，要求新旧元素总数相等，按原有元素序列重解释；它不是转置。例如形状 `(2,4)` 展平再 reshape 成 `(4,2)`，行序列仍是 `4,5,6,7,8,9,10,11`。`ct.transpose(x)` 交换矩阵的两个轴；`ct.permute(x, axes)` 按指定轴序重排，适用于 rank 大于 2 的 tile。两者操作的是 tile 的逻辑坐标，不会自动把结果写回一块新的全局内存；若随后要在不同线程/寄存器布局间交换数据，实际指令与代价由编译结果决定，不能因为 API 名称像 view 就假定零成本。

下面程序把逻辑 view、overlap、reshape、transpose 和 3D permute 写成可运行的 cuTile kernel，并让八个 tile block 对同一计数位置各加 4。三个输出各有独立的接口合同：`flat_out` 保持 tile 行优先次序，`transpose_out` 展示轴交换，`permuted_out` 展示轴序 `(2,0,1)`；计数结果是 `8×4=32`。

<details>
<summary>完整 view 与 atomic 示例</summary>

<!-- source-check: examples/cuda_tile_views_atomics.py -->
```python
#!/usr/bin/env python3
"""Exercise cuTile tiled views, shape operations, and reduction atomics."""

import sys


def main() -> int:
    try:
        import torch
        import cuda.tile as ct
    except ImportError as error:
        print(f"SKIP: PyTorch or cuTile is unavailable: {error}")
        return 77

    if not torch.cuda.is_available():
        print("SKIP: no usable CUDA device")
        return 77

    @ct.kernel
    def transform(source, flat_out, transpose_out, permuted_out):
        bid = ct.bid(0)
        source_view = source.tiled_view(
            (2, 4),
            padding_mode=ct.PaddingMode.ZERO,
            traversal_steps=(1, 4),
        )
        tile = source_view.load((bid, 0))
        flat_view = flat_out.tiled_view((1, 8))
        transposed_view = transpose_out.tiled_view((1, 8))
        flat_view.store((bid, 0), tile.reshape((1, 8)))
        transposed_view.store((bid, 0), ct.transpose(tile).reshape((1, 8)))

        if bid == 0:
            cube = ct.arange(8, dtype=ct.int32).reshape((2, 2, 2))
            permuted = ct.permute(cube, (2, 0, 1))
            permuted_out.tiled_view((8,)).store(0, permuted.reshape((8,)))

    @ct.kernel
    def add_one_tile_per_block(counter):
        view = counter.tiled_view((1,))
        increment = ct.full((1,), 4, dtype=ct.int32)
        # No old value is needed: use TiledView's reduction-style atomic form.
        view.atomic_store_add(0, increment)

    source = torch.arange(16, dtype=torch.int32, device="cuda").reshape(4, 4)
    flat_out = torch.full((2, 8), -1, dtype=torch.int32, device="cuda")
    transpose_out = torch.full((2, 8), -1, dtype=torch.int32, device="cuda")
    permuted_out = torch.full((8,), -1, dtype=torch.int32, device="cuda")
    counter = torch.zeros((1,), dtype=torch.int32, device="cuda")
    stream = torch.cuda.current_stream()

    ct.launch(stream, (2,), transform,
              (source, flat_out, transpose_out, permuted_out))
    ct.launch(stream, (8,), add_one_tile_per_block, (counter,))
    stream.synchronize()

    expected_flat = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10, 11]],
        dtype=torch.int32,
        device="cuda",
    )
    expected_transpose = torch.tensor(
        [[0, 4, 1, 5, 2, 6, 3, 7], [4, 8, 5, 9, 6, 10, 7, 11]],
        dtype=torch.int32,
        device="cuda",
    )
    expected_permute = torch.tensor(
        [0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(flat_out, expected_flat, rtol=0, atol=0)
    torch.testing.assert_close(transpose_out, expected_transpose, rtol=0, atol=0)
    torch.testing.assert_close(permuted_out, expected_permute, rtol=0, atol=0)
    torch.testing.assert_close(
        counter, torch.tensor([32], dtype=torch.int32, device="cuda"), rtol=0, atol=0
    )
    print("view/reshape/transpose/permute/atomic checks: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

</details>

CPU 手算可独立核对变换：tile `[[4,5,6,7],[8,9,10,11]]` 展平为 `[4,5,6,7,8,9,10,11]`；转置后按行展平为 `[4,8,5,9,6,10,7,11]`。三维 `cube=arange(8).reshape(2,2,2)` 使用轴序 `(2,0,1)` 后展平为 `[0,2,4,6,1,3,5,7]`。CPU 检查只验证坐标公式，不模拟 GPU atomic 的并发执行或硬件布局。

原子 API 也要按对象区分。`ct.atomic_add(array, indices, update)` 按数组元素坐标逐项执行原子读—改—写，并返回每项更新前的值；默认检查索引，默认 memory order/scope 为 acquire-release/device。整次 tile 调用不是一个事务：不同元素独立原子化，元素间顺序未定义。若调用者不需要旧值，`TiledView.atomic_store_add(tile_index, update_tile)` 是 reduction-style 形式，不返回旧值，适合原子累计。两种调用都不会建立“整个 CTA 已到达”的屏障，也不会让普通 load/store 自动获得全局顺序。

cuTile 的执行层次只有逻辑 block（编译器映射到 CTA）和 tile 数据，没有 SIMT 的显式 thread/lane，也不允许 tile kernel 内显式 block barrier 或线程间共享同步。普通 load/store 默认 `MemoryOrder.WEAK`，不提供跨 block 同步；free atomic 默认 `ACQ_REL` memory order 和 `DEVICE` scope。跨 block 共享更新要用文档支持的 atomic 及与消费者相符的 memory order/scope；若 producer kernel 完成后再由 consumer kernel 读，按同一 CUDA stream 顺序 launch 可建立 kernel 间先后关系，跨 stream 则用 event/dependency 建立先后。不要在 host 侧同步、kernel 间顺序和 tile 内部同步三者之间混用概念。

现场运行：从仓库根目录执行 `python roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/cuda_tile_views_atomics.py`；在 CUDA 13.3/cuTile 1.4 系列环境检查逐元素输出和计数。运行不依赖 Atomics 到达顺序来验证旧值，只核对整数加法的最终交换律结果；本机 CPU 模型见同目录的 shape/address 检查程序，不构成编译或设备结果。

<details>
<summary>纯 CPU 坐标模型</summary>

<!-- source-check: examples/cuda_tile_views_cpu_model.py -->
```python
#!/usr/bin/env python3
"""Check cuTile view coordinates and shape transforms without CUDA."""


def main() -> None:
    array = [[row * 4 + col for col in range(4)] for row in range(4)]

    def read_window(tile_row: int) -> list[list[int]]:
        origin_row = tile_row * 1  # traversal_steps[0]
        origin_col = 0 * 4
        return [
            array[row][origin_col:origin_col + 4]
            for row in range(origin_row, origin_row + 2)
        ]

    first = read_window(0)
    second = read_window(1)
    assert first == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert second == [[4, 5, 6, 7], [8, 9, 10, 11]]
    assert first[1] == second[0]  # the one-row traversal step overlaps

    flattened = [value for row in second for value in row]
    assert flattened == [4, 5, 6, 7, 8, 9, 10, 11]
    transposed = [second[row][col] for col in range(4) for row in range(2)]
    assert transposed == [4, 8, 5, 9, 6, 10, 7, 11]

    cube = [[[i * 4 + j * 2 + k for k in range(2)]
             for j in range(2)] for i in range(2)]
    permuted = [cube[j][k][i] for i in range(2)
                for j in range(2) for k in range(2)]
    assert permuted == [0, 2, 4, 6, 1, 3, 5, 7]

    # Eight blocks each add four to one counter; this is only arithmetic,
    # not a CPU simulation of CUDA's concurrent atomic implementation.
    assert sum(4 for _ in range(8)) == 32
    print("CPU tile-view coordinate model: PASS")


if __name__ == "__main__":
    main()
```

</details>

## 13. CUDA C++：执行空间、对象生命周期与设备链接

`.cu` 不是“C++ 文件里某些函数碰巧跑到 GPU”。`nvcc` 分离 host/device 代码，再分别交给 CUDA 前端和受支持的 host compiler。`__host__`、`__device__`、`__global__` 指定函数的编译/调用空间。一个只有 `__host__` 的 helper 被 kernel 调用，会在 device 编译阶段失败；`__host__ __device__` 函数要对两次编译都成立，必要时用 `#if defined(__CUDA_ARCH__)` 选择编译期分支。`__CUDA_ARCH__` 是设备编译宏，不是运行时 GPU 查询。

模板也不绕过执行空间合同：`__host__ __device__` 模板会按 host/device 调用点实例化；若 device 调用得到的 specialization 只有 host 版本，仍会失败。扩展 lambda 需显式标注执行空间，并用 `--extended-lambda` 编译。捕获值会成为闭包对象成员，捕获 host 指针不会自动变成 device pointer。

下面完整程序让 kernel 调用另一个 translation unit 中定义的 device 函数，并展示固定/动态 shared storage、模板和 host/device lambda。`DeviceBuffer` 是 host RAII owner；kernel 参数是只含设备指针和长度的平凡 `DeviceSpan`。这把管理内存的 C++ 对象和设备访问的轻量 descriptor 分开。

<details>
<summary>跨 translation unit 的 device 函数声明</summary>

<!-- source-check: examples/cpp_device_ops.cuh -->
```cpp
#pragma once

// Device functions have external linkage by default; spell it out here.
extern __device__ int scale_in_other_tu(int value);
```

</details>

<details>
<summary>device 函数定义所在的 translation unit</summary>

<!-- source-check: examples/cpp_device_ops.cu -->
```cpp
#include "cpp_device_ops.cuh"

extern __device__ int scale_in_other_tu(int value) {
    return value * 3;
}
```

</details>

<details>
<summary>host/device 调用、shared storage 与异步 owner 的完整程序</summary>

<!-- source-check: examples/cpp_language_support.cu -->
```cpp
#include "cpp_device_ops.cuh"
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

template <typename T>
__host__ __device__ constexpr T twice(T value) { return value + value; }

__host__ __device__ int apply_host_device_lambda(int value) {
    auto double_value = [] __host__ __device__ (int x) { return twice(x); };
    return double_value(value);
}

struct DeviceSpan {
    int* data;
    int size;
    __host__ __device__ int load(int i) const { return data[i]; }
    __host__ __device__ void store(int i, int x) const { data[i] = x; }
};

static_assert(std::is_trivially_copyable<DeviceSpan>::value, "plain kernel descriptor");
static_assert(std::is_trivially_destructible<DeviceSpan>::value, "no async destructor");

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t n) : pointer_(nullptr) {
        const cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&pointer_), n * sizeof(int));
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() noexcept {
        if (pointer_) {
            const cudaError_t e = cudaFree(pointer_);
            if (e != cudaSuccess) std::fprintf(stderr, "cudaFree: %s\n", cudaGetErrorString(e));
        }
    }
    DeviceSpan view(int n) const { return DeviceSpan{pointer_, n}; }
    int* data() const { return pointer_; }
private:
    int* pointer_;
};

__global__ void apply_static_shared(DeviceSpan in, DeviceSpan out) {
    // Contract: this kernel is launched with exactly 128 threads per block.
    __shared__ int staging[128];
    const int lane = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + lane;
    staging[lane] = i < in.size ? in.load(i) : 0;
    __syncthreads();
    if (i < in.size) out.store(i, scale_in_other_tu(apply_host_device_lambda(staging[lane])));
}

__global__ void apply_dynamic_shared(DeviceSpan in, DeviceSpan out) {
    extern __shared__ int staging[];
    const int lane = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + lane;
    staging[lane] = i < in.size ? in.load(i) : 0;
    __syncthreads();
    if (i < in.size) out.store(i, scale_in_other_tu(apply_host_device_lambda(staging[lane])));
}

static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
    }
}

int main(int argc, char** argv) {
  try {
    constexpr int n = 5, threads = 128;
    const bool use_static = argc > 1 && std::string(argv[1]) == "--static";
    const std::vector<int> h_in{1, 2, 3, 4, 5};
    std::vector<int> h_out(n, -1);
    {
        DeviceBuffer d_in(n), d_out(n);
        check(cudaMemcpy(d_in.data(), h_in.data(), n * sizeof(int), cudaMemcpyHostToDevice), "H2D");
        const DeviceSpan in = d_in.view(n), out = d_out.view(n);
        const int blocks = (n + threads - 1) / threads;
        if (use_static) apply_static_shared<<<blocks, threads>>>(in, out);
        else apply_dynamic_shared<<<blocks, threads, threads * sizeof(int)>>>(in, out);
        check(cudaGetLastError(), "launch");
        check(cudaDeviceSynchronize(), "kernel completion");
        check(cudaMemcpy(h_out.data(), d_out.data(), n * sizeof(int), cudaMemcpyDeviceToHost), "D2H");
        for (int i = 0; i < n; ++i) {
            if (h_out[i] != h_in[i] * 6) return EXIT_FAILURE;
        }
        std::puts(use_static ? "static shared: PASS" : "dynamic shared: PASS");
        // Owners leave scope after the explicit GPU completion boundary.
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "CUDA C++ example failed: %s\n", error.what());
    return EXIT_FAILURE;
  }
}
```

</details>

`twice<T>` 是编译期泛型；host/device lambda 调用它，kernel 再调用另一个 TU 的 `scale_in_other_tu`。device function 默认有 external linkage；示例仍在声明处显式写 `extern` 以强调该 ABI 边界。CUDA 13 的默认 internal-linkage 变化涉及 `__global__` functions 与 `__device__`、`__constant__`、`__managed__` variables，不能把它概括成“所有 device 符号都自动导出/隐藏”。NVCC 默认 whole-program device compilation 只假设使用单元内有定义；跨文件函数或变量要按符号类别正确声明，并启用 relocatable device code（RDC），随后由 device linker 解析。

对象生命周期解释了为何 kernel 参数不应承担资源所有权。Runtime 将 kernel 参数按原始字节复制到设备参数区，不会按标准 C++ 语义在 device 端运行用户自定义 copy constructor；kernel 是异步的，非平凡析构还可能在设备完成前于 host 执行。若参数析构有副作用或释放指针，就会和 GPU 使用期重叠。示例中的 `DeviceBuffer` 不跨 ABI 边界，只传平凡、无析构副作用的 `DeviceSpan`；`check()` 将 CUDA 错误抛为 host 异常，`main()` 捕获并报告失败，使 owner 在栈展开时照常析构。正常路径则先同步并 D2H 校验，之后才离开 owner 作用域。另一个限制是，带 `__device__`、`__shared__`、`__constant__`、`__managed__` 或 `__tile__` memory-space 的 class-type variable 不能有非空 constructor/destructor；这和上面 host launch 参数的 raw-byte copy 是两条不同的合同。异步生产代码应以 stream/event 排序保证最后一次访问先于 free。

固定 shared kernel 将 `staging` 声明为 128 个元素，因此它的 launch contract 是每 block 恰好 128 threads；改变 `threads` 时不能只改 launch 参数，还需同时改静态数组容量及索引合同。动态版本的 `extern __shared__` 容量由第三个 launch 参数的字节数指定，示例传入 `threads*sizeof(int)`。

storage duration 和 CUDA memory space 是两条维度：自动局部 `index` 在每个 thread 有独立语义，编译器可放寄存器，压力过大时 spill 到 local memory；固定 `__shared__` 数组按 CTA 分配；`extern __shared__` 数组仍是每 CTA shared storage，但其字节数来自 launch 配置。模块级 `__device__`、`__constant__` 不是 thread-local；函数内 `static` device 变量属于 device execution-space 的持久共享状态，不是每个 thread/CTA 各一份，且禁止动态初始化，多个 thread 更新仍需原子/同步。`__managed__` 支持 host/device 访问但不等于 shared memory。SIMT 的 device heap `new/delete` 和动态 shared memory不是一种分配；cuTile 还禁止普通非 placement `new/delete`。

函数指针与虚调用须按执行空间审查。host 不能取得 `__device__` 函数地址；`__global__` kernel 地址也不能在 host/device 两侧互换。SIMT 的间接调用只能使用对应 device execution space 中有效的目标；例如 `nvstd::function` 可在 host/device 各自包装 callable，但两个空间的实例不能在运行时互传，也不能作为由 host launch 的 `__global__` 参数。Tile 代码明确不支持函数指针表达式/调用或 virtual invocation。SIMT 的虚函数调用则要保证 override 的 execution space 一致，而且不能把 polymorphic object 从 host 按值传给 kernel：这属于未定义行为，因为对象布局/vtable 不是可移植的跨空间参数。对有限分支，模板、直接函数调用或显式 `switch` 通常比跨空间指针更可审查。

device code 不支持 C++ exceptions/RTTI：`throw`、`try/catch`、`typeid` 和 `dynamic_cast` 不能进入设备执行路径；这不等于 host 代码不能使用异常。普通 `std::vector` 是 host-side 容器，不是 GPU 数组；标准库函数是否可在 device 调用取决于具体声明与 host compiler，优先使用 CUDA C++ 标准库 `cuda::std` 中明确支持的实现。模板/`constexpr` 语法通过，不代表函数体在另一个 execution space 有效。

将 `sm_XX` 换成目标设备的 Compute Capability，从仓库根目录编译并运行：

```bash
nvcc -std=c++17 --extended-lambda -arch=sm_XX -dc -I roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/cpp_device_ops.cu -o cpp_device_ops.o
nvcc -std=c++17 --extended-lambda -arch=sm_XX -dc -I roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/cpp_language_support.cu -o cpp_language_support.o
nvcc -std=c++17 -arch=sm_XX -dlink cpp_device_ops.o cpp_language_support.o -o cpp_device_link.o
nvcc -arch=sm_XX cpp_language_support.o cpp_device_ops.o cpp_device_link.o -o cpp_language_support
./cpp_language_support
./cpp_language_support --static
```

`-dc` 生成含 relocatable device code 的 host object；`-dlink` 将这些 object 中的 device code/symbol 合并成可执行 device image，再交给最后的 host link。也可对所有 `.cu` 使用 `-rdc=true` 并由 NVCC 自动完成 device-link 阶段。若跨 TU 声明、定义或 linkage 不一致，错误应在 device link 阶段定位，而不是只改 host `-L/-l` 参数。

## 14. Error Log：从错误码追到更具体的 API 诊断

CUDA API 返回码保留了机器可处理的类别，但单看 `CUDA_ERROR_INVALID_VALUE` 之类的值通常不能说明哪个参数错、错误属于哪个入口。CUDA 12.9 起的 Error Log Management 为被覆盖的 Driver API 错误提供英文诊断；在启动进程前将 `CUDA_LOG_FILE` 设为 `stderr`、`stdout` 或文件路径，可在错误发生时输出 `[Time][TID][Source][Severity][API Entry Point] Message`。没有错误时不保证有任何日志；错误日志也不会替代检查每次 API 返回码、launch 后错误或 stream 同步错误。

日志缓冲区可用 `cuLogsDumpToFile` 或 `cuLogsDumpToMemory` 读取。`CUlogIterator` 记住读取位置：先用 `cuLogsCurrent(&iterator, 0)` 记录当前末尾，再以该 iterator dump 新日志，读取后 iterator 会推进到末尾；传空 iterator 则导出最多 100 条的完整缓存。内存导出缓冲区最大 25,600 bytes，输入容量不足会先放截断提示并丢弃较老条目。若库要集成自己的处理器，可按 `void callback(void* userData, CUlogLevel level, char* message, size_t length)` 注册 `cuLogsRegisterCallback`，保存返回的 handle 并用 `cuLogsUnregisterCallback` 成对注销。Driver 13.3 的 Error Log 接口尚未覆盖所有 API，路径有效性也可能等到实际生成日志时才验证；它只在 Driver API 提供，不能假定 Runtime 有等价接口。

下面程序故意触发手册记录的“向空 buffer dump”无效调用，再从 log ring buffer 读回诊断。这个例子针对工具链 13.3 中新增的 Driver logging API；错误调用是实验输入，正常工程代码不应复制。`cuGetErrorName` 仍展示机器错误码，dump 的文本补充具体 API 入口与错误原因。

<details>
<summary>Error Log Driver API 完整探针</summary>

<!-- source-check: examples/error_log_probe.cu -->
```cpp
#include <cuda.h>

#include <cstdio>
#include <cstdlib>

int main() {
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(result, &name);
        std::fprintf(stderr, "cuInit: %s\n", name ? name : "unknown");
        return EXIT_FAILURE;
    }

    // The CUDA Programming Guide documents this invalid call as a source of
    // the Error Log message "buffer cannot be NULL".
    size_t invalid_capacity = 0;
    result = cuLogsDumpToMemory(nullptr, nullptr, &invalid_capacity, 0);
    if (result != CUDA_ERROR_INVALID_VALUE) {
        const char* name = nullptr;
        cuGetErrorName(result, &name);
        std::fprintf(stderr, "expected CUDA_ERROR_INVALID_VALUE, got %s\n",
                     name ? name : "unknown");
        return EXIT_FAILURE;
    }

    char log_buffer[25600] = {};
    size_t bytes_written = sizeof(log_buffer);
    result = cuLogsDumpToMemory(nullptr, log_buffer, &bytes_written, 0);
    if (result != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(result, &name);
        std::fprintf(stderr, "cuLogsDumpToMemory: %s\n", name ? name : "unknown");
        return EXIT_FAILURE;
    }
    if (bytes_written != 0) std::printf("%s", log_buffer);
    return EXIT_SUCCESS;
}
```

</details>

Linux 服务器可从仓库根目录运行 `nvcc -std=c++17 roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/error_log_probe.cu -lcuda -o error_log_probe`，再执行 `CUDA_LOG_FILE=stderr ./error_log_probe`；PowerShell 对应为 `$env:CUDA_LOG_FILE='stderr'; .\error_log_probe.exe`。程序也会主动 dump 内存缓冲，因此即使没设置环境变量仍有一次可读输出。若某个 CUDA API 没接入 Error Log，仍须从返回码、`cudaGetLastError()` 和实际同步边界定位，不能把日志缺席当作没有错误。

```bash
nvcc -std=c++17 roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/error_log_probe.cu -lcuda -o error_log_probe
CUDA_LOG_FILE=stderr ./error_log_probe
```

## 15. Driver Entry Points 与版本查询

查询“CUDA 版本”时要区分三个数字：`CUDA_VERSION` 是编译时 Toolkit header 版本；`cudaRuntimeGetVersion` 报告进程链接/加载的 Runtime API 版本；`cuDriverGetVersion` 报告已安装 Driver 支持的 CUDA Driver API 版本。它不是驱动程序包的 marketing version，也不能代替设备的 Compute Capability 查询。

Driver Entry Point Access 允许程序在运行时取得 Driver 函数地址，以在同一 driver ABI 下做可选特性查询或调用。函数指针必须使用 Toolkit 的 `cudaTypedefs.h` 中对应版本 `PFN_...` 类型，并向 `cuGetProcAddress` 传与 typedef 匹配的 ABI introduction version。下面按 CUDA 11.2 ABI 解析 `cuMemAllocAsync`：`PFN_cuMemAllocAsync_v11020` 配 `11020`。不能把 `CUDA_VERSION` 或本机 `cuDriverGetVersion` 动态值直接塞进请求，也不能把返回地址 cast 成自猜的 prototype；较新的驱动可能存在签名不同的后续 ABI。`CUresult` 表示查找调用本身是否有效；`CUdriverProcAddressQueryResult` 另说明 symbol lookup 的原因，例如请求版本不足或当前 driver 找不到 symbol；所以两个结果与返回函数指针都要检查。

<details>
<summary>版本和 Driver entry-point ABI 查询完整程序</summary>

<!-- source-check: examples/driver_entry_point_query.cu -->
```cpp
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>

int main() {
    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) != cudaSuccess) {
        std::fprintf(stderr, "cudaRuntimeGetVersion failed\n");
        return EXIT_FAILURE;
    }
    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuInit failed; check the installed NVIDIA driver\n");
        return EXIT_FAILURE;
    }
    int driver_api_version = 0;
    if (cuDriverGetVersion(&driver_api_version) != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuDriverGetVersion failed\n");
        return EXIT_FAILURE;
    }

    std::printf("headers CUDA_VERSION=%d, linked runtime=%d, driver API=%d\n",
                CUDA_VERSION, runtime_version, driver_api_version);

    constexpr int kRequiredAbi = 11020;  // CUDA 11.2 ABI for cuMemAllocAsync.
    if (driver_api_version < kRequiredAbi) {
        std::printf("driver API is older than requested ABI %d\n", kRequiredAbi);
        return 77;
    }

    PFN_cuMemAllocAsync_v11020 allocate_async = nullptr;
    CUdriverProcAddressQueryResult symbol_status{};
    const CUresult lookup = cuGetProcAddress(
        "cuMemAllocAsync",
        &allocate_async,
        kRequiredAbi,
        CU_GET_PROC_ADDRESS_DEFAULT,
        &symbol_status);
    if (lookup != CUDA_SUCCESS || allocate_async == nullptr) {
        std::fprintf(stderr,
                     "cuGetProcAddress failed: CUresult=%d symbolStatus=%d functionFound=%s\n",
                     static_cast<int>(lookup), static_cast<int>(symbol_status),
                     allocate_async ? "yes" : "no");
        return EXIT_FAILURE;
    }

    std::puts("resolved cuMemAllocAsync using PFN_cuMemAllocAsync_v11020 / ABI 11020");
    return EXIT_SUCCESS;
}
```

</details>

Linux 服务器可用 `nvcc -std=c++17 roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/driver_entry_point_query.cu -lcuda -o driver_entry_point_query` 编译，然后运行。成功解析只证明当前 driver 暴露了被请求的这个 ABI entry point；若要真正调用异步分配，还必须先选择正确 context/stream，并确认设备 memory-pool 支持、所有异步使用在 stream 有序 free 之前结束。Toolkit 13.3 的 `<cudaTypedefs.h>` 提供函数指针 typedef；较老 Toolkit 缺少该 typedef 时，应按其文档提供的 ABI 声明/版本路径处理，而不是随意强制转换。

```bash
nvcc -std=c++17 roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/driver_entry_point_query.cu -lcuda -o driver_entry_point_query
./driver_entry_point_query
```

## LeetGPU：正确性与代码归档

线程调度、occupancy、PTX/SASS 检查和分歧没有一个可以代表所有架构的独立平台题目；本章不虚构平台实验。需要用矩阵乘题练习线程映射和 accumulator 时，可直达 [LeetGPU MatMul #02](https://leetgpu.com/challenges/matrix-multiplication)，从空题面实现并单独保存平台原始 `solve`/kernel。已归档的 IEEE FP32 实现、mask、tile 和 grid 可作为阅读对照；平台成绩仍以实际提交记录为准。

## 服务器：真实性能

本地有 CUDA Driver、Numba-CUDA 或 cuTile 的设备环境可分别运行本章新增的资源生命周期、SIMT 尾块与 view/atomic 示例；CPU 坐标模型不要求安装 CUDA：

```bash
python roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/cuda_python_bindings_lifecycle.py
python roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/numba_simt_vector_add.py
python roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/cuda_tile_views_atomics.py
python roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/cuda_tile_views_cpu_model.py
```

cuTile 向量加法可从仓库根目录直接运行：

~~~bash
python roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/tile_vector_add.py --help
python roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/tile_vector_add.py
compute-sanitizer --tool memcheck --error-exitcode=1 python roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/tile_vector_add.py
~~~

运行环境需要相互兼容的 PyTorch CUDA、cuTile、驱动和 GPU。程序检查长度 1、127、128、129、1025 的结果和尾部哨兵，长度零在 host 跳过；依赖或 GPU 缺失返回 77。C++ tile 示例另需支持 CUDA Tile C++ 的 Toolkit，不能用 Python 包的存在推断 C++ 编译器支持。平台编辑器若不支持 cuTile，继续用它支持的 CUDA/Triton 提交，cuTile 对照在服务器执行即可。

这份小例程不计时。做性能对照时，先分别预热 CUDA SIMT、Triton、cuTile，再用同一数组长度和 dtype 测稳定 kernel time；保留 JIT 时间作为另一项开销。更换 tile 大小后同时查看生成代码与资源，避免把首次编译或空输入跳过当成加速。

服务器测量应在同一目标设备上固定输入 shape、dtype、block/tile、warmup、iteration 和同步范围，至少比较依赖链与独立链、分歧与 predication 的正确性和 kernel time，并记录 registers/thread、active warps、eligible/stall 分类、branch efficiency、内存吞吐和 wave tail 形状。源码、PTX、SASS 检查用于解释 counter，不替代测量。依赖链与独立链按各自的源级工作量归一化；数学等价的分支/选择实现可比较相同任务耗时。

## 参考阅读

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 与 [NVCC compiler guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html)
- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)、[Error Log Management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/error-log-management.html)、[Driver Entry Point Access](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html) 与 [`cuda.bindings` 13.3.2 Driver reference](https://nvidia.github.io/cuda-python/cuda-bindings/13.3.2/module/driver.html)
- [cuTile Python TiledView](https://docs.nvidia.com/cuda/cutile-python/data/tiled_view.html) 与 [memory model](https://docs.nvidia.com/cuda/cutile-python/memory_model.html)
- [Numba-CUDA kernel API](https://nvidia.github.io/numba-cuda/reference/kernel.html) 与 [host/stream API](https://nvidia.github.io/numba-cuda/reference/host.html)
- [CuPy Stream API](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html) 与 [`cupy.zeros`](https://docs.cupy.dev/en/v13.2.0/reference/generated/cupy.zeros.html)
