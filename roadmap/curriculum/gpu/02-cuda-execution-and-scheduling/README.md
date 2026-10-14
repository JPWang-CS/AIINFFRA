# 第 2 章：CUDA 执行模型与指令调度

CUDA 按线程坐标生成线性编号，每 32 个线程组成一个 warp。线程块驻留到 SM 后，调度器在这些线程束中选择下一条指令可执行的 warp 发射；若线程等待数据，其余就绪线程束仍可继续推进。依赖链、分支和浮点计算方式由此影响实际执行过程。

CUDA 先按 x、再按 y、最后按 z 的顺序为 block 内线程编号；每 32 个线程组成一个 warp（线程束）。下面用二维 block 计算这一分组关系。

## 1. 线程编号与 warp 分组

先看一个二维线程块：`dim3 block(4, 2)`。这里 `x=4`、`y=2`，所以总共有 8 个线程。每个线程有一对坐标 `(threadIdx.x, threadIdx.y)`：

| `y` | `x = 0` | `x = 1` | `x = 2` | `x = 3` |
|---:|---:|---:|---:|---:|
| 0 | (0,0) | (1,0) | (2,0) | (3,0) |
| 1 | (0,1) | (1,1) | (2,1) | (3,1) |

线性编号先让 x 从 0 增加到 3，再把 y 加 1。于是 `(0,0)→0`、`(1,0)→1`、`(2,0)→2`、`(3,0)→3`、`(0,1)→4`，最后 `(3,1)→7`。如果用这个编号访问向量，线程 0 处理元素 0，线程 1 处理元素 1。坐标和编号描述的是同一个线程：坐标方便表达二维位置，编号方便顺序分组。

CUDA 把同一个 block 中连续 32 个线性编号归为一个 warp（线程束）。warp 是硬件调度线程的基本组，一个 lane（线程束内位置）对应组内的一条线程，编号为 0 到 31。`block(4,2)` 只有 8 个线程，所以它们占据一个 warp 的前 8 个 lane；本 block 的线程数不会因此补到 32，另一个 block 的线程也不会拼进来。

把 block 的 x 方向设为 16：`dim3 block(16,4)` 有 64 个线程。第一个 warp 包含 `threadIdx.y=0` 和 `threadIdx.y=1` 各 16 个线程；第二个 warp 包含 y=2 和 y=3 的线程。warp 按线性编号每 32 个线程划分；若这些线程处理矩阵，实际对应位置由 kernel 下标决定。

<figure class="diagram-frame">
<img src="assets/warp-linearization.svg" alt="上图展示 block(4,2) 的 8 个线程属于同一 warp 的前 8 个 lane；下图展示 block(16,4) 中 threadIdx.y 不同的四排线程分成两个 warp，每个 warp 包含相邻两排的各 16 个线程。">
<figcaption>格内数字是线程的线性编号。下图的蓝色、橙色边框分别标出 warp 0 和 warp 1；每个边框内包含 32 个线程。</figcaption>
</figure>

现在可以写出一般公式。对二维 block，线性线程编号 `tid = threadIdx.x + blockDim.x × threadIdx.y`。对三维 block，再加上 `blockDim.x × blockDim.y × threadIdx.z`，即：

$$
tid = threadIdx.x + blockDim.x\,threadIdx.y + blockDim.x\,blockDim.y\,threadIdx.z
$$

`warp_id` 表示当前线程属于 block 内第几个 warp；`lane_id` 表示它在该 warp 中的位置：

$$
warp\_id = \left\lfloor\frac{tid}{32}\right\rfloor,\qquad lane\_id = tid\bmod 32
$$

除以 32 得到 warp 编号，取余 32 得到 lane 编号。对 `block(16,4)`，线程 `(x=7,y=1)` 的 tid 为 23，因此属于 warp 0、lane 23。线程 `(x=7,y=2)` 的 tid 为 39，属于 warp 1、lane 7。

每个 block 都从 tid=0 开始独立编号和分组，一个 warp 的线程全部来自同一个 block。

```cpp
__global__ void record_mapping(int* warp_ids, int* lanes) {
    int tid = threadIdx.x + blockDim.x * threadIdx.y;
    warp_ids[tid] = tid / 32;
    lanes[tid] = tid % 32;
}

// 输出数组各有 64 个 int；只启动一个 16×4 block，覆盖 tid=0..63。
dim3 block(16, 4);
record_mapping<<<1, block>>>(warp_ids, lanes);
```

线程坐标通过 kernel 的下标公式对应到数据。下面的 GEMM 让每个线程计算一个输出元素，可以直接看清这层对应关系。

矩阵乘法通常称为 GEMM（General Matrix Multiplication）。下面计算 `A[M,N] × B[N,K] = C[M,K]`：输出的每个元素是 A 的一行与 B 的一列对应相乘后求和。这个简单实现让一个线程计算一个输出元素，因而可以直接从代码看出线程坐标与矩阵坐标的关系。

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

这份代码中，`k` 是输出列，`m` 是输出行；两者分别由 block 坐标和块内线程坐标计算。`idx=m*K+k` 是行优先数组 C 中的元素位置。每个有效线程沿归约维 `n` 求和，最后写一个 `C[m,k]`。`sum` 从零开始，写回时覆盖旧值。边界判断把行或列越界的线程排除，保护 A、B、C 的访问。

在上述 kernel 中，`threadIdx.x` 增加 1 就向右移动一个输出列，`threadIdx.y` 增加 1 就移动到下一输出行；每个线程负责一个 `C[m,k]`。因此 `blockDim.x` 是**一个 block 在输出行中覆盖的列数**。先看没有越界线程的内部 block：

| 线程块 | warp 0 的 32 个线程 | 对输出 C 的覆盖 |
|---|---|---|
| `block(16,16)` | `y=0,x=0…15`；`y=1,x=0…15` | 相邻两行，各 16 列 |
| `block(32,8)` | `y=0,x=0…31` | 同一行的连续 32 列 |

例如 C 有 64 列，16 列宽的 block 沿列方向排 4 个，32 列宽的 block 排 2 个。表中的“一行”是 warp 当前处理的输出片段；完整矩阵行有 64 个元素。线程与数据的位置关系由下标公式确定。

取 `M=3,N=2,K=5`，负责 `C[1,2]` 的线程先算 `idx=1*5+2=7`。循环第 0 次读取 `A[2]` 和 `B[2]`，第 1 次读取 `A[3]` 和 `B[7]`，得到 `C[7]=A[2]B[2]+A[3]B[7]`。若算出的行号为 3 或列号为 5，`if` 条件为假，这个线程跳过计算和写回。

数据块的元素数与线程数由索引映射建立关系。例如 NVIDIA transpose 示例使用 32×32 数据块和 32×16（512 个线程）的 block；每个线程循环搬运两个元素。具体映射见 [CUDA Samples transpose](https://github.com/NVIDIA/cuda-samples/tree/5443602d89ed99aede2e4b7bf329daddeadb320e/cpp/6_Performance/transpose)。

Triton MatMul 用 `pid_m`、`pid_k` 选择输出 tile。程序描述的是一个 program 处理哪些元素；Triton 编译器再安排这些元素由哪些线程和 lane 处理。

## 2. 线程块驻留与 warp 发射

### 2.1 已分配资源：驻留状态

kernel 启动后，设备把 block 分配给 SM。SM 要先为这个 block 留出线程状态、寄存器和共享内存；资源分配完成后，block 处于驻留状态（resident），其中的 warp 才能参与调度。一个 SM 可以同时驻留多个 block，数量由这些资源的上限决定。

占用率（occupancy）是当前驻留 warp 数占 SM 最大 warp 数的比例。例如 SM 最多支持 64 个 warp、当前驻留 32 个时，占用率为 50%；其中部分 warp 仍可能等待数据。

设一个 block 包含 $T_{block}$ 个线程，编译器为每个线程分配 $R_{thread}$ 个 32 位寄存器。该 block 的寄存器用量 $R_{block}$ 和 warp 数量 $W_{block}$ 可先估算为：

$$
R_{block}\approx T_{block}\times R_{thread},\qquad
W_{block}=\left\lceil\frac{T_{block}}{32}\right\rceil
$$

先算出一个 block 的需求，再分别检查 SM 的寄存器、共享内存、线程、warp 和 block 上限。最先用完的资源决定能放下几个 block。寄存器分配还要按硬件的分配单位取整，所以实际用量以编译报告为准。寄存器不足时，编译器可能把部分值移到 local memory；这个过程称为寄存器溢出（spill），会增加设备内存读写。

### 2.2 下一条指令可执行：就绪状态

> [!IMPORTANT] 从驻留到执行
> block 取得所需资源后驻留在 SM 上；其中的 warp 在下一条指令条件满足时就绪，由调度器选中并发射。等待、就绪和发射期间，block 持续占用驻留资源。

就绪表示 warp 的下一条指令已满足执行条件。例如，前一条加载的数据已返回，或参与同步的线程都已到达。调度器从就绪 warp 中选择指令，并发送给有空位的执行管线。

指令间的数据依赖会决定下一条指令何时就绪。下面比较两段求和代码；[fmaf(a,b,x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44fmaffff) 表示融合乘加 $a\times b+x$，乘加结果只舍入一次，简称 FMA（Fused Multiply-Add）。第一段每次更新同一个累加值；第二段维护四个独立累加值，互不等待时可交错执行。

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

第一段的每次 FMA 都依赖上一次 x。第二段有四个独立累加器：等待 x0 的结果时，调度器可发出 x1、x2 或 x3 的指令。这种同一线程内可交错执行的工作称为指令级并行（Instruction-Level Parallelism，ILP）。

四条累加链会多占寄存器，也改变最终的浮点求和顺序。正确性比较应使用明确的误差容限；性能比较则同时看耗时、寄存器数和驻留 warp 数。若寄存器增加使 SM 能驻留的 warp 大幅减少，拆累加链也可能变慢。

### 2.3 发出指令：Issue

指令发射是调度器把指令送入执行管线。显存加载发射后，线程等待数据返回；期间调度器可发出其他就绪 warp 的指令。若当前 warp 的后续指令不依赖该数据，它也能继续执行。

例如，warp A 读取显存后等待结果供加法使用；warp B 的乘法输入已准备好，调度器便可先发出 B 的指令。A 的数据返回后，其加法指令也就绪。A、B 的线程状态和寄存器值仍保存在 SM 上。

分析性能时依次查看资源允许驻留多少 warp、其中多少 warp 就绪，以及调度器发射了多少指令。驻留量较低时检查寄存器与共享内存占用；就绪量较低时检查数据依赖、访存等待和同步。

一个 warp 因数据依赖而等待时，SM 可发出同一 SM 上其他已驻留且就绪 warp 的指令。Ascend C 的 AIV/AIC 工作分配由软件模型组织；这里描述的是 GPU 在 SM 上选择 warp 发射指令，两种机制位于不同层次。

如果所有驻留 warp 都在等数据，即使占用率达到 100%，执行管线也会空闲。少量 warp 若已有足够的独立指令填满管线，提高占用率的收益就较小。

把 block 从 128 个线程改为 256 个线程，会改变 grid 中的 block 数、每个 block 的资源需求和尾块大小。记录这些量并比较实际耗时，才能判断配置效果。

## 3. SIMT 与分支执行

SIMT（Single Instruction, Multiple Threads，单指令多线程）让程序员按单线程描述计算，硬件再以 warp 为组执行。同一条加载指令可由 32 个线程使用各自地址，例如读取连续的 32 个 `input[global_id]` 元素。

当一个 warp 中的线程选择不同分支时，warp 会分别处理各自采用的指令路径，这称为分支分歧（divergence）。活动掩码（active mask）标出当前路径参与的 lane；其他 lane 暂停该路径，之后在共同代码处继续执行，这一步称为重汇合（reconvergence）。

<figure class="diagram-frame">
<img src="assets/warp-branch-lanes.svg" alt="一个 warp 的 32 个 lane 按 global_id 奇偶分成两条分支：16 个偶数 lane 活动于 even 路径，16 个奇数 lane 活动于 odd 路径，两条路径之后汇合到共同写回。">
<figcaption>活动 lane 只参加自己选择的路径。图示假定显式分支保留；编译器也可能改写短分支，实际形式要看生成指令。</figcaption>
</figure>

```cpp
if ((global_id & 1) == 0) {
    output[global_id] = expensive_even(input[global_id]);
} else {
    output[global_id] = expensive_odd(input[global_id]);
}
```

假设 32 个 lane 对应连续 `global_id`，其中 16 个选择 even，另 16 个选择 odd。保留显式分支时，even 指令由偶数 lane 执行，odd 指令由奇数 lane 执行；每个线程只计算所选路径。两个路径越长，warp 处理的指令越多。若不同 warp 分别统一选择 even 和 odd，每个 warp 内只需处理一条路径。

### 3.1 谓词执行与短分支

短分支中，编译器可能用谓词执行（predication）把条件附在指令上。指令按条件掩码作用于相应 lane，未参与的 lane 不产生该指令的读写结果；这种形式可减少显式跳转。

下面这份源码先做乘法，再做加法，最后选择一个值。按源码计算顺序，每个线程都先算了两个候选值：

```cpp
float even_value = input[i] * 2.0f;
float odd_value = input[i] + 3.0f;
output[i] = ((i & 1) == 0) ? even_value : odd_value;
```

这段代码按源码先计算两个候选值，再选择输出。编译器可能保留两次计算，也可能把其中的运算改写为带谓词的指令；查看生成指令可确认采用了哪种形式及各路径的活动 lane。

短分支的计算量较小时，谓词执行可省去跳转开销。长分支在整个 warp 选择同一路径时，可跳过另一条路径；warp 内两条路径都有线程时，则需要分别处理两组指令。

### 3.2 独立线程调度与 warp 协作

从计算能力（Compute Capability，CC）7.0（Volta）开始，GPU 为每个线程记录指令位置和执行状态，这称为独立线程调度（Independent Thread Scheduling，ITS）。warp 中部分线程等待时，其他线程可继续；发射指令时，硬件仍按活动 lane 组成 SIMT 执行组。

写代码时要注意：**同一个 warp 中的线程，也需要同步才能保证通过内存交换的数据先写后读。** 例如，线程 0 写共享内存，其他线程随后读取；两步之间用 `__syncwarp()` 等待写入完成：

```cpp
// 示例条件：一个 32 线程的一维 block，所有线程均执行下面的同步。
// 读完之前，没有其他代码覆盖 shared_value。
__shared__ float shared_value;
if (threadIdx.x == 0) shared_value = value;
__syncwarp(0xffffffffu);
float x = shared_value;
```

`0xffffffffu` 的 32 个 bit 全为 1，表示这个 warp 的 32 个线程都参与同步。线程 0 完成写入后也必须执行 `__syncwarp()`；其他线程通过同步后再读，才能保证读到这次写入的值。同步放在 `if` 外，因为读者和写者都需要到达这里。

每个线程都有自己的一份局部变量。普通标量通常放在寄存器中；通过 [__shfl_sync](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-shuffle-functions)，其他线程可以取得指定 lane 的寄存器值。下面把 lane 0 的 `value` 传给这个 warp 的所有参与线程：

```cpp
// 前置条件：完整 warp 的 32 个线程均到达这里，lane 0 提供有效 value。
const unsigned mask = 0xffffffffu;
int x = __shfl_sync(mask, value, 0);
```

每个线程都把自己的 `value` 传入，但最后一个参数 0 指定从 lane 0 取值。例如 lane 0 的值是 10，lane 1 的值是 20，调用后每个参与线程得到的 x 都是 10。若需要两者，可以分别调用 `__shfl_sync(mask, value, 0)` 和 `__shfl_sync(mask, value, 1)`。

`mask` 的位指定参与线程，最后的 lane 编号指定值的来源。所有参与线程要使用相同 mask 执行对应调用，来源 lane 也要参加并已经给 value 赋值。`__shfl_sync` 自带这次寄存器交换需要的同步。若某个局部变量被溢出到 local memory，编译器会先取得它的寄存器操作数，再执行 shuffle。

例如完整的一维 warp 中，每个 lane 先生成自己的值；每个线程执行两次 shuffle，就能分别取得 lane 0 和 lane 1 的值：

```cpp
// 前置条件：blockDim.x 是 32 的倍数，且这个完整 warp 的所有 lane 都执行两次调用。
const unsigned mask = 0xffffffffu;
int lane = threadIdx.x & 31;
int value = 10 + lane;
int from_lane_0 = __shfl_sync(mask, value, 0);
int from_lane_1 = __shfl_sync(mask, value, 1);
```

在这个例子中，lane 0 的 `value` 是 10，lane 1 的 `value` 是 11，因此每个参与线程的 `from_lane_0` 都是 10，`from_lane_1` 都是 11。二维或三维 block 先按 CUDA 线性编号计算 `tid`，再用 `tid % 32` 得到 lane 编号。

共享内存例子用 [__syncwarp](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#synchronization-functions) 保证先写后读；shuffle 例子用指令自身的同步保证寄存器交换。若同一个 block 的不同 warp 通过共享内存传值，通常在写入与读取之间使用 `__syncthreads()`。

参加同步的线程要包含数据的写入者和读取者。对于没有有效输入的线程，先给后续读取的位置写好填充值，再按同步函数的要求决定是否可以退出。地址越界和多个线程争写同一位置，则需要另外修正索引或定义原子更新规则。


## 4. Block 波次与尾部利用率

<details>
<summary>展开波次公式、算例与负载均衡方法</summary>

假设设备有 $S$ 个 SM，每个 SM 最多同时驻留这个 kernel 的 $B$ 个 block，则同时能放下 $S\times B$ 个 block。再假设各 block 耗时相近，把这一批容量称为“一波”。对于包含 $G>0$ 个 block 的 grid，估算波数为：

$$
waves=\left\lceil\frac{G}{S\times B}\right\rceil
$$

最后一波中剩余的 block 数为：

$$
G-(waves-1)(S\times B)
$$

当最后一波不足 $S\times B$ 个 block 时，部分资源会提前空闲，形成尾效应。实际设备会在一个 block 结束后继续安排后续 block，并不会等同一波全部结束才开始下一波。上面的公式用于估算等时任务的尾部，实际时间还受各 block 的工作量影响。

block 内的尾部也会留下无效线程位置，例如最后一个 warp 只有少量有效 lane，或部分线程的元素索引超出数组范围。

例如 `S=4,B=2`，每波容量为 8。10 个等时 block 可以看成第一波 8 个、最后一波 2 个；最后只用了 8 个位置中的 2 个，至多涉及 2 个 SM。8 个 block 刚好满一波，12 个则是 8 个加 4 个。若每个 block 都耗时 t、且并发不改变耗时，10 个 block 的这个简化模型需要约 2t。

矩阵计算有两类尾部。tile 内的尾部发生在边界形状，例如 tile 覆盖 32 列而矩阵只剩 5 列，其余位置由 mask 排除。设备调度的尾部发生在 grid 末尾，例如只剩一两个 block 时，其他 SM 没有新任务。用整除和非整除的矩阵形状分别观察这两类情况。

尾部可以从任务划分、block 工作量和资源占用三处处理：

| 做法 | 适用情况 | 需要检查 |
|---|---|---|
| 合并多个独立的小任务 | 单个矩阵或请求太小，单独启动会留下大量空闲资源 | 合并后地址、边界和每个任务的工作量是否正确 |
| 动态领取逻辑任务 | 任务耗时不均，静态分配给线程或 block 后会有少数执行者拖到最后 | 任务领取开销、同步开销与任务粒度 |
| 调整 tile 或每个 block 的工作量 | block 太少，或每个 block 占用资源过多 | 寄存器、共享内存、驻留 block 数和总工作量 |
| 为边界形状使用专门 tile/kernel | 主体形状规则，只有边缘剩余区域较小 | 额外分支或启动成本是否低于省下的无效工作 |

动态领取任务适用于工作量差异较大的逻辑任务：线程或 block 完成当前任务后，可从队列领取下一项。规则工作量的普通 kernel 由硬件调度器在 block 结束、资源释放后安排尚未运行的 block。

例如，17 个耗时均为 `t` 的不可拆分 block 分给 8 个并行位置时，至少一个位置要顺序执行 3 个 block，理想情况下总时间下界约为 `3t`。此估算假设 8 个位置始终可用，且每个 block 耗时不随并发变化。补足空 block 不会减少已有计算量。persistent kernel（常驻内核）让一组 block 持续驻留并循环领取逻辑任务，适用于任务粒度和调度开销经评估后有收益的场景。

</details>

## 5. 计算管线：普通 CUDA Core、FMA 与 Tensor Core

官方文档：[fmaf：参数、舍入与特殊值](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44fmaffff) · [Tensor Core MMA：矩阵形状、输入类型与精度规则](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)

普通算术指令和矩阵乘加指令使用不同的计算资源。下面先看普通 FP32 算术：`fmaf(a,b,acc)` 计算 $a\times b+acc$，乘加作为一个运算只舍入一次：

```cpp
float acc = 0.0f;
for (int k = 0; k < K; ++k) {
    acc = fmaf(a[k], b[k], acc);
}
```

上面的循环一次处理一对 a、b。Tensor Core 的矩阵乘加指令（Matrix Multiply-Accumulate，MMA）则一次处理一个矩阵小块，例如把 A 的一个块与 B 的一个块相乘，再加到累加器矩阵上。使用这类指令，需要把数据排成它支持的形状、类型和布局。

CUDA Core 算术和 Tensor Core 矩阵指令的处理规模、吞吐率与舍入规则不同。检查生成指令时，普通 FP32 FMA 和矩阵 MMA 应分别识别；后面的类型说明也按这两条路径展开。

### 5.1 输入、累加器与输出类型

FP16（半精度浮点）和 BF16（一种 16 位浮点格式）常用来保存输入，每个元素占 2 字节。读入后，代码可先转成 FP32 再计算，也可将低精度矩阵块交给 MMA 指令。先看 FP16 输入、FP32 乘加的顺序：

$$
\begin{aligned}
a_f&=\operatorname{FP32}(a_{in}),\\
b_f&=\operatorname{FP32}(b_{in}),\\
acc_{fp32}&\leftarrow\operatorname{fmaf}(a_f,b_f,acc_{fp32}).
\end{aligned}
$$

公式中 $a_{in}$、$b_{in}$ 是输入数组中的 FP16 值，$a_f$、$b_f$ 是转换后的 FP32 值。累加完成后，按输出数组类型写回：

$$
C_{out}=\operatorname{cast}_{out}(acc_{fp32}).
$$

以下示例使用 [__half2float](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html) 转换输入，再调用 [fmaf](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44fmaffff)。每个线程计算一对输入的乘积，输出为 FP32：

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

沿代码逐步看：`a[i]`、`b[i]` 是 FP16，每个输入元素占 2 字节；`__half2float` 把它们转成 FP32 的 af、bf；`fmaf` 执行 FP32 乘加；结果写入 float 数组 c，每个输出元素占 4 字节。这里节省的是输入数组的存储和加载数据量。

例如，原始值 `1.0001` 保存成 FP16 时舍入为 `1.0`；之后转成 FP32，参与运算的仍是 `1.0`。因此，输入存储类型为 FP16，乘加运算和累加器类型为 FP32。

另一种写法把 FP16/BF16 矩阵块直接作为 MMA 操作数，并使用 FP32 累加器。矩阵指令按其数值规则计算；检查生成的算术指令可确认程序采用显式 FP32 乘加还是矩阵乘加。

下面的 FP16 分块矩阵乘法使用 `A[M,K] × B[K,N] = C[M,N]`，K 是归约维。`threadIdx.x` 对应输出行 m，`threadIdx.y` 对应输出列 n。输入先搬入 half 类型的共享内存 As、Bs，读取时转成 float，由 float 类型的 sum 累加，最后写回 half。

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

两份代码的类型可以直接对照：

| 实现 | 输入数组 | 计算时的 a、b | 累加器 | 输出数组 |
|---|---|---|---|---|
| 前面的 `half_input_float_accum` | FP16 | 显式转成 FP32 | FP32 FMA 的第三个操作数为 0 | FP32 |
| 上面的 FP16 tiled GEMM | FP16 | 显式转成 FP32 | float 类型的 sum | FP16 |

输出数组改为 half 会改变写回类型；累加精度由 sum 或矩阵指令的累加器类型决定。共享内存 As、Bs 每个元素占 2 字节，分析 bank 访问时按此宽度计算。

<details>
<summary>延伸：GEMM 的 alpha、beta 与旧输出</summary>

tiled GEMM 的写回公式是 `C=alpha*sum+beta*C_old`。`alpha` 缩放新结果，`beta` 缩放调用前的旧输出。原代码在 beta 为零时仍读取 C 并计算 `beta*C_old`，因此调用前需初始化 C；若旧值为 NaN，乘积仍会得到 NaN。代码注释中的带宽倍数需在设备上测量；算法估算字节数与 profiler 记录的 DRAM 字节数分别报告。

</details>

### 5.2 Triton `tl.dot` 的 `input_precision`

> [!WARNING] FP32 输入也要确认乘法精度
> `tl.dot` 处理 FP32 输入时可使用 TF32 精度。用 `input_precision` 指定这次矩阵乘法允许的输入精度。

对 FP32 输入，[tl.dot](https://triton-lang.org/main/python-api/generated/triton.language.dot.html) 的 `input_precision` 决定乘法精度。例如 PyTorch 关闭 TF32、Triton 开启 TF32 时，两边的计算路径不同；长归约的误差可能超过测试容差。要求 IEEE FP32 乘法时，Triton 写成：

```python
acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
acc += tl.dot(tile_a, tile_b, input_precision="ieee")
```

同时在 PyTorch reference 侧固定：

```python
torch.backends.cuda.matmul.allow_tf32 = False
```

`input_precision="ieee"` 指定这次点积的乘法精度；`tile_a`、`tile_b`、`acc` 和输出数组的类型分别由代码确定。例如 FP32 累加结果写入 FP16 输出时，会舍入为 FP16。

小矩阵通过、长归约失败时，固定输入、矩阵形状和容差，分别运行 IEEE 与 TF32 版本，比较误差变化，再检查地址、mask、累加顺序和输出转换。FP32 输出保留写回前的累加结果，但乘法与求和路径仍决定最终误差。

## 6. 依赖链、分支与执行调度实验

本实验在 NVIDIA GeForce RTX 3090、CUDA 12.4（`V12.4.131`）上以 `sm_86` 本机目标编译。四个 kernel 各自与 CPU 参考结果比较，再测平均耗时和按源码计算的 FMA 吞吐。命令参数依次为 `n`、`steps`、`repeats`：`257 / 20 / 5` 每个 block 启动 256 个线程，grid 有 2 个 block，尾块只有 1 个有效线程；`1,048,576 / 200 / 50` 用于比较吞吐。以下命令从仓库根目录运行，其中 `cd` 将终端切换到本章目录。

```bash
cd roadmap/curriculum/gpu/02-cuda-execution-and-scheduling
nvcc -O3 -std=c++17 -lineinfo --resource-usage -arch=sm_86 \
  examples/execution_and_scheduling.cu -o /tmp/cuda-execution-sm86
/tmp/cuda-execution-sm86 257 20 5
compute-sanitizer --tool memcheck --error-exitcode=1 /tmp/cuda-execution-sm86 257 20 5
/tmp/cuda-execution-sm86 1048576 200 50
```

程序先计算 CPU 参考值：两个累加版本使用 `std::fma`，分支版本使用普通乘加表达式。每个 kernel 预热一次，再重复启动 `repeats` 次，用 CUDA event 测量总时间并除以重复次数，得到平均单次耗时 `ms`。输入、输出拷贝在计时区间外，连续提交之间的空隙计入总时间。测量后，程序把 GPU 结果拷回主机，逐项比较参考值并计算最大绝对误差 `max_abs_error`。结果有限、误差不超过 `1e-4` 且计时有效时，打印 PASS。`source_FMA_GFLOP_s` 按源码中的 FMA 数计算每秒十亿次浮点操作；一个 FMA 按一次乘法和一次加法计两次操作。

### 6.1 依赖链与四条独立累加链

依赖链每个元素执行 steps 次 FMA，每次使用上一步的 x。独立版本维护 x0、x1、x2、x3，每轮分别更新，再求和、减去 `0.6f` 并乘以 `0.25f`。steps 为 200 时，两者每元素分别执行 200 次和 800 次 FMA；各自与对应的 CPU 参考值比较，因为计算公式不同，输出值也不同。

<!-- source-check: examples/execution_and_scheduling.cu -->
~~~cpp
__global__ void dependent_chain_kernel(const float* input, float* output,
                                       int n, int steps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = input[i];
    for (int step = 0; step < steps; ++step) {
        x = fmaf(x, 1.000001f, 0.00001f);
    }
    output[i] = x;
}

__global__ void independent_accumulators_kernel(const float* input,
                                                float* output, int n,
                                                int steps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x0 = input[i];
    float x1 = input[i] + 0.1f;
    float x2 = input[i] + 0.2f;
    float x3 = input[i] + 0.3f;
    for (int step = 0; step < steps; ++step) {
        x0 = fmaf(x0, 1.000001f, 0.00001f);
        x1 = fmaf(x1, 1.000001f, 0.00001f);
        x2 = fmaf(x2, 1.000001f, 0.00001f);
        x3 = fmaf(x3, 1.000001f, 0.00001f);
    }
    output[i] = 0.25f * (x0 + x1 + x2 + x3 - 0.6f);
}
~~~

本次大输入的输出为：

~~~text
dependent      0.0201 ms max_abs_error=0 PASS source_FMA_GFLOP_s=20855.398
independent    0.0576 ms max_abs_error=0 PASS source_FMA_GFLOP_s=29132.291
~~~

每个版本与自己的参考函数比较；两个累加版本的初值和计算路径不同。dependent 每元素做 `200` 次 FMA，共 `209,715,200` 个 FMA，即 `419,430,400` FLOPs，耗时 20.1 us。independent 每元素做 `800` 次 FMA，共 `838,860,800` 个 FMA，即 `1,677,721,600` FLOPs，耗时 57.6 us。四倍源码工作用了约 2.87 倍时间，所以按源码 FMA 数计，单位时间工作量约高 40%，对应约 20.9 TFLOP/s 和 29.1 TFLOP/s。四个累加器属于同一线程，彼此不依赖，增加可连续发射的独立计算指令。

本次编译报告显示，单链每线程使用 9 个寄存器，四链使用 14 个。按 [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#occupancy)，RTX 3090 每个 SM 最多驻留 48 个 warp，并有 65,536 个 32 位寄存器。每个 block 的 256 个线程组成 8 个 warp，因此 warp 数量上限对应每 SM 6 个 block。未计分配取整时，单链每块使用 `9×256=2304` 个寄存器，四链每块使用 `14×256=3584` 个；六个 block 分别需要 13,824 和 21,504 个。实际寄存器分配按硬件粒度取整，可用 occupancy API 查询驻留上限。

小输入 `257 / 20 / 5` 中，dependent 为 0.0035 ms、error 0、PASS、2.953 GFLOP/s；independent 为 0.0029 ms、error 0、PASS、14.342 GFLOP/s。

### 6.2 分支与双路径选择

偶数位置输出 even_path，奇数位置输出 odd_path。分支版本按条件调用对应函数；`predicated_select_kernel` 先算两个候选值再选结果。PTX/SASS 可显示最终采用的跳转、谓词和算术指令；比较时间时一并记录保留的算术量。

<!-- source-check: examples/execution_and_scheduling.cu -->
~~~cpp
__device__ __forceinline__ float even_path(float x) {
    return fmaf(x, 1.25f, 0.5f);
}

__device__ __forceinline__ float odd_path(float x) {
    return fmaf(x, 0.75f, -0.25f);
}

__global__ void divergent_branch_kernel(const float* input, float* output,
                                        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if ((i & 1) == 0) {
        output[i] = even_path(input[i]);
    } else {
        output[i] = odd_path(input[i]);
    }
}

__global__ void predicated_select_kernel(const float* input, float* output,
                                         int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float even_value = even_path(input[i]);
    const float odd_value = odd_path(input[i]);
    output[i] = ((i & 1) == 0) ? even_value : odd_value;
}
~~~

sm86 native 大输入输出如下：

~~~text
divergent      0.0118 ms max_abs_error=1.1920929e-07 PASS
predicated     0.0113 ms max_abs_error=1.1920929e-07 PASS
~~~

偶数位置走 `even_path(x)=fmaf(x,1.25,0.5)`，奇数位置走 `odd_path(x)=fmaf(x,0.75,-0.25)`。divergent 版本按条件只计算对应路径；predicated 版本先计算两条路径再选择。CPU reference 使用普通乘加，GPU 使用 FMA，因此出现 `1.1920929e-07` 的舍入差；`1e-4` 容差下两者都 PASS。大输入两者分别为 11.8 us 和 11.3 us，差 0.5 us。

`steps` 只控制两个累加 kernel；这里的两条候选路径各只有一次 FMA。程序只为累加链打印 `source_FMA_GFLOP_s`，这两行只打印平均时间、最大误差和 PASS。

小输入 `257 / 20 / 5` 中，divergent 和 predicated 都是 0.0027 ms、error `5.9604645e-08`、PASS。

本次两种写法相差 0.5 μs。候选路径各只有一次 FMA，数组读写、kernel 启动和任务提交占据共同开销。编译器也可能将短分支改写为谓词指令；SASS 中的条件跳转、谓词和 FMA 可用于确认最终指令。相关规则见 [CUDA 12.4 SIMT Architecture](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#simt-architecture) 和 [Control Flow Instructions](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#control-flow-instructions)。

为了把路径差异拉长，新增 probe 仍调用同一个 `branch_probe_kernel`，只改变 `flags`。`flags[i]=1` 调 `even_chain`，`flags[i]=0` 调 `odd_chain`；每个函数的循环都依赖上一步的 `x`，每个线程做 `steps` 次 FMA。`__noinline__` 保留函数调用边界，`#pragma unroll 1` 保留运行时循环。

<!-- source-check: examples/execution_and_scheduling.cu -->
~~~cpp
void make_probe_flags(int n, bool warp_uniform,
                      std::vector<unsigned char>& flags) {
    flags.resize(n);
    for (int i = 0; i < n; ++i) {
        flags[i] = warp_uniform ? (((i / 32) & 1) == 0)
                                : ((i & 1) == 0);
    }
}
~~~

`warp_uniform` 中 i=0–31 全走 even，i=32–63 全走 odd；`lane_split` 中每个 warp 的偶数 lane 走 even，奇数 lane 走 odd。两种 flags 使用相同 kernel、相同输入规模、相同线程配置和相同每线程 steps；`n=1,048,576` 可被 64 整除，两条路径各有 524,288 个元素。

<!-- source-check: examples/execution_and_scheduling.cu -->
~~~cpp
__device__ __noinline__ float even_chain(float x, int steps) {
#pragma unroll 1
    for (int step = 0; step < steps; ++step) {
        x = fmaf(x, 1.000001f, 0.00001f);
    }
    return x;
}

__device__ __noinline__ float odd_chain(float x, int steps) {
#pragma unroll 1
    for (int step = 0; step < steps; ++step) {
        x = fmaf(x, 0.999999f, -0.00001f);
    }
    return x;
}

__global__ void branch_probe_kernel(const float* input,
                                    const unsigned char* flags,
                                    float* output, int n, int steps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    output[i] = flags[i] ? even_chain(input[i], steps)
                         : odd_chain(input[i], steps);
}
~~~

沿着一个线程的执行顺序看：先用 block 编号和线程编号算出元素位置 `i`，越过数组边界的线程立即退出；有效线程读取 `flags[i]` 和 `input[i]`，进入对应的累加函数，完成循环后把返回值写入 `output[i]`。循环中的 `x` 是该线程的局部计算值，每次 FMA 都使用上一次的结果。两种模式中，每个线程都只计算自己选中的一条链，因此总有效 FMA 数都是 `n×steps`；相同位置的元素可以选中不同的链，输出分别与各自的 CPU 参考值比较。

从仓库根目录将终端切换到 `roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/`，再运行脚本：

~~~bash
bash examples/run_branch_probe.sh
~~~

脚本默认用 `-arch=sm_86` 编译，先跑 CPU self-test，再用 `n=257、steps=8、repeats=3、rounds=3` 检查尾块，并运行 Compute Sanitizer memcheck；随后保存 SASS，分别测量 `steps=1/32/256` 的大输入。缺少检查工具时，脚本会打印相应的 SKIP。本次服务器完成了 memcheck，最后打印的结果目录为 `/tmp/branch-probe-NEQldh`。

服务器先输出环境和资源字段：

~~~text
branch_probe GPU=NVIDIA GeForce RTX 3090 cc=8.6 driver=13020 runtime=12040
branch_probe n=1048576 steps=256 repeats=50 rounds=9 block=256 grid=4096
branch_probe kernel numRegs=10 sharedBytes=0 localBytes=0 theoretical_blocks_per_sm=6
~~~

`cc=8.6` 对应本次 `sm_86` 编译目标。`driver=13020` 表示驱动支持 CUDA 13.2，`runtime=12040` 表示使用 CUDA 12.4 Runtime；版本号含义见 [CUDA Runtime Version Management](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-runtime-api/group__CUDART____VERSION.html)。大输入有 1,048,576 个元素，每个 block 包含 256 个线程，因此 `grid=1048576/256=4096`，没有不完整的尾块。

编译时的 `ptxas` 报告与运行时资源查询相互对应：

~~~text
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, 384 bytes cmem[0]
~~~

| 字段 | 含义 |
|---|---|
| `numRegs=10` | kernel 每线程使用的寄存器数 |
| `sharedBytes=0` / `localBytes=0` | 静态共享内存为 0，每线程本地内存为 0 |
| `stack frame=0` / `spill stores=0` / `spill loads=0` | 编译产物没有栈帧和寄存器溢出读写 |
| `cmem[0]=384` | kernel 参数等使用的常量内存，不是每线程 384 字节 |
| `theoretical_blocks_per_sm=6` | occupancy API 按资源计算的驻留上限；每 block 有 8 个 warp，6 个 block 共 48 个 warp，不是计时期间的实测驻留数 |

输入、flags 和输出由 `cudaMalloc` 在运行时分配。编译报告开头的 `0 bytes gmem` 是该编译阶段的静态字段；程序运行时仍分配并访问这些设备数组。报告也列出同一源文件中的四个原始 kernel；本次脚本实际启动 `branch_probe_kernel`。

小输入首先给出工具自检和两种模式的数值检查结果：

~~~text
branch_probe self-test PASS
branch_probe pattern=uniform_pre even=129 odd=128 max_abs_error=0 guard=PASS PASS
branch_probe pattern=split_pre even=129 odd=128 max_abs_error=0 guard=PASS PASS
~~~

CPU self-test 检查 flags 计数、参考值、非法参数、哨兵和 NaN 逻辑。`_pre` 表示正式计时前的 GPU 输出检查。`n=257` 需要两个 block：第一个处理 256 个元素，第二个只有一个有效线程，其余线程在边界判断处退出。两种 flags 都分配了 129 个 even 元素和 128 个 odd 元素。

`max_abs_error=0` 表示这次所有输出都与 CPU `std::fma` 参考值一致。输出区前后各放置 32 个 float 哨兵，`guard=PASS` 表示这些保护值没有被写坏。每轮计时结束后也会检查保护区，最后再检查最后执行模式的数值结果。`overall PASS` 汇总数值、保护区和有限且大于零的计时检查；三组大输入也都通过了这些检查。

Compute Sanitizer 对同一个小输入报告：

~~~text
========= ERROR SUMMARY: 0 errors
~~~

它在运行期间检查内存访问，插桩增加了额外耗时。因此性能表使用普通运行的数据，memcheck 记录用于检查内存错误。

大输入计时开始前，两种模式各预热 3 次。每一轮分别启动 50 次，CUDA event 总时间除以 50，得到该轮的平均单次耗时；再对 9 个轮均值取中位数。预热、数据拷贝和结果检查都在计时区间外，连续启动之间的提交空隙会计入时间。先后次序每轮交换，输出中的前两轮例如：

~~~text
branch_probe round=0 order=uniform-split uniform_ms=0.076534 split_ms=0.147640
branch_probe round=1 order=split-uniform uniform_ms=0.076554 split_ms=0.147844
~~~

本次大输入的正常计时为：

| steps | uniform median (ms) | split median (ms) | split / uniform |
|---:|---:|---:|---:|
| 1 | 0.013455 | 0.013578 | 1.009132 |
| 32 | 0.016404 | 0.024617 | 1.500624 |
| 256 | 0.076534 | 0.147702 | 1.929890 |

steps 从 1 增加到 32、256 时，split 相对 uniform 的额外耗时由约 0.9% 增加到 50.1%、93.0%。短路径中，启动、访存和函数调用等共同开销稀释了分歧的影响；计算链变长后，两种路径组织的差距逐渐显现。

uniform 的一个 warp 内，32 个线程走同一条路径；split 中，两组 16 个线程分别走两条路径，执行一条时另一组不参与。两种模式的有效计算量相同，但 split 需要执行两条路径的 warp 指令，因此长路径耗时接近 uniform 的两倍。

优化重点是减少 warp 内部的长路径分歧。若先对数据分组，分组和输出回排也应计入总时间。Profiler 可检查 FMA 循环每条 warp 指令的活跃线程数：uniform 预期 32，split 预期 16。

<details>
<summary>统计修正与原始记录</summary>

旧实现从未排序的原始 vector 取 `front/back`，导致 min/max 误取首末轮；中位数和比值计算正确。以下数字为原始九轮输出的复算：

~~~text
uniform min_ms=0.076104 median_ms=0.076534 max_ms=0.076776
split   min_ms=0.146903 median_ms=0.147702 max_ms=0.147884
~~~

原始记录见[实验记录](../../../../notes/cuda/execution-and-scheduling-rtx3090-2026-10-09.md)、[branch probe 日志](../../../../notes/cuda/logs/2026-10-10-branch-probe-rtx3090.txt)和[sm86 原始日志](../../../../notes/cuda/logs/2026-10-09-execution-and-scheduling-rtx3090-sm86.txt)。对应的 SASS 保存在 `/tmp/branch-probe-NEQldh/branch_probe.sass.txt`。

</details>

下面保留原始四项对照的完整程序；受控实验使用上面的 `--branch-probe` 入口。

<details>
<summary>完整程序：原始四项对照</summary>

<!-- source-check: examples/execution_and_scheduling_baseline_2026_10_09.cu -->
~~~cpp
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t error__ = (call);                                        \
        if (error__ != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n",                  \
                         __FILE__, __LINE__, cudaGetErrorString(error__));    \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (false)

__global__ void dependent_chain_kernel(const float* input, float* output,
                                       int n, int steps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = input[i];
    for (int step = 0; step < steps; ++step) {
        x = fmaf(x, 1.000001f, 0.00001f);
    }
    output[i] = x;
}

__global__ void independent_accumulators_kernel(const float* input,
                                                float* output, int n,
                                                int steps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x0 = input[i];
    float x1 = input[i] + 0.1f;
    float x2 = input[i] + 0.2f;
    float x3 = input[i] + 0.3f;
    for (int step = 0; step < steps; ++step) {
        x0 = fmaf(x0, 1.000001f, 0.00001f);
        x1 = fmaf(x1, 1.000001f, 0.00001f);
        x2 = fmaf(x2, 1.000001f, 0.00001f);
        x3 = fmaf(x3, 1.000001f, 0.00001f);
    }
    output[i] = 0.25f * (x0 + x1 + x2 + x3 - 0.6f);
}

__device__ __forceinline__ float even_path(float x) {
    return fmaf(x, 1.25f, 0.5f);
}

__device__ __forceinline__ float odd_path(float x) {
    return fmaf(x, 0.75f, -0.25f);
}

__global__ void divergent_branch_kernel(const float* input, float* output,
                                        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if ((i & 1) == 0) {
        output[i] = even_path(input[i]);
    } else {
        output[i] = odd_path(input[i]);
    }
}

__global__ void predicated_select_kernel(const float* input, float* output,
                                         int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float even_value = even_path(input[i]);
    const float odd_value = odd_path(input[i]);
    output[i] = ((i & 1) == 0) ? even_value : odd_value;
}

void reference_chain(const std::vector<float>& input,
                     std::vector<float>& output, int steps) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        for (int step = 0; step < steps; ++step) {
            x = std::fma(x, 1.000001f, 0.00001f);
        }
        output[i] = x;
    }
}

void reference_independent(const std::vector<float>& input,
                           std::vector<float>& output, int steps) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        float x0 = input[i];
        float x1 = input[i] + 0.1f;
        float x2 = input[i] + 0.2f;
        float x3 = input[i] + 0.3f;
        for (int step = 0; step < steps; ++step) {
            x0 = std::fma(x0, 1.000001f, 0.00001f);
            x1 = std::fma(x1, 1.000001f, 0.00001f);
            x2 = std::fma(x2, 1.000001f, 0.00001f);
            x3 = std::fma(x3, 1.000001f, 0.00001f);
        }
        output[i] = 0.25f * (x0 + x1 + x2 + x3 - 0.6f);
    }
}

void reference_branch(const std::vector<float>& input,
                      std::vector<float>& output) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = (i & 1) ? (input[i] * 0.75f - 0.25f)
                            : (input[i] * 1.25f + 0.5f);
    }
}

float max_abs_error(const std::vector<float>& actual,
                    const std::vector<float>& expected) {
    float error = 0.0f;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            return std::numeric_limits<float>::infinity();
        }
        error = std::max(error, std::fabs(actual[i] - expected[i]));
    }
    return error;
}

template <typename Launch>
float measure(Launch launch, float* device_output,
              std::vector<float>& host_output, int repeats) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(host_output.data(), device_output,
                          host_output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms / static_cast<float>(repeats);
}

}  // namespace

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 1 << 20;
    const int steps = argc > 2 ? std::atoi(argv[2]) : 200;
    const int repeats = argc > 3 ? std::atoi(argv[3]) : 50;
    if (n <= 0 || n > std::numeric_limits<int>::max() - 255 ||
        steps <= 0 || steps > std::numeric_limits<int>::max() / 4 ||
        repeats <= 0) return EXIT_FAILURE;

    std::vector<float> input(n), output(n), reference(n);
    for (int i = 0; i < n; ++i) input[i] = 0.001f * static_cast<float>(i % 1000);

    float* device_input = nullptr;
    float* device_output = nullptr;
    CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    bool all_ok = true;
    auto run = [&](const char* name, int source_fmas, auto launch, auto reference_fn) {
        reference_fn(input, reference);
        const float ms = measure(launch, device_output, output, repeats);
        const float error = max_abs_error(output, reference);
        const bool ok = std::isfinite(error) && error <= 1.0e-4f &&
                        std::isfinite(ms) && ms > 0.0f;
        all_ok = all_ok && ok;
        std::printf("%-14s %.4f ms max_abs_error=%.8g %s", name, ms,
                    error, ok ? "PASS" : "FAIL");
        if (source_fmas > 0 && ms > 0.0f) {
            // Source-level arithmetic count, not an instruction counter.
            const double gflops = 2.0 * n * source_fmas / (ms * 1.0e6);
            std::printf(" source_FMA_GFLOP_s=%.3f", gflops);
        }
        std::printf("\n");
    };

    run("dependent", steps, [&] {
        dependent_chain_kernel<<<blocks, threads>>>(device_input, device_output,
                                                     n, steps);
    }, [&](const auto& a, auto& b) { reference_chain(a, b, steps); });

    run("independent", 4 * steps, [&] {
        independent_accumulators_kernel<<<blocks, threads>>>(
            device_input, device_output, n, steps);
    }, [&](const auto& a, auto& b) { reference_independent(a, b, steps); });

    run("divergent", 0, [&] {
        divergent_branch_kernel<<<blocks, threads>>>(device_input, device_output,
                                                      n);
    }, [&](const auto& a, auto& b) { reference_branch(a, b); });

    run("predicated", 0, [&] {
        predicated_select_kernel<<<blocks, threads>>>(
            device_input, device_output, n);
    }, [&](const auto& a, auto& b) { reference_branch(a, b); });

    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
~~~

</details>

第 6 节把前面的依赖链、分支和类型问题放进同一组实验，保留每个 kernel 的输入、CPU 参考、误差和测量口径。完成主线后，可展开第 4 节，把 grid 大小和驻留 block 上限换成任务量估算。

## 7. 从 CUDA 源码到 PTX 与 SASS

CUDA 源文件包含 CPU 侧控制代码和 GPU kernel。沿第 6 节的单链 FMA 例子看编译过程，可以分清两边的代码如何进入最终程序，以及 PTX、SASS 分别能说明什么。

<!-- source-check: examples/execution_and_scheduling.cu -->
~~~cpp
__global__ void dependent_chain_kernel(const float* input, float* output,
                                       int n, int steps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = input[i];
    for (int step = 0; step < steps; ++step) {
        x = fmaf(x, 1.000001f, 0.00001f);
    }
    output[i] = x;
}
~~~

下面只展示 host 启动部分。示例假定 `device_input` 已填入有效输入、`device_output` 已分配；重点是配置 launch、检查提交状态并等待执行完成：

```cpp
const int n = 1 << 20;
const int threads = 256;
const int blocks = (n + threads - 1) / threads;
dependent_chain_kernel<<<blocks, threads>>>(
    device_input, device_output, n, 200);
cudaError_t launch_status = cudaGetLastError();
if (launch_status == cudaSuccess) {
    launch_status = cudaDeviceSynchronize();
}
```

这段 host 代码和 CUDA Runtime API 调用在 CPU 上运行；带 `__global__` 的函数体在 GPU 上运行。三尖括号里的 `blocks`、`threads` 设置 grid 和 block，括号里的参数才是传给 kernel 的值。launch 返回后先检查配置错误；同步调用再报告此前 kernel 的执行错误。

完整程序应在 `launch_status` 非 `cudaSuccess` 时报告错误并停止，不读取输出；同步状态也要检查成功后，才能进行结果比较。

`nvcc` 是编译阶段的驱动，负责解析 `.cu` 并协调主机与设备编译；CUDA driver 在运行时根据当前 GPU 选择兼容 cubin，或在受支持条件下将 PTX JIT 后加载。普通 `gcc`/`g++` 或 `cl` 不直接识别 `<<<...>>>`。例如 host 代码中的 `dependent_chain_kernel<<<blocks, threads>>>(...)` 会先由 `nvcc` 转换成标准 host C++ 调用包装；`nvcc` 同时把已编译的设备映像嵌入合成后的 host 代码，再交给 host compiler 编译。

| 阶段 | 编译器处理 |
|---|---|
| 解析与分离 | `nvcc` 识别 `.cu` 中的 host/device 区域，并分别预处理。 |
| device 编译 | NVCC device 前端生成 PTX；`ptxas` 为目标 `sm_XX` 生成 cubin；fatbin 收集 cubin，也可包含 PTX 回退。 |
| host 合成 | `nvcc` 将 CUDA 扩展（包括 kernel launch）改写为标准 C++ host constructs，并把 fatbin 映像嵌入合成代码。 |
| host 编译与链接 | `gcc`/`g++` 或 `cl` 编译带嵌入设备映像的 host 代码生成 object；链接时加入 CUDA Runtime，生成 `.exe` 或 Linux 可执行文件。 |
| 装载与执行 | CUDA driver 为当前 GPU 选择兼容 cubin；没有合适 cubin 时，只有 fatbin 含受支持的 PTX 且 driver 能处理，才会执行 PTX JIT。 |

分离编译时，host object 与 relocatable device code 可先分别生成，再由设备链接阶段解析跨编译单元的 device 调用。

常见后缀对应不同产物：`.cpp` 通常是主机 C++ 源码；`.cu` 可同时包含主机代码和 kernel；`.ptx` 是虚拟指令集文本；`.cubin` 是目标 GPU 的设备二进制；`.fatbin` 可装多个 cubin 和/或 PTX；`.o` 是编译得到的目标文件，可能含主机代码或可重定位设备代码；`.exe` 是 Windows 可执行程序。用 `cuobjdump` 可查看可执行文件内包含的设备目标。

PTX（Parallel Thread Execution）是面向虚拟 ISA（指令集架构，Instruction Set Architecture）的中间表示，使用虚拟寄存器。`ptxas` 按目标架构分配物理寄存器、选择指令并安排控制流；SASS 是目标 GPU 机器指令的可读反汇编，GPU 执行 cubin 中编码的机器指令。编译器可合并、展开或删除源代码操作，一条 PTX 指令也可生成多条 SASS。每线程物理寄存器数应读 `ptxas` 或资源查询报告。

下面的 PTX 与 SASS 片段依据本节源代码整理，用于追踪索引、加载、累加和写回。寄存器编号标示数据角色；地址构造和参数初始化另有省略。实际生成结果随编译器、Toolkit、编译选项和目标 GPU 而变化。

真实 `.ptx` 文件顶部还包含 `.version`、`.target` 和 `.address_size`，分别声明 PTX 版本、虚拟或目标架构，以及地址宽度；以下只摘出 kernel 主体。

语法上，`.visible .entry` 声明 kernel 入口，`.param` 声明入口参数，`.reg` 声明虚拟寄存器；`%p<2>` 这类范围声明会编号为 `%p0`、`%p1`，`%r<5>` 则编号为 `%r0` 到 `%r4`。`.b32`、`.b64` 表示位宽，`.f32` 表示 32 位浮点，`.u32`/`.s32` 分别表示无符号/有符号 32 位整数；`%p` 是 predicate（谓词）寄存器。`r`、`rd`、`f` 是寄存器名的常见约定，具体类型由声明与指令后缀决定。

```ptx
// 教学示意：虚拟寄存器和指令序列会随实际编译改变。
.visible .entry dependent_chain(
    .param .u64 input,
    .param .u64 output,
    .param .s32 n,
    .param .s32 steps
)
{
.reg .pred %p<2>;
.reg .b32 %r<5>;
.reg .b64 %rd<5>;
.reg .f32 %f<3>;
ld.param.u64 %rd0, [input];
ld.param.u64 %rd1, [output];
cvta.to.global.u64 %rd0, %rd0;
cvta.to.global.u64 %rd1, %rd1;
ld.param.s32 %r3, [n];
ld.param.s32 %r4, [steps];
mov.u32 %r0, %ctaid.x;
mov.u32 %r1, %ntid.x;
mov.u32 %r2, %tid.x;
mad.lo.u32 %r0, %r0, %r1, %r2;
setp.ge.s32 %p0, %r0, %r3;
@%p0 bra DONE;
mul.wide.s32 %rd2, %r0, 4;
add.s64 %rd3, %rd0, %rd2;
ld.global.f32 %f0, [%rd3];
mov.f32 %f1, 0f3F800008;
mov.f32 %f2, 0f3727C5AC;
mov.s32 %r2, 0;
LOOP:
setp.ge.s32 %p1, %r2, %r4;
@%p1 bra STORE;
fma.rn.f32 %f0, %f0, %f1, %f2;
add.s32 %r2, %r2, 1;
bra LOOP;
STORE:
add.s64 %rd4, %rd1, %rd2;
st.global.f32 [%rd4], %f0;
DONE:
ret;
}
```

`%tid.x`、`%ctaid.x`、`%ntid.x` 是线程、block 和 block 尺寸的特殊寄存器；`mov` 将它们读入普通虚拟寄存器，`mad.lo` 计算线性索引。`setp` 设置 predicate，`@%p0 bra` 在索引越界时跳过读写。`mul.wide` 把 `float` 元素索引变成 64 位字节偏移；`%rd2` 始终保存偏移，`%rd3`、`%rd4` 分别保存输入和输出地址。`ld.global`/`st.global` 标出 global 地址空间。循环 predicate 在 `step >= steps` 时退出；`fma.rn.f32` 执行 FP32 融合乘加并按最近偶数舍入，下一轮继续使用 `%f0`，保留单链依赖。PTX 的 `0f` 前缀表示后续 8 个十六进制数按 IEEE 754 FP32 位型解释：`0f3F800008` 对应 `1.000001f`，`0f3727C5AC` 对应 `0.00001f`。

SASS 使用目标 GPU 的物理寄存器和指令编码。表中省略地址构造、常量初始化及循环入口判断；64 位输入、输出地址分别占 `R6:R7`、`R8:R9` 寄存器对，访存指令用寄存器对起始编号表示地址。循环片段采用 post-test 形式，因此先检查 `steps > 0`；本节传入 200，若 steps 为 0 则直接写回输入值。

| 指令示意 | 数据角色与作用 |
|---|---|
| `S2R R3, SR_TID.X` | 读取 `threadIdx.x`；后续索引计算省略，线性索引放在 `R4`，元素数 `n` 放在 `R5` |
| `ISETP.GE.AND P0, PT, R4, R5, PT`；`@P0 BRA DONE` | 比较 `i >= n`；越界线程跳过读写 |
| `LDG.E R0, [R6]` | 从输入地址寄存器对读取 `x` 到 `R0` |
| `ISETP.GT.AND P2, PT, R11, RZ, PT`；`@!P2 BRA STORE` | 循环入口前检查 `steps > 0`；零步时直接写回 `x` |
| `FFMA R0, R0, R1, R2` | 令 `R1` 为 scale、`R2` 为 bias，执行 `x = x * scale + bias` |
| `IADD3 R10, R10, 0x1, RZ` | 将循环计数器 `R10` 加一；`R11` 保存 `steps` |
| `ISETP.LT.AND P1, PT, R10, R11, PT`；`@P1 BRA LOOP` | post-test 回边：`step < steps` 时再执行一轮；否则落到 store |
| `STG.E [R8], R0` | 将结果写入输出地址寄存器对 |

`PT` 和 `RZ` 分别是 predicate 真值与零寄存器。真实反汇编可能重排地址计算、合并操作，或把边界处理改成谓词化访存；应从目标代码确认分支结构和 FMA 是否仍沿同一累加值形成依赖链。表中的寄存器编号是自洽的教学角色分配，不对应某次实际分配。

`.reg .f32` 等 PTX 声明使用的是虚拟寄存器；实际每线程物理寄存器数应读 `ptxas` 报告或 `cuobjdump --dump-resource-usage`。`local` 访问也不自动等于 spill：动态寻址的局部数组可能使用 local memory，spill 是寄存器分配不足时的另一种来源。应结合 spill/local 字节报告、SASS 中的 local load/store，以及对应源码对象一起判断。

静态编译与反汇编需要 CUDA Toolkit 和主机编译器；kernel 执行、数值验证、Compute Sanitizer 与性能测量还需要兼容设备和驱动。`-O3` 设置主机编译器优化级别；设备代码优化由 CUDA 编译流程及目标架构选项控制。`-lineinfo` 加入源码行映射，源码与机器指令之间通常不是一对一关系。`-G` 生成设备调试代码，会改变优化和资源用量；性能测试应使用实际发布配置。

先用 `nvcc --list-gpu-code` 查看当前 Toolkit 支持生成的 `sm_XX`。命令中的 `XX` 替换为目标计算能力主、次版本的拼接值，例如 CC 8.6 对应 `86`，CC 8.9 对应 `89`。实际值按设备和当前 Toolkit 确认。以下 PowerShell 命令在本章目录运行；编译产物使用相同 Toolkit、主机编译器、优化级别和目标架构，便于比较对应代码：

```powershell
# 示例为 CC 8.6；按目标设备改成实际支持的 sm_XX。
nvcc --list-gpu-code
$arch = "sm_86"
nvcc -O3 -std=c++17 -lineinfo --resource-usage -arch=$arch `
  examples/execution_and_scheduling.cu -o execution_check.exe
nvcc -O3 -std=c++17 -lineinfo -ptx -arch=$arch `
  examples/execution_and_scheduling.cu -o execution_check.ptx
nvcc -O3 -std=c++17 -lineinfo -cubin -arch=$arch `
  examples/execution_and_scheduling.cu -o execution_check.cubin
cuobjdump --dump-ptx execution_check.exe > execution_check.ptx.txt
cuobjdump --dump-sass execution_check.exe > execution_check.sass.txt
cuobjdump --dump-resource-usage execution_check.exe
nvdisasm --print-code --print-line-info execution_check.cubin
```

从仓库根目录切换到本章目录后，Linux 可使用相同选项：

```bash
# 示例为 CC 8.6；按目标设备改成实际支持的 sm_XX。
nvcc --list-gpu-code
arch=sm_86
nvcc -O3 -std=c++17 -lineinfo --resource-usage -arch="$arch" \
  examples/execution_and_scheduling.cu -o execution_check
nvcc -O3 -std=c++17 -lineinfo -ptx -arch="$arch" \
  examples/execution_and_scheduling.cu -o execution_check.ptx
nvcc -O3 -std=c++17 -lineinfo -cubin -arch="$arch" \
  examples/execution_and_scheduling.cu -o execution_check.cubin
cuobjdump --dump-ptx execution_check > execution_check.ptx.txt
cuobjdump --dump-sass execution_check > execution_check.sass.txt
cuobjdump --dump-resource-usage execution_check
nvdisasm --print-code --print-line-info execution_check.cubin
ldd ./execution_check
readelf -d ./execution_check
./execution_check 257 20 5
compute-sanitizer --tool memcheck --error-exitcode=1 ./execution_check 257 20 5
```

上面的编译、反汇编、`ldd` 与 `readelf` 检查可在没有 GPU 的主机上完成；最后两条会启动 kernel 并运行 sanitizer，需要兼容设备与驱动。

以下命令在本章目录运行。需要同时嵌入指定架构 cubin 与可供兼容设备 JIT 的 PTX 时，可构建 fatbin：

```bash
nvcc -O3 -std=c++17 -lineinfo \
  -gencode 'arch=compute_86,code=[sm_86,compute_86]' \
  examples/execution_and_scheduling.cu -o execution_fallback
cuobjdump --list-elf execution_fallback
cuobjdump --list-ptx execution_fallback
```

上例以 CC 8.6 展示成对填写的目标值；按实际设备与 Toolkit 支持结果，同时替换 `compute_86` 和 `sm_86` 中的版本号。`compute_XX` 是虚拟计算能力和 PTX 目标，`sm_XX` 是实际 GPU 的 cubin 目标。PTX 回退只有在 fatbin 确实包含该 PTX，且目标驱动支持时才可用。PowerShell 可用相同的 `-gencode` 选项，并将输出名改为 `.exe`。

检查依赖链时，在 PTX 中定位循环回边与 `fma.rn.f32`，在 SASS 中查找 `FFMA`、循环计数和分支；若编译器展开循环或合并指令，应以真实控制流及读写同一累加寄存器的依赖为准。检查路径选择时沿 predicate 设置和条件 `BRA` 追踪两条路径；若变成谓词执行或选择指令，就记录实际形式，不从 C++ 函数名推断。检查 local/spill 时对照 `ptxas` 的 spill bytes/local bytes、SASS 中 local load/store 与源码里的动态索引对象。`--resource-usage`、`cuobjdump`、`nvdisasm` 用于查看资源和代码；`ldd`、`readelf -d` 检查 Linux host 动态库。主机崩溃时可运行 `gdb --args ./execution_check 257 20 5`，再用 `run` 和 `bt` 查看调用栈。`CUDA_LAUNCH_BLOCKING=1` 和 Compute Sanitizer 可帮助定位设备执行错误，但会改变执行条件，正式计时应关闭。

Triton 的 PTX/cubin 通常由运行时编译并写入缓存。先运行目标 kernel，再按 Triton 版本、设备、数据类型、输入形状和编译配置定位对应缓存产物；缓存目录结构会随版本变化。将缓存中的 PTX/cubin 与本次启动的输入配置对应，再检查指令和资源。

编译阶段可对照 [NVCC compilation trajectory](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#the-cuda-compilation-trajectory)；fatbin、cubin 与反汇编工具见 [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)；PTX 的指令、特殊寄存器和修饰符见 [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)。

### 7.1 编译、链接与运行时错误

一个 CUDA 程序经过源代码编译、链接、程序装载和设备执行。错误发生在哪个阶段，决定下一步检查什么。

| 现象 | 所处阶段 | 首先检查 |
|---|---|---|
| 未声明标识符、模板实例化失败、设备函数调用主机函数 | 编译 | 声明、头文件、函数执行空间、编译目标和 Toolkit 版本 |
| `undefined reference` 或设备链接找不到符号 | 链接 | 定义是否进入构建、声明与定义是否一致、库是否链接、跨文件设备调用是否启用 RDC |
| 程序启动时找不到共享库 | 装载 | 可执行文件依赖的库及其搜索路径，并核对库版本 |
| kernel 启动配置失败 | 提交 | grid/block、共享内存用量、目标架构与可用设备代码 |
| 在同步或后续 API 才报告非法访问 | 设备执行 | 同一 stream 之前提交的 kernel、地址与边界；同步点可能只是报告错误的位置 |

RDC（Relocatable Device Code，可重定位设备代码）支持设备端函数跨编译单元链接，通常用 `-rdc=true` 或分阶段 `-dc`、`-dlink` 构建。`-lcuda` 用于链接主机侧 Driver API；设备函数定义仍须加入设备链接输入。

在 Linux 服务器上，从仓库根目录切换到本章目录后，可执行以下检查并检查生成文件：

```bash
nvcc -std=c++17 -O2 -g -lineinfo examples/execution_and_scheduling.cu -o execution_check
ldd ./execution_check
readelf -d ./execution_check
./execution_check 257 20 5
compute-sanitizer --tool memcheck --error-exitcode=1 ./execution_check 257 20 5
```

`ldd` 检查主机动态库实际从哪里加载，`readelf -d` 检查可执行文件的动态链接信息。主机代码崩溃时，用 `gdb --args ./execution_check 257 20 5` 启动，执行 run、bt 查看调用栈。设备访存报错则用小输入和 Compute Sanitizer 查 kernel 地址。两类工具分别定位主机调用和设备访问。

缩小问题时，只保留一个失败形状、一个实现和固定输入；记录完整命令、首个错误及对应源码位置。修复后重新进行数值比较和边界测试，再恢复异步执行与性能计时。

## 8. 从编译产物到 kernel 启动

程序装载设备代码后，CPU 侧通过 Runtime API 或 Driver API 建立启动所需的设备上下文、代码模块和 kernel 句柄，再把 launch 请求提交到 stream。两套 API 都控制 GPU；Runtime 封装常见步骤，Driver API 则让程序显式管理这些对象。

以向量加法 `vector_add` 为例，主机先选择 GPU，再取得该设备的 context（上下文）。上下文管理设备显存分配、已装载代码和执行资源。程序随后把设备代码装入 module（代码包），按函数符号 `vector_add` 查找入口，Driver 返回 `CUfunction` 句柄供后续启动使用。

主机随后准备参数，把启动请求排进 stream（执行队列）。GPU 按 stream 中的顺序执行，主机在同步或读回结果时等待完成。CUDA C++ 的 `kernel<<<grid, block, shared_bytes, stream>>>(args)` 启动语法由编译器生成主机侧调用，Runtime 负责传参和代码装载等步骤；Driver API 则由程序显式调用选设备、取得 context、装载 module、查找函数和启动 kernel 的接口。

一个 CPU 线程调用 Driver API 时，使用的是它当前选中的 current context；另一个 CPU 线程可以选中自己的 context。Runtime 通常共享设备的 primary context（主上下文）：`cuDevicePrimaryCtxRetain` 取得使用权，`cuCtxSetCurrent` 将它设为当前环境，最后用对应的 release 归还使用权。本例在同一 context 中分配显存、取得函数和启动 kernel，保持资源的所属关系一致；自行创建的 context 则在使用结束后调用对应的 destroy 操作。

<details>
<summary>PTX、Driver 启动参数与完整向量加法程序</summary>

下面程序嵌入一段 PTX，按“取得主上下文→装载模块→查找 kernel→分配显存→传参启动→复制结果”的顺序执行。PTX 声明三个 64 位设备地址和一个 32 位整数 n，主机侧按相同宽度准备参数。

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

启动参数数组 `kernel_params` 中，每一项指向主机上保存参数值的变量。`&d_a` 指向主机变量 d_a；Driver 从那里取出 d_a 保存的设备地址值，复制到 kernel 参数区。GPU 用这个设备地址访问数组，不使用主机地址 &d_a。

这个例子使用同步 HtoD（Host to Device，CPU 到 GPU）和 DtoH（Device to Host，GPU 到 CPU）复制数据，主机缓冲区由 std::vector 保存。改用异步复制时，页锁定内存指传输期间不会被操作系统换出的主机内存；应核对该 API 的要求，并让缓冲区一直存活到传输完成。

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

装载 PTX 时为当前 GPU 编译设备代码，称为 JIT（即时编译）。`cuModuleLoadDataEx` 的 numOptions 指定配置项数量，options 和 optionValues 分别传选项及对应值；本例没有配置，前两者传 `0,nullptr`，optionValues 也传 nullptr。启动 kernel 后，主机继续执行，因此检查结果前先等待 stream 完成。

释放资源时先等设备结束使用，再销毁 stream、释放显存、卸载 module，最后 release 主上下文。主上下文由 retain/release 配对管理；自行创建的 context 使用对应的 destroy 操作。从仓库根目录运行以下编译与执行命令，源文件路径相对于仓库根目录。

```bash
nvcc -std=c++17 -O2 roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/driver_api_vector_add.cu -lcuda -o driver_api_vector_add
./driver_api_vector_add --help
./driver_api_vector_add 257
```

程序找不到设备时退出 77，表示跳过；其他 Driver 错误返回失败。PTX 声明 `.target sm_70` 和 `.version 7.0`，Driver 装载时为兼容的实际 GPU 进行 JIT。编译这个主机程序仍需要 Toolkit 头文件、主机编译工具和兼容驱动。

</details>

### 8.1 Runtime 初始化、lazy loading 与错误定位

Runtime 初始化会准备运行时状态和上下文；设备代码装载则把某个 kernel 的代码准备好。这两项开销可能发生在不同位置。

Lazy loading（延迟装载）在首次使用 kernel 时装载代码；eager 模式提前装载。CUDA 11.7 引入此功能，Linux 从 12.2、Windows 从 12.3 起默认采用 lazy。进程启动前可设置 `CUDA_MODULE_LOADING=LAZY` 或 `EAGER`，运行时可用 `cuModuleGetLoadingMode` 查询。

`cudaFree(nullptr)` 可触发 Runtime/context 初始化。若要提前装载指定 kernel，再调用 `cudaFuncGetAttributes` 或 `cuModuleGetFunction`。

下面示例比较首次启动 add_one 与提前装载。传入 `--preload` 时，程序先调用 `cudaFuncGetAttributes`。启动后检查 `cudaGetLastError()`，同步时再检查设备执行错误。调试时可设置 `CUDA_LAUNCH_BLOCKING=1`；正式计时使用默认异步执行。从仓库根目录运行下方命令，源文件路径相对于仓库根目录。


<details>
<summary>Runtime 初始化、预载与错误检查示例</summary>

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
</details>


比较首次调用延迟时，分别计时：Runtime/context 初始化、函数预载、首次启动和预热后的重复启动。这样能判断额外时间花在初始化、JIT/代码装载还是 kernel 本身。若错误在同步处报告，检查同一 stream 中此前提交的操作；同步函数可能只是等到了并报告了那个错误。

## 9. 从线程分工到数据分块

tile 是 kernel 按数据块组织的一组逻辑值。普通 CUDA SIMT kernel 写明每个线程负责的元素，再安排 warp 和 block 协作；CUDA Tile 把一组元素交给编译器，由编译器安排硬件线程、加载和计算指令。

例如向量加法，SIMT 写法给每个线程计算 `blockIdx.x*blockDim.x+threadIdx.x`；tile 写法按逻辑块编号 `bid` 选择第一个或后续的 128 元素块，整块相加后写回。两种写法都要选择块大小、数据类型和 grid，并处理最后一个不完整块。

CUDA Tile 与 Triton 都可以把一组元素作为逻辑工作块表达；具体接口、编译版本和设备支持范围放在 9.6 的接口补充中，算法部分先按数据责任和地址关系阅读。

### 9.1 从一个元素到一块数据

长度 256 的向量可以由第 0 个 tile 负责逻辑元素 0..127、第 1 个 tile 负责逻辑元素 128..255。128 是本例的逻辑元素数；物理线程数和线程如何取得这些值由编译结果决定。SIMT 版本把责任写到每个线程，tile 版本把同一责任写成一整块数据。

普通 CUDA C++ 显式写一个线程的元素索引：

~~~cpp
__global__ void add_simt(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
// n > 0；a、b、c 指向至少 n 个 float，c 不与输入重叠。
add_simt<<<(n + 127) / 128, 128, 0, stream>>>(a, b, c, n);
~~~
### 9.2 坐标怎样落到元素和地址

Array 是设备内存中的数组；形状说明各维的元素数，row-major（按行连续）布局把同一行相邻元素放在连续位置。tile 坐标选择数据块，局部坐标选择块内元素，随后按数组形状和步长计算地址。步长表示相邻索引之间跨过的元素数或字节数，单位由接口决定；下面公式以元素为单位，换算字节地址时再乘 `sizeof(T)`。

设 Array 的形状为 `(M,N)`，tile 的形状为 `(TM,TN)`。tile 坐标 `(bm,bn)` 中的局部元素 `(i,j)` 对应：

$$
r=b_mT_M+i,\qquad c=b_nT_N+j.
$$

对于 row-major 连续数组，它的元素偏移是：

$$
\operatorname{offset}(i,j)=(b_mT_M+i)N+(b_nT_N+j).
$$

取 `A[10,16]`、tile `(2,4)`、tile 坐标 `(1,2)`，该 tile 从元素 `(2,8)` 开始，包含以下八个元素：

| tile 内局部行 | 第 0 列 | 第 1 列 | 第 2 列 | 第 3 列 |
|---|---|---|---|---|
| 0 | A[2,8]，偏移 40 | A[2,9]，41 | A[2,10]，42 | A[2,11]，43 |
| 1 | A[3,8]，偏移 56 | A[3,9]，57 | A[3,10]，58 | A[3,11]，59 |

规则 tile-space load 用块坐标 `(1,2)` 选择整块。gather 是为每个元素提供自己的索引，从不连续位置读取数据。Triton 的指针写法则构造每个元素地址。使用哪个接口，就按哪个接口的坐标含义传参。

规则二维访问把形状和步长交给编译器，有利于选择适合整块搬运的指令；支持的 GPU 上可采用 Tensor Memory Accelerator（TMA，专用的张量搬运单元）。Gather 适合查表、置换和由数据决定的索引。TMA 的实际选用结果仍以生成指令为准。

### 9.3 逻辑 tile、Array 与硬件线程资源

前面的二维示例中，`tile.shape=(2,4)` 表示两行四列、共八个逻辑值；编译器把这些值的加载、计算和写回分配给硬件线程，并安排寄存器等资源。

下面换成一维向量。源数组可以是任意合法长度，例如长度 1025 的向量，用长度 128 的 tile 处理时，最后一块只有一个有效元素。Triton 的向量加一先计算这组元素的地址，再读出值：

~~~python
offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
ptrs = a + offsets                         # 指针组成的 tile，还没有读取 a
x = tl.load(ptrs, offsets < N, other=0.0)   # 数据组成的 tile
y = x + 1.0                               # 新的逻辑值
tl.store(c + offsets, y, offsets < N)       # 写入 c 的有效元素
~~~

`ptrs` 保存一组地址，尚未读取数据；`tl.load` 得到这些地址中的值，`x + 1.0` 产生新的逻辑值，`tl.store` 才写回有效元素。编译器再把这组访问分配给线程；要确认某个 lane 访问哪些元素，需要检查编译后的布局。

### 9.4 尾块与加载前保护

mask 是标记每个位置是否有效的布尔条件。长度 129、tile 长度 128 时，grid 有两个 tile block。第二个 block 只有第一个元素有效，其余 127 个加载位置由 mask 排除并填零，输出只存第一个位置。长度为零时在 host 直接返回，不发起零大小 grid，也不读取一个完全处于 Array 外的 tile。

grid 必须按数组长度正确计算。上述规则处理的是与数组边界部分相交的尾块。gather 按每个元素自己的索引读取，scatter 按每个元素自己的目标索引写回；这两类访问以及完全在数组外的块，需要按相应接口的边界规则检查。

归约是把一组元素合成为一个结果。求和与 GEMM 的无效归约项可填零；求最大值通常填负无穷，例如有效值全是负数时，填零会错误地把最大值变成零。Softmax 则应先排除无效位置，再让它们的指数贡献为零。

`ct.where(mask,x,y)` 按 mask 选择已计算的值。加载前应用边界 mask，写回时也对输出位置应用 mask，这样越界位置不会参与读写。

### 9.5 分块大小、数据复用与 FP32 累加器

采用 `A[M,N] @ B[N,K] = C[M,K]`，N 是归约维。对每个输出 `C[m,k]`，程序沿 `n=0..N-1` 计算 `A[m,n] * B[n,k]` 并求和；累加器（accumulator）保存已经算出的部分和。BM、BN、BK 分别表示输出行块、归约维块和输出列块的尺寸。一个 tile block 负责 BM 行、BK 列输出，每轮加载 A 的 BM×BN 块和 B 的 BN×BK 块，让已加载的数据参与多次乘加，再把结果加入同一个累加器。

同一块 A 或 B 数据被多个乘加复用时，读取流量相对减少。累加器要保存 BM×BK 个 FP32 部分和，因此增大 BM 或 BK 会增大它；增大 BN 则扩大每轮加载的输入块，不改变输出块的大小。这些选择会影响寄存器、共享内存等资源用量。下面的完整 cuTile GEMM 代码把这个循环写成具体接口：

<details>
<summary>cuTile GEMM kernel 与接口细节</summary>

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

输入是连续二维 FP16 数组，累加器是 FP32，grid 为 `(ceil(M/BM),ceil(K/BK))`，即行、列方向的输出块数分别向上取整。Tile 大小按目标矩阵指令的支持范围选择。`ct.mma` 返回新累加器，所以每轮要把返回值赋给 acc。完成全部归约后再转换输出，保留归约过程的 FP32 累加精度。

</details>

它与 Triton 的 `acc += tl.dot(tile_a,tile_b)` 表达相同的分块算法。比较两个实现时，固定形状和数值类型，再看各自的数据复用、生成指令、寄存器与共享内存用量，以及实际时间。

以输入元素占 s 字节计算，一轮理想 tile 加载需要：

$$
B_{\mathrm{load}}=s(B_MB_N+B_NB_K),\qquad
F_{\mathrm{tile}}=2B_MB_NB_K.
$$

增大 BM 或 BK 可让已加载数据参加更多乘加，提高这部分计算与读取的比值；同时 acc 要保存更多 FP32 值，占用更多资源。完整性能估算还要加入输出写回和缓存行为，并检查资源增加后还能驻留多少 block。

### 9.6 cuTile 接口、编译提示与性能证据

CUDA Tile 的 Python 接口是 `cuda.tile`；C++ 接口使用 `cuda_tile.h` 和 `cuda::tiles`。下面的 C++ 示例按 CUDA Toolkit 13.3 编写，运行前应核对 Toolkit、Python 包、驱动和 GPU 的支持范围；Triton 使用自己的编译系统。
<details>
<summary>cuTile 向量加法接口、完整 Python 测试与编译提示</summary>


在读取 C++ 示例前，先对应接口和动作：

| 接口或参数 | 作用 |
|---|---|
| `tensor_span` | 描述指针及其数组维度 |
| `partition_view` | 按指定 `shape` 观察并分割该数组 |
| `extents` | 给出各维长度 |
| `128_ic` | 编译期整数常量 128 |
| `bid` | 当前逻辑块的编号 |
| `load_masked` / `store_masked` | 加载或写回时保护尾部无效位置 |
| `assume_aligned` | 声明地址已经满足对齐条件，不会自动修复地址 |

CUDA Tile C++ 则为指针附上数组形状，再定义分块：

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
// n > 0；a、b、c 分别由 cudaMalloc 分配，并满足此处的对齐与不重叠要求。
add_tile<<<(n + 127) / 128, 1, 0, stream>>>(a, b, c, n);
~~~

`128_ic` 在编译时确定 tile 大小，n 在运行时给出数组长度。`load_masked` 负责部分尾块；普通 load 要求整块有效。`assume_aligned` 只告知编译器地址已经对齐，不会替程序调整地址；这里的 16 字节对齐表示地址是 16 的整数倍。比如原始 FP32 指针 a 对齐到 16 字节，a+1 会偏移 4 字节，此时需要重新检查对齐。

CUDA Tile 接口要求 tile 各维大小为 2 的幂。源数组可使用其他合法长度，尾块由 mask 或相应边界模式处理。

下面的 Python 程序用 PyTorch 分配数组，用 cuTile kernel 做加法。测试输入选用可精确表示的小整数，便于逐元素检查。输出额外分配七个元素作为哨兵区，传入 kernel 的数组视图只包含前 n 个元素；运行后检查哨兵是否仍完整。

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

`ct.load` 用数组长度判断尾部，index 表示 tile 编号。示例加载时把尾部无效位置填零，存储时只写有效元素。`ct.Constant[int]` 让 tile 大小成为编译期参数；改变它可能生成另一份专门编译的 kernel。

编译 hint 是给编译器的建议性信息。occupancy hint 给出希望每个 SM 驻留多少个 CTA，CTA 在 CUDA 中就是 thread block；latency hint 用 1 到 10 表示预期 DRAM（这里指显存）流量的轻重，值较大时编译器可能增加预取深度。预取是提前加载后续计算要用的数据，以便搬运与当前计算重叠。allow_tma 只允许编译器在支持的设备上考虑 TMA 搬运。具体定义见 [cuTile 性能提示](https://docs.nvidia.com/cuda/cutile-python/performance.html)，实际采用哪些指令仍看生成代码。

Triton 的 num_warps 指定每个 program 使用的 warp 数，num_stages 控制编译器安排的流水阶段数；它们与 cuTile 的上述提示是不同参数。修改后，比较生成代码，再测寄存器、共享内存、驻留 warp、访存指令和时间，判断这次参数调整的实际效果。

Host 程序可在同一 stream 中先后启动 SIMT kernel 和 tile kernel，处理同一设备数组。设备端函数调用须遵循各自接口规则；C++ tile 函数与普通 `__device__` 函数分开调用。

</details>


### 9.7 高级 view、shape 变换与原子更新

滑动窗口的相邻 tile 可能重叠；不同 block 也可能同时把部分结果累加到同一个输出位置。前者要按坐标计算每个窗口覆盖的元素，后者要用原子加法让同一位置的并发更新不丢失。

view（视图）规定数组地址与坐标之间的关系，不复制数据。Array 保存设备数组的信息，Tile 保存当前计算的一组值，TiledView 规定怎样把数组分成 tile；真正的 `load` 才按这个关系读取值。这里的步长也称 `stride`：例如行跨度为 4 个 `int32` 元素，就是 16 字节。

对连续 row-major 的二维数组 `A[R,C]`，元素 `(i,j)` 的字节地址为：

$$
addr(i,j)=base+(iC+j)\,sizeof(T).
$$

Tile 形状为 `(r,c)`，起点步长为 `(u,v)` 时，第 `(p,q)` 个 tile 从元素坐标 `(pu,qv)` 开始。默认步长等于 tile 大小，相邻块正好衔接。若 u 小于 r，相邻块会重叠；u 大于 r，则留出空隙。`view.load((1,0))` 选择行方向的第二个 tile，元素起点按上述公式计算。

以 `A=arange(16).reshape(4,4)` 为例，每个 int32 占 4 字节，跨到下一行要加 16 字节。`tiled_view((2,4),traversal_steps=(1,4))` 每块取两行、四列，但行起点每次只加 1。因此 tile `(1,0)` 覆盖 `[[4,5,6,7],[8,9,10,11]]`，与前一个 tile 重叠一行。可以先在 CPU 上手算这些元素位置，再检查 GPU 编译结果中的线程分配。

`reshape` 保持元素序列，只改变分组方式，并要求元素总数相等。例如 `(2,4)` 改为 `(4,2)`，展平序列仍是 `4,5,6,7,8,9,10,11`。`transpose` 交换两个轴，`permute` 按指定次序排列多个轴；轴就是行、列等索引方向，交换轴会改变元素的逻辑位置。

这些操作先得到新的 tile 值，store 才把值写回数组。编译器可能用寄存器重排或线程间交换实现变换，实际成本需要检查生成指令。

下面程序把逻辑 view、重叠访问、reshape、transpose 和三维 permute 写成 cuTile kernel。`A=arange(16).reshape(4,4)` 的第二个窗口从第 1 行开始，因此与第一个窗口重叠一行；另一个 kernel 让 8 个 tile block 对同一计数位置各加 4，原子加法的结果为 `8×4=32`。三个输出的含义分别为：`flat_out` 保持 tile 行优先次序，`transpose_out` 展示轴交换，`permuted_out` 展示轴序 `(2,0,1)`。

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

<details>
<summary>原子接口、内存顺序与跨 block 依赖</summary>

`ct.atomic_add(array,indices,update)` 对每个指定数组元素做原子加法，返回该元素更新前的值。默认检查索引，默认内存顺序为 acquire-release（约束相关读写的先后关系），作用范围为 device（同一 GPU 上的参与者）。原子性针对每个元素分别成立；一个 tile 中多个元素的更新没有统一的先后顺序。

若只需要累计，不需要旧值，可用 `TiledView.atomic_store_add(tile_index,update_tile)`。两种接口都保护各自的原子更新；让所有线程或 block 等到一起，则需要另行安排同步和 kernel 顺序。

cuTile 代码按逻辑 block 和 tile 描述计算，线程分配由编译器完成。kernel 的 load/store 按接口定义的内存顺序执行；需要让另一个 block 等待写入完成时，应按文档设置原子操作的内存顺序和作用范围，或用 stream/event 建立 kernel 间依赖。原子更新保证同一元素的更新顺序，不负责让所有线程或 block 集体等待。

若一个 kernel 产生数组，另一个 kernel 读取它，可以把两次启动排到同一 stream，建立先后顺序。使用不同 stream 时，则用 event 或依赖关系让读取等待写入完成。主机想检查结果时，还要等设备工作结束。

</details>

从仓库根目录运行 `python roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/cuda_tile_views_atomics.py`。在 CUDA 13.3/cuTile 1.4 系列环境中，检查数组变换结果及计数 32。原子加法的到达次序可以不同，整数总和应相同。下面的 CPU 模型只计算坐标与预期值，设备程序再检验实际执行结果。

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

以下内容适合需要从 Python 或 C++ 直接管理 CUDA 资源，或要追查 Driver 返回码与 ABI 兼容时阅读。它们沿用前面解释的 context、stream、设备地址和编译链接概念。

## 10. Python 与 C++ 的调用边界和资源生命周期

同一张 GPU 上的计算，无论从 Python 还是 C++ 发起，都要处理设备地址、context、stream、完成顺序和资源生命周期。绑定接口是某种语言对现有函数接口的包装，例如用 Python 调用 CUDA 的 C 接口；不同语言可以使用不同包装，但必须遵守同样的资源和执行顺序。

### 10.1 Python host 控制与 device kernel 的边界

普通 Python 函数在 CPU 上执行。Python 可以调用已有的 CUDA 算子，也可以通过专用的 kernel 接口编写 GPU 代码；这种描述设备计算的专用语言或接口称为 DSL（Domain-Specific Language，领域专用语言）。绑定 API 则在主机侧选择设备和 context、分配资源并提交 stream 操作。下表按用途区分这些接口：

| 接口 | 用途 | 例子 |
|---|---|---|
| `cuda.bindings` | 在 Python 中调用 Driver/Runtime API | 分配显存、创建 stream、装载模块、启动 kernel |
| `cuda.core` | 用更高层的 Python 对象管理 CUDA 资源 | 管理设备、内存和执行资源 |
| `cuda.lang`、Numba-CUDA `@cuda.jit` | 定义按线程编写的 SIMT kernel | 每线程计算一个元素下标 |
| `cuda.tile` | 定义按 tile 编写的 kernel | 每个 tile block 计算一组元素 |
| CuPy、PyTorch | 管理数组并调用计算实现 | 分配设备张量、管理当前 stream |

这些库可以共同处理同一批数据。例如 PyTorch 分配数组，另一库提交 kernel，再由 PyTorch 读取结果；读取必须等待此前的写入完成，后面会说明如何用 stream 或 event 建立这个顺序。

NumPy 数组保存在主机内存，其 data 地址指向 CPU 可以访问的位置。Driver 分配得到的 `CUdeviceptr` 是设备地址，应交给匹配的设备和 context 使用。对于 FP32 向量，第 i 个元素相对起点偏移 4i 字节；C++ 的 float 指针加 i 会自动乘 4，Driver 的 CUdeviceptr 按字节计数，调用者自己计算字节偏移。

下面程序用 `cuda.bindings.driver` 取得主上下文，保存调用者原来的 current context，再把主上下文设为当前。随后创建 stream、分配显存、排入清零操作，等待完成后复制回 NumPy 数组检查。结束时释放自己创建的资源并恢复调用者原来的 context。这个例子展示的是主机 API 控制过程。

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

`checked` 取出 binding 返回元组中的错误码，检查成功后再返回其他结果。Context、stream 等句柄只作为 CUDA API 的资源标识使用。取得主上下文与 release 配对，`cuCtxSetCurrent` 则负责选中它；程序最后恢复此前保存的 context。

`cuMemsetD32Async` 将清零排到 stream 中，返回时设备可能还未写完。因此程序先同步，再用同步复制 `cuMemcpyDtoH` 读回普通 NumPy 数组。哨兵值从 `0xA5A5A5A5` 变为 0 并通过断言，才确认这次清零及读回成功。若改为异步读回，还需按 API 要求使用页锁定主机缓冲区，并保留它直到传输完成。

cuTile 的 `@ct.kernel` 定义设备计算，主机把数组和 stream 交给 `ct.launch` 启动。Numba-CUDA 的 `@cuda.jit` 则按线程写 kernel，用 `i=cuda.grid(1)` 计算当前线程的元素下标。

下面 Numba 程序处理 257 个 FP32 元素，用 `if i<n` 保护数组访问。最后一个 block 中，部分线程的索引已经越界；这些线程必须在加载之前跳过读写。

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

块数用整数式 `(n+threads-1)//threads` 向上取整。n=257、threads=128 时，启动 3 个 block、共 384 个线程；只有索引 0..256 有效，257..383 的线程跳过数组访问。

输出末尾额外保留一个 FP32 哨兵，检查 out[n] 是否被尾块写坏。主机和设备数组都明确使用 np.float32，使参数类型和复制字节数一致。

使用这些库时，要处理三项设置：Numba kernel 用 `i<n` 保护边界；CuPy 分配 FP32 数组时写 `cp.zeros(shape,dtype=np.float32)`，其默认浮点类型通常是 FP64；等待单个 stream 用 `stream.synchronize()`。CuPy 设备级同步由 `cp.cuda.Device` 提供，Numba 使用 `cuda.synchronize()`。

Numba kernel 可以使用 CuPy 的当前 stream：先取得 `cp_stream=cp.cuda.get_current_stream()`，再用 `cuda.external_stream(cp_stream.ptr)` 包装，启动时传入这个 stream。这个包装对象只是借用 CuPy 的 stream，不负责销毁它；两边要使用同一设备和 context，CuPy stream 的所有者也要一直存活。

跨库传递数组时，CUDA Array Interface（CUDA 数组接口，CAI）提供地址、形状、数据类型和 stream 等信息。例如 CuPy 在 stream A 中写入数组后，Numba 可在同一 stream 读取，或先等待对应 event；读取方按 CAI 提供的执行顺序获取数据。

按参数手算一次：设备地址 P 指向 4 字节分配，`cuMemsetD32Async(P,0,1,S)` 将一个 32 位值写成零，操作排入 stream S。同步后读回并断言零值，才检查到实际写入。释放 P 前要等设备完成使用。

运行上述两份设备程序需要相应 Python 库、兼容驱动和 GPU。主机侧语法及坐标计算可以单独检查，设备访问和数值输出留到对应环境执行。


### 10.2 C++ 执行位置、显存管理与设备链接

NVCC 编译 .cu 文件时，分别处理主机代码和设备代码。函数前的限定符说明它在哪一侧使用：

| 限定符 | 执行位置与调用方式 |
|---|---|
| `__host__` | 在 CPU 上执行，由主机代码调用 |
| `__device__` | 在 GPU 上执行，由设备代码调用 |
| `__global__` | 定义 GPU kernel，由启动语法或相应启动 API 提交 |
| `__host__ __device__` | 分别生成主机和设备可调用的版本 |

Kernel 调用辅助函数时，该函数必须有设备端版本。双限定符函数要让函数体分别通过两侧编译；必要时用 `#if defined(__CUDA_ARCH__)` 选择设备编译分支。这个宏表示当前设备编译目标，运行时查询 GPU 则使用设备属性 API。

模板按参数类型生成相应函数实例，例如同一份乘二函数分别用于 int 和 float；设备调用的实例也须通过设备编译。lambda 是可保存外部值的小函数对象；扩展 lambda 用限定符标明执行位置，并需 `--extended-lambda` 编译选项。捕获主机指针时，lambda 保存的仍是主机地址；设备代码要访问设备数组，应传入相应设备地址。

显存管理对象 `DeviceBuffer` 负责分配和释放显存；传给 kernel 的 `DeviceSpan` 只携带设备地址和长度，不会替调用方释放显存。异步操作完成前，管理对象和 stream 都必须继续存活。对象创建时执行构造函数、结束生命周期时执行析构函数，把资源的取得和释放放进这两个过程，称为 RAII（Resource Acquisition Is Initialization，对象生命周期管理资源）。

一个 `.cu` 连同它包含的头文件一起形成一个编译输入，称为 translation unit（编译单元）。跨 `.cu` 文件调用设备函数时，要启用 RDC（可重定位设备代码）：先分别编译各文件中的设备代码，再由设备链接器连接这些定义。下面程序把这些调用关系和两种共享内存分配放在一起。

<details>
<summary>供另一个编译单元使用的设备函数声明</summary>

<!-- source-check: examples/cpp_device_ops.cuh -->
```cpp
#pragma once

// Device functions have external linkage by default; spell it out here.
extern __device__ int scale_in_other_tu(int value);
```

</details>

<details>
<summary>设备函数定义所在的编译单元</summary>

<!-- source-check: examples/cpp_device_ops.cu -->
```cpp
#include "cpp_device_ops.cuh"

extern __device__ int scale_in_other_tu(int value) {
    return value * 3;
}
```

</details>

<details>
<summary>主机与设备调用、共享内存和显存管理的完整程序</summary>

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

`twice<T>` 在调用处生成对应类型的函数实例，主机和设备 lambda 分别调用它。Kernel 还调用另一个 `.cu` 文件中的 `scale_in_other_tu`；声明、定义和设备链接输入必须一致。

<details>
<summary>C++ 补充：链接属性、设备对象和间接调用限制</summary>

设备函数默认具有 external linkage（外部链接属性，可被其他编译单元引用），示例在声明中明确写 `extern`。CUDA 13 调整了部分 kernel 和设备变量的默认链接属性；跨文件构建时，应分别核对 `__global__` 函数与 `__device__`、`__constant__`、`__managed__` 变量的链接规则。

Runtime 把 kernel 参数的字节复制到设备参数区；这个过程不会替程序在 GPU 上调用对象的自定义复制构造函数。若把负责释放显存的对象传进去，主机上的临时对象可能在 kernel 完成前析构，从而提前释放 GPU 仍在使用的内存。

所以示例把 DeviceBuffer 留在主机，只传入简单的 DeviceSpan。程序先等待设备完成、复制结果，再让管理对象析构。使用 stream/event 安排异步执行时，也必须让释放发生在最后一次访问结束之后。

带 `__device__`、`__shared__`、`__constant__`、`__managed__` 或 `__tile__` 存储空间限定符的类类型变量，不支持非空构造函数和析构函数。因此，需要在析构时释放资源的管理对象留在主机侧；这个限制针对这些存储空间中的类变量，不扩展到所有 tile 局部类。

固定版本声明 staging[128]，并用线程编号访问它；示例按 128 个线程启动。改变线程数时，要同时检查数组容量、初始化范围和读取位置。动态版本使用 `extern __shared__`，启动的第三个参数指定每块字节数，这里传 `threads*sizeof(int)`。

变量保存在哪里、由谁使用，可以逐项看声明：

| 声明 | 保存与使用方式 |
|---|---|
| 普通局部 index | 每线程自己的值，通常使用寄存器，溢出时使用 local memory |
| 固定 `__shared__` 数组 | 每个 block 一份，大小在编译时确定 |
| `extern __shared__` 数组 | 每个 block 一份，大小由启动参数指定 |
| 模块级 `__device__`、`__constant__` | 设备代码中的全局数据，多个线程可访问 |
| 设备函数内 static | 持久的共享状态，按规则静态初始化；并发修改需同步或原子操作 |
| `__managed__` | 可供主机和设备访问的统一内存对象 |

设备端 new/delete 使用设备堆，动态共享内存则来自 block 启动配置。cuTile 设备代码禁止普通的非 placement new/delete。

函数指针必须指向当前执行侧可调用的函数。主机端和设备端分别使用各自的函数地址；kernel 的主机侧入口与设备侧入口也属于不同地址空间。SIMT 设备代码可按支持规则间接调用；例如 `nvstd::function` 可分别包装主机函数和设备函数，对象各自在对应执行侧使用。

虚函数调用还依赖对象的虚函数表。设备端对象及 override 必须使用一致的执行限定符；把主机上构造的多态对象按值传给 kernel 使用，会产生未定义行为。cuTile 则不支持函数指针调用和虚调用。只有少量可选操作时，模板、直接调用或 switch 更容易检查。

CUDA 设备代码支持的 C++ 子集不包括异常处理和运行时类型识别（RTTI）；`throw`、`try/catch`、`typeid` 和 `dynamic_cast` 用于主机代码时则按主机编译器规则处理。

std::vector 等普通容器通常管理主机数据。设备函数需要使用明确支持设备执行的 API，例如 CUDA C++ 标准库 cuda::std 中的相应实现。模板和 constexpr 的函数体同样要满足调用侧的编译要求。

</details>

Compute Capability（计算能力）是 GPU 支持的 CUDA 硬件功能版本，例如 8.9 对应编译目标 sm_89。将命令中的占位符 `sm_XX` 换成实际目标，再从仓库根目录编译并运行：

```bash
nvcc -std=c++17 --extended-lambda -arch=sm_XX -dc -I roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/cpp_device_ops.cu -o cpp_device_ops.o
nvcc -std=c++17 --extended-lambda -arch=sm_XX -dc -I roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/cpp_language_support.cu -o cpp_language_support.o
nvcc -std=c++17 -arch=sm_XX -dlink cpp_device_ops.o cpp_language_support.o -o cpp_device_link.o
nvcc -arch=sm_XX cpp_language_support.o cpp_device_ops.o cpp_device_link.o -o cpp_language_support
./cpp_language_support
./cpp_language_support --static
```

命令按三步进行：-dc 分别编译各个 .cu 文件并保留可链接的设备代码，-dlink 连接其中的设备函数和变量，最后完成主机链接。也可用 `-rdc=true` 让 NVCC 安排设备链接。若报告设备符号缺失，先核对声明、定义和参与链接的文件；主机库搜索参数 -L/-l 解决的是另一侧的链接问题。

## 11. 错误定位与 Driver 接口兼容性

当错误出现在 Driver 调用、启动或同步边界时，可从两方面继续定位：返回码标明错误类别，Error Log 记录 API 入口、参数和诊断文本；版本查询与 Driver entry point（运行时取得函数地址）用于核对头文件、Runtime、驱动和函数接口的匹配情况。

### 11.1 Error Log：从错误类别追到具体调用

第 7.1 节按编译、链接、装载、启动和执行阶段分类错误；本节从已取得的 Driver 返回码继续查具体调用。例如 `CUDA_ERROR_INVALID_VALUE` 表示参数无效，Error Log Management（错误日志管理）还可给出对应入口和诊断文本。本节探针使用 Toolkit 13.3 接口。

启动进程前把 `CUDA_LOG_FILE` 设为 stderr、stdout 或文件路径，可记录时间、线程、来源、级别、API 入口和消息。程序检查 Driver 返回码、kernel 启动状态和同步处返回的执行错误；日志提供对应调用的补充诊断。

`cuLogsDumpToFile` 将缓冲日志写入文件，`cuLogsDumpToMemory` 将日志写入主机缓冲区。只读取新日志时，先用 `cuLogsCurrent(&iterator,0)` 保存当前位置，再从该 `CUlogIterator` 导出；导出后 iterator 推进到下一条消息。传入空 iterator 时读取当前缓存，最多保留 100 条记录。

提供 25,600 字节主机缓冲区可容纳最多 100 条保留记录。缓冲区较小时，导出结果会带截断提示并舍弃较旧条目。`cuLogsRegisterCallback` 可注册回调函数，库产生日志时调用该函数；用返回句柄调用 `cuLogsUnregisterCallback` 注销。回调签名为 `void callback(void* userData, CUlogLevel level, char* message, size_t length)`。

Driver 13.3 的部分接口不提供 Error Log 诊断；文件路径也可能在首次写入时才检查。定位时结合返回码和实际日志内容。

下面的探针向 `cuLogsDumpToMemory` 传入空缓冲区，确认它返回参数无效；随后导出日志查看调用入口和诊断文本。`cuGetErrorName` 返回错误名称。

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

Linux 服务器从仓库根目录执行下面命令。PowerShell 设置环境变量的写法是 `$env:CUDA_LOG_FILE='stderr'`，再运行 error_log_probe.exe。程序还会主动导出内存缓冲，所以没有环境变量时也能检查导出内容。没有日志的调用继续按返回码排查。

```bash
nvcc -std=c++17 roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/error_log_probe.cu -lcuda -o error_log_probe
CUDA_LOG_FILE=stderr ./error_log_probe
```


### 11.2 Driver Entry Points、版本查询与 ABI

版本查询和按 Driver entry point 取得函数指针，是兼容性问题的另一层。函数调用的参数、返回值等二进制传递约定称为 ABI（Application Binary Interface，应用二进制接口）；headers 是编译时包含的头文件，PFN 类型则规定指定版本的参数类型和返回类型。下面先区分各版本数字，再按指定 ABI 取得地址。

三个版本查询分别报告以下环境版本：

| 查询 | 回答的问题 |
|---|---|
| CUDA_VERSION | 编译时用了哪个 Toolkit 的头文件？ |
| cudaRuntimeGetVersion | 进程使用哪个 Runtime API 版本？ |
| cuDriverGetVersion | 已安装驱动支持哪个 Driver API 版本？ |

驱动安装包版本与 GPU 计算能力由各自查询报告；记录时保留查询名称，便于辨认版本所属组件。

Driver Entry Point Access 在运行时取得 Driver 函数地址，程序再通过函数指针调用。指针类型应匹配请求的 ABI 版本；它用于 CPU 调用 Driver 函数。`CUfunction` 则是标识 GPU kernel 入口的句柄。

Toolkit 的 `cudaTypedefs.h` 为不同 ABI 版本提供 `PFN_...` 函数指针类型。下面查询 CUDA 11.2 版 `cuMemAllocAsync`，使用 `PFN_cuMemAllocAsync_v11020`，并向 `cuGetProcAddress` 请求 ABI 版本 11020。这个值表示所需接口版本。

查找后检查三项：`CUresult` 返回码、`CUdriverProcAddressQueryResult` 状态和函数指针。它们分别报告查询结果、符号查找状态和可调用地址是否取得。

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

从仓库根目录运行下面命令，查询请求的接口是否可用。若随后调用异步分配，再选择 context 和 stream、查询设备内存池支持，并按 stream 顺序在最后一次使用后释放内存。

本例使用 Toolkit 13.3 的 cudaTypedefs.h。较旧 Toolkit 若缺少相应类型定义，应使用该版本支持的声明与接口，避免用不匹配的函数指针强行调用。

```bash
nvcc -std=c++17 roadmap/curriculum/gpu/02-cuda-execution-and-scheduling/examples/driver_entry_point_query.cu -lcuda -o driver_entry_point_query
./driver_entry_point_query
```


## 参考阅读

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 与 [NVCC compiler guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html)
- [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)、[Error Log Management](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/error-log-management.html)、[Driver Entry Point Access](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/driver-entry-point-access.html) 与 [`cuda.bindings` 13.3.2 Driver reference](https://nvidia.github.io/cuda-python/cuda-bindings/13.3.2/module/driver.html)
- [cuTile Python TiledView](https://docs.nvidia.com/cuda/cutile-python/data/tiled_view.html) 与 [memory model](https://docs.nvidia.com/cuda/cutile-python/memory_model.html)
- [Numba-CUDA kernel API](https://nvidia.github.io/numba-cuda/reference/kernel.html) 与 [host/stream API](https://nvidia.github.io/numba-cuda/reference/host.html)
- [CuPy Stream API](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html) 与 [`cupy.zeros`](https://docs.cupy.dev/en/v13.2.0/reference/generated/cupy.zeros.html)
