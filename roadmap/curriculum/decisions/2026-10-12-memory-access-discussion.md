# 访存与布局讨论记录（2026-10-12）

记录日期：2026-10-12（Asia/Shanghai）

本记录整理本轮关于张量连续布局、别名承诺、低精度格式、向量化、warp 请求合并、lane 地址映射、Ascend 与 CUDA 访存层次、shared-memory bank 的问答。下列问题保留为用户提出的疑问，不表示用户已完成学习验收；技术说明为 Agent 整理，并按官方接口文档、标准文本或地址计算核对。

## 已确认的技术结论

### 1. `contiguous()` 与 transpose view

**用户原话（疑问）：**“contiguous() 的作用是将只有shape和stride变化 但是读取顺序不连续的数据块 变为读取连续的数据块吗”

**Agent 整理：**transpose 通常创建共享原 storage 的 view，只改 shape/stride，不搬动元素。例：`a` 是连续的 2×3 张量，值按行存为 `0,1,2,3,4,5`。`b=a.T` 的逻辑 shape 是 3×2、stride 为 `(1,3)`，按逻辑行读取是 `[[0,3],[1,4],[2,5]]`；底层数据仍未重排。

默认 `b.contiguous()` 返回逻辑值相同、row-major 连续的新 storage，其 shape 为 3×2、stride 为 `(2,1)`，物理元素顺序是 `0,3,1,4,2,5`。若张量已经满足请求的 memory format，PyTorch 会直接返回原张量；否则复制生成连续布局。该操作不是原地转置，也不保证之后任意 kernel 的线程映射都能合并访存。

### 2. `__restrict__` 的别名承诺

**用户原话（疑问）：**“__restrict__ 什么情况会用到 不能取别名有什么好处”

**Agent 整理：**`__restrict__` 向编译器承诺：在该函数执行期间，受限指针所访问并可能修改的对象不会被另一个无关指针以冲突方式访问。它约束的是内存访问关系，不是指针变量名；从受限指针派生地址仍可使用。只读输入可以被多个读取者共享；`const` 只禁止通过该指针写入，不代表没有别名能修改对象。

下面是 Agent 整理的设备函数片段，用于说明单线程内的读写依赖：

```cpp
__device__ float calc(const float* a, float* b, int i) {
    float x = a[i];
    b[i] = x + 1.0f;
    float y = a[i];
    return x + y;
}
```

若最初 `a[i]=3`，且 `a[i]`、`b[i]` 是不同位置，返回值为 `3+3=6`；若两者指向同一位置，写入后 `a[i]` 变成 4，返回值为 `3+4=7`。这个未加限定的函数必须保留两种调用的语义。

把参数改为 `const float* __restrict__ a` 和 `float* __restrict__ b` 后，合法调用保证这里的输出写入不影响输入读取，编译器便有机会复用 `x`，省去第二次 load。若实际调用违反承诺，让这两个访问位置重叠，就属于未定义行为；运行时通常不会自动报错。承诺也可能增加同时存活的值与寄存器压力，不能据此直接断言更快。

### 3. FP16、BF16 与指数/尾数

**用户原话（疑问）：**“bf16 是尾数位多还是指数位 是尾数位吧”

**Agent 整理：**BF16 的指数位与 FP32 相同，尾数字段更短。下表的尾数是 fraction 字段位数，不含正规数隐含的最高有效位。

| 格式 | 符号位 | 指数位 | 尾数字段 | `[1,2)` 中相邻数间距 |
|---|---:|---:|---:|---:|
| FP32 | 1 | 8 | 23 | `2^-23` |
| FP16 | 1 | 5 | 10 | `2^-10` |
| BF16 | 1 | 8 | 7 | `2^-7` |

BF16 的指数范围接近 FP32，适合保留较大动态范围；FP16 尾数多 3 位，在 `[1,2)` 内的间距是 BF16 的八分之一，精度更细。两种 16 位格式的范围和精度取舍不同，不能把 BF16 描述成尾数位更多。

### 4. `float4` 与向量化宽度

**用户原话（疑问）：**“float4 向量化常用来减少指令条数 这个是什么意思”“最多只能float4吗”

**Agent 整理：**`float4` 是由四个 FP32 组成的 16B 向量类型。满足对齐、连续访问和边界条件时，每线程的一次 16B load/store 有机会替代四次 4B 标量访问，减少访存指令数；它不会减少需要搬运的总字节数，也不会自动把四路算术融合成一条运算。编译器也可能把多个标量访问合并，最终应检查生成指令。

CUDA 的内置浮点向量类型到 `float4`，没有内置 `float8` 类型；一个线程可以处理多个 `float4`，也可使用更宽的低层指令。常见 Ampere、Ada、Hopper global-to-register 访存路径的向量宽度上限为 128 bit（16B）。PTX ISA 8.8 为 `sm_100+` 增加 `ld.global.v8.f32` 这类 256 bit（32B）形式；它需要相应 Toolkit、目标架构和对齐条件，不能推广为所有 GPU、所有地址空间或所有指令都支持 32B，也不意味着硬件只发出一个 32B DRAM 事务。上述架构边界是文档结论，本轮未用 NVCC 或设备实测。

### 5. 请求合并与向量化的区别

**用户原话（疑问）：**“这里不是说尽可能合并吗”

**Agent 整理：**coalescing（请求合并）统计一条 warp 访存指令中所有 active lane 的地址覆盖；向量化则改变单线程一条指令读写的数据宽度。两者相关，但不是同一层次。

32 个 lane 各执行一次标量 FP32 load，首地址 32B 对齐且 lane 地址连续时，有效数据 128B，覆盖 4 个 32B sector。若每个 lane 各读一个 `float4`，warp 总有效数据为 512B，覆盖 16 个 32B sector；这不表示产生一个 512B 事务。sector 覆盖数用于描述请求地址范围，不等于实际 DRAM 流量；缓存命中、对齐和后端事务处理都会影响真实传输。

### 6. lane 地址公式与例子修正

**用户原话（疑问，保留两段）：**

> 一维 contiguous copy 中，lane l 访问 base + (pid×BLOCK+l)×4。当一个 warp 内的 l=0..31 连续时

> 这段是不是语法有问题

**Agent 整理：**问题在于原式混用了 Triton program、CUDA lane 和字节地址；“warp 内 `l=0..31` 连续”也没有说明 lane 与元素索引的关系。本章已有一维 `copy_f32` 的 CUDA 索引为 `i=blockIdx.x*blockDim.x+threadIdx.x`，`src[i]` 是按 `float` 元素索引，指针运算会自动乘 `sizeof(float)`。

设 `i0` 为该 warp lane 0 负责的首个全局元素索引，`base_addr` 为 `src[0]` 的数值字节地址。一次读指令中 lane `l` 访问元素 `i0+l`，其字节地址为：

```text
addr(l) = base_addr + 4 * (i0 + l),  l = 0..31
```

当 `blockDim.x=128` 时，每个 block 有四个 warp；warp 0 的 `i0=blockIdx.x*128`，warp 1 的 `i0=blockIdx.x*128+32`。每个 warp 的 lane 仍为 0..31，首元素索引还要包含该 warp 在 block 中的偏移。一个完整 warp 的 128B 访问若按 32B 对齐，覆盖 4 个 sector；若 FP32 首地址只按 4B 对齐且相对 32B 边界偏移 4B，会覆盖 5 个 sector。尾部 mask/条件会让无效 lane 不发出该次请求，只统计有效 lane 覆盖的 sector。同一组地址换由不同 lane 访问不会增加 sector 覆盖；硬件不要求 lane 按顺序执行。

本章 Triton 示例中的 `tl.arange` 生成逻辑 tile 索引，不保证索引与硬件 lane 一一对应。实际布局取决于编译器及配置；这里的 CUDA 例子只用于手算地址，不是 Triton 的固定 lane 映射。

本轮已在 [访存与布局 §4/§4.1](../operators/01-memory-and-layout/README.md) 修正相关说明；只改文档，未修改 `copy_alignment.cu` 或其他原始 kernel。

### 7. Ascend 多核访存与 CUDA warp 合并

**用户原话（疑问）：**“cuda 没有地址冲突问题吗 多个thread读一个总线的数据发生冲突 例如 ascend不同核通常不会读一个连续行或者连续列”

**Agent 整理：**两类现象涉及不同的并行层次。CUDA coalescing 观察一个 warp 的一条访存指令如何覆盖地址区段；不同 warp、CTA 或 SM 仍会竞争缓存、显存分区、通道和互联资源。Ascend C 的 AI Core/AIV 数据搬运约束也不能只按逻辑“行/列”判断。

Ascend 官方指导在指定产品与访存场景中建议避免多个核访问同一个对齐的 512B 地址区间，并说明此类请求可能串行；示例使用了特定的 cache/L2 hint 设置。该结论限定于文档所述产品、指令与内存路径，不能推导为所有 Ascend 型号或所有缓存条件下的通则，也不是本轮用户实测。CUDA 中多个线程只读同一地址通常不构成数据竞争，但仍可能受请求合并与缓存系统影响；并发写或读写重叠则需要原子操作、同步或明确的数据所有权。

### 8. Shared-memory bank conflict

**用户原话（疑问）：**“这里的blank flict是什么呢 行冲突 列冲突 还是都会”

**Agent 整理：**这里指 bank conflict（bank 冲突）。在典型的 32-bank、每 bank 按 4B word 服务的 FP32 标量访问模型中：

```text
bank(byte_address) = floor(byte_address / 4) mod 32
```

对 `float tile[32][32]`，假设行主序，`tile[0][lane]` 将 32 个 lane 分到 32 个 bank；`tile[lane][0]` 则让它们落到同一 bank 的不同 word，形成典型 32-way conflict。若所有 lane 读取同一 bank 的同一 word，硬件可广播，不是这种冲突。`float tile[32][33]` 把行跨度增加 1 个 word 后，列访问可依次落到不同 bank。

假设数组首元素位于 bank 0，`tile[r][c]` 在这两种布局中的 bank 分别为 `(32*r+c)%32` 和 `(33*r+c)%32`。取 `c=0`，前者所有行都落入 bank 0，后者行 0～31 分别落入 bank 0～31。同一长行若每隔 32 个 FP32 读取一个元素，也会访问同一个 bank 的不同 word。

冲突不由“行访问”或“列访问”这个名称决定，而由一条 shared-memory 指令中 lane 到地址的映射决定。FP16、64 位值、向量访问或不同 bank width 的请求可能覆盖多个 bank，必须按目标架构与实际指令重新分析，不能直接套用上述 FP32 scalar 模型。

### 9. 不同 warp 的缓存复用与三种地址分布

**用户问题（概括）：**不同 warp 是否共享缓存；将 §4.2 中关于相同地址、连续访问和同行/同列的三段说明改成直接易懂的说法。

**用户原话（疑问）：**“一个warp内有访存的缓存 不同warp有吗”

**Agent 整理：**GPU 没有给每个 warp 单独分配一份 L1/L2 缓存。L1 放在各个 SM 上，L2 由整个 GPU 的 SM 共用。

| 两个 warp 的位置 | 可以利用的缓存 |
|---|---|
| 在同一个 SM 上 | 共用该 SM 的 L1，也可以利用全 GPU 共用的 L2 |
| 在不同 SM 上 | 各用自己 SM 的 L1，可以利用全 GPU 共用的 L2 |

同一个 SM 上的 warp 即使来自不同 block，也使用该 SM 的 L1。相比之下，普通 shared memory 按 block 分配，不同 block 不会因为在同一个 SM 上就自动共享彼此的 shared 数据。warp 复用的是缓存中的数据，不是另一个 warp 的寄存器。

例如，warp 0 在 SM 0 上先读取 `A[0]`，warp 1 随后在 SM 0 上也读取 `A[0]`。如果数据还在 SM 0 的 L1，且这次读取使用 L1，warp 1 可以直接从 L1 取到。另一个 SM 上的 warp 不能直接使用 SM 0 的 L1；如果它自己的 L1 未命中，而 L2 仍有这份数据，就可以从 L2 取，不必再访问显存。L1/L2 都未命中时，才需要继续从显存取回。具体使用哪些缓存，由访存指令和缓存策略决定。

每个 warp 仍会执行自己的读取指令。后来的 warp 命中缓存，只是省去了再次取显存数据的过程，不会把两个 warp 合成一个更大的 warp。数据被其他数据替换后，后续读取还要重新取回。

同一个 warp 的 32 个线程在一条指令中读同一个地址，可以把这些请求一起处理；不同 warp 先后读取这个地址，则可以复用缓存中的数据。前者是请求合并，后者是缓存命中。这里讨论已准备好、存放在 GPU 显存中的只读输入；有读写依赖时仍需正确同步。

正文用连续 `float` 数组 `A`、每行 `N=64`、`lane=0..31` 解释三种访问：所有线程读 `A[0]`；相邻线程读 `A[lane]`；线程分别读 `A[lane*N]`。第三种访问的下标为 `0、64、128……`，相邻元素地址相差 256B，每个地址落在不同的 32B 段中。收尾说明 block 内 shared-memory 复用要考虑搬运与同步成本。

访存与布局正文 §4.2 已说明缓存归属与三个读取例子，本节保留更完整的讨论。缓存归属依据 [CUDA Programming Guide：Caches](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#caches)。本次只补记讨论，没有新增 GPU 测量，也没有推进学习状态。

## 提案

可复用现有 `copy_alignment.cu` 对比 aligned/scalar 与 `float4` 路径，并用偏移 0/1、尾部长度 1/3/4/5 等输入检查对齐和 mask；也可复用转置示例比较 `tile[32][32]` 与 `tile[32][33]`，观察 lane 地址映射和 bank 计数。它们是现有实验的可选分析角度，不新建学习单元、源码或测量记录。

## 待定与待测

- 不同 Toolkit/目标架构下 scalar 与 `float4` 最终生成的访存指令、每线程寄存器数及指令数。
- 目标 GPU 上两种 copy 布局的实际耗时、L1/L2 命中和内存 sector/DRAM 流量；sector 覆盖不能替代流量计数器。
- 实际 shared-memory bank 计数、转置 `tile[32][33]` 的性能变化，以及不同元素宽度/指令形式的 bank 映射。
- 不同 warp/CTA/SM 访问模式下的缓存与显存资源争用；现有对照不构成任何 GPU 性能结果。

本轮没有 NVCC/GPU 执行，也未做 sanitizer、profiler 或 benchmark；不新增正确性或性能状态，不推进 `PATH.md`/`NOW.md`。既有学习和实验记录保持原样。

## 资料与代码入口

- [PyTorch `Tensor.contiguous`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html) · [PyTorch Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)
- [CUDA C++ `__restrict__` 指针](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#restrict-pointers) · [C11 N1570 §6.7.3.1](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf)
- [CUDA Best Practices：Coalesced Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory) · [Shared Memory Banks](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks)
- [PTX ISA：Load Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)
- [Ascend C：避免同地址访问](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/programug/Ascendcopdevg/docs/zh/guide/%E7%AE%97%E5%AD%90%E5%AE%9E%E8%B7%B5%E5%8F%82%E8%80%83/SIMD%E7%AE%97%E5%AD%90%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96/%E5%86%85%E5%AD%98%E8%AE%BF%E9%97%AE/%E9%81%BF%E5%85%8D%E5%90%8C%E5%9C%B0%E5%9D%80%E8%AE%BF%E9%97%AE.md)
- 本地 CUDA 访存教材：[CUDA Programming Guide 13.3](../../../downloads/cuda-programming-guide.pdf)；实验源码：[copy_alignment.cu](../operators/01-memory-and-layout/examples/copy_alignment.cu)
