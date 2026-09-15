# 第三章 寄存器、Shared Memory与存储层次

> 状态：`WIP`（Work in Progress，进行中）。正文已建立，尚未完成用户阅读和真实GPU验收。

本章专门处理GPU算子最常见也最容易被一句话带过的资源：寄存器、local memory、shared memory、L1/L2和设备显存。重点是物理归属、可见范围、编译分配、数据复用和性能代价。

术语：Register File（寄存器文件）是SM上的物理寄存器资源；Local Memory（局部内存）是线程私有地址空间；Shared Memory（共享内存）是CTA内可编程片上存储；Level-1/Level-2 Cache（L1/L2，一/二级缓存）分别位于SM附近和全GPU共享路径。

## 1. 存储层次不是一条简单的速度阶梯

常见示意图会写“register最快、global最慢”，但高性能代码还必须考虑容量、作用域、是否由程序显式管理以及实际数据路径。

| 层次 | 物理/逻辑位置 | 主要作用域 | 谁管理 | 典型用途 |
|---|---|---|---|---|
| Register | SM片上register file | 单thread | 编译器分配 | accumulator、地址、局部值 |
| Local memory | 地址空间属于thread，实际可落到显存/cache | 单thread | 编译器 | register spill、大型局部数组 |
| Shared memory | SM统一数据缓存中的可配置片上区域 | 单CTA；新架构可扩展cluster协作 | 程序显式 | tile、跨warppartial、重排 |
| L1/data cache | 每SM附近 | 当前SM访问 | 硬件 | global/local访问缓存 |
| L2 | 所有SM共享 | 整个GPU | 硬件与部分策略API | 跨CTA复用、显存流量缓冲 |
| Global memory | HBM/GDDR等设备显存 | 整个设备 | 程序分配、硬件访问 | 大矩阵、权重、KV Cache |
| Host memory | CPU内存 | Host与映射设备 | OS/Runtime | 输入输出、控制数据 |

Local memory这个名字最容易误导。它是“线程私有地址空间”，不是SM上的一小块快速内存；发生register spill时，数据会通过cache访问设备内存。

### Coalescing到底合并什么

一个warp发出global load时，硬件观察32个lane请求的地址，把它们覆盖的地址segment转换成一个或多个memory transaction。连续、对齐的访问通常只传输少量必要segment；大stride或散乱访问需要更多transaction。

```text
连续float访问：lane 0..31 → 128 B连续范围
stride=8访问： lane 0,8,16,...元素 → 跨越更大地址范围
```

两种情况都执行32次逻辑load，但第二种可能搬运更多未使用字节。Profiler中的requested bytes与actual sectors/bytes可以揭示这种浪费。

### Shared memory解决什么，又会带来什么

Shared memory常用于把global中的二维tile以合并方式搬入，再按计算需要的顺序反复读取。例如矩阵转置：global读和写很难同时连续，可以先连续读入shared，转置索引后再连续写出。

Shared memory按bank并行服务访问。同一warp访问不同bank时吞吐较好；多个lane访问同一bank的不同地址会产生bank conflict并被拆分。二维tile常通过padding改变每行stride：

```cpp
__shared__ float tile[TILE][TILE + 1];
```

多出的1列不是存数据需要，而是避免转置访问时多线程落到同一bank。是否有效要结合元素宽度、bank组织和实际指令检查，不能把`+1`当万能公式。

### Register为什么既宝贵又危险

Register提供低延迟局部存储，GEMM accumulator通常尽量保留其中。但register file由所有驻留warp共享。每线程register增加时，每SM可同时驻留的warp可能阶梯式下降；超过编译或硬件预算还可能spill到local memory。

这就是大tile的双刃剑：提高数据复用和指令级并行，同时扩大accumulator和地址状态。优化时要同时看吞吐、register数、spill和eligible warp，不能只追求更大tile。

## Register file到底属于谁

Streaming Multiprocessor（SM，流式多处理器）内部有一组物理register file。CUDA给程序员的语义却是“每个thread拥有自己的register值”。两句话并不矛盾：物理容量由同一SM上的全部resident warp共享，逻辑值仍属于单个thread，其他thread不能通过普通指针读取。

```text
SM register file（有限的物理容量）
├── CTA 0 / warp 0 / thread 0 的局部值
├── CTA 0 / warp 0 / thread 1 的局部值
├── CTA 0 / warp 1 的局部值
└── CTA 1 的全部resident thread局部值
```

源码没有`register malloc`。编译器根据中间表示、变量活跃区间、循环展开、内联和目标架构分配register。下面这些对象通常消耗register：

- 指针和地址偏移；
- 循环计数、predicate和mask；
- 每线程局部标量或向量；
- 从global/shared载入后等待使用的数据；
- GEMM或Attention的accumulator；
- 异步流水的descriptor、barrier状态和多stage地址。

变量个数不等于register数。两个不同时存活的变量可以复用同一物理register；一个宽向量、展开数组或大accumulator可能展开成很多register。

## 活跃区间为什么影响register pressure

```cpp
float a = load_a();
float b = load_b();
float c = expensive_compute(a);
float d = expensive_compute(b);
store(c + d);
```

如果`a`和`b`在很长一段代码中都必须保留，它们的live range（活跃区间）重叠。编译器需要同时保存更多值。调整计算顺序，让一个值使用完再载入另一个值，可能缩短live range；但也可能减少Instruction-Level Parallelism（ILP，指令级并行），所以仍要实测。

循环展开也有同样的权衡。展开可以减少分支并暴露独立指令，却会让多个迭代的数据同时存活，增加register需求和代码体积。

## GEMM accumulator为什么特别吃register

设一个thread负责$M_t\times N_t$个输出元素，仅accumulator就至少需要$M_tN_t$个对应精度的逻辑值。若使用32位浮点累加，粗略下界为：

$$
R_{acc}\ge M_t\times N_t
$$

这还没有包括A/B fragment、指针、mask和流水状态。扩大输出tile虽然增加复用，也会让accumulator面积按两个方向相乘增长。这解释了为什么MatMul把输出列tile翻倍后，register压力可能比shared增加更先成为限制。

在Triton中：

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
```

这是program级tensor抽象，不表示单个thread拥有完整`BLOCK_M × BLOCK_N`数组；编译器会根据layout把元素分配给warps和lanes。但BLOCK尺寸、`num_warps`和layout共同决定每线程最终持有多少accumulator fragment，必须检查编译结果。

## Register分配为什么呈阶梯变化

硬件通常按warp或更大粒度分配register，而不是精确到“每线程多1个就只多32个”。因此每线程register从某个数增加1，可能跨过分配粒度，导致每CTA资源突然增加，resident CTA数量下降。

理论估算：

$$
R_{CTA}\approx R_{thread}\times Threads_{CTA}
$$

真实值还要向架构规定的分配粒度取整。不能只用除法得出最终occupancy，要结合目标Compute Capability、Occupancy API或Nsight Compute。

CUDA编译可查看：

```bash
nvcc kernel.cu -arch=sm_XX -Xptxas=-v -lineinfo -o kernel
```

输出中的register、stack frame、spill stores和spill loads是第一层证据。Triton则查看编译元数据、生成的PTX/SASS以及Profiler local-memory流量。

## Spill到local memory发生了什么

当编译器无法把活跃值全部保留在register，或大型局部数组需要可寻址空间时，会使用local memory（线程私有地址空间）。“local”描述可见性，不表示物理上靠近thread。

```text
thread局部值
→ local address
→ L1/L2 cache
→ cache miss时访问设备DRAM
```

Spill的代价不是固定的：若数据命中cache，代价低于DRAM miss；若每个thread访问不同local地址，还会增加大量memory instruction和cache压力。Profiler中应关注local load/store、L1/L2流量和相关stall，而不是只看编译器报告出现“spill”二字。

限制register数量也不是通用优化：

```bash
nvcc kernel.cu --maxrregcount=64 ...
```

它可能提高理论occupancy，也可能强制更多spill并使性能下降。正确实验要比较register、resident warp、local traffic和总耗时四项。

## Shared Memory的物理资源与程序语义

Shared Memory（共享内存）是SM上的可编程片上存储，同一CTA内thread可通过地址读写并协作。不同CTA即使同时驻留在同一SM，也不能把普通shared数组当作彼此共享的数据结构。

静态与动态shared：

```cpp
__shared__ float fixed_tile[32][33];
extern __shared__ unsigned char dynamic_smem[];

kernel<<<grid, block, dynamic_bytes, stream>>>(...);
```

两者都会进入每CTA shared资源账本。某些架构允许配置统一L1/shared物理资源的划分，但具体容量、默认每block上限和opt-in上限必须按目标设备查询。

## Bank conflict要按warp请求理解

Shared memory分成多个bank，目标是让warp内多个lane同时访问不同bank。设bank数量为$B$，bank宽度对应$w$字节，地址到bank的简化映射可以写成：

$$
bank(address)=\left\lfloor\frac{address}{w}\right\rfloor\bmod B
$$

同一warp访问同一bank的不同地址时，请求需要拆分；多个lane读取完全相同地址则可能走broadcast。实际规则随架构和访问宽度变化，所以公式用于理解，不代替官方文档与Profiler。

矩阵转置中，若二维shared tile每行正好跨过完整bank周期，按列访问会让多个lane落到同一bank。把行stride改成`TILE + 1`会打散映射：

```cpp
__shared__ float tile[TILE][TILE + 1];
```

这会增加少量shared容量，换取更少bank conflict。实验要同时记录shared transaction、bank conflict指标与耗时。

## Global、L1与L2之间的数据路径

Global Memory（全局内存）是CUDA地址空间，通常由设备DRAM承载。SM发出的global请求先经过合并与cache路径，再到L2和memory partition。L2由全GPU共享，是不同SM请求汇合的重要层级。

```text
warp的32个地址
→ 合并成若干memory sector/transaction
→ L1（取决于指令和cache策略）
→ L2
→ DRAM partition
```

Coalescing关心同一warp地址覆盖了多少transaction；cache locality关心之后是否再次使用相同cache line。两者不能混为一谈：连续访问可能coalesced但没有复用，散乱访问也可能因缓存命中减少DRAM流量，却仍产生较多memory instruction。

## 一个贯穿实验：Transpose

按四版实现观察存储层次：

1. 直接读取并转置写回：一侧访问连续，另一侧大stride；
2. Shared tile：global两侧都按连续方向搬运；
3. Shared tile但无padding：观察bank conflict；
4. `TILE+1` padding：比较bank conflict和有效带宽。

每版记录：数学正确性、requested/actual DRAM bytes、L1/L2命中、shared transactions、bank conflict、register数和耗时。这样才能区分收益来自coalescing、cache还是shared访问变化。

## Triton中的对应关系

`tl.arange`和广播表达的是一个program处理的逻辑tile；`tl.load`的pointer tensor描述每个逻辑元素地址。编译器再把这些元素分给warp/lane并选择load指令。

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
ptrs = x + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
tile = tl.load(ptrs, mask=mask)
```

`ptrs`是地址tile，不是已经搬入的矩阵块。地址表达式决定相邻lane最终访问是否连续；`tile`的生命周期、重复使用和`tl.dot`布局会影响register/shared分配。必须从生成代码和Profiler确认，不能只根据Python形状猜物理存储。

## 本章实验与验收补充

- 写一个register-pressure sweep：扩大每thread accumulator或unroll，记录register、spill、occupancy和时间；
- 写一个shared bank-conflict对照；
- 写一个stride/coalescing sweep；
- 用现有MatMul解释当前accumulator为何可能达到255 registers/thread；
- 能回答“local memory为什么不local”“高register一定坏吗”“shared能否跨CTA共享”。

## 章节导航

[上一章](../02-execution-model-and-instruction-scheduling/README.md) · [返回第一篇目录](../README.md) · [下一章](../04-compute-pipelines-numerics-and-tensor-cores/README.md)

