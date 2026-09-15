# 第二章 CUDA执行模型与指令调度

> 状态：`WIP`（Work in Progress，进行中）。正文已建立，尚未完成用户阅读和真实GPU验收。

本章从一次kernel launch开始，追踪grid怎样拆成线程块、线程块怎样驻留到流式多处理器、线程怎样组成warp、调度器怎样选择可发射warp，并把CUDA层次映射到Triton program。

术语：Compute Unified Device Architecture（CUDA，统一计算设备架构）；Cooperative Thread Array（CTA，协作线程阵列，对应thread block）；Single Instruction, Multiple Threads（SIMT，单指令多线程）；SM为Streaming Multiprocessor（流式多处理器）。Kernel是GPU核函数，grid是一次launch的全部CTA，warp是32个lane组成的调度分组。

## 1. 从CPU发起kernel到SM执行，中间发生什么

GPU不是独立运行一个Python函数。Host程序先通过CUDA Runtime或Driver准备参数、内存和执行配置，再向某个stream提交kernel。GPU前端接收任务后，把grid中的CTA分发给能够容纳它们的SM。

```text
Host线程
  │  kernel<<<grid, block, shared, stream>>>(args)
  ▼
CUDA Runtime / Driver
  │  建立launch、参数和依赖
  ▼
GPU工作分发前端
  │  从grid选择尚未执行的CTA
  ▼
GPC中的SM
  │  分配warp槽、register和shared memory
  ▼
Warp Scheduler选择ready warp
  │
  ├── Load/Store pipe：地址与访存
  ├── FP/INT pipe：普通算术和逻辑
  ├── SFU：exp、rsqrt等特殊函数
  └── Tensor pipe：矩阵乘加
```

CTA一旦驻留到SM，它使用的register和shared memory通常会保持到CTA结束。SM不能因为某个warp正在等待内存，就把这个CTA的上下文换回Host内存再装另一个；隐藏延迟依赖同一SM上其他ready warp。

### Grid、CTA、warp和thread怎样对应

CUDA源码看到的是thread，硬件发射看到的主要是warp：

```text
Grid
├── CTA 0 → warp 0, warp 1, ...
├── CTA 1 → warp 0, warp 1, ...
└── CTA n → ...
```

一个CTA的线程按线性thread id连续分组，每32个线程形成一个warp。若block有100个线程，会形成4个warp，最后一个warp只有4个有效lane，其余lane从一开始就不做有效工作。

线程到数据的映射决定访存形态。以一维Vector Add为例：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) c[i] = a[i] + b[i];
```

同一warp的lane 0..31访问连续的`a[i]`、`b[i]`和`c[i]`，硬件可以把请求合并成较少的memory transaction。如果改成`i * stride`，每个lane访问相距很远的位置，请求会覆盖更多segment，传输大量未使用字节。

### CTA为什么不是“软件循环的一次迭代”

CTA是资源分配与同步边界。同一CTA内线程可以使用shared memory并调用CTA barrier；普通kernel中不同CTA没有可靠执行顺序，也不能用一个CTA写flag、另一个CTA自旋读取来替代全局同步。

这条限制直接塑造算法：长Reduction常先让每个CTA产生partial，再启动第二个kernel归约；Split-K GEMM让多个CTA计算partial C，最后还需要atomic或额外归约。

## 2. SM内部如何选择和执行warp

一个SM可以同时驻留多个CTA和多个warp。每个warp保留程序计数器、active mask和register状态。每个周期，warp scheduler从满足依赖、操作数和执行资源条件的warp中选择指令发射。

“active warp”只表示warp驻留；“eligible warp”才表示它当前具备发射条件。一个kernel理论occupancy很高，但大量warp都在等待内存或长依赖链，scheduler仍可能找不到足够eligible warp。

常见等待原因可以按因果理解：

| 等待来源 | 代码中的原因 | 可尝试的方向 |
|---|---|---|
| 长内存延迟 | 重复global load、低cache命中 | 合并访问、复用、增加可隐藏延迟的warp |
| 数据依赖 | 连续指令依赖同一个累加器 | 增加独立accumulator或ILP |
| Barrier | CTA内warp到达时间不一致 | 减少不必要同步、平衡warp工作量 |
| 执行管线忙 | 大量同类指令争用同一pipe | 与其他pipe交错或改变算法 |
| 资源不足 | register/shared限制驻留 | 调整tile、stage、unroll和block |

### SIMT与分支发散

SIMT允许每个thread拥有自己的控制流，但warp一次发射一条指令。当同一warp的线程进入不同分支时，硬件用active mask分别执行路径：

```cpp
if (threadIdx.x % 2 == 0) {
    y = expensive_a(x);
} else {
    y = expensive_b(x);
}
```

若两个分支都很长，warp通常要先后完成两条路径，执行单元在每条路径上只服务部分lane。分支本身不是一定慢：如果整个warp选择相同路径，就没有lane级分歧；短小分支也可能被predication处理。

因此优化目标不是“消灭所有if”，而是让同一warp的控制流和数据访问尽量一致，并用Profiler确认分歧是否真的成为瓶颈。

## 3. 从二维thread坐标到线性warp

线程块可以是一维、二维或三维，但warp分组依据线性thread id：

$$
tid_{linear}=threadIdx.x+blockDim.x\times(threadIdx.y+blockDim.y\times threadIdx.z)
$$

连续32个线性id组成一个warp。二维block若`blockDim.x`不是warp size的合理倍数，一个warp可能跨多行，进而改变分支和访存形态。维度写法帮助映射问题，硬件调度仍按线性warp组织。

## 4. Grid大小决定有没有足够工作

设grid含$B$个CTA、设备有$S$个SM，每SM受资源限制可同时驻留$r$个CTA。粗略看，一轮可并行处理$S\times r$个CTA；超过部分形成后续wave。

```text
B << S       ：许多SM无工作，典型small-grid问题
B ≈ S        ：每SM约一个CTA，隐藏barrier或长延迟的机会有限
B >> S       ：形成多轮wave，通常更容易填满设备
```

Grid大不等于一定快。CTA太小会增加调度与边界开销；CTA太大又可能因register/shared限制驻留。正确配置来自算子形状和资源账本。

## 5. Warp Scheduler实际在选择什么

Resident warp已经占用SM资源；active warp尚未退出；eligible warp还必须满足操作数就绪、barrier已通过且目标执行pipe可接收。Scheduler从eligible集合发射下一条指令。

因此一个Profiler现象要这样解释：

```text
active warps多但eligible少
→ 不是“线程不够”
→ 可能是memory dependency、long scoreboard、barrier或执行pipe争用
→ 再根据warp-state与memory/pipe指标区分
```

增加occupancy只在能提供更多ready warp时帮助隐藏延迟。若所有warp都在等同一个CTA barrier，增加同CTA线程没有用；若计算pipe已饱和，更多eligible warp也不会提高吞吐。

## 6. Divergence的具体执行过程

```cpp
if (x > 0) y = f(x);
else       y = g(x);
```

若一个warp内两类thread同时存在，硬件维护不同路径对应的active mask，分阶段执行$f$和$g$。分歧成本取决于两条路径指令量、参与lane比例和重汇合位置。把数据按warp分组，使同一warp选择相同分支，可能比删除`if`更有效。

边界mask是常见且通常可接受的分歧：只有最后一个CTA的部分lane越界。若整个grid大部分warp都满载，尾部损失很小；若shape本身很小，尾部warp比例就会显著上升。

## 7. Triton program怎样映射

Triton的一个program instance通常对应一个独立tile任务，`tl.program_id(axis)`选择tile坐标：

```python
pid = tl.program_id(0)
offs = pid * BLOCK + tl.arange(0, BLOCK)
mask = offs < n
x = tl.load(x_ptr + offs, mask=mask)
```

`BLOCK`是逻辑tile宽度，不等于CUDA thread block中的thread数。`num_warps`告诉编译器该program使用多少warp协作，逻辑元素由编译器布局到lanes。分析Triton时要区分：

```text
program数量 → grid并行度
BLOCK形状   → 单program工作量与数据布局
num_warps   → warp协作与资源分配
num_stages  → pipeline和片上buffer压力
```

## 8. 三个实验

1. Vector Add固定数据量，改变BLOCK，观察program数、尾部mask和带宽；
2. Reduction固定行宽，比较一个warp和多个warp协作，观察shuffle/shared/barrier；
3. MatMul固定tile，改变`num_warps`，同时记录register、occupancy、eligible warp与耗时。

实验只改变一个变量；否则无法判断收益来自grid覆盖、warp协作还是资源变化。

## 章节导航

[上一章](../01-gpu-hardware-map-and-generations/README.md) · [返回第一篇目录](../README.md) · [下一章](../03-registers-shared-memory-and-memory-hierarchy/README.md)

