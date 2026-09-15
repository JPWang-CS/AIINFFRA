# 第六章 同步、并发、异步搬运与流水

> 状态：`WIP`（Work in Progress，进行中）。正文已建立，尚未完成用户阅读和真实GPU验收。

本章解释从CTA barrier到Host stream，再到异步global-to-shared copy、TMA和warp specialization。所有异步机制都从发起者、完成条件、可见范围和buffer生命周期四个问题出发。

术语：Tensor Memory Accelerator（TMA，张量内存加速器）；Memory Barrier（mbarrier，内存屏障对象）；Warpgroup Matrix Multiply-Accumulate（WGMMA，warp group矩阵乘加）；barrier、fence和atomic分别回答会合、可见顺序和不可分更新问题。

## 从同步问题开始

高性能流水首先是正确性问题：谁拥有buffer、谁写入、谁等待、完成对哪些线程可见。下面沿普通barrier、双缓冲、异步copy、TMA和warp specialization逐层展开。

## Barrier、fence和atomic不是同一种同步

Barrier要求一组参与线程在某点会合，常见CTA级`__syncthreads()`还提供相应shared/global可见性保证。Memory fence约束内存操作顺序，却不保证其他线程已经走到同一位置。Atomic保证某个read-modify-write不可被并发更新拆开，但多个atomic之间仍需考虑顺序和scope。

```text
barrier：大家都到了吗？
fence：我的写入按要求可见了吗？
atomic：这一次更新会不会被其他更新打断？
```

把fence当barrier会让消费者提前读取；把barrier放在分歧路径中，若并非所有要求参与的线程到达，可能死锁。

## Stream与Event是Host提交层的并发

CUDA stream是有序命令队列，同一stream内操作保持顺序，不同stream在依赖和硬件资源允许时可以重叠。Event既可计时，也可建立跨stream依赖。

```text
stream 0: H2D chunk0 → kernel0 → D2H chunk0
stream 1:             H2D chunk1 → kernel1 → D2H chunk1
```

是否真正重叠取决于copy engine、kernel资源、Host内存是否pinned和依赖关系，必须用Nsight Systems时间线验证。

## 双缓冲先画状态机

两个buffer交替承担load和compute：

```text
buffer 0: EMPTY → LOADING → READY → COMPUTING → EMPTY
buffer 1:        EMPTY   → LOADING → READY → COMPUTING
```

Producer不能覆盖consumer尚未读完的buffer；consumer不能读取尚未完成搬运的数据。多stage只是在这个状态机中增加buffer数量，shared占用和barrier状态也随之增加。

## cp.async改变global到shared的流水

传统路径常由线程执行global load到register，再store到shared。异步copy允许组织global-to-shared搬运并按group提交/等待，使后续tile搬运与当前tile计算重叠。它减少某些中间register和同步开销，但不会消除DRAM流量。

优化要检查：load latency是否被覆盖、stage增加后shared/register是否限制resident CTA、wait位置是否过早、bank conflict是否让异步请求拆分。

## TMA与mbarrier

Tensor Memory Accelerator（TMA，张量内存加速器）用tensor map描述多维shape、stride和layout，由专用机制执行更大粒度搬运。Memory barrier（mbarrier，内存屏障对象）用于表达异步完成和阶段状态。

TMA减少线程重复地址计算，但把复杂度转移到descriptor、barrier、buffer生命周期和producer/consumer分工。它是架构专用路径；没有对应硬件时只能读官方定义、代码和编译目标，不能记录实测吞吐。

## Warp specialization

Producer warp负责提交搬运，consumer warp或warp group负责矩阵计算。两类warp执行不同代码，但这是按warp划分的稳定角色，不是让同一warp内部lane长期分歧。

```text
Producer：等待空buffer → 提交TMA → 通知READY
Consumer：等待READY → WGMMA/compute → 通知EMPTY
```

性能取决于producer数量、搬运与计算时长、buffer深度和occupancy。Producer过多会浪费计算warp，过少又可能断供。

## 本章实验

1. 用错误shared读写构造缺失barrier，再用sanitizer验证；
2. 实现单buffer与双buffer GEMM，画时间线并比较资源；
3. Sweep stage数，记录shared、register、resident CTA和stall；
4. 对支持的目标检查`cp.async`或TMA/WGMMA指令；
5. 无对应硬件时只完成代码/指令阅读，明确未实测边界。

## 一个可推导的两stage主循环

下面是状态逻辑，不对应某个特定API：

```text
预取 tile[0] 到 buffer[0]
等待 buffer[0] READY

for k_tile:
    向 buffer[next] 提交下一tile搬运
    用 buffer[curr] 做计算
    等待 buffer[next] READY
    确认 buffer[curr] 已不再被consumer使用
    swap(curr, next)
```

如果“等待下一tile”放在当前计算之前，copy和compute就会重新串行；如果过晚复用`buffer[curr]`，producer可能覆盖consumer仍在读取的数据。优化流水必须先验证状态正确，再谈重叠。

## CTA barrier的CUDA例子

```cpp
shared[threadIdx.x] = input[idx];
__syncthreads();
float x = shared[peer];
```

Barrier前的写入由不同thread完成，barrier后所有参与thread才读取。若越界thread在`__syncthreads()`前直接`return`，同一CTA中其他thread可能永久等待。正确做法通常是让所有thread到达barrier，仅用predicate控制load/store值。

## Warp级同步不能默认省略

Warp内shuffle通过register交换数据，但参与mask必须准确表示共同参与的lane。Shared memory的warp内producer-consumer在现代执行模型下也应使用明确的warp同步；“同一warp天然同时执行”不是可以依赖的通用正确性证明。

## Pipeline深度怎样计算资源代价

若每stage保存A/B tile，元素宽度为$b$，stage数为$s$：

$$
S_{pipeline}\approx s\times(B_M\cdot B_K+B_K\cdot B_N)\times b
$$

Stage从2变3会把这部分shared增加约50%，但并不保证隐藏更多有效延迟。若因此resident CTA从2降到1，整体并行度可能下降。还要计算barrier对象、padding和编译器额外buffer。

## 用时间线判断是否真正重叠

Nsight Systems适合看Host提交、stream、memcpy和kernel间空洞；Nsight Compute与指令检查适合看单kernel内部异步copy、barrier stall和memory/Tensor pipe。仅凭总耗时下降，无法证明收益来自流水重叠，也可能来自编译变化或频率波动。

期望证据链：

```text
假设：copy latency没有被覆盖
→ 改动：增加一个stage或调整wait位置
→ 预期：memory dependency stall下降，shared增加
→ 检查：指令存在、资源变化、stall变化、总耗时
→ 结论：是否支持假设
```

## Cluster的边界

Thread Block Cluster（线程块集群）允许一组CTA在新的硬件层级共同调度，并使用cluster同步或Distributed Shared Memory（分布式共享内存）。它不是任意grid级全局同步；cluster大小、调度和资源受目标架构限制。

没有cluster支持时，跨CTA协作仍通常依靠global memory、atomic、多kernel边界或cooperative launch。代码必须按编译目标和Runtime能力选择路径。



## 章节导航

[上一章](../05-resource-model-occupancy-and-roofline/README.md) · [返回第一篇目录](../README.md) · [下一章](../07-cuda-triton-compilation-ptx-sass-and-libraries/README.md)

