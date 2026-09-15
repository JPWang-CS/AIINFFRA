# 第七章 CUDA、Triton编译、PTX/SASS与高性能库

> 状态：`WIP`（Work in Progress，进行中）。正文已建立，尚未完成用户阅读和真实GPU验收。

本章从CUDA/Triton源码追到中间表示、PTX、cubin和SASS，并说明不同编译目标与高性能库怎样影响功能兼容和性能可移植性。

术语：Intermediate Representation（IR，中间表示）；Parallel Thread Execution（PTX，并行线程执行虚拟指令集）；CUDA Binary（cubin，目标GPU二进制）；SASS（常称Shader Assembly，GPU机器指令）；Just-In-Time Compilation（JIT，即时编译）。

## 编译与性能可移植性

CUDA程序能够跨多代GPU运行，是软件兼容性；同一份kernel在多代GPU上都达到最佳性能，是性能可移植性。两者不是一回事。

要把这件事讲清楚，必须先知道源码怎样变成GPU真正执行的指令。

## 1. 从CUDA源码到机器指令

```text
CUDA C++ / Triton等高层源码
          ↓ 前端与优化
PTX：面向虚拟计算架构的中间ISA
          ↓ ptxas或Driver JIT
cubin：面向具体sm目标的设备二进制
          ↓ GPU执行
SASS：实际机器指令
```

可执行文件或动态库通常把一份或多份PTX/cubin装进fatbin。运行时会选择与当前设备匹配的cubin；没有合适cubin但存在兼容PTX时，Driver可以JIT生成目标机器码。

因此PTX不是最终硬件指令，SASS也不是可以随意跨架构运行的通用汇编。

### 三个常用编译命令

```bash
# 只生成PTX
nvcc kernel.cu -arch=compute_80 -ptx -o kernel.ptx

# 为具体目标生成cubin
nvcc kernel.cu -arch=sm_80 -cubin -o kernel.cubin

# 查看支持的虚拟和真实目标
nvcc --list-gpu-arch
nvcc --list-gpu-code
```

需要同时发布多个目标时使用`-gencode`。实际命令必须根据所用Toolkit支持的目标调整，不能从旧文章原样复制。

```bash
nvcc kernel.cu \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_90,code=compute_90
```

最后一项保留PTX fallback。它可以提高未来设备运行的可能性，但JIT得到的是“能针对新目标装配的代码”，不会自动把旧算法重写成TMA、WGMMA或新的低精度schedule。

## 2. baseline、family-specific和architecture-specific

较新的CUDA把功能目标分得更细：

- baseline目标，例如`compute_100`，面向承诺向后兼容的基础能力；
- family-specific目标，例如`compute_100f`，允许同一家族共享的专用能力；
- architecture-specific目标，例如`compute_100a`，允许某个精确计算能力的完整专用能力。

能力越专用，编译器可使用的指令越多，二进制兼容范围通常越窄。`a`后缀不是“更快模式”的开关；源码必须真的使用相关机制，目标设备也必须精确匹配。

从Compute Capability 9.0开始，一些专用能力不再保证后续所有架构都原样兼容。从10.0开始又增加family-specific层次。因此“更高CC一定是低CC的完整超集”不能再无条件用于所有专用指令。

## 3. 为什么能运行却可能很慢

假设一份Ampere时期的GEMM在更新GPU上正常运行。它可能仍然：

- 用线程参与地址计算和global-to-shared copy，而没有使用TMA；
- 使用warp级MMA，而没有采用warp-group矩阵路径；
- tile太小，无法填充新增计算吞吐；
- stage、register和shared配置不适合新的资源比例；
- grid对新的SM数量来说太小；
- baseline仍是旧库或不同dtype，比较口径已经失效。

兼容性只保证程序语义有机会延续，硬件利用率需要重新设计和测量。

## 4. 架构变化怎样落到kernel决策

拿到一个kernel，按五个问题检查：

1. 算法瓶颈是计算、DRAM、cache、同步、launch还是并行度？
2. 主计算使用普通算术、SFU还是矩阵管线？
3. 数据通过普通load、异步copy还是tensor descriptor搬运？
4. CTA、warp、warp group或cluster怎样分工？
5. 编译目标是否允许并实际生成预期指令？

不同算子的答案不同：

| 算子 | 首要检查 | 代际变化可能带来的重写 |
|---|---|---|
| Copy/Vector | transaction、L2、DRAM | 访问宽度、持久化策略、内存系统 |
| Softmax/Norm | reduce、SFU、load/store | warp协作、shared/register、长行划分 |
| GEMM | MMA、tile、copy pipeline | 新dtype、TMA/WGMMA、片上layout |
| Prefill Attention | QK/PV矩阵计算与在线归一化 | work partition、异步搬运、warp specialization |
| Decode Attention | KV读取、小M并行、split/reduce | page/layout、KV量化、跨CTA归约 |
| Quantized GEMM | packing、scale、低精度MMA | 新格式、block scaling、fused dequant |

“换一代GPU会不会更快”不是完整问题。对于launch-bound的小kernel，矩阵峰值没有意义；对于KV带宽受限的Decode，重点可能是扫描字节、cache和量化；对于GEMM，才需要重点检查矩阵管线和供数能力。

## 5. 怎样证明生成了预期代码

CUDA代码可以用编译器和二进制工具检查：

```bash
nvcc kernel.cu -O3 -arch=sm_XX -Xptxas=-v -lineinfo -o kernel
cuobjdump --dump-ptx kernel > kernel.ptx.txt
cuobjdump --dump-sass kernel > kernel.sass.txt
```

`-Xptxas=-v`还能显示register、shared和spill等编译资源。看到MMA指令只证明指令存在，不证明矩阵管线利用率高；还需要Profiler观察执行次数、吞吐、stall和数据供应。

Triton会根据target JIT编译。第一次运行可能包含编译开销，缓存也与代码、版本和目标有关。课程后续应保存Triton IR、PTX或编译元数据，并把warmup放在正式计时之外。

## 6. 一次跨设备迁移的完整流程

以MatMul为例：

```text
固定数学语义、shape、stride、dtype和误差
        ↓
在新设备重新建立同语义强baseline
        ↓
查询CC、register/shared、SM数和显存系统
        ↓
编译并确认目标、资源与实际指令
        ↓
重新选择tile、warp、stage和work partition
        ↓
用Profiler验证计算管线、memory和stall
        ↓
做多shape回归，保留失败配置
```

不能只复制旧tile，然后用两台设备的毫秒数作结论。设备峰值、SM数、内存和频率不同，原始时间甚至不是可解释的归一化指标。

## 证据边界

没有对应硬件时，可以确认官方功能定义、阅读作者实现、编译支持的目标，并分析代码依赖；不能声称真实occupancy、时延、吞吐和利用率。

证据强度从定义到性能逐步增加：

```text
官方文档 → 功能与兼容范围
作者论文/代码 → 某种实现及其报告环境
PTX/SASS → 当前构建实际生成的指令
Profiler → 当前运行使用了哪些资源
真实benchmark → 当前环境中的性能结果
```

## 练习

1. PTX、cubin、fatbin和SASS分别是什么？
2. PTX fallback为什么不能自动获得新架构的最佳schedule？
3. `compute_100`、`compute_100f`和`compute_100a`的兼容范围为什么不同？
4. 怎样证明一个MatMul确实生成了矩阵指令？怎样证明它用得高效？
5. 把一个旧kernel迁移到新设备时，哪些实验条件必须保持不变？
6. 选择Softmax、GEMM或Decode Attention，写出其跨架构迁移检查表。

官方资料：[CUDA Platform](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html) · [NVCC](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html) · [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) · [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)

## 章节导航

[上一章](../06-synchronization-concurrency-async-copy-and-pipelines/README.md) · [返回第一篇目录](../README.md) · [下一章](../08-benchmark-profiling-correctness-and-debugging/README.md)

