# 第一篇 GPU硬件与性能基础

本篇从 CUDA 程序的执行过程出发，逐步建立线程、存储、协作和性能分析的模型。Vector Add、矩阵转置、Reduction、Softmax、MatMul 和 Attention 是贯穿案例；硬件能力以 CUDA 手册为依据，优化结论回到代码与实测。

网页课程入口：[GPU 与 CUDA](course-site/index.html?chapter=1)。

## 课程入口

| 章 | 内容 | 实践主线 |
|---|---|---|
| [第一章：从 CUDA 程序看 GPU 的整体结构](01-gpu-hardware-map-and-generations/README.md) | 系统、程序、索引、架构与版本 | 完整 Vector Add 程序、地址手算 |
| [第二章：线程执行、指令调度与计算管线](02-cuda-execution-and-scheduling/README.md) | 驻留与就绪、依赖链、分歧、数值格式、编译产物 | 执行对照、Triton MatMul 精度问题 |
| [第三章：寄存器、存储层次与数据访问](03-registers-and-memory-system/README.md) | 活跃区间、local、sector、bank、复用 | Matrix Transpose、寄存器探针 |
| [第四章：同步、线程协作与异步流水线](04-synchronization-and-asynchronous-execution/README.md) | 原子与内存序、归约、stream/event、异步搬运 | 消息发布、尾块归约、双缓冲流水 |
| [第五章：资源模型、性能分析与优化方法](05-performance-analysis-and-optimization/README.md) | occupancy、Roofline、正确计时、Profiler、优化证据 | 资源账本、MatMul sweep 与 Nsys 复盘 |

阅读顺序是执行 → 存储 → 协作 → 性能证据。每个机制通过具体索引、算例和源代码解释；性能结论保留输入形状、精度、硬件与计时口径。

## CUDA 资料

- [本地 CUDA Programming Guide 13.3](../../../downloads/cuda-programming-guide.pdf)
- [CUDA Programming Guide 在线版](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
- [CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [CUDA GPU 计算能力表](https://developer.nvidia.com/cuda-gpus)

[返回课程总路由](../README.md)
