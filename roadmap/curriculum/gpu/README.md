# 第一篇 GPU硬件与性能基础

本篇从 CUDA 程序的执行过程出发，逐步建立线程、存储、协作和性能分析的模型。Vector Add、矩阵转置、Reduction、Softmax、MatMul 和 Attention 是贯穿案例；硬件能力以 CUDA 手册为依据，优化结论回到代码与实测。

网页课程入口：[GPU 与 CUDA](course-site/index.html?chapter=1)。

## 课程入口

| 章 | 内容 | 实践主线 |
|---|---|---|
| [第一章：从 CUDA 程序看 GPU 的整体结构](01-gpu-hardware-map-and-generations/README.md) | 系统、程序、索引、架构与版本 | 完整 Vector Add 程序、地址手算 |
| [第二章：CUDA 执行模型与指令调度](02-cuda-execution-and-scheduling/README.md) | 线程编号、warp 分组、驻留与就绪、依赖链、分支、数值类型 | 线程映射手算、四种核函数的执行对照 |
| [第三章：寄存器文件、地址空间与内存系统](03-registers-and-memory-system/README.md) | 寄存器用量、存储范围、合并访问、共享内存冲突 | Matrix Transpose、寄存器探针 |
| [第四章：线程协作、同步与异步执行](04-synchronization-and-asynchronous-execution/README.md) | 共享数据读写、归约、warp 交换、stream/event、异步搬运 | Reduction、消息发布、双缓冲流水 |
| [第五章：GPU kernel 的计时与性能分析](05-performance-analysis-and-optimization/README.md) | 计时范围、occupancy、Roofline、精度、性能分析工具 | 资源用量手算、MatMul 对照、PyTorch 到 GPU kernel 的追踪 |

阅读顺序是执行 → 存储 → 协作 → 性能证据。每个机制通过具体索引、算例和源代码解释；性能结论保留输入形状、精度、硬件与计时口径。

## CUDA 资料

- [本地 CUDA Programming Guide 13.3](../../../downloads/cuda-programming-guide.pdf)
- [CUDA Programming Guide 在线版](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
- [CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [CUDA GPU 计算能力表](https://developer.nvidia.com/cuda-gpus)
- [AIInfraGuide 学习路线](https://caomaolufei.github.io/AIInfraGuide/guides/ai-infra%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/)：课程层次与主题覆盖参考。
- [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)：设备查询、转置和 stream 示例已结合到相应正文。
- [GPU MODE](https://github.com/gpu-mode/lectures)：第五章结合其 PyTorch profiling 方法开展实测。

[返回课程总路由](../README.md)
