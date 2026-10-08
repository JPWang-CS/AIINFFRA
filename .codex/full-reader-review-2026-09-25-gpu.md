# GPU 基础五章全文走读记录（2026-09-25）

本轮按 README 源文件逐段分块读取到 EOF，重点检查“前提 → 定义 → 公式 → 代码/数据流 → 理解检验”的连续性。`source-check` 代码、公式、历史实验与空白测量表均保留；未生成网页、未执行 GPU benchmark、未修改 PATH/NOW 或原始 solutions/reference。

## 01-gpu-hardware-map-and-generations/README.md

- 总行数：789；已读至 EOF。
- 完整章节覆盖：CPU/GPU 任务分工（1.1–1.4）；kernel/grid/block/thread 与 SM 驻留（2.1–2.4）；Vector Add 完整程序、主机/设备指针、复制/异步性、错误检查、编译（3.1–3.5）；一维/二维索引、元素/字节偏移、Triton 地址与多元素线程（4.1–4.5）；架构/计算能力、PTX/cubin/fatbin、JIT、设备查询与编译环境（5.1–5.5）；总结、参考与导航。
- 具体观察：第 1.3 节准确区分 SM/GPC/L1/L2/寄存器/shared/global，但初读时容易把资源类别直接当作某个 kernel 的并发状态；已补一句将资源概览与第 2 章驻留/就绪/发射衔接。
- 具体观察：Vector Add 代码明确区分 `cudaGetLastError`、`cudaDeviceSynchronize`、D2H 与数值 reference；普通 `cudaMemcpy` 的方向/主机内存条件已有限定，未改代码。
- 具体观察：二维 3×5 矩阵、`offset < M*N` 的错误示例和矩形 tile 的尾部说明能把元素坐标、字节地址和 stride 连起来；RTX 4090 696 GB/s 记录明确绑定 shape/线程数/0 error，空表仍留空。
- 修改位置/类型：仅 1.3 末尾新增阅读桥接句；未改标题、锚点、公式、代码或实验数据。
- 主要 primary 来源：NVIDIA [Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)、[Writing CUDA SIMT Kernels](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/02-basics/writing-cuda-kernels.html)、[NVCC compiler guide](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)。核对线程块调度无固定顺序、occupancy 定义与资源约束、目标代码/工具链边界。
- 未确认项：无本机 nvcc/GPU 重新编译；历史性能记录不作新成绩。

## 02-cuda-execution-and-scheduling/README.md

- 总行数：1798；已读至 EOF。
- 完整章节覆盖：线程线性编号与 warp（1）；驻留/就绪/发射（2.1–2.3）；SIMT、分支、谓词、ITS（3–3.2）；波次（4）；FMA/Tensor Core/dtype/Triton precision（5.1–5.2）；依赖链/分支实验（6.1–6.2）；PTX/SASS 与错误定位（7）；CUDA Tile（8.1–8.6）；Driver/Runtime/context/module（9）；CUDA Python 控制面与 kernel DSL（10）；cuTile view/shape/atomic（11）；CUDA C++ execution space、生命周期、device link（12）；Error Log（13）；Driver Entry Points/版本查询（14）；参考。
- 具体观察：SIMT 段已经把活动掩码、地址、谓词、发射/完成、ITS 与 `__shfl_sync`/`__syncwarp` 的边界写清；source-check 代码保留。
- 具体观察：section 6.2 的两个 kernel 数学输出相同，但 predicated-select 源码无条件计算 even/odd 两边，原句“完成相同逐元素运算，可比较 kernel 时间”会误导为同算术工作量对照；已改为“语义相同、工作量可能不同，需结合 PTX/SASS/指令数解释”。
- 具体观察：CUDA Tile、Driver API、cuTile、Python binding、Green/cluster/TMA 等扩展都反复声明教学映射不等于性能或设备验证；未发现应修改的历史表格或 source-check 实现。
- 修改位置/类型：section 6.2 解释段；其余为完整阅读记录，无标题/锚点/代码改动。
- 主要 primary 来源：NVIDIA [Writing CUDA SIMT Kernels](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/02-basics/writing-cuda-kernels.html)、[CUDA Programming Guide 13.2 synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf)、[CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)。核对 block 调度无序、occupancy 资源含义、`__syncwarp` 内存顺序和 API 生命周期。
- 未确认项：未编译运行 CUDA 13.3/cuTile/Driver API 示例；性能表仍按现场填写。

## 03-registers-and-memory-system/README.md

- 总行数：1498；已读至 EOF。
- 完整章节覆盖：线程私有值/寄存器与活跃区间（1–3）；地址空间与缓存（4.1–4.4）；32B sector（5）；shared bank/broadcast/padding（6）；矩形转置与三条代码路径（7）；cache/shared staging（8）；PTX/SASS（9）；UVA/Unified Memory/映射内存（10.1–10.3）；Thread Block Cluster/DSM/TMA multicast（11.1–11.5）；Green Context（12）；Extended GPU Memory/NUMA（13）；参考。
- 具体观察：作用域、地址空间、物理存储路径的先后关系已清楚；寄存器估算、local spill、32B sector 手算均声明为趋势/模型，不把公式当作编译器或硬件实测。
- 具体观察：转置章节明确区分输入越界与输出越界，解释提前 return 为什么可能漏写合法输出；naive/unpadded/padded 的线程组织差异与空白实测表保留。
- 具体观察：DSM/TMA multicast、barrier transaction bytes、TMA/MMA 两个完成点、Green Context/EGM 都有能力条件与未验证边界；未发现确定性公式或 source-check 错误。
- 修改位置/类型：本轮无必要正文改动；仅完整覆盖记录。
- 主要 primary 来源：NVIDIA [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)、[CUDA Samples transpose](https://github.com/NVIDIA/cuda-samples/tree/5443602d89ed99aede2e4b7bf329daddeadb320e/cpp/6_Performance/transpose)、CUDA Driver/Runtime VMM 与 Green Context 文档（正文参考链接）。核对 shared bank、VMM 权限、cluster/TMA 生命周期和资源分组边界。
- 未确认项：未在目标设备上运行 DSM/TMA/Green Context/EGM；空白性能表不作结论。

## 04-synchronization-and-asynchronous-execution/README.md

- 总行数：2976；已读至 EOF。
- 完整章节覆盖：shared handoff（1）；block/warp reduction 与练习（2–3）；stream/event/batch copy（4.1–4.2）；双缓冲（5）；cp.async pipeline（6.1–6.4）；异步 barrier（7）；TMA/tensor map/swizzle/cluster（8.1–8.11）；stream-ordered allocation（9.1–9.4）；CUDA Graph（10.1–10.12）；VMM（11.1）；IPC/pool IPC（12.1）；PDL（13）；Memory Synchronization Domains（14）；CLC；外部 graphics/semaphore interop；参考。
- 具体观察：正文持续区分 barrier 会合、memory ordering、atomic、fence、异步事务完成和 source-read completion；`cp.async`/TMA/Graph/IPC 各自的完成边界没有被压成“API 返回即完成”。
- 具体观察：大量完整程序都将 correctness、skip、现场性能和历史空表分开，尤其 reduction 的 CPU partial 仅作为 block 同步教学，不冒充完整 GPU reduction。
- 具体观察：TMA multicast transaction bytes、barrier phase、MMA 读完 shared 后才能复用、Graph capture 依赖闭合、VMM map/access 权限和 IPC importer 生命周期均给出具体失效条件；未发现需改 source-check 代码的确定性问题。
- 修改位置/类型：本轮无必要正文改动；仅完整覆盖记录。
- 主要 primary 来源：NVIDIA [CUDA Programming Guide 13.3](https://docs.nvidia.com/cuda/cuda-programming-guide/)、[CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)、[CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)、固定版本 CUTLASS/CUDA Samples 链接。核对 stream/event、barrier、TMA、Graph、VMM、IPC 和 external semaphore 的官方边界。
- 未确认项：未执行 CUDA 13.3 编译、设备运行、Nsight 或 sanitizer；本轮没有新增性能数字。

## 05-performance-analysis-and-optimization/README.md

- 总行数：620；已读至 EOF。
- 完整章节覆盖：Event/计时与 shape 回归（1.1–1.2）；occupancy 手算与 kernel 时间（2.1–2.2）；Roofline/Vector Add/GEMM（3.1–3.2）；浮点精度与 correctness（4）；MatMul sweep、Nsight Systems trace 与复测（5.1–5.4）；优化假设/指标证据（6）；PTX/SASS/profiler（7）；请求时延、TTFT/TPOT、Amdahl、MFU/带宽（拓展）；参考。
- 具体观察：历史 MatMul 数据保留 shape/FP32/IEEE/计时口径与“无 NCU counters”的限制；空表明确留给新设备实测，未将 profiler 时间线当硬件计数器。
- 具体观察：Roofline 的 840 GB/s、占用率课堂假设与 GEMM FLOPs/bytes 公式分别标出算例/条件；FLOPs、算法下界 bytes、实测 DRAM/L2 bytes 的口径已分开。
- 具体观察：TTFT/TPOT 对单 token、分位数与 Amdahl 的边界明确；未发现需改 source-check 代码或历史实测数字的确定性问题。
- 修改位置/类型：本轮无必要正文改动；第 5 章仅完整覆盖记录。
- 主要 primary 来源：NVIDIA [CUDA Events/Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)、[Nsight Systems](https://docs.nvidia.com/nsight-systems/)、[Nsight Compute](https://docs.nvidia.com/nsight-compute/)、Triton [fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)。核对 Event 时间语义、occupancy 资源约束、profiler 分工和 Roofline 口径。
- 未确认项：未重跑 MatMul、Nsight 或 GPU benchmark；保留历史数据，不新增成绩。

## 本轮验证

- 五篇 README 均已从首行读到 EOF；标题、锚点、公式、source-check 代码与实验状态未重构。
- 本轮允许写集仅为五篇 README 与本记录；未改 PATH/NOW、原始实现、生成站点或构建脚本。
- `git diff --check` 的限定路径检查应由主代理在最终合并前重跑；本轮此前审阅的 GPU 五章 scoped diff 无新增空白错误。
