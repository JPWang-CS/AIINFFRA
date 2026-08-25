# GPU 基础补强路线：层次、架构与性能因果链

> 定位：补齐现有算子路线中的 GPU 体系知识，不另起第三条主线，不改变 [NOW.md](../NOW.md) 当前的 Triton MatMul。
> 知识入口：[GPU 架构详解](../notes/cuda/gpu-architecture-layers.md)。
> 方法：Just-in-Time 学习——当前算子遇到哪一层，就做对应最小实验。

---

## 1. 为什么补、补到哪里

现有路线已经覆盖 CUDA/Triton 算子，但 GPU 知识分散在 memory、warp、GEMM 和 profiling 中。补强后的目标不是背芯片框图，而是形成一条可验证的解释链：

```text
模型/算子工作量
-> CUDA/Triton 的 grid、tile、warp 映射
-> SM 上的调度与资源占用
-> register/shared/L2/HBM 数据路径
-> 架构代际提供的能力
-> profiler 指标
-> 优化决策与实测结果
```

完成后应能：

- 画出系统层、芯片层、SM 层、指令层四层图，并避免混淆逻辑层级与物理层级；
- 解释 3090（SM86）、A100（SM80）、H100（SM90）、B200（SM100）为什么不能只比较 CUDA Core 数；
- 用 Compute Capability 判断可用特性和资源上限；
- 根据 kernel 的数据量、FLOPs、访问模式和资源用量预测瓶颈，再用 profiler 验证；
- 把昇腾的 tiling、片上缓存和搬运流水经验迁移到 CUDA/Triton，同时指出类比边界。

---

## 2. 课程结构：四层 + 一条代际线

| 单元 | 核心问题 | 最小实验 | 验收产出 | 挂载点 |
|------|----------|----------|----------|--------|
| G0 系统层 | CPU、PCIe/NVLink、GPU、HBM 如何连接？ | `nvidia-smi topo -m`、H2D/D2H bandwidth | 一张数据路径图 + 3 个带宽数字 | B1 MatMul 前后穿插 |
| G1 执行层 | Grid/Block/Warp 如何落到 SM？ | 改 block size，记录 occupancy/耗时 | 能解释 block≠SM、warp≠block | B1 MatMul |
| G2 存储层 | register/shared/L2/HBM 如何影响性能？ | vector add stride；tiled GEMM tile 对比 | transaction、复用、spill 的证据 | B1 MatMul、B2 Softmax |
| G3 流水层 | 指令如何被调度，延迟如何隐藏？ | 看 PTX/SASS 类别；Nsight stall | 一条源码→指令→counter 因果链 | B2 Softmax、B3 FlashAttention |
| G4 计算层 | CUDA core、Tensor Core、SFU 各做什么？ | FP32/FP16 matmul 对比 | dtype、吞吐、精度三列表 | B1 MatMul、B3 FA |
| G5 代际线 | Ampere→Hopper→Blackwell 改了哪些编程决策？ | 对 tuning guide 做差异表 | 一张 capability/feature 表 | 随 B1/B3/M3/M4 渐进学习 |

---

## 3. 当前执行顺序（不打断 Triton MatMul）

### 第 1 次：随 B1 MatMul，2–3 小时

只学当前 RTX 3090 真正会遇到的部分：

1. 读 [GPU 架构详解](../notes/cuda/gpu-architecture-layers.md) 第 1–4、6、8 节。
2. 在 AutoDL 记录 `nvidia-smi`、`deviceQuery` 或等价属性：GPU 型号、CC、SM 数、每 SM shared memory、warp size。
3. 给 Triton MatMul 的每组配置同时记录 `BLOCK_M/N/K`、`num_warps`、GFLOPS。
4. 选最快和最慢两组，回答：工作量相同，差异来自数据复用、并行度、资源压力还是测量噪声？

完成定义：

- [ ] 能画 `program instance -> CTA/block -> SM -> warp` 映射；
- [ ] 不再用 A100 的资源上限解释 3090；
- [ ] 留下至少 3 组 MatMul 配置与 GFLOPS；
- [ ] 形成 1 分钟口径：为什么更大的 tile 不一定更快。

### 第 2 次：随 B2 Fused Softmax，2 小时

1. 用 arithmetic intensity 预测 softmax 是 memory-bound 还是 compute/SFU-bound。
2. 比较一行一个 program 时，行宽改变对 occupancy、register 和访存的影响。
3. 用 Nsight Compute 或 Triton profiler 记录至少一组 memory throughput、SM utilization；有条件再看 spills/stalls。
4. 把结果与 Vector Add 的 840.1 GB/s 基线比较，解释为什么两个 memory-heavy kernel 的有效带宽不必相同。

完成定义：

- [ ] 一张 roofline 手算；
- [ ] 一组 profiler 证据；
- [ ] 能区分 occupancy、utilization、efficiency。

### 第 3 次：随 B3 FlashAttention，半天

1. 复习 Ampere `cp.async` 的目的：global→shared 搬运与计算重叠。
2. 学 Hopper TMA、mbarrier、warp specialization、Thread Block Cluster/DSM 的概念边界。
3. 对照 FA2 的 work partitioning，画 producer/consumer warp 与 Q/K/V tile 数据流。
4. Blackwell 只补 TMEM、第五代 Tensor Core 和数据中心 SM100 vs 消费级 SM120 的差异，不手写架构特化 PTX。

完成定义：

- [ ] 能讲清 `cp.async` 与 TMA 的差别；
- [ ] 能讲清 block 内 shared memory 与 cluster DSM 的作用域；
- [ ] 能解释为什么现代 GEMM/attention 越来越强调 warp specialization。

### 第 4 次：进入 M3/M4 时，半天

1. 画 PCIe、NVLink、NVSwitch、HBM 与 NIC 的层次图。
2. 实测或查目标实例的 GPU-GPU 拓扑与 P2P 能力。
3. 将 TP/PP/EP 通信量映射到链路，区分链路带宽、单 GPU HBM 带宽和聚合带宽。

完成定义：

- [ ] `nvidia-smi topo -m` 图能读懂；
- [ ] 给定 TP all-reduce 大小能估算理论通信下界；
- [ ] 能解释 NVLink 快不代表 kernel 的 HBM 瓶颈消失。

---

## 4. 架构代际的学习优先级

### 必学：Ampere

- 当前 RTX 3090 是 SM86，实验数字以它为准。
- 掌握 warp/SM 资源、L1/shared、L2、GDDR/HBM 差异、Tensor Core、async copy。
- A100（SM80）作为数据中心对照，但资源上限不能直接套给 3090。

### 紧接主线学：Hopper

- TMA：多维 global↔shared bulk copy，减少地址计算、register 和 SM 搬运开销。
- Thread Block Cluster + DSM：增加 block 之上的协作层次。
- Warp-group MMA、FP8 Transformer Engine：理解现代 GEMM/attention 的执行和精度路线。

### 建立增量认知：Blackwell

- 第五代 Tensor Core、NVFP4/中他低精度、TMEM、NVLink 5。
- SM100 数据中心和 SM120 消费级不是同一个编译目标；架构名不能替代 Compute Capability。
- 当前只要求能读文档、识别能力、理解为何高性能库会出现架构专用 kernel。

### 暂不深挖

- 手写 Tensor Core PTX、WGMMA/TCGEN05 指令细节；
- GPC/TPC 图形管线细节；
- 所有历史架构的 SKU 参数背诵；
- 在没有对应 GPU 时做无法验证的极限优化。

---

## 5. 每个 GPU 知识单元的统一验收

中中，凡是可执行的 CUDA/Triton 算子或内核实验，统一先在 LeetGPU 完成正确性验收，再上真实 GPU 做 benchmark；LeetGPU 未通过时不进入真实卡。纯架构阅读、论文理解和 profiler 读数实验不适用 LeetGPU，但仍需保留验证证据。

| 维度 | 必须留下 |
|------|----------|
| 图 | 一张层次图或数据流图 |
| 数 | 至少一个真实 GPU 规格和一个实测性能数字 |
| 证据 | profiler、编译资源或 PTX/SASS 中至少一种 |
| 对比 | NVIDIA ↔ Ascend，或 Ampere ↔ Hopper/Blackwell |
| 口径 | 1 分钟回答“是什么、为什么影响性能、怎么验证” |

建议实验记录模板：

```text
GPU / CC:
kernel / shape / dtype:
launch or Triton config:
FLOPs / bytes / predicted bottleneck:
register / shared / occupancy:
time / GB/s / GFLOPS:
profiler evidence:
conclusion and next single-variable experiment:
```

---

## 6. 精选资料，而不是资料堆积

### 主线必读

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)：逻辑层级、SIMT、memory hierarchy、cluster 的语义权威源。
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)：优化原则与验证方法。
- [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)：当前 3090/A100 的代际基线，注意 SM80 与 SM86 差异。
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)：TMA、cluster/DSM 与现代 pipeline。
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)：只读增量与兼容性章节。

### GitHub 路由

- [gpu-mode/lectures](https://github.com/gpu-mode/lectures)：4 架构、8 性能清单、16 profiling、23 Tensor Core、29 Triton internals。
- [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)：`deviceQuery`、bandwidth、async copy、cooperative groups；按实验拿样例，不通刷。
- [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA)：MatMul 后读优化阶梯，只对照思路，不复制当前作业。
- [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)：进入 Hopper/Blackwell 时读 hierarchy、CuTe layout 和高性能 GEMM pipeline。

资料纪律：官方文档负责事实，GitHub 负责实验和读码，仓库自己的 benchmark 负责结论。

---

## 7. 与现有路线的关系

```text
当前 B1 Triton MatMul
  + G0/G1/G2/G4 的 Ampere 最小实验
      -> B2 Fused Softmax + G2/G3 profiling
          -> B3 FlashAttention + G3/G5 Hopper
              -> M3/M4 系统与分布式 + interconnect/topology
```

GPU 基础补强不是新的并行大计划。一次只做当前算子的挂载单元；`PATH.md` 仍是唯一进度源，`NOW.md` 仍决定当前焦点。
