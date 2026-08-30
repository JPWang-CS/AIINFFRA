# GPU 底层架构与性能优化课程：从 SM 到集群

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

## 2. 全盘能力结构：九层 + 一条代际线

| 单元 | 核心问题 | 必须掌握 | 最小实验 | 挂载点 |
|------|----------|----------|----------|--------|
| G0 整机与拓扑 | CPU、GPU、显存、PCIe/NVLink/NIC 如何连接？ | NUMA、P2P、链路带宽与 HBM 带宽的区别 | topology + H2D/D2H/P2P | B1、M4 |
| G1 编程与执行 | Grid/Cluster/Block/Warp/Thread 如何执行？ | SIMT、ITS、divergence、同步作用域、atomics | block size/divergence microbench | B1、B2 |
| G2 SM 微架构 | block 如何驻留，warp 如何发射？ | scheduler、pipeline、latency/throughput、ILP/TLP、occupancy | block/warp/寄存器 sweep | B1 |
| G3 存储系统 | register/shared/L1/L2/HBM 如何搬数据？ | coalescing、alignment、sector、bank conflict、spill、cache reuse | stride/copy/tiled GEMM | B1、B2 |
| G4 计算与数值 | CUDA Core、SFU、Tensor Core 做什么？ | FP32/TF32/FP16/BF16/FP8/FP4、累加精度、MMA tile | 精度—误差—吞吐对照 | B1、B3、量化 |
| G5 指令与编译 | CUDA/Triton 如何变成机器指令？ | CUDA→PTX→SASS；Triton→MLIR→PTX；load/MMA/barrier 指令族 | PTX/SASS 对照 | B1、B3 |
| G6 性能模型 | 为什么慢，优化上限在哪里？ | workload、arithmetic intensity、roofline、SOL、stall、tail effect | 手算 + Nsight 证据 | 所有算子 |
| G7 Runtime 与并发 | kernel 之外还有哪些瓶颈？ | launch、stream、event、async copy、CUDA Graph、pinned/managed memory | overlap/launch microbench | B2、M3 |
| G8 库与系统 | 高性能实现如何组织？ | cuBLAS/cuDNN、CUTLASS/CuTe、CUB/CCCL、Triton、NCCL 的分工 | library baseline + 读码 | B3、M3、M4 |
| 代际线 | Volta→Ampere→Hopper→Blackwell 改了什么？ | Tensor Core、`cp.async`、TMA/WGMMA/cluster、TMEM/TCGen05 | capability/feature 差异表 | 随 G0–G8 挂载 |

这九层不是九门并行课程。每次只从当前算子向下追到真正限制它的层，再回到代码验证。例如当前 MatMul 只打开 G1/G2/G3/G4/G6；NCCL、CUDA Graph、Blackwell 指令不会插队。

---

## 3. 优化能力矩阵：必须覆盖，但按主线解锁

| 优化层 | 典型手段 | 先问什么 | 证据 | 主线挂载 |
|--------|----------|----------|------|----------|
| 算法/数学 | 少算、稀疏、在线算法、近似、重计算 | 工作量本身能否下降？ | FLOPs/bytes/误差 | Softmax、Attention、量化 |
| 数据布局 | contiguous、transpose、packing、padding | warp 地址是否连续、对齐？ | memory transactions、L2/HBM traffic | Vector Add、MatMul |
| 分块与复用 | register/shared tiling、persistent tile | 数据能复用几次？ | arithmetic intensity、traffic | MatMul、Attention |
| 并行映射 | block/warp/thread、program ordering | 并行度够吗，尾块浪费吗？ | waves、active warps、tail effect | 全部算子 |
| 流水与隐藏延迟 | double buffer、`cp.async`、TMA、warp specialization | 搬运能否和计算重叠？ | eligible warps、stall、timeline | MatMul、FA |
| 指令效率 | vector load、FMA/MMA、fast math、减少地址计算 | 是否发出了目标指令？ | PTX/SASS、instruction mix | MatMul、Softmax |
| 资源平衡 | register/shared/occupancy、spill | 更大 tile 的收益是否超过驻留损失？ | registers、shared、occupancy、spill | MatMul、Softmax |
| 融合与调度 | kernel fusion、persistent kernel、减少 launch | 中间结果和 launch 能否消除？ | Nsight Systems、HBM traffic | Fused Softmax、MLP |
| Runtime/图 | stream、event、CUDA Graph、allocator | 小 kernel 是否被 CPU/launch 限制？ | CPU/GPU timeline | 推理系统 |
| 多 GPU/集群 | collective、overlap、topology-aware parallelism | 通信量、路径、slow rank 在哪？ | algbw/busbw、topology、scaling | M4 |

固定诊断顺序：

```text
正确性/数值语义
-> workload 与理论下界
-> 系统 timeline（launch/copy/communication）
-> kernel roofline（memory/compute/latency）
-> SM 资源、warp stall、指令与源码
-> 单变量修改并复测
```

不要从某个 profiler counter 直接跳到优化动作，也不要把 occupancy 当目标函数。

### 3.1 硬件机制必须落到优化动作

| 硬件机制 | 对代码的控制旋钮 | 先看的证据 | 典型算子 |
|----------|------------------|------------|----------|
| warp scheduler / SMSP | `num_warps`、block size、ILP、每线程工作量 | eligible/active warps、issue slot、stall | reduction、MatMul |
| register file | tile、accumulator、unroll、局部数组 | registers/thread、spill、resident blocks | MatMul、Softmax、FA |
| shared memory / banks | shared tile、padding、layout、pipeline stages | shared traffic、bank conflict、容量 | GEMM、reduction、FA |
| L1/L2/HBM | coalescing、vector load、program ordering、persistent tile | sectors、hit rate、requested/actual bytes、GB/s | elementwise、GEMM、attention |
| FP/INT/SFU pipelines | 指令替换、fast math、减少地址计算 | instruction mix、pipeline utilization | Softmax、activation |
| Tensor Core | dtype、layout、tile、MMA 路径、累加精度 | MMA 指令、Tensor utilization、误差 | MatMul、MLP、Attention |
| async copy/TMA | stages、double buffer、producer/consumer warps | scoreboard stall、overlap、barrier wait | GEMM、FA |
| grid/SM 数量 | grid size、persistent scheduling、split-K、grouped ordering | waves、tail effect、SM utilization | GEMM、MoE |
| launch/stream/graph | fusion、batch、stream、CUDA Graph | CPU/GPU timeline、launch gap | 小算子、decode |
| PCIe/NVLink/NIC | rank placement、parallel degree、collective、overlap | topology、algbw/busbw、slow rank | TP/EP/DP |

固定写法：每个优化提交都必须写成“硬件假设 → 代码旋钮 → 预期 counter 变化 → 实测”。如果只有参数变化和耗时，没有硬件假设与证据，只算 autotune 结果，不算掌握了优化。

### 3.2 核心算子的极致性能阶梯

极致性能不是所有题都无限打磨。只选择四类锚点：`MatMul`、`Softmax/Norm`、`FlashAttention`、`Fused MLP/GQA`；其他算子完成正确性、服务器 baseline 和一次性能解释即可。

核心锚点在 `LEETGPU_PASS` 后，服务器章节按下面的内部阶梯迭代；lesson 仍然只显示“LeetGPU”和“服务器”两个验收段，不新增重复章节。

```text
P0 固定数值语义、shape 集和强 baseline
-> P1 算 FLOPs/bytes/理论屋顶与测得屋顶
-> P2 建立最小可解释 kernel
-> P3 sweep tile/warp/stage，找资源边界
-> P4 Nsight Systems 排除 launch/copy 问题
-> P5 Nsight Compute 定位 roofline/stall/traffic/resource
-> P6 检查 PTX/SASS 是否出现目标 load/MMA/barrier
-> P7 做布局、流水、融合、persistent/架构特化
-> P8 多 shape 回归、跨架构复测、形成停止结论
```

受限云容器没有 NCU counters 时，P4 可以形成 `P0-lite` 证据：保存完整 Nsight Systems raw log，按 Grid/Block 从 trace 排除 correctness 小 kernel，记录 kernel duration 分布、Reg/Trd、dynamic shared memory 与 API launch/sync。它可以支持资源假设和单变量实验，但不能替代 P5 的 achieved occupancy、warp stall、L2/DRAM counter。

| 锚点 | 主指标 | 强 baseline | 极致目标的判断方式 |
|------|----------|-------------|--------------------|
| MatMul | GFLOPS/TFLOPS、误差 | 同精度 cuBLAS/`torch.mm` | 主流 shape 达到强 baseline 的 80% 为合格，90% 为冲刺；差距必须有 counter 解释 |
| Softmax/Norm | 有效 GB/s、µs | PyTorch + Triton/库强实现 | 接近实测 copy roof，且在目标行宽稳定胜过 unfused baseline |
| FlashAttention | TFLOPS、HBM bytes、显存、误差 | 官方 FA/Triton 实现 | 多序列长度比较，不只看单 shape；解释 pipeline 与非 MMA 开销 |
| Fused MLP/GQA | tokens/s、µs、HBM bytes | unfused PyTorch/框架 kernel | 融合后减少真实 traffic/launch，并在目标 serving shape 获益 |

这些百分比是工程门槛，不是硬件定律。若 shape 太小、数值语义不同、库使用了不可用的架构特化路径，就记录原因并换成同语义 reference。禁止用 TF32 成绩冒充 IEEE FP32 优化，也禁止只挑一个有利 shape。

停止条件满足其一：已经接近同语义强 baseline/测得 roof；连续两轮有依据的单变量优化收益小于噪声；继续优化需要当前 GPU 不支持的指令；或收益只存在于非目标 shape。停止时必须留下尚存差距和下一代架构可能的突破点。

---

## 4. 实验梯：从微架构现象到真实 ML 算子

| 阶 | 实验 | 只改变的变量 | 必须留下 | 挂载点 |
|---:|------|--------------|----------|--------|
| L0 | `deviceQuery` + topology + bandwidth | GPU/链路 | CC、SM、cache/shared、3 类带宽 | 当前服务器首次使用 |
| L1 | Vector Add contiguous/stride/unaligned | 地址模式 | GB/s + transaction 差异 | B1 已有代码增量 |
| L2 | 分支与同步 microbench | divergence/sync | warp efficiency/stall | B2 前 |
| L3 | Reduction：shared→warp shuffle | reduction 层次 | latency、sync、bank conflict | B2 Softmax |
| L4 | MatMul：naive→tile→register tile | tile/复用 | GFLOPS、traffic、resource | 当前 B1 |
| L5 | MatMul：IEEE→TF32→FP16/BF16 | 数值路径 | error + instruction + throughput | 当前 B1 后半 |
| L6 | async pipeline | stage/copy path | overlap、stall、shared/register | B3 前 |
| L7 | Fused Softmax / Norm | fusion/row mapping | bytes、occupancy、GB/s | B2/B5 |
| L8 | FlashAttention | work partition/pipeline | HBM traffic、TFLOPS、占用 | B3 |
| L9 | CUDA Graph / multi-stream | launch/overlap | Nsight Systems timeline | M3 |
| L10 | P2P + NCCL collectives | message size/topology | algbw/busbw/latency | M4 |
| L11 | CUTLASS/CuTe 架构读码 | pipeline/layout | 自写 kernel 与库差距解释 | B3 后 |

算子实验仍只有两个验收段：**LeetGPU：正确性与原始代码归档**、**服务器：真实性能**。L0/L2/L9/L10 这类无 LeetGPU 题面的系统或微实验，使用 reference/cuda-samples/nccl-tests 做正确性门，不伪造 `LEETGPU_PASS` 状态。

---

## 5. 当前执行顺序（不打断 Triton MatMul）

### 第 1 次：随 B1 MatMul，2–3 小时

只学当前 RTX 3090 真正会遇到的部分，并把“硬件资源 → 精度选择 → 性能证据”连成一个实验：

1. 读 [GPU 架构详解](../notes/cuda/gpu-architecture-layers.md) 第 1–5、8、10 节；先能画 `program -> CTA/block -> SM -> warp`，再讨论 tile 和 profiler。
2. 在 AutoDL 记录 `nvidia-smi`、`deviceQuery` 或等价属性：GPU 型号、CC、SM 数、每 SM shared memory、register file、warp size、显存带宽。**3090（SM86）与 A100（SM80）分开记录，不混用资源上限。**
3. 建立 IEEE FP32 基线：Triton 用 `tl.dot(..., input_precision="ieee")`，PyTorch 关闭 TF32；固定 shape、warmup、repeats 和正确性 reference，只比较 tile/warp/stage。
4. 对至少 3 组 `BLOCK_M/N/K`、`num_warps`、`num_stages` 记录 GFLOPS、耗时；先从 `64×32×64`，再测 `128×32×64`、`128×32×128`。选最快和最慢两组，回答差异来自数据复用、并行度、register/shared 压力还是测量噪声。
5. 在 IEEE 基线单独归档后，才做 TF32 对照：保持 shape 和 tile 不变，分别记录 `max_abs_error`、`max_rel_error`、耗时和 GFLOPS。TF32 是另一条精度路径，**不得与 IEEE FP32 成绩混写或直接排名**。
6. 对最快 IEEE 配置做一次 Nsight Compute：先看 roofline、occupancy、registers/thread、shared memory/block、memory workload；用 profiler 证据验证前面的推断，而不是用 occupancy 数字单独判快慢。

完成定义：

- [ ] 能画 `program instance -> CTA/block -> SM -> warp` 映射；
- [ ] 不再用 A100 的资源上限解释 3090；
- [ ] 留下至少 3 组 MatMul 配置与 GFLOPS；
- [ ] 留下一组 IEEE FP32 与 TF32 的误差—吞吐对照，能解释二者为什么不能混为同一成绩；
- [ ] 有一份 roofline/occupancy/memory 证据，并形成 1 分钟口径：为什么更大的 tile 不一定更快。

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

## 6. 架构代际的学习优先级

### 必学：Ampere

- 当前 RTX 3090 是 SM86，实验数字以它为准。
- 掌握 warp/SM 资源、L1/shared、L2、GDDR/HBM 差异、Tensor Core、async copy。
- A100（SM80）作为数据中心对照，但资源上限不能直接套给 3090。

### 紧接主线学：Hopper

- TMA：多维 global↔shared bulk copy，减少地址计算、register 和 SM 搬运开销。
- Thread Block Cluster + DSM：增加 block 之上的协作层次。
- Warp-group MMA、FP8 Transformer Engine：理解现代 GEMM/attention 的执行和精度路线。

### 建立增量认知：Blackwell

- 第五代 Tensor Core、NVFP4/其他低精度、TMEM、NVLink 5。
- SM100 数据中心和 SM120 消费级不是同一个编译目标；架构名不能替代 Compute Capability。
- 当前只要求能读文档、识别能力、理解为何高性能库会出现架构专用 kernel。

### 暂不深挖

- 手写 Tensor Core PTX、WGMMA/TCGEN05 指令细节；
- GPC/TPC 图形管线细节；
- 所有历史架构的 SKU 参数背诵；
- 在没有对应 GPU 时做无法验证的极限优化。

---

## 7. 每个 GPU 知识单元的统一验收

凡是可执行的 CUDA/Triton 算子或内核实验，统一分为两段：LeetGPU 正确性与代码归档，再上真实 GPU 做 benchmark；LeetGPU 未通过时不进入真实卡。纯架构阅读、论文理解和 profiler 读数实验不适用 LeetGPU，但仍需保留验证证据。

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

## 8. 精选资料，而不是资料堆积

### 主线必读

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)：逻辑层级、SIMT、memory hierarchy、cluster 的语义权威源。
- [CUDA C++ Memory Model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html)：thread scope、atomic、barrier、fence 和 happens-before。
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)：优化原则与验证方法。
- [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)：当前 3090/A100 的代际基线，注意 SM80 与 SM86 差异。
- [Triton Matrix Multiplication Tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)：tile、`tl.dot`、program ordering 与 autotune 的官方实现路线；只用于提交后的对照。
- [PyTorch FP32 MatMul Precision](https://docs.pytorch.org/docs/main/generated/torch.set_float32_matmul_precision.html)：`highest` / `high` / `medium` 的精度与 TF32 语义；每次 Torch 对照都要记录该设置。
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：roofline、memory workload、occupancy 的指标解释；用它验证而不是猜测瓶颈。
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)：只在 G5 对照 load/store、MMA、barrier 和 async 指令，不从头背 ISA。
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)：G7/M3 观察 CPU launch、stream、copy、graph 和多 GPU timeline。
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)：TMA、cluster/DSM 与现代 pipeline。
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)：只读增量与兼容性章节。

### GitHub 路由

- [gpu-mode/lectures](https://github.com/gpu-mode/lectures)：4 架构、8 性能清单、16 profiling、23 Tensor Core、29 Triton internals。
- [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)：`deviceQuery`、bandwidth、async copy、cooperative groups；按实验拿样例，不通刷。
- [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA)：MatMul 后读优化阶梯，只对照思路，不复制当前作业。
- [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)：进入 Hopper/Blackwell 时读 hierarchy、CuTe layout 和高性能 GEMM pipeline。
- [NVIDIA/cccl](https://github.com/NVIDIA/cccl)：CUB/Thrust/libcu++ 的 scan、reduce、memory primitive；到 reduction/通用并行原语时读。
- [NVIDIA/accelerated-computing-hub](https://github.com/NVIDIA/accelerated-computing-hub)：NVIDIA 维护的课程与实验索引，只按当前单元取用。

### 学习网站与课程路由

| 当前问题 | 只读这些 | 对应实验 |
|----------|----------|----------|
| 不懂 Grid/Block/SM/内存 | Programming Guide + GPU MODE 4 | L0/L1 |
| kernel 慢但不知道为什么 | Best Practices + GPU MODE 8/16 + Nsight Guide | L2–L8 |
| reduction/scan | GPU MODE 9/20/21 + CCCL/CUB | L3 |
| Tensor Core/GEMM | GPU MODE 23 + SGEMM_CUDA；之后 CUTLASS | L4/L5/L11 |
| Triton 编译和调优 | Triton 官方教程 + GPU MODE 14/29 | B1–B5 |
| SASS/微架构 | GPU MODE 37 + PTX ISA；只识别关键指令 | L5/L6 |
| 通信 | GPU MODE 17/67 + NCCL 官方文档 | L10 |

PMPP 用作体系教材：线程/内存/性能、卷积、reduction、scan 对应章节按实验读；不要求先通读整本再写代码。

资料纪律：官方文档负责事实，GitHub 负责实验和读码，仓库自己的 benchmark 负责结论。

---

## 9. 与现有路线的关系

```text
当前 B1 Triton MatMul
  + G0/G1/G2/G4 的 Ampere 最小实验
      -> B2 Fused Softmax + G2/G3 profiling
          -> B3 FlashAttention + G3/G5 Hopper
              -> M3/M4 系统与分布式 + interconnect/topology
```

GPU 基础补强不是新的并行大计划。一次只做当前算子的挂载单元；`PATH.md` 仍是唯一进度源，`NOW.md` 仍决定当前焦点。

---

## 10. 最终能力门槛

完成整个 PATH 后，应能独立完成下面闭环，而不只是会调用框架：

1. 从模型算子写出 FLOPs、bytes、数值误差与 roofline 预判。
2. 选择 CUDA/Triton/库实现，画出 program/block/warp 与数据 tile 映射。
3. 用 LeetGPU 或 reference 建正确性门，并归档原始实现。
4. 在真实 GPU 上建立可复现 baseline，区分架构、精度和环境。
5. 用 Nsight Systems→Compute→PTX/SASS 逐层定位，不靠猜 counter。
6. 做单变量优化，解释收益、退化和跨架构不可迁移部分。
7. 把单卡结论扩展到 fusion、runtime、通信与多机多卡。
