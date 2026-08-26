# GPU 架构详解：系统、芯片、SM、指令与代际

> 新版主笔记。旧的 [NVIDIA vs Da Vinci 对比](gpu-architecture.md) 只作早期速记；学习任务见 [GPU 基础补强路线](../../roadmap/gpu-foundations.md)。
> 具体资源数字随 SKU 改变，以 `cudaGetDeviceProperties`、Compute Capability 和官方 tuning guide 为准。

---

## 1. 四层架构，三张不同的图

| 层次 | 对象 | 核心问题 |
|------|------|----------|
| 系统层 | CPU、PCIe/NVLink、GPU、HBM | 数据在哪里，跨设备搬多少？ |
| 芯片层 | GPC、TPC、SM、L2、内存控制器 | block 被调度到哪里？ |
| SM 层 | warp scheduler、执行单元、register、L1/shared | occupancy 与 stall 从哪来？ |
| 指令层 | PTX、SASS、LD/ST、FFMA、MMA、异步搬运 | 源码最终发出什么？ |

```text
CUDA 逻辑层级：Grid -> [Cluster] -> Block -> Warp -> Thread
硬件物理层级：GPU -> GPC -> TPC -> SM -> execution units
数据存储层级：HBM -> L2 -> L1/Shared -> Register
```

关键映射：

- 一个 block 在任一时刻驻留于一个 SM，不能跨 SM 执行。
- 一个 SM 可驻留多个 block，受 threads、warps、register、shared memory 等共同限制。
- block 内线程按连续 ID 划成 32-thread warp。
- Hopper（CC 9.0）起可选 Thread Block Cluster；同 cluster 的 blocks 被调度到一个 GPC，并可使用 Distributed Shared Memory。

---

## 2. 从整机到 SM

```text
Host CPU / DRAM
    | PCIe 或 NVLink-C2C
    v
GPU
├── GPC
│   └── TPC
│       └── SM
│           ├── warp schedulers / dispatch
│           ├── FP/INT、LD-ST、SFU pipelines
│           ├── Tensor Cores
│           ├── register file
│           └── unified L1 / texture / shared memory
├── 全 GPU 共享 L2
├── memory controllers
└── GDDR/HBM
```

- PCIe/NVLink 是设备间互连，HBM/GDDR 是 GPU 本地显存；链路带宽不等于显存带宽。
- GPC/TPC 是物理组织边界；初期知道位置即可，不必学图形管线。
- SM 是 CUDA 计算、驻留和调度的核心资源容器。
- warp 遇到内存或依赖等待时，scheduler 可选择另一个 ready warp，用并发隐藏延迟。

核心权衡：

```text
更大 tile / 更多 register / 更多 shared memory
  -> 可能提高复用和指令效率
  -> 也可能降低 resident blocks/warps
```

目标是让瓶颈资源保持忙碌，不是盲目追求 100% occupancy。

---

## 3. 执行模型

| CUDA 概念 | 语义 | 硬件事实 |
|-----------|------|----------|
| Grid | 一次 kernel 的全部 blocks | blocks 分发到可用 SM |
| Cluster | 可选 block 组，CC 9.0+ | 同 cluster 驻留于一个 GPC，可跨 block 协作 |
| Block / CTA | 可同步并共享 shared memory 的线程组 | 生命周期内驻留同一 SM |
| Warp | 32 个连续 thread | SM 的基本 SIMT 执行组 |
| Thread | 逻辑标量上下文 | 私有 register/local state 与独立控制流语义 |

SIMT 不是简单 SIMD：SIMT 暴露独立 thread 状态和控制流，硬件将其组成 warp 执行。warp 内分支不同会产生 divergence。Volta 起有 Independent Thread Scheduling，不能依赖旧式“warp 内天然同步”的未定义写法；warp 协作应用带 mask 的 primitives 或明确同步。

active blocks/SM 取以下上限的最小值：每 SM 最大 blocks/warps/threads、register 使用、shared-memory 使用及其他架构限制。

必须区分：

- occupancy：active warps / hardware maximum；
- utilization：硬件单元有多忙；
- efficiency：完成同样工作用了多少有效 transaction / instruction。

低 occupancy 的大 tile 可能因复用充分而更快；高 occupancy、低利用率也可能存在长依赖链或低效访存。

---

## 4. 数据层次

| 空间 | 作用域 | 管理者 | 常见问题 |
|------|--------|--------|----------|
| Register | thread | 编译器 | register pressure、spill |
| Local memory | thread 语义，物理走显存/cache 路径 | 编译器/硬件 | spill、动态索引 |
| Shared memory | block | 程序员 | bank conflict、同步、容量压低 occupancy |
| Distributed Shared Memory | cluster，Hopper+ | 程序员 | 跨 block 访问、cluster occupancy |
| L1/Texture | SM | 硬件 + carveout | 命中率、与 shared 容量权衡 |
| L2 | 全 GPU | 硬件，可给 persistence hint | 工作集、跨 SM 复用、thrashing |
| Global memory | 全 GPU | 程序员分配、硬件 transaction | coalescing、对齐、带宽、延迟 |
| Constant/Texture | 特定只读模式 | 程序员 + 硬件 | 只在合适模式下有优势 |

四个关键点：

1. coalescing 决定 warp 地址如何组成 memory transactions；不要死记“必然一次 128B”，粒度依架构、指令、cache 和数据宽度变化。
2. shared bank conflict 是同 warp 多个不同地址落到同 bank 后的串行化；相同地址广播是特殊情况。
3. spill 会生成 local-memory load/store；local 是地址空间名称，不是片上小缓存。
4. tiling 用片上容量换数据复用和更高 arithmetic intensity。

```text
普通 load：global -> L1/L2 命中或 HBM -> register
高复用：global -> shared tile -> compute -> register accumulator -> global
```

---

## 5. 计算单元、搬运与指令

| 单元 | 操作 | ML 例子 |
|------|------|---------|
| FP/INT pipelines | 算术与地址计算 | elementwise、索引、归约 |
| LD/ST units | 内存指令 | global/shared load-store |
| SFU | exp、sin、倒数等 | softmax、激活 |
| Tensor Cores | 小矩阵 MMA | GEMM、QK/AV、MLP |
| Async/copy mechanisms | 数据搬运 | global→shared pipeline、TMA |

CUDA Core 数不能单独预测性能；还取决于 dtype、指令吞吐、Tensor Core、内存、occupancy 和 workload。

### GEMM 的精度路径决定计算单元

“输入 tensor 的 dtype 是 `float32`”并不能单独决定走哪一种硬件路径；还要看 matmul 的内部精度策略。Ampere 上应把下面三种情况分开记录：

| 路径 | 乘法输入精度 | 主要计算单元 | 累加器 | Triton / PyTorch 控制 | 适用场景 |
|---|---|---|---|---|---|
| IEEE FP32 | FP32，23-bit mantissa | FP32 CUDA pipelines | FP32 | `tl.dot(..., input_precision="ieee")`；PyTorch `"highest"` / 禁用 TF32 | 数值 reference、严格容差 |
| TF32 | FP32 range、10-bit mantissa | Tensor Cores 的 TF32 MMA | FP32 | Triton 默认/TF32 路径；PyTorch `"high"` 或允许 TF32 | 多数 DL FP32 GEMM 的吞吐优先路径 |
| FP16 / BF16 | 低精度输入 | Tensor Cores | 常用 FP32 | 输入 dtype + `tl.dot` | 训练/推理主流路径，需按模型容差验证 |

TF32 不是把 tensor 存成新 dtype：FP32 tensor 保持原样，硬件在矩阵乘内部只读取 10 位 mantissa，保留 FP32 的 8-bit exponent 范围，并以 FP32 累加。它用较小的输入精度交换 Tensor Core 吞吐，因此必须同时报告误差和性能；不能把 TF32 GFLOPS 与 IEEE FP32 GFLOPS 当作同一精度成绩。

对当前 Triton MatMul 的诊断顺序固定：先在 IEEE FP32 下调 tile/warp/stage，确认数据复用与资源占用；再固定最快 tile，单独开启 TF32 对照 `max_abs_error`、`max_rel_error` 和吞吐。若结果变快，先归因于“计算单元从 CUDA FP32 pipeline 转到 Tensor Core”，不能误归因给 tile 优化。

```text
基础：load tile -> __syncthreads -> compute -> __syncthreads
Ampere：cp.async / pipeline，让 global->shared 与计算重叠
Hopper：TMA 做多维 bulk tensor copy，配合 mbarrier、warp specialization
Blackwell：现代 pipeline 上加入 TMEM 等矩阵计算数据通路
```

这与昇腾 `CopyIn -> Compute -> CopyOut`、double buffer 的本质相通，但 CUDA 的 thread/warp 角色、同步作用域和接口不同。

编译链：

```text
CUDA C++ / Triton -> PTX 虚拟 ISA -> SASS 目标机器指令
```

- `nvcc -ptx` 或 Triton cache 看 PTX；`cuobjdump --dump-sass` / `nvdisasm` 看 SASS。
- 初期只识别 load-store、算术、控制流、同步、MMA、异步搬运六类。
- PTX 前向 JIT 不代表所有架构特化指令都可移植；`sm_90a`、`sm_100a` 等目标需单独处理。

---

## 6. 架构代际：只记改变编程决策的差异

| 代际 | 常见 CC / 代表卡 | ML kernel 关键变化 | 当前深度 |
|------|------------------|--------------------|----------|
| Volta | 7.0 / V100 | 第一代 Tensor Core、Independent Thread Scheduling | 起点，知道即可 |
| Turing | 7.5 / T4、RTX 20 | Tensor Core 类型扩展 | 知道即可 |
| Ampere DC | 8.0 / A100 | TF32、BF16、`cp.async`、更大 L2/shared、MIG | 必学基线 |
| Ampere 消费级 | 8.6 / RTX 30、A10 | 资源上限不同，如最大 48 warps/SM、shared 更小 | 当前 3090 实验基线 |
| Ada | 8.9 / RTX 40、L40 | 延续模型，硬件与低精度能力演进 | 会查 CC 和实测 |
| Hopper | 9.0 / H100、H200 | TMA、block cluster、DSM、warp-group MMA、FP8 | FA/CUTLASS 阶段重点 |
| Blackwell DC | 10.x / B200、B300 | 第五代 Tensor Core、TMEM、更新低精度与互连 | 架构增量 |
| Blackwell 消费级 | 12.0 / RTX 50 | 与 SM100 不同目标，资源上限也不同 | 知道同代不等于同 CC |

| 对比 | RTX 3090 | A100 | H100 | B200 |
|------|----------|------|------|------|
| 定位 | 消费级 Ampere | 数据中心 Ampere | 数据中心 Hopper | 数据中心 Blackwell |
| CC | 8.6 | 8.0 | 9.0 | 10.0 |
| 显存 | GDDR6X | HBM2e | HBM3/HBM2e（依 SKU） | HBM3e |
| 关键能力 | Ampere Tensor Core/async 基础 | 更大 shared/L2 | TMA、cluster/DSM、WGMMA、FP8 | TMEM、新 Tensor Core/低精度 |

3090 和 A100 都叫 Ampere，但 CC、最大 active warps、shared memory、显存和数据中心特性不同。不能拿 A100 的上限直接解释 3090 的 occupancy 或带宽。

跨架构稳定原则：合并访问、减少传输、提高复用、控制 divergence/同步/资源占用，用 roofline + profiler 判断瓶颈。

必须重测：最佳 tile/`num_warps`/stages、register/shared/occupancy、Tensor Core dtype 与指令、实际 HBM/L2/互连带宽。

---

## 7. NVIDIA 与昇腾：迁移本质，限制类比

| 本质 | NVIDIA | 昇腾 | 提示 |
|------|--------|------|------|
| 计算资源容器 | SM | AI Core | 可类比调度资源，不等价微架构 |
| 矩阵计算 | Tensor Core / MMA | Cube / Mmad | 都依赖 tile、layout、精度、累加类型 |
| 向量/标量 | CUDA pipelines、SFU | Vector / Scalar Unit | 运算可映射，调度不同 |
| 片上显式存储 | Shared、register fragments | UB、L1、L0A/B/C | 都做分块复用，可见层次不同 |
| 搬运流水 | async copy、TMA、pipeline | DataCopy、TQue/TPipe | double buffer 思路可迁移 |
| 任务划分 | grid/block/warp/thread | block_dim/AI Core + 向量化 | CUDA 多暴露 thread/warp 层 |

危险类比：

- register 不等于 Ascend L0；前者是 thread 私有、编译器分配资源，后者是专用矩阵数据通路存储。
- CUDA warp 不等于 Ascend block；warp 是 32-thread 硬件执行组。
- `__syncthreads()` 不直接等同某个 Ascend API；先问同步实体、作用域、内存可见性。

---

## 8. 底层微架构概念：学到能解释 counter

| 概念 | 应理解到的深度 | 常见误区 | 对应证据 |
|------|----------------|----------|----------|
| latency / throughput | 单条指令延迟与流水线单位时间吞吐不是一回事 | 只背“某指令几 cycle” | dependency stall、instruction throughput |
| ILP / TLP | 单 warp 独立指令与多 resident warp 都能隐藏延迟 | occupancy 越高一定越快 | eligible warps、issue slot、active warps |
| scheduler / SMSP | SM 内有多个调度分区，warp 归属与资源分区依架构 | 把 SM 当作一个串行核心 | issue active、warp state |
| scoreboarding | 数据/内存依赖未满足时 warp 不能发射下一条相关指令 | 所有 stall 都是“访存慢” | long/short scoreboard + source correlation |
| predication/divergence | 分支可用 predication 或分路径执行，代价取决于路径与 active lanes | 看到 `if` 就等于严重 divergence | branch efficiency、active threads |
| memory transaction/sector | warp 请求会按地址、宽度、cache/架构组成 transaction | 固定背“一次一定 128B” | requested vs transferred bytes、sectors |
| bank conflict | shared bank 映射造成同 warp 串行，广播是例外 | 把所有 shared 慢归因于 conflict | shared wavefront/conflict metrics |
| register allocation | 编译器按 thread 分配，block 粒度量化影响驻留；过量会 spill | 只看源码变量数量 | ptxas、registers/thread、local traffic |
| wave / tail effect | grid 的最后一波 block 不能填满所有 SM | 只看平均 occupancy | waves per SM、grid size、timeline |
| cache locality | L1/L2 命中既由访问模式也由并发工作集决定 | 命中率越高必然越快 | hit rate + bytes + latency + reuse distance |
| memory consistency | barrier、fence、atomic 的同步实体/作用域/可见性不同 | `__syncthreads()` 等于全 GPU 同步 | racecheck、thread scope、happens-before |

学习边界：这些概念用于建立“源码 → 指令 → pipeline → counter”的解释链，不要求背未公开的内部实现。不同架构的 scheduler 数量、吞吐和资源量必须查对应 Compute Capability 文档或实测。

### 一次 kernel 的完整路径

```text
CPU launch
-> grid blocks 排队
-> block 取得 SM 的 register/shared/warp slots
-> warp scheduler 选择 ready warp
-> 指令发往 FP/INT/LDST/SFU/Tensor pipeline
-> load 经 L1/L2 到 HBM，或访问 shared/register
-> block 完成释放资源
-> 最后一波 block 决定 tail
```

这条路径对应五类性能上限：launch/并行度不足、memory、compute instruction、dependency latency、resource/tail。优化前先确定属于哪一类。

---

## 9. 全栈优化层次

```text
数学/算法：少算、少存、允许什么误差
    ↓
图与算子：融合、重计算、布局、稀疏
    ↓
kernel：tile、warp、memory、pipeline、instruction
    ↓
runtime：launch、stream、graph、allocator、copy overlap
    ↓
单机多卡：P2P、NVLink/NVSwitch、collective
    ↓
多节点：NIC、PCIe/NUMA、RDMA、topology、parallel strategy
```

越靠上潜在收益通常越大，但数值和系统约束也越强；越靠下越接近硬件，收益更依赖具体架构。不能用 kernel 微优化补救错误的算法工作量，也不能用 NCCL 参数补救不合理的并行策略。

| 症状 | 第一检查层 | 常见下一步 |
|------|------------|------------|
| 小 shape GPU 很闲 | launch/runtime | fusion、batch、CUDA Graph |
| HBM 接近上限 | 算法/数据复用 | fusion、tiling、少读写 |
| compute 接近上限 | 数值/指令 | Tensor Core、低精度、减少非 MMA 指令 |
| 两者都低 | 并行/依赖/资源 | grid、stall、divergence、spill、tail |
| 单卡快、多卡慢 | 通信/拓扑 | collective 量、overlap、rank placement |
| 换 GPU 后退化 | 代际/资源 | 重查 CC、tile、shared/register、指令路径 |

---

## 10. Profiling 因果链

```text
FLOPs/bytes -> 理论瓶颈 -> launch/tile/access
-> register/shared/warps -> profiler 证据
-> 单变量修改 -> 正确性与性能复测
```

| 工具 | 回答的问题 |
|------|------------|
| `nvidia-smi` | SKU、驱动、显存与系统视图 |
| `deviceQuery` | CC 与资源上限 |
| Nsight Systems | launch、stream、copy、kernel 时间线 |
| Nsight Compute | 吞吐、stall、occupancy、访存 |
| `nvcc -Xptxas -v` | register、shared、spill |
| `cuobjdump` / `nvdisasm` | 是否出现预期指令类别 |

每次至少回答：是 launch/latency/memory/compute 哪类瓶颈？哪个资源接近上限？stall 对应哪段源码？tile 复用收益是否大于 occupancy 损失？

---

## 11. 精选资料

官方主线：

1. [CUDA Programming Guide：Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
2. [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
3. [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
4. [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
5. [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)

GitHub：

- [gpu-mode/lectures](https://github.com/gpu-mode/lectures)：lecture 4 架构、8 性能、16 profiling、23 Tensor Core、29 Triton internals。
- [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)：`deviceQuery`、bandwidth、async copy、cooperative groups。
- [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA)：coalescing、shared/register tiling、vectorized access。
- [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)：Hopper/Blackwell 的 hierarchy、CuTe layout、TMA/WGMMA，暂不作为入门作业。
- [NVIDIA/cccl](https://github.com/NVIDIA/cccl)：CUB、Thrust、libcu++；归约、scan、同步与通用 primitive 的生产级参考。
- [NVIDIA/accelerated-computing-hub](https://github.com/NVIDIA/accelerated-computing-hub)：NVIDIA 维护的 GPU 教学资源入口。
- PMPP 第 4–6 章：compute architecture、memory architecture、performance considerations。

官方文档定事实，GitHub 学实验和读码，目标 GPU benchmark 负责最终结论。

---

## 12. 一分钟口径

> CUDA 的软件层次是 grid、可选 cluster、block、warp、thread，硬件核心是 SM。一个 block 驻留在一个 SM，SM 将 threads 划成 32-thread warps，通过多个 resident warps 隐藏延迟。性能同时取决于 HBM/L2/shared/register 的数据复用，以及 register/shared 对 occupancy 的限制。Ampere强化异步 global-to-shared pipeline，Hopper加入 TMA、block cluster 和 DSM，Blackwell进一步强化 Tensor Core、低精度和 TMEM。跨架构原则稳定，但最佳 tile 和资源配置必须按 Compute Capability 与真实 GPU 重新 profile。
