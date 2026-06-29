# GPU/ML 系统工程师面试准备指南
## 面向：Ascend NPU → NVIDIA GPU/ML Systems 转型工程师

---

## 目标岗位分析

**最常见职位名称：**
- ML Systems Engineer / ML Infrastructure Engineer
- GPU Software Engineer / CUDA Engineer
- Inference Engineer / Model Optimization Engineer
- Compiler Engineer (ML) / Kernel Engineer
- AI Framework Engineer

**主要招聘公司：**
- 大厂：NVIDIA, Google DeepMind, Meta AI, Microsoft, Amazon (AWS Inferentia/Trainium), Apple
- AI 独角兽：Anthropic, OpenAI, Mistral, Cohere, Databricks, Anyscale
- 推理专属：Groq, Cerebras, Together AI, Fireworks AI, Baseten
- 国内出海：字节跳动 (TikTok)、阿里云、百度、华为海外、DeepSeek

---

## 硬技能要求（频率排序）

| 排名 | 技能 | 出现频率 | 说明 |
|------|------|:--------:|------|
| 1 | CUDA 编程（kernel 编写、优化） | ★★★★★ | 几乎所有 GPU 岗位必考 |
| 2 | PyTorch internals / autograd | ★★★★★ | ML infra 岗位标配 |
| 3 | 推理优化（TensorRT, vLLM, quantization） | ★★★★☆ | LLM 推理岗高频 |
| 4 | **Triton kernel 编写** | ★★★★☆ | **2024 年后快速上升** |
| 5 | 分布式训练（Megatron, DeepSpeed, FSDP） | ★★★★☆ | 大模型团队必考 |
| 6 | 内存优化（FlashAttention, KV cache） | ★★★★☆ | LLM 方向必备 |
| 7 | C++/系统编程 | ★★★★☆ | 编译器/runtime 岗 |
| 8 | 性能分析（Nsight, perf, profiler） | ★★★☆☆ | 优化岗位重点 |
| 9 | MLIR / XLA / 编译器基础 | ★★★☆☆ | Google/Apple 偏好 |
| 10 | 网络通信（NCCL, RDMA, InfiniBand） | ★★★☆☆ | 分布式专向岗位 |

---

## CUDA / GPU 高频面试题（含答案）

**Q1：CUDA 线程层次是什么？**

一个 kernel 启动一个 grid，grid 由若干 block 组成，每个 block 最多 1024 个 thread。32 个 thread 组成一个 warp，是 GPU 调度的最小单元。warp 内所有 thread 执行同一条指令（SIMT 模型），如果存在分支会序列化执行（warp divergence），导致性能下降。

**Q2：什么是 memory coalescing？**

当一个 warp 的 32 个 thread 访问 global memory 时，如果地址连续且对齐（128 byte cacheline），硬件合并为 1 次事务（coalesced）；若地址分散，每个 thread 触发独立事务，带宽利用率降至 1/32。优化 memory layout（转置、padding）是 kernel 优化的第一步。

**Q3：shared memory bank conflict 是什么？**

Shared memory 分为 32 个 bank，若同一 warp 内多个 thread 访问同一 bank 的不同地址，会串行化（bank conflict）。解决方法：对 shared memory 数组加 padding（`float smem[32][33]`，多一列打破对齐）。

**Q4：occupancy 如何影响性能？**

Occupancy = SM 上活跃 warp / 最大支持 warp。Occupancy 高 → 能更好地隐藏内存延迟（latency hiding）。但高 occupancy ≠ 高性能：compute-bound kernel 可在低 occupancy 下跑满算力。限制 occupancy 的因素：registers、shared memory 用量、block size。

**Q5：A100 各级内存容量和延迟？**

Global (HBM2): 80GB，~600-800 cycle；L2 cache: 40MB，~200 cycle；L1/shared: 192KB/SM，~20-30 cycle；Registers: 256KB/SM，~1 cycle。优化原则：将热数据放 registers 和 shared memory，减少 global memory 访问。

**Q6：warp divergence 如何避免？**

Warp divergence 发生时硬件序列化两路，最坏算力减半。避免：1) 让同一 warp 内 thread 走相同分支（基于 `threadIdx` 的条件）；2) 用 predication 代替分支；3) 将不同行为的 thread 重排到不同 warp。

**Q7：tensor core 如何工作？**

Tensor core 是专门执行矩阵乘加（$D = A \times B + C$）的硬件单元，A100 FP16 达 312 TFLOPS（vs CUDA core 的 19.5 TFLOPS）。WMMA API 让 32 个 thread 协作操作 16×16 tile。实际通过 cuBLAS / CUTLASS / Triton（`tl.dot` 自动生成 `mma.sync`）调用。

**Q8：INT8 量化的 CUDA 层变化？**

INT8 推理用 `dp4a` 指令（4 个 INT8 的点积，A100 理论 624 TOPS ≈ FP16 的 2×）。挑战：per-channel/per-token scaling 控制误差；矩阵维度需 16/64 对齐；非矩阵算子（layernorm, softmax）仍需 FP16 格式转换。

**Q9：FlashAttention 为什么快？**

IO-aware tiling：将 Q/K/V 分块加载到 SRAM（不写回 HBM），用 online softmax 跨块增量更新，最终只写一次输出。HBM 读写从 O(N²) 降到 O(N)，A100 实现 3-4× 端到端加速，显存从 O(N²) 降到 O(N)。

**Q10：PCIe vs NVLink 的影响？**

PCIe 4.0 x16 双向 ~32 GB/s，NVLink 4.0 (H100) 单 GPU 总带宽 ~900 GB/s。单节点多 GPU 训练 NVLink 通信不成瓶颈；PCIe 互联则 all-reduce 严重受限，需用 gradient compression 或 ZeRO。跨节点走 InfiniBand HDR（200 Gb/s ≈ 25 GB/s），是大模型训练主要通信瓶颈。

**Q11：如何 profile 一个 CUDA kernel？**

`nsys profile` → 生成 timeline（看 kernel 占比、stream 并发、数据传输）；`ncu` (Nsight Compute) → 深度分析 SOL（Speed of Light）：compute SOL 低 = memory-bound，memory SOL 低 = latency-bound。关注：Memory Throughput、Compute Throughput、Warp State（stalled reason）。

**Q12：persistent kernel 是什么？**

常驻 GPU 的 kernel，通过工作队列从 CPU 获取任务，避免每次 kernel launch 开销（~5-20μs）。适合大量小 kernel 连续调用（decoder 逐 token 生成）或极低延迟在线推理。代价是占用 SM 资源影响并发。

**Q13：CUDA stream 和异步执行？**

同一 stream 内操作顺序执行；不同 stream 可并发（计算+数据传输重叠）。`cudaMemcpyAsync` + stream 实现 double buffering，掩盖 PCIe 传输延迟。常见用法：多 stream pipeline 将大 batch 切成 micro-batch 交错执行。

**Q14：解释 Triton 和 CUDA 的区别。**

Triton 是 block-level 编程模型：你指定 tile 对数据的操作（`tl.load`、`tl.dot`、`tl.store`），编译器自动处理 thread 分配、shared memory promotion、coalescing、syncthreads。性能接近手写 CUDA（~93-95%），开发时间降 10×。Flash Attention 2 就是用 Triton 写的。

**Q15：GQA 的 KV cache 节省怎么算？**

GQA 将 H 个 Q heads 分为 G 组，每组共享一个 KV head（G 个 KV heads，$G < H$）。KV cache 节省比例 $= G/H$。LLaMA 2 70B ($G=8, H=64$): 节省 87.5%，每 token KV 从 2.56 MB 降到 320 KB。

---

## 推理系统高频面试题（含答案）

**Q1：continuous batching 是什么？**

标准 batching 等一个 batch 所有 sequence 完成才释放，导致短 sequence 提前结束后 GPU 空转。Continuous batching 每个 decode step 后检查完成的 sequence，立即用新 request 填充，KV cache slot 动态分配。实测吞吐提升 5-23×，GPU 利用率从 30-40% 提升到 80%+。

**Q2：KV cache 占多少显存？优化手段？**

公式：$2 \times \text{num\_layers} \times \text{num\_kv\_heads} \times \text{head\_dim} \times \text{seq\_len} \times \text{batch} \times \text{dtype\_bytes}$。LLaMA-2-70B (GQA, FP16), seq=4096, batch=1 ≈ 4 GB；batch=32 则 128 GB。优化：GQA/MQA（减少 kv_heads）、KV cache 量化（INT8/FP4）、PagedAttention（减少碎片）、prefix caching（复用相同 prompt）。

**Q3：speculative decoding 的原理？**

小草稿模型（draft model）快速生成 k 个 token，大目标模型并行验证（一次 forward）。rejection sampling 保证输出分布与 target 等价。适用条件：小大模型同家族（acceptance rate 高）、batch size 小（目标模型 memory-bound）。代码生成典型加速比 2-3×，通用文本 1.5-2×。

**Q4：prefill 和 decode 为什么要分离？**

Prefill 是 compute-bound（大矩阵乘），decode 是 memory-bandwidth bound（矩阵-向量乘）。混合 serving 时相互干扰：长 prefill 阻塞 decode（TTFT 毛刺）；显存争抢。PD 分离（Splitwise/DistServe）将两者放不同 GPU 集群，P99 TTFT 降低 4-5×，整体吞吐提升 2×。KV cache 用 RDMA 传输（~10ms，远小于 prefill 耗时）。

**Q5：PagedAttention 解决什么问题？**

传统 KV cache 预分配连续显存，内部碎片（平均 60-80% 浪费）+ 外部碎片（奇怪大小空洞）。PagedAttention 借鉴 OS 分页，KV cache 切成固定大小 block（vLLM 默认 16 tokens），block table 维护逻辑→物理映射。碎片从 60-80% 降到 <4%，支持 copy-on-write 共享 prefix。

**Q6：W4A16 vs W8A8 量化的区别？**

W4A16（权重 INT4，激活 FP16）：显存减 4×，每次 GEMM 需先反量化到 FP16，适合 memory-bound（小 batch）。W8A8（权重激活都 INT8）：利用 INT8 Tensor Core（~2× FP16 吞吐），适合 compute-bound（大 batch）。实践：单 token 推理用 W4A16，吞吐优化用 W8A8。

**Q7：chunked prefill 解决什么？**

超长 prompt 单次 prefill 耗时数秒，阻塞所有 decode 请求。Chunked prefill 将 prompt 切成小块（如 512 tokens），与 decode step 交替执行，消除延迟毛刺。vLLM 0.4+ 支持（`--enable-chunked-prefill`）。

**Q8：FP8 推理（H100）的注意点？**

H100 原生 FP8 tensor core：E4M3（前向激活）和 E5M2（梯度），理论 ~3958 TFLOPS（vs BF16 的 989）。挑战：FP8 动态范围窄，激活 outlier 需 per-tensor/per-channel scaling；Transformer Engine（NVIDIA）自动管理 scaling factor。LLM 推理 FP8 精度损失通常 <0.5% MMLU。

**Q9：tensor parallelism vs pipeline parallelism 推理 tradeoff？**

TP 将矩阵按列/行切分，每层 all-reduce，latency 与 GPU 数量线性相关（NVLink 依赖）；适合低延迟推理。PP 按层分组，跨节点通信量小，但有 pipeline bubble，单 token 延迟高；适合超大模型或跨节点场景。推理服务通常优先 TP。

---

## 分布式训练高频面试题（含答案）

**Q1：DP/TP/PP 各适用什么场景？**

Data Parallel（DP）：模型放入单卡，scale batch size，通信量 = 梯度大小（2×参数量）。Tensor Parallel（TP）：模型放不进单卡，权重矩阵按维度切分，单节点 NVLink 互联，每层需 all-reduce。Pipeline Parallel（PP）：模型极大，按层切分，跨节点通信量小（只传激活），有 pipeline bubble（效率 50-70%）。实际大模型用 3D parallelism（DP×TP×PP）。

**Q2：ZeRO 三个阶段？**

Stage 1：切分 optimizer states（Adam 的 momentum + variance），4× 显存节省，通信量不变。Stage 2：额外切分梯度，8× 节省，通信 +50%（Reduce-Scatter 替代 AllReduce）。Stage 3：额外切分参数，Nd× 节省（N = GPU 数），forward 时 AllGather 参数、backward 后 Reduce-Scatter 梯度，通信约 3× DP。PyTorch FSDP = ZeRO-3 官方实现。

**Q3：gradient checkpointing 的 tradeoff？**

前向只保留部分层的激活，反向时重新计算中间激活，激活显存从 O(layers) 降到 O(√layers)。代价：额外 ~33% 计算（重算前向）。Selective checkpointing（只重算 attention，跳过 FFN）在节省和计算间取平衡。长上下文训练中激活是主要瓶颈，必须用。

**Q4：Ring all-reduce 的原理？**

N 个 GPU 排成环，两阶段：Reduce-Scatter（每 GPU 收到一份 reduced 结果）+ All-Gather（广播给所有 GPU）。每个 GPU 通信量 = 2(N-1)/N × model_size ≈ 2× model size，与 GPU 数量无关（bandwidth-optimal）。NCCL 在 NVLink 上接近理论带宽上限。

**Q5：混合精度训练（AMP）为什么不全用 FP16？**

FP16 动态范围（~65504）容易溢出/下溢（小梯度归零）；Adam 的参数更新量可能比参数小 1000×，需 FP32 精度。AMP 用 FP16 做前向/反向（利用 Tensor Core），用 FP32 保存 master weights 和 optimizer state。Loss scaling（GradScaler）动态调整 loss 放大倍数，防止梯度下溢。

**Q6：sequence parallelism 是什么？**

Tensor Parallelism 切 hidden dim，但 layernorm 和 dropout 的激活在 sequence 维度无法用 TP 切分，导致每卡仍存完整 sequence 激活。SP 将这些算子的激活也按 sequence 切分，进一步降低显存。超长上下文（>32K）时激活成为主要瓶颈，TP+SP 联合是标准做法（Megatron-LM）。

---

## 系统设计题

### 设计题 1：设计 LLM 推理服务（70B 模型，QPS=100，P99 latency < 2s）

**关键考量点**：
1. **部署**：LLaMA-2-70B FP16 需 140GB → W4A16 量化到 35GB 可单卡 A100 80GB；或双卡 TP=2
2. **Batching**：continuous batching（vLLM）+ PagedAttention，GPU 利用率 80%+
3. **调度**：在线估算 KV cache 需求，抢占式调度（recomputaion preemption）防 OOM
4. **长 prompt**：chunked prefill 防阻塞；prefix caching 加速重复 system prompt
5. **扩展**：多实例 + load balancer，按 KV cache 使用量做 routing
6. **监控**：TTFT / TBT / KV cache 使用率 / 队列深度

### 设计题 2：100B 参数模型分布式训练（64 GPU，A100）

**关键考量点**：
1. **并行策略**：TP=8（节点内 NVLink），PP=8（跨节点），DP=8 × 8 = 64 GPU
2. **显存**：100B × 16 bytes (Adam FP16) = 1.6 TB，ZeRO-3 降至 1.6TB/64 = 25 GB/GPU（可行）
3. **通信**：TP all-reduce（节点内，快）；PP P2P 传激活（跨节点，少量）；DP all-reduce（跨节点，梯度）
4. **激活**：gradient checkpointing（每 √layers 层一个 checkpoint）
5. **混合精度**：BF16 forward + FP32 optimizer state（on GPU），或 ZeRO-Offload 把 optimizer 卸到 CPU

### 设计题 3：Triton GEMM kernel 优化

**要说清楚的点**：
1. block tiling（BLOCK_M、BLOCK_N、BLOCK_K 的选择）
2. `tl.dot` 自动调用 Tensor Core
3. L2 cache 优化（GROUP_M swizzle）
4. Autotuning 策略（以上参数的搜索空间）
5. 对比 cuBLAS：Triton 可以轻松自定义（fused softmax、custom dtype）

---

## Ascend → GPU 叙事框架

### 核心叙事

> "我有昇腾 NPU 算子开发经验，迁移到 GPU 生态时发现核心概念高度同构——只是工具链不同。Ascend 上的 L1 Buffer tiling 对应 CUDA 的 shared memory tiling；Pipe 流水对应 double buffering；Cube Unit 对应 Tensor Core。这让我用 1-2 个月就上手了 GPU 内核优化，而不是从零开始。"

### 具体话术模板

**"你有 GPU 经验吗？"**  
"我有半年的 CUDA + Triton 实战经验，实现了 GEMM（naive → tiled → FP16）、Softmax、在读 Flash Attention 源码。在此之前有 2 年昇腾 NPU 算子开发，两个平台的优化思路高度同构，所以上手很快。"

**"为什么从 Ascend 转 GPU？"**  
"GPU 是业界事实标准，生态更完整（CUDA / Triton / vLLM / PyTorch）。我的核心竞争力不是绑定某个 vendor，而是理解算子优化的方法论——访存分析、tiling 策略、计算-访存 tradeoff，这些在任何平台上都适用。"

**"解释一个你优化过的算子"**  
"以 GEMM 为例：v0 naive 每 thread 直接读 global memory，算术强度 0.25 FLOP/byte，完全 memory-bound。v1 shared memory tiling 把 TILE×TILE 的数据搬进片上，算术强度提升到 TILE/2，实测 GFLOPS 提升 5-10×。这和 Ascend 上做 L1 Buffer tiling 是完全相同的思路——把数据搬进片上，多次复用，减少 HBM 访问。"

### 简历加分项（按优先级）

1. **能展示的代码**：GitHub 有 GEMM/Softmax/Attention 的 Triton 实现，并附 benchmark（比 PyTorch 快多少）
2. **Profiling 数据**：Nsight Compute 截图，能指出瓶颈在哪、如何改进
3. **LeetGPU 通关记录**：fp16 GEMM、Flash Attention 等 Hard 题
4. **论文阅读**：能复述 Flash Attention 2 的 work partitioning 改进（说明读得深）
5. **系统级理解**：能讲 vLLM 的 continuous batching + PagedAttention 工作原理
6. **Cross-platform 经验**：Ascend 经验是差异化优势，主动提，说明理解"异构计算本质"

---

*最后激活时间：求职前 2-3 个月。保持刷题 + 系统复习 + 项目 demo 展示。*
