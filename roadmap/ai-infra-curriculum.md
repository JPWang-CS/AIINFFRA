# PATH 执行参考：AI Infra 全阶段学习计划

> 定位：把 [PATH.md](../PATH.md) 的知识地图展开成可执行的学习计划，覆盖算子、模型结构、推理、训练、面试。
> 约束：本文件不是另一条学习线。进度仍以 [PATH.md](../PATH.md) 为唯一权威源，当前焦点由 [NOW.md](../NOW.md) 指定，本文件只负责“任务、步骤、验收、示例”细化。
> 参考：[AIInfraGuide](https://github.com/caomaolufei/AIInfraGuide) · [Triton Language Docs](https://triton-lang.org/) · [vLLM](https://github.com/vllm-project/vllm)

---

## 1. 怎么用这份计划

1. 先看“当前主线”，现在只做当前主线，不并行推进多套大计划。
2. 每个模块都按“目标 -> 任务 -> 步骤 -> 完成定义 -> 验收”推进。
3. 代码跑通后进 `solutions/`，笔记理解后进 `notes/`，进度再更新到 `PATH.md`。
4. 所有示例数字都要能自己手算一遍，不能只背结论。

### 每个学习单元统一产出

| 产出 | 要求 |
|------|------|
| 可运行代码 | CUDA 进 `solutions/cuda/`，Triton/Python 进 `solutions/triton/` |
| 当前代码入口 | lesson 直接放当前代码快照或路径，并标明 `WIP` / 通过状态 |
| 笔记 | 能讲清“解决什么问题、核心思路、关键数字、取舍” |
| 验证 | 正确性对比 + 性能数字 |
| 归档索引 | 题目/题号 → 原始 `solve` → 本地文件 → 正确性/性能证据 |
| 面试口径 | 能用 1 分钟讲清：是什么、为什么、怎么验证 |

### 统一验收基线

- 正确性：和 PyTorch / CPU reference 对齐。
- 性能：至少记录一个数字，如 GFLOPS、GB/s、耗时、显存。
- 理解：能不看材料画图或推导公式。

---

## 2. 当前主线

**现在进入 PATH B：Triton 实现阶段。**

统一章节规则只有两段：**LeetGPU：正确性与代码归档 → 服务器：真实性能**。LeetGPU 未通过、原始 `solve` 未归档时，不进入服务器；服务器 benchmark 未完成时，不标记为完整产物。纯 CUDA kernel、warp shuffle、手写 FlashAttention 等底层深钻暂不插队，放到 Triton 主线阶段性完成后。

优先级：
```text
Triton vec add -> matmul -> fused softmax -> flash attention -> GQA/fused MLP
```

A4/A5 不再作为当前主线，只作为背景收尾：
```text
A4 1-pass true online 落盘
A4 warp shuffle + benchmark
A5 Flash Attention 注释笔记
```

---

## 3. 文件归属

| 文件 | 职责 |
|------|------|
| [PATH.md](../PATH.md) | 唯一进度源 |
| [NOW.md](../NOW.md) | 当前焦点 |
| [lessons/](../lessons/) | 主题课 |
| [solutions/](../solutions/) | 自己写的代码 |
| [reference/](../reference/) | 参考实现 |
| [notes/](../notes/) | 知识笔记 |
| 本文件 | 全阶段执行计划 |

---

## 4. 大模型内容板块映射

[notes/llm/](../notes/llm/README.md) 是内容聚合，不是另一条学习线。

| 子板块 | 内容 | 进入阶段 |
|--------|------|---------|
| [模型结构](../notes/llm/architectures.md) | Transformer、GQA/MoE/SSM、HF config | M1.5 |
| [推理系统](../notes/llm/inference-systems.md) | vLLM、PagedAttention、调度、量化 | M3 |
| [训练系统](../notes/llm/training-systems.md) | 显存账本、FSDP/ZeRO、TP/PP/EP | M4 |
| [面试](../notes/llm/interview.md) | 高频题、系统设计、叙事 | M5 |
| [论文](../notes/llm/papers.md) | 精读顺序和清单 | 随 M1.5/M3/M4 滚动 |

---

## 5. 全阶段总览

| 模块 | 主题 | 状态 | 目标 |
|------|------|:--:|------|
| M0 | A4/A5 背景收尾 | ⏳ 穿插 | 不阻塞 Triton |
| M1 | CUDA 算子优化 | ⏳ 后续 | 建立 kernel 底层感觉 |
| M1.5 | 模型结构理论 | ⏳ 滚动 | 能读懂模型 config |
| M2 | Triton 实现 | 🚧 当前 | 能写常见 ML 算子 |
| M2.5 | 最新模型与算子构建 | 🚧 并行强化 | 能构建 GQA/MLA/MoE/FlashAttention/PagedAttention |
| M3 | 推理系统 | ⏳ 后续 | 能讲 vLLM 链路 |
| M4 | 训练/分布式 | ⏳ 后续 | 能算显存和通信 |
| M5 | 面试冲刺 | ⏳ 后续 | 项目 + 题库 + 叙事 |

---

## M0：A4/A5 背景收尾

目标：把 Softmax 优化和 Flash Attention 阅读收完，但不抢 Triton 主线时间。

### M0.1 Softmax 1-pass true online 落盘

任务：
- 把 LeetGPU 上实践过的 `maxSumkernel` 落盘为 `solutions/cuda/softmax/softmax_1pass.cu`。
- 保持 `extern "C" void solve(const float*, float*, int)` 签名。

步骤：
1. 每个线程串行扫一段元素，维护 `(m, s)`。
2. block 内 tree reduce 合并 `(m, s)`。
3. 空线程用 `-INFINITY` 哨兵，避免 `exp(NaN)`。
4. 用 `normalize_kernel` 多 block 并行归一化。
5. 用 `KERNEL=softmax_1pass.cu ./run.sh` 验证。

完成定义：
- [ ] 代码在仓库
- [ ] 精度误差 < 1e-4
- [ ] 记录耗时或带宽

### M0.2 Softmax 三版 benchmark

任务：
- 对比 `softmax_naive`、`softmax_online`、`softmax_opt`。
- 分别记录耗时、有效带宽 GB/s、误差。

关键公式：
```text
有效带宽 = 实际读写字节 / 耗时
3-pass: 3 次读 input + 1 次写 output
2-pass fused: 2 次读 input + 1 次写 output
```

完成定义：
- [ ] 三个版本都有数字
- [ ] 能解释为什么 online 比 3-pass 少读一次

### M0.3 A5 Flash Attention 注释

任务：
- 读 `reference/cuda/flash_attention/flash_attn.cu`。
- 标注每个 `__syncthreads` 的作用。
- 解释 Q/K/V tile、online softmax、register accumulator。

完成定义：
- [ ] 生成注释版笔记
- [ ] 能画出 `Br x Bc` tile 的数据流

---

## M1：CUDA 算子优化

目标：用 CUDA 建立 kernel 底层感觉，为理解 Triton 生成代码打底。

GPU 体系主课入口：[GPU 底层架构与性能优化课程](gpu-foundations.md) · [GPU 架构知识图](../notes/cuda/gpu-architecture-layers.md)。它覆盖整机拓扑、执行模型、SM 微架构、存储、数值/Tensor Core、编译指令、性能模型、runtime、库与集群九层，但按当前算子 Just-in-Time 解锁，不另开第三条主线：B1 MatMul 学 Ampere/SM/内存/Tensor Core/roofline，B2 Softmax 学 reduction 与 profiling，B3 FlashAttention 学 async pipeline/Hopper，M3 学 runtime/CUDA Graph，M4 学互联和通信；Blackwell 做架构增量。

### 底层能力挂载表

| PATH 节点 | 同步解锁的 GPU 底层能力 | 实验证据 |
|-----------|--------------------------|----------|
| B1 MatMul | G1 执行、G2 SM、G3 存储、G4 数值/Tensor Core、G6 roofline | tile/warp/stage sweep；IEEE/TF32 对照；Nsight Compute |
| B2 Softmax | divergence、reduction、SFU、register pressure、memory-bound | shared/warp reduction；GB/s；stall/traffic |
| B3 FlashAttention | async copy、double buffer、warp specialization、Hopper TMA/WGMMA | Q/K/V 数据流；pipeline/timeline；HBM traffic |
| B5 GQA/MLP | fusion、layout、quantized MMA | 中间张量 bytes；融合前后性能/误差 |
| M3 推理 | launch、stream、CUDA Graph、allocator、KV cache locality | Nsight Systems；TTFT/TPOT/tokens/s |
| M4 分布式 | PCIe/NVLink/NVSwitch/NIC、P2P、NCCL、RDMA | topology；algbw/busbw；scaling efficiency |

### 分阶段任务表

| 算子 | 必须做到 | 可选深钻 |
|------|---------|---------|
| Reduce | shared memory tree reduce、warp shuffle | grid-wide reduce |
| GEMM | naive、tiled、fp16 | vec4、double buffer、tensor core |
| Softmax | 3-pass、2-pass fused、1-pass online | benchmark |
| Flash Attention | 读 CUDA、对照论文 | 手写简化版 |
| LayerNorm/RMSNorm | 读参考、写一版 | 融合 residual |
| Profiling | Nsight Systems/Compute 能跑 | roofline 分析 |

### 极致性能锚点

不是每道题都无限优化。全路线固定四类锚点：MatMul、Softmax/Norm、FlashAttention、Fused MLP/GQA。它们完成 LeetGPU 正确性和原始代码归档后，在服务器阶段执行 [P0–P8 极致性能阶梯](gpu-foundations.md#32-核心算子的极致性能阶梯)：建立同语义强 baseline 与实测 roof，逐层优化 tile/layout/pipeline/instruction/fusion，并用 Nsight 和 PTX/SASS 证明原因，最后做多 shape 回归和停止判断。其余算子只要求可靠 baseline 与一次瓶颈解释，避免主线被无底洞式调参拖住。

### 分阶段完成定义

每个算子都要满足：
- [ ] LeetGPU：平台正确性通过且原始 `solve`/kernel 已归档；无对应题面时使用明确的 reference gate
- [ ] 性能：有 GFLOPS 或 GB/s
- [ ] 瓶颈：用 workload/roofline/profiler 证据区分 launch、memory、compute、latency/resource
- [ ] 面试：能讲 1 分钟优化过程

---

## M1.5：模型结构理论

目标：能对着一个最新开源模型 config 讲清结构。

### 分阶段任务表

| 任务 | 最小产出 | 验收 |
|------|---------|------|
| LLaMA/Qwen config 分析 | HF config 笔记 | 能讲 attention、norm、位置编码、FFN |
| DeepSeek MLA/MoE | 笔记 | 能手算 KV cache，能讲 expert 并行 |
| GPT/Claude/Gemini 趋势 | 一页趋势图 | 能区分公开事实和传闻 |
| Mamba/SSM | 对比笔记 | 能对比 Attention 和 SSM |
| 模型追踪表 | 更新状态 | 每个家族有结构记录 |

### 关键示例：KV cache 手算

```text
KV_bytes = 2 * num_kv_heads * head_dim * seq_len * num_layers * dtype_bytes
```

例：`seq=4096, layers=32, head_dim=128, fp16`

| 变体 | num_kv_heads | 计算 | 结果 |
|------|-------------|------|------|
| MHA | 32 | 2*32*128*4096*32*2 | 2 GiB |
| GQA(8) | 8 | 2*8*128*4096*32*2 | 512 MiB |
| MQA | 1 | 2*1*128*4096*32*2 | 64 MiB |

结论：Decode 阶段每 token 都要读全部历史 KV，KV 越小，吞吐越高。

---

## M2：Triton 实现阶段

目标：从 CUDA thread 级思维切换到 Triton block/tile 级思维，完成常见 ML 算子。

### 核心概念

| 概念 | 一句话 | 对应 CUDA |
|------|--------|-----------|
| `tl.program_id(0)` | 当前 block 编号 | `blockIdx.x` |
| `tl.arange(0, N)` | 生成索引向量 | 一个 block 的 thread 索引集合 |
| `tl.load/store` | 整块读写 | 手动线程循环 |
| `tl.dot` | 矩阵乘 | 手写 GEMM |
| `tl.constexpr` | 编译期常量 | 模板参数 |

调试配套课：[Lesson 07 — Triton Debugging](../lessons/07-triton-debugging.md)。先用最小 case、interpreter、打印和断言定位正确性，再使用 `compute-sanitizer` 查 GPU 内存问题，最后才做 GFLOPS/occupancy 调优。

### B1：Vector Add

目标：理解 grid、block、mask。

步骤：
1. `python -c "import triton, torch"` 确认环境。
2. 写 `add_kernel`：`pid -> offsets -> mask -> load -> add -> store`。
3. 用 `triton.cdiv(N, BLOCK_SIZE)` 计算 grid。
4. 和 `x + y` 对齐。

代码骨架：

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

完成定义：
- [x] LeetGPU Triton Vector Add 通过
- [x] 和 PyTorch 对齐（`assert_close`，N=1/256/257/1000/2^20）
- [x] AutoDL 实际 GPU benchmark：RTX 3090，840.1 GB/s
- [ ] 原始 LeetGPU `solve` 单独归档（当前仓库只有 wrapper）
- [ ] 能解释 mask 的作用

### B1（续）：MatMul

目标：写出 tiled GEMM。

步骤：
1. `pid_m`, `pid_k` 划分 C 的输出 tile。
2. 沿 N 归约维循环，`acc += tl.dot(a, b)`。
3. 记录 GFLOPS。
4. 调整 `BLOCK_M/BLOCK_N/BLOCK_K`。

```python
acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
for n in range(0, N, BLOCK_N):
    a = tl.load(a_ptrs)   # [BLOCK_M, BLOCK_N]
    b = tl.load(b_ptrs)   # [BLOCK_N, BLOCK_K]
    acc += tl.dot(a, b, input_precision='ieee')
```

当前单元卡（2026-08-28）：LeetGPU #02 已通过并归档为 [`solutions/triton/matmul_leetgpu.py`](../solutions/triton/matmul_leetgpu.py)，状态 `LEETGPU_PASS`；SuccessPublicTrace 为 A100-80GB、24.54 ms、55.3th percentile。服务器适配版 [`solutions/triton/matmul.py`](../solutions/triton/matmul.py) 已在 RTX 3090 `GPU_VALIDATED`。历史 [`solutions/triton/matmul_leetgpu_wip.py`](../solutions/triton/matmul_leetgpu_wip.py) 保留默认 TF32 失败案例（4×4 最大绝对误差 `0.1275177001953125`）。MatMul 单元总体为 `GPU_VALIDATED`，尚未 `COMPLETE`；下一步是 Nsight Compute / P0–P8 性能分析与最终口径。

完成定义：
- [x] LeetGPU #02 通过并归档原始 `solve`/kernel：[`solutions/triton/matmul_leetgpu.py`](../solutions/triton/matmul_leetgpu.py)，`LEETGPU_PASS`
- [x] 和 `A @ B` 对齐：LeetGPU SuccessPublicTrace
- [x] 服务器适配版 [`solutions/triton/matmul.py`](../solutions/triton/matmul.py) 在 RTX 3090 `GPU_VALIDATED`
- [x] 记录正确性和 GFLOPS：最佳 22.033 ms / 18,713.5 GFLOPS；`128×64×128` 因 shared memory 131,072 B > 101,376 B 编译失败
- [ ] Nsight Compute / P0–P8 性能分析与最终口径（完成后才可标记 `COMPLETE`）

### B3：Fused Softmax

目标：把 max/exp/sum/div 合成一个 kernel。

```python
x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
x_max = tl.max(x, axis=0)
num = tl.exp(x - x_max)
denom = tl.sum(num, axis=0)
y = num / denom
```

完成定义：
- [ ] 和 `torch.softmax` 对齐
- [ ] 能解释 max trick
- [ ] 记录提速

### B4：Flash Attention

目标：用 Triton 实现 tiling + online softmax。

关键状态：
```python
m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
l = tl.zeros((BLOCK_M,), dtype=tl.float32)
acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
```

循环内更新：
```text
m_new = max(m, rowmax(scores))
correction = exp(m - m_new)
l = l * correction + rowsum(exp(scores - m_new))
acc = acc * correction + exp(scores - m_new) @ V
m = m_new
```

完成定义：
- [ ] 与 PyTorch ref 对齐
- [ ] 记录显存和速度
- [ ] 能解释 causal 为什么省一半

### B5：GQA / Fused MLP

GQA：
- Q head 数 > KV head 数。
- 多组 Q head 共用一组 KV head。
- 实现时按 group 映射。

Fused MLP：
- gate/up/down 三个 Linear 合成一个 kernel。
- 减少中间激活写回 HBM。

完成定义：
- [ ] 正确性 + autotune
- [ ] 能讲 GQA 为什么省 KV cache
- [ ] 能讲 fused kernel 为什么快

### 常见坑

| 坑 | 解法 |
|----|------|
| `tl.arange` 非 2 的幂 | block size 用 2 的幂 |
| 忘了 mask | 所有 load/store 带 mask |
| max 用 0 填充 | max 用 -inf，sum 用 0 |
| 指针按字节偏移 | Triton 按元素数偏移 |
| `tl.dot` 形状不对 | 确认 tile 形状和 dtype |
| 只跑不验证 | 每个 kernel 对比 PyTorch |

---


## M2.5：最新模型与算子构建（并行强化）

> 内容入口：[大模型构建能力](../notes/llm/operator-building.md)
> 原则：不另起学习线，构建任务挂回 Triton 主线和模型结构/推理阶段。

### 构建任务

| 任务 | 最小产出 | 验收 |
|------|---------|------|
| RoPE | PyTorch/Triton 实现 | 和 HF 对齐 |
| RMSNorm | CUDA/Triton 实现 | 和 `torch.nn.RMSNorm` 对齐 |
| GQA | Triton attention | KV cache 尺寸符合手算 |
| SwiGLU / fused MLP | Triton 实现 | 和 `nn.Linear` 对齐 |
| MLA 简化版 | PyTorch 低秩 KV | 能手算节省显存 |
| MoE router + top-k | Python/Triton | 能画 token 到 expert 映射 |
| FlashAttention 1/2 | Triton 实现 | 和 PyTorch ref 对齐，记录显存/速度 |
| PagedAttention block table | Python 模拟 | 能画逻辑到物理映射 |
| 量化 scale/dequant | Python/Torch | 能手算 scale/zero_point |
| 投机解码 | Python 模拟 | 记录接受率和加速比 |
| SSM/Mamba 简化版 | Python scan | 能和 Attention 对比复杂度 |

### 完成标准

- [ ] 有代码
- [ ] 和 reference 对齐
- [ ] 有 GFLOPS / GB/s / 显存 / 误差
- [ ] 能讲 1 分钟构建思路
- [ ] 不是只读笔记，至少有一个简化实现

### 面试对应

- GQA/MLA：KV cache、Decode 带宽
- MoE：AllToAll、负载均衡
- FlashAttention：tiling、online softmax
- PagedAttention：block table、碎片
- 量化：scale、误差、kernel
- 投机解码：接受率、吞吐

## M3：推理系统

目标：能画 vLLM 完整链路，能解释核心机制。

### vLLM 链路

```text
请求 -> Scheduler -> Worker -> ModelRunner -> Attention -> KV Cache -> 返回
```

### 分阶段任务表

| 主题 | 核心问题 | 验收 |
|------|---------|------|
| Prefill vs Decode | 两阶段瓶颈为什么不同 | 能画计算/带宽对比 |
| PagedAttention | block table 怎么映射 | 能画逻辑到物理映射 |
| Continuous Batching | iteration 调度怎么提高利用率 | 能解释完成即退出 |
| Chunked Prefill | 怎么减少 prefill/decode 干扰 | 能画时间线 |
| Prefix Cache | 相同前缀怎么复用 | 能解释 hash/radix |
| Quantization | 权重/激活/KV 怎么量化 | 能对比 W4A16/W8A8 |
| Speculative Decoding | draft-verify 为什么快 | 能说接受率影响 |
| PD 分离 | 为什么拆 prefill/decode | 能说 KV 传输挑战 |

### 分阶段完成定义

- [ ] 读 `vllm/core/scheduler.py`
- [ ] 读 `vllm/attention/ops/paged_attn.py`
- [ ] 跑一次 vLLM benchmark
- [ ] 记录 TTFT / TPOT / throughput
- [ ] 能画一张请求生命周期图

---

## M4：训练系统 / 分布式训练

目标：能算训练显存、能画并行通信图、能跑最小 demo。

执行入口：[分布式训练基础](distributed.md) → [多机多卡专项](multi-node-multi-gpu.md)；实验使用 [统一执行系统](execution-system.md) 和 [分布式记录模板](../templates/distributed-record.md)。先单机 collective/并行语义，再进入多节点 NCCL/RDMA 和混合并行，不直接从框架参数跳到大集群。

### 训练显存手算

以 7B 模型、混合精度为例：

```text
FP16/BF16 参数：14 GB
FP32 master weight：28 GB
梯度：14 GB
Adam m：28 GB
Adam v：28 GB
合计约：112 GB
```

结论：所以大模型训练必须用 ZeRO/FSDP 或 Offload。

### 分阶段任务表

| 主题 | 最小产出 | 验收 |
|------|---------|------|
| NCCL/集合通信 | 通信原语图 | 能画 AllReduce/AllGather/ReduceScatter |
| DDP | 最小 demo | 能说梯度 AllReduce |
| FSDP/ZeRO | 显存账本 | 能说 Stage 1/2/3 分片什么 |
| TP | 单卡模拟 | 能画 Column/Row parallel |
| PP | 调度图 | 能讲 GPipe/1F1B |
| EP | MoE 通信 | 能画 AllToAll |
| 3D parallel | 拓扑设计 | 能设计 TP×PP×DP |

### 分阶段完成定义

- [ ] 手算 7B 显存
- [ ] 画 TP=2、PP=4、DP=4 通信图
- [ ] 有 GPU 时跑 DDP/FSDP demo
- [ ] 能解释 FSDP 为什么 AllGather + ReduceScatter

---

## M5：求职冲刺

目标：把仓库内容变成可讲项目、可答面试、可展示的 portfolio。

### 项目清单

| 项目 | 内容 | 展示 |
|------|------|------|
| GEMM | naive -> tiled -> fp16 | GFLOPS 提升 |
| Softmax | 3-pass -> online -> benchmark | GB/s |
| Triton GEMM/Softmax | 与 PyTorch 对比 | 正确性 + 提速 |
| Flash Attention | Triton 实现 | 显存/速度 |
| vLLM | 推理链路分析 | TTFT/TPOT |

### 面试准备

- 按 [interviews.md](interviews.md) 刷题。
- 每个题准备 1 分钟和 3 分钟版本。
- 系统设计题要能画图、算数字、说取舍。
- Ascend→GPU 叙事固定为“异构计算本质”。

### 分阶段完成定义

- [ ] 3-5 个可讲项目
- [ ] 每个项目有数字
- [ ] 高频题能讲清
- [ ] 能设计一个 LLM 推理服务

---

## 6. 建议推进节奏

| 周 | 焦点 | 最小产出 |
|----|------|---------|
| W1 | Triton vec add + matmul | 两个 kernel 跑通 |
| W2 | MatMul 极致调优 + fused softmax | 同精度强 baseline、roofline、BLOCK/warp/stage、Nsight 证据 |
| W3 | Triton Flash Attention | 正确性、显存、官方强 baseline、pipeline 分析 |
| W4 | causal + GQA/fused MLP | 正确性 + autotune |
| W5 | Triton benchmark 复盘 | 性能数字汇总 |
| W6 | A4/A5 背景收尾 | 1-pass 落盘或 A5 注释 |
| W7 | vLLM 入门 + PagedAttention | 跑一次 vLLM |
| W8 | Scheduler + continuous batching | 画调度循环 |
| W9 | 量化 + speculative decoding | 取舍表 |
| W10 | PD 分离 + benchmark | TTFT/TPOT |
| W11 | NCCL + DDP/FSDP | 显存账本 + demo |
| W12 | TP/PP/EP + 面试 | 通信图 + 高频题 |

---

## 7. 关键示例与已验证数字

### 7.1 KV cache

```text
KV = 2 * 8 * 128 * 4096 * 32 * 2 = 512 MiB
```

这是 GQA(8)、`seq=4096`、32 层、`head_dim=128`、FP16 的结果。

### 7.2 7B 训练显存

```text
14 + 28 + 14 + 28 + 28 = 112 GB
```

这是混合精度 + AdamW 的常见基线账本。

### 7.3 Ring AllReduce

```text
每 GPU 通信量 ≈ 2(N-1)/N * 模型大小
```

GPU 越多，每卡通信量不会按 GPU 数线性增长，这是 ring 的优势。

### 7.4 Flash Attention

```text
显存：O(N²) -> O(N)
HBM 读写：O(N²) -> O(N)
```

原因：不落完整 `QK^T` 矩阵，分块 + online softmax。

---

## 8. 外部参考

- [AIInfraGuide](https://github.com/caomaolufei/AIInfraGuide)
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Triton Language Docs](https://triton-lang.org/)
- [vLLM](https://github.com/vllm-project/vllm)
- [CUDA MODE Lectures](https://github.com/cuda-mode/lectures)

## 更新记录

- 2026-08-03：新增 PATH 执行参考。
- 2026-08-06：补全 M0-M5 全阶段计划，扩展 Triton、推理、训练、面试内容，加入关键数字。
