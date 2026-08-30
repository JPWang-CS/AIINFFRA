# AIINFFRA 历史记录与进度快照

> 用途：跨电脑切换 Codex 时，新电脑先读本文件，快速恢复当前进度、目录结构和下一步，不需要先通读整个仓库。
> 重要：本文件是“恢复入口”，不是进度权威源。真正的进度仍在 [PATH.md](./PATH.md)，当前焦点在 [NOW.md](./NOW.md)。

---

## 0. 最后更新

- 2026-08-30（用户决定 MatMul 先阶段性收口：LeetGPU `LEETGPU_PASS`、RTX 3090 `GPU_VALIDATED` baseline 已完成，当前最佳 20.830 ms / 19,794.1 GFLOPS / `torch.mm` 80.3%；Nsight Systems P0-lite 已归档。剩余 NCU counters、PTX/SASS、spill/occupancy、多 shape 回归和完整 P0–P8 极致优化转入 GPU 优化篇，不再阻塞主线；当前焦点切换为 B2 Triton Fused Softmax）
- 2026-08-30（Triton MatMul Nsight Systems P0-lite：完整归档 s3/s2 两份 raw log，并按 Grid=64×16、Block=256 从 trace 剔除 4 次 correctness，重算 60 次大 shape。s3 mean 21.208 ms、255 regs/thread、0.098 MB DymSMem；s2 mean 22.362 ms、255 regs/thread、0.049 MB，慢 5.44%。结论：降低 stages 虽将 shared memory 减半，但未解除 register bottleneck，pipeline 变浅反而退化；下一步缩小输出列 tile 验证 accumulator/register pressure。NCU counters 因 AutoDL 权限阻塞，证据边界保持 P0-lite）
- 2026-08-28（Triton MatMul LeetGPU `SuccessPublicTrace` 已归档：LeetGPU #02、Triton、A100-80GB、24.54 ms、55.3th percentile；最终版仅将 `tl.dot` 指定为 `input_precision='ieee'`，历史 WIP 保留默认 TF32 失败证据（4×4 最大绝对误差 0.1275177001953125）；服务器适配版已在 RTX 3090 `GPU_VALIDATED`，MatMul 单元总体 `GPU_VALIDATED` 但尚未 `COMPLETE`，下一步 Nsight Compute / P0–P8；Vector Add 原始 `solve` 归档缺口保持不变）
- 2026-08-28（保存进度：当前主线仍为 Triton MatMul；理论侧 FlashAttention-2 统一笔记 [notes/algorithms/flash-attention-2.md](./notes/algorithms/flash-attention-2.md) 用户阅读约 50%，状态仍为 WIP/🚧，未视为已读完或已掌握；下一步继续阅读统一笔记后半部分，结合公式与 Triton/CUDA 代码映射）
- 2026-08-26（进一步把硬件知识与算子优化绑定：确定 MatMul、Softmax/Norm、FlashAttention、Fused MLP/GQA 四类极致性能锚点，新增“硬件机制→代码旋钮→counter→实测”映射、P0–P8 优化阶梯、强 baseline/roof 与停止条件）
- 2026-08-26（Triton MatMul IEEE FP32 配置 sweep：RTX 3090 最佳为 BLOCK_M=128、BLOCK_N=32、BLOCK_K=256、w8、s3，22.033 ms / 18,713.5 GFLOPS，为 torch.mm 的 77.8%；128×64×128 因 shared memory 131,072 B 超过 101,376 B 上限而编译失败）
- 2026-08-26（新增 `notes/triton/matmul-performance-analysis.md`：记录 MatMul sweep 的硬件解释、资源失败原因、Nsight Compute P0、邻域搜索、L2 排布、TF32 对照和 autotune 后续顺序）
- 2026-08-26（GPU 底层架构与优化全盘纳入主计划：统一为 G0–G8 九层能力、L0–L11 实验梯和十层优化矩阵；按 B1/B2/B3/M3/M4 Just-in-Time 挂载，不改变当前 Triton MatMul 焦点）
- 2026-08-26（Triton MatMul 服务器适配版在 AutoDL RTX 3090 完成 4 组边界正确性测试；IEEE FP32 下 Triton 24.681 ms / 16,706 GFLOPS，torch.mm 17.120 ms / 24,083.3 GFLOPS，约为 69.4%；LeetGPU 页面仍无法运行，原始代码保持 WIP）
- 2026-08-26（全盘重构章节规则：统一 WIP → LEETGPU_PASS → GPU_VALIDATED → COMPLETE 状态；lesson 只保留 LeetGPU 与服务器两段；校准 A1/A3 代码产物错配、旧状态、运行环境和链接问题）
- 2026-08-26（复盘发现 MatMul lesson 内嵌的是用户本轮 LeetGPU 编辑器快照，而 `solutions/triton/matmul.py` 仍是另一份 WIP；已在章节标明来源和未同步状态，后续通过后以原始平台版本统一归档）
- 2026-08-26（全盘校准发现两处旧产物错配：A1 CUDA Vector Add 只有 Lesson 01 代码快照、没有本地归档；PATH 原 A3 的 float tiled 名称实际对应 fp16 文件，已改为 A3=fp16 已完成、A3+=float 计划项）
- 2026-08-26（按反馈把当前 MatMul LeetGPU 草稿代码快照直接放进 Lesson 06 的 5.5 章节；明确标注未通过，后续修正后继续更新快照）
- 2026-08-26（根据反馈简化 Lesson 06 MatMul 结构：删除无独立价值的“5.5 三步走”，改为“5.5 LeetGPU：正确性与代码归档”和“5.6 服务器：真实性能”两章；题目、当前代码、归档要求直接放进对应章节）
- 2026-08-26（补齐 LeetGPU 代码归档规则：学习计划必须一眼列出题目入口、通过后的原始 `solve`/kernel、本地 `solutions/` 文件和正确性/性能证据；发现 Vector Add 当前本地文件是 wrapper，LeetGPU 原始代码未单独归档，已明确标记缺失，不再把 wrapper 当作平台版本）
- 2026-08-26（用户继续编写 Triton MatMul LeetGPU 题：已写 M/K 输出 tile、FP32 accumulator、N 维归约循环和 `tl.dot` 框架；尚未通过。当前问题为 `offset/offs` 命名不一致、`tl.arange` 归约偏移写法、A/B load 边界 mask、C 的 masked store 与循环作用域；通过前不做本地同步或 AutoDL benchmark）
- 2026-08-24（全盘复盘并统一 B1 Vector Add 验收记录格式；纠正本次 AutoDL 实际卡型为 RTX 3090，并改为运行时动态记录 GPU 型号；修正 README、课程、路线图和恢复入口中的旧状态，当前焦点为 Triton MatMul）
- 2026-08-24（开始阅读 Triton MatMul；尚未创建或编写 `solutions/triton/matmul.py`，本日学习到此结束）
- 2026-08-23（AutoDL RTX 3090 完成 Triton Vector Add 真实 GPU 验收：正确性通过，Triton 840.1 GB/s，`torch.add` 843.0 GB/s；当前焦点切换到 Triton MatMul）
- 2026-08-20：理论线完成 DeepSeek-V3.2 config + KV Cache / 权重显存 / FLOPs 三笔手算；下一步进入 FA2 → MLA → DSA。
- 2026-08-20（全盘校准：算子主线固定为 Triton B1，统一执行顺序为“自己写 → LeetGPU → 真实 GPU → benchmark”；纯 CUDA kernel 后置；新增最新论文/模型资料快照 `notes/llm/updates/2026-08-20.md`，同步 PATH/NOW/课程与构建路线）
- 2026-08-20（Triton Vector Add 由用户自己完成并通过 LeetGPU；B1 剩余真实 GPU 验证与 GB/s 记录）

- 2026-08-14（确认 DeepSeek-V4：2026-04-24 预览开源、07-31 Flash 正式、08-13 V4-Pro-0813 正式；核心 = CSA + HCA（MLA 骨架）+ Lightning Indexer + mHC/Muon + MXFP4；1M ctx 下 prefill ≈ V3.2 的 27%、KV ≈ 10%；主线不切 V4，改为 V3.2 打底、V4 做增量，挂主线 A 第 3 步；新笔记 deepseek-v4.md，tracker / 架构地图 / algorithms README / NOW 同步）
- 2026-08-13（理论线重构为"主干 + 枝干 + 字典"：主干=模型主线，枝干=必要小模块按挂载点学；主线 A 第 1 步手算是热身，A5/FA1 在第 2 步注意力接续（FA2 → MLA → DSA），训练侧枝干 A1 挪到 serving 之后；新笔记：FA2、FA4/FlexAttention、GDN/Qwen3.5、DSA、SageAttention3/Kascade、优化器 Adam/AdamW【枝干 A1】）
- 2026-08-10
- 当前主线：PATH B Triton 实现（B1 vec_add 待用户自己写，matmul 下一步）
- 并行强化：最新模型与算子构建能力（GQA/MLA/MoE/FlashAttention/PagedAttention 等）
- 当前状态：A5 读码完成；Triton Vector Add 已由用户自己写完、通过 LeetGPU，并在 AutoDL RTX 3090 完成 benchmark（840.1 GB/s vs `torch.add` 843.0 GB/s）；当前进入 MatMul；softmax_1pass 仍为后置 CUDA 草稿；本机 venv 已就绪（torch CPU + triton-windows）

---

## 1. 一句话概况

从昇腾 NPU 算子开发转向 NVIDIA GPU/ML 系统工程师方向，Triton 是主力，CUDA 作为底层，当前进入 Triton 实现阶段。

学习路线：

```text
A CUDA 打底 -> B Triton -> C 推理系统 -> D 分布式 -> E Agent
```

---

## 2. 当前主线

算子线现在做 PATH B：Triton 实现；理论线并行推进主线 A（DeepSeek-V3.2 → V4 增量）。

完成顺序：

```text
1. Triton Vector Add
2. Triton MatMul
3. Triton Fused Softmax
4. Triton Flash Attention
5. Triton GQA / Fused MLP
```

代码落盘位置：`solutions/triton/`

详细任务和验收：

- [roadmap/ai-infra-curriculum.md](./roadmap/ai-infra-curriculum.md)
- [lessons/06-triton-intro.md](./lessons/06-triton-intro.md)
- [solutions/triton/README.md](./solutions/triton/README.md)

---

## 3. 已完成进度

### 算子线

| 阶段 | 状态 | 日期/说明 |
|------|:--:|------|
| A1 CUDA 基础 + Vector Add | ✅ | LeetGPU 跑通 |
| A2 GEMM naive | ✅ | 2026-06-16，LeetGPU `2_matrix_multiplication` |
| A2+ GEMM fp16 naive | ✅ | 2026-06-22 |
| A3 GEMM tiled | ✅ | LeetGPU 跑通 |
| A3+ GEMM fp16 tiled + benchmark | ✅ | 4090 实测 tiled 约 0.6x naive，L2/occupancy 结论 |
| A4 Softmax 3-pass | ✅ | 2026-07-01，LeetGPU `5_softmax` |
| A4 Softmax 2-pass fused online | ✅ | 2026-07-11，`softmax_online.cu` |
| A4 1-pass true online | 🚧 | Agent 草稿 `softmax_1pass.cu`（2026-08-09，算法模拟+编译通过），待用户重写 |
| A4 warp shuffle / benchmark | ⏳ | 待做 |
| A5 Flash Attention 读码 | ✅ | 2026-08-10，逐段注释完成，[阅读笔记](./notes/cuda/flash-attn-reading.md)，发现 2 个真实 bug |
| B1 Triton vec_add + matmul | `GPU_VALIDATED` 已阶段性收口 | Vector Add 技术验收完成（840.1 GB/s），但原始 LeetGPU `solve` 归档缺失；MatMul baseline 已完成，剩余 P0–P8 优化延期至 GPU 优化篇 |
| B2-B5 Triton 实现 | B2 当前 | B2 Fused Softmax 下一单元 |

### 存档：A4 Softmax 详情（2026-07-01 ~ 08-09）

> 课程：[Lesson 04](./lessons/04-softmax.md) · 周报：[2026-07-22](./weekly/2026-07-22-softmax-online.md)

| 优化 | 说明 | 状态 |
|------|------|:--:|
| 3-pass naive | findMax → countSum → normalize，~1ms | ✅ `softmax_naive.cu` 2026-07-01 |
| 2-pass fused online | 一趟出 (partial_max, partial_sum) → host merge → normalize | ✅ `softmax_online.cu` 2026-07-11 |
| true online 1-pass | per-thread K-element scan + tree reduce merge (m,s) pair → `maxSumkernel` | 🚧 Agent 草稿 `softmax_1pass.cu` 2026-08-09（算法模拟 + 编译通过），待用户重写 |
| warp shuffle reduce | `__shfl_down_sync` 替代 shared memory 归约 | ⏳ 暂缓（用户决定） |
| benchmark 对比 | 3-pass vs online vs warp shuffle → ncu 分析带宽 | ⏳ 暂缓（用户决定） |

**2026-07-10 实践要点**：

- per-thread online scan：逐元素维护 `(m, s)` pair，公式 `m_new=max(m,val), s=s·exp(m-m_new)+exp(val-m_new)`
- tree reduce merge 公式：`s_new = s_a·exp(m_a-m_new) + s_b·exp(m_b-m_new)`，满足交换律+结合律
- 哨兵 NaN：两个空线程 merge 时 `-inf - (-inf) = NaN` → `if (m_a == -INFINITY)` 跳过
- `__syncthreads()`：同 block 所有线程必须全部到达，否则死锁；不能提前 `return`
- Device 指针：kernel 写入的 device 指针不能在 host 直接读，必须 `cudaMemcpy`；`cudaMalloc` 用 `cudaFree`
- 性能陷阱：normalize 必须多 block 并行，单线程串行 N 个 `expf` 会崩到 60ms
- LeetGPU 通过方案：3-pass（`findMax_kernel` + `countSum_kernel` + `softmax_kernel`）~1ms baseline

LeetGPU `5_softmax` 贴 `solve()` 提交；服务器 `KERNEL=xxx.cu ./run.sh` 测精度 + 带宽（harness → `solutions/cuda/softmax/main.cu` + `run.sh`）。

### 理论线

| 主题 | 状态 | 说明 |
|------|:--:|------|
| Online Softmax | ✅ 已掌握 | 能推公式，能讲 HBM 优化 |
| Parallel Reduce | ✅ 已掌握 | 树状 reduce + warp shuffle |
| Flash Attention 机制 | ✅ FA1 已消化 | 2026-08-10 经 A5 读码 + 问答消化；FA2/3 待补 |
| INT8/FP8 量化 | 🚧 草稿 | 待消化 |
| MoE 推理 | 🚧 草稿 | 待消化 |
| Speculative Decoding | 🚧 草稿 | 待消化 |
| PD 分离 | 🚧 草稿 | 待消化 |
| MLA / DeepSeek | 🚧（DeepSeek-V3.2 config + 三笔手算 ✅ 2026-08-20；FA2 / MLA / DSA 待学） | 下一步：注意力主线 |
| 最新模型结构 | 🚧 草稿 | 已补全详细内容 |
| 剩余理论速览 | 🚧 草稿 | 已分类补全 |
| FA2 / FA4 / GDN / DSA / SageAttention3 | 🚧 草稿 2026-08-13 | 随主线步骤消化，不单独排队 |
| 优化器 Adam/AdamW | 🚧 草稿 2026-08-13 | 枝干 A1 第 1 段（主线 A serving 之后） |
| DeepSeek-V4（CSA+HCA） | 🚧 草稿 2026-08-14 | 主线 A 第 3 步：CSA/HCA → mHC/Muon → MXFP4/混合精度 |

---

## 4. 大模型板块

`notes/llm/` 是内容聚合板块，不独立维护进度。

| 文件 | 内容 |
|------|------|
| [notes/llm/README.md](./notes/llm/README.md) | 板块入口和 PATH 映射 |
| [notes/llm/architectures.md](./notes/llm/architectures.md) | 模型结构 |
| [notes/llm/inference-systems.md](./notes/llm/inference-systems.md) | 推理系统 |
| [notes/llm/training-systems.md](./notes/llm/training-systems.md) | 训练系统 |
| [notes/llm/interview.md](./notes/llm/interview.md) | 面试 |
| [notes/llm/papers.md](./notes/llm/papers.md) | 论文 |
| [notes/llm/operator-building.md](./notes/llm/operator-building.md) | 最新模型与算子构建路线 |

---

## 5. 学习计划结构

所有学习计划挂在 `roadmap/` 下，当前总入口：

```text
roadmap/ai-infra-curriculum.md   # 总执行计划 M0-M5
roadmap/vllm.md                  # 推理系统源码深挖
roadmap/distributed.md           # 分布式训练
roadmap/agents.md                # Agent 实验室
roadmap/interviews.md            # 面试
roadmap/leetgpu-ladder.md        # 可选 CUDA 深钻
```

---

## 6. 重要约束

- `PATH.md` 是唯一进度权威源。
- `NOW.md` 决定当前学什么。
- `notes/llm/` 是内容聚合，不是另一条学习线。
- 算子线写代码，理论线写笔记。
- 代码从空文件开始写，参考实现只用于对照。
- 每个学习单元至少要有：正确性、性能数字、可讲清的面试口径。
- 除非用户明确要求，不要修改 `NOW.md` 和 `PATH.md`。

---

## 7. 最近变更记录

| 日期 | 内容 |
|------|------|
| 2026-08-24 | 全盘复盘 B1 进度，统一 Vector Add 验收格式并清理旧的“benchmark 待做”记录；当前焦点为 Triton MatMul |
| 2026-08-23 | Triton Vector Add 在 AutoDL RTX 3090 完成真实 GPU 正确性与带宽 benchmark；当前进入 MatMul |
| 2026-08-14 | V4 资料入库（发布线 / CSA+HCA / mHC/Muon / MXFP4 / MegaMoE / TileLang / 磁盘 KV）；tracker、架构地图、algorithms README、NOW 同步；新增 [deepseek-v4.md](./notes/algorithms/deepseek-v4.md) 草稿 |
| 2026-08-10 | NOW.md 瘦身：已完成单元移入 HISTORY 存档，NOW 只留当前焦点 + 历史跳转 |
| 2026-08-10 | 校准 B 线流程：自己写 → LeetGPU → 真实 GPU → 性能分析 |
| 2026-08-10 | A5 读码完成（笔记 + 2 个 bug）；`vector_add.py` 为 Agent 草稿待用户重写；本机 venv 搭好 |
| 2026-08-08 | 修复 `lessons/02/03/05` 中公式写在代码块内的问题，公式恢复正常渲染 |
| 2026-08-09 | A4 1-pass true online：Agent 起草 `softmax_1pass.cu`（算法模拟+编译通过），待用户重写；跳过三版 benchmark |
| 2026-08-06 | 补全所有学习计划，当前主线切到 Triton 实现 |
| 2026-08-06 | 新增 AGENTS.md、progress-resume/triton-guide skill，优化 coach agent |
| 2026-08-06 | 新增最新模型与算子构建能力路线，接入 M2.5 |
| 2026-08-06 | 新增 `notes/llm/` 大模型内容板块 |
| 2026-08-06 | 新增 `solutions/triton/` Triton 代码落盘入口 |
| 2026-08-03 | 新增 PATH 执行参考，补模型结构与理论速览 |
| 2026-07-22 | Softmax 2-pass fused 记录与 A5 准备 |
| 2026-07-11 | 完成 `softmax_online.cu` |
| 2026-07-01 | Softmax 3-pass naive LeetGPU 跑通 |
| 2026-06-22 | GEMM fp16 naive/tiled 跑通 |
| 2026-06-16 | GEMM naive 跑通 |

---

## 8. 新电脑恢复步骤

1. 读本文件，恢复上下文。
2. 读 [roadmap/ai-infra-curriculum.md](./roadmap/ai-infra-curriculum.md) 当前主线。
3. 读 [NOW.md](./NOW.md) 和 [PATH.md](./PATH.md) 确认最新状态。
4. 检查 `git status` 和 `git diff`，确认是否有未提交改动。
5. 算子线从当前 [NOW.md](./NOW.md) 的 Triton MatMul 开始；理论线按主线 A（DeepSeek-V3.2 → V4 增量）。

---

## 9. 关键文件速查

| 目的 | 文件 |
|------|------|
| 当前学什么 | [NOW.md](./NOW.md) |
| 权威进度 | [PATH.md](./PATH.md) |
| 总学习计划 | [roadmap/ai-infra-curriculum.md](./roadmap/ai-infra-curriculum.md) |
| Triton 课程 | [lessons/06-triton-intro.md](./lessons/06-triton-intro.md) |
| Triton 代码位置 | [solutions/triton/](./solutions/triton/) |
| 大模型板块 | [notes/llm/README.md](./notes/llm/README.md) |
| 理论线 | [notes/algorithms/README.md](./notes/algorithms/README.md) |
| 面试 | [roadmap/interviews.md](./roadmap/interviews.md) |
