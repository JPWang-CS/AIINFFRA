# AIINFFRA 历史记录与进度快照

> 用途：跨电脑切换 Codex 时，新电脑先读本文件，快速恢复当前进度、目录结构和下一步，不需要先通读整个仓库。
> 重要：本文件是“恢复入口”，不是进度权威源。真正的进度仍在 [PATH.md](./PATH.md)，当前焦点在 [NOW.md](./NOW.md)。

---

## 0. 最后更新

- 2026-08-20：理论线完成 DeepSeek-V3.2 config + KV Cache / 权重显存 / FLOPs 三笔手算；下一步进入 FA2 → MLA → DSA。
- 2026-08-20（全盘校准：算子主线固定为 Triton B1，统一执行顺序为“自己写 → LeetGPU → 真实 GPU → benchmark”；纯 CUDA kernel 后置；新增最新论文/模型资料快照 `notes/llm/updates/2026-08-20.md`，同步 PATH/NOW/课程与构建路线）
- 2026-08-20（Triton Vector Add 由用户自己完成并通过 LeetGPU；B1 剩余真实 GPU 验证与 GB/s 记录）

- 2026-08-14（确认 DeepSeek-V4：2026-04-24 预览开源、07-31 Flash 正式、08-13 V4-Pro-0813 正式；核心 = CSA + HCA（MLA 骨架）+ Lightning Indexer + mHC/Muon + MXFP4；1M ctx 下 prefill ≈ V3.2 的 27%、KV ≈ 10%；主线不切 V4，改为 V3.2 打底、V4 做增量，挂主线 A 第 3 步；新笔记 deepseek-v4.md，tracker / 架构地图 / algorithms README / NOW 同步）
- 2026-08-13（理论线重构为"主干 + 枝干 + 字典"：主干=模型主线，枝干=必要小模块按挂载点学；主线 A 第 1 步手算是热身，A5/FA1 在第 2 步注意力接续（FA2 → MLA → DSA），训练侧枝干 A1 挪到 serving 之后；新笔记：FA2、FA4/FlexAttention、GDN/Qwen3.5、DSA、SageAttention3/Kascade、优化器 Adam/AdamW【枝干 A1】）
- 2026-08-10
- 当前主线：PATH B Triton 实现（B1 vec_add 待用户自己写，matmul 下一步）
- 并行强化：最新模型与算子构建能力（GQA/MLA/MoE/FlashAttention/PagedAttention 等）
- 当前状态：A5 读码完成；Triton Vector Add 已由用户自己写完并通过 LeetGPU，待真实 GPU benchmark；softmax_1pass 仍为后置 CUDA 草稿；本机 venv 已就绪（torch CPU + triton-windows）

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
| B1 Triton vec_add + matmul | 🚧 当前 | Vector Add 已由用户自己写完并通过 LeetGPU（2026-08-20）；真实 GPU benchmark 与 matmul 待做 |
| B2-B5 Triton 实现 | ⏳ | 待做 |

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
5. 算子线从 Triton Vector Add 开始；理论线按 [NOW.md](./NOW.md) 走主线 A（DeepSeek-V3.2 → V4 增量）。

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
