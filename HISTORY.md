# AIINFFRA 历史记录与进度快照

> 用途：跨电脑切换 Codex 时，新电脑先读本文件，快速恢复当前进度、目录结构和下一步，不需要先通读整个仓库。
> 重要：本文件是“恢复入口”，不是进度权威源。真正的进度仍在 [PATH.md](./PATH.md)，当前焦点在 [NOW.md](./NOW.md)。

---

## 0. 最后更新

- 2026-08-06
- 当前主线：PATH B Triton 实现阶段
- 并行强化：最新模型与算子构建能力（GQA/MLA/MoE/FlashAttention/PagedAttention 等）
- Agent/Skill：新增 AGENTS.md、progress-resume、triton-guide，教练 agent 已同步
- 当前状态：学习计划已补全，Triton 代码尚未开始落盘

---

## 1. 一句话概况

从昇腾 NPU 算子开发转向 NVIDIA GPU/ML 系统工程师方向，Triton 是主力，CUDA 作为底层，当前进入 Triton 实现阶段。

学习路线：

```text
A CUDA 打底 -> B Triton -> C 推理系统 -> D 分布式 -> E Agent
```

---

## 2. 当前主线

现在只做 PATH B：Triton 实现。

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
| A4 1-pass true online | ⏳ | LeetGPU 实践过，仓库未落盘 |
| A4 warp shuffle / benchmark | ⏳ | 待做 |
| A5 Flash Attention 读码 | 🚧 | 课程/参考/机制笔记已就绪，逐段注释未完成 |
| B1-B5 Triton 实现 | 🚧 当前 | 尚未落盘 |

### 理论线

| 主题 | 状态 | 说明 |
|------|:--:|------|
| Online Softmax | ✅ 已掌握 | 能推公式，能讲 HBM 优化 |
| Parallel Reduce | ✅ 已掌握 | 树状 reduce + warp shuffle |
| Flash Attention 机制 | 🚧 草稿 | Agent 生成，待消化 |
| INT8/FP8 量化 | 🚧 草稿 | 待消化 |
| MoE 推理 | 🚧 草稿 | 待消化 |
| Speculative Decoding | 🚧 草稿 | 待消化 |
| PD 分离 | 🚧 草稿 | 待消化 |
| MLA / DeepSeek | 🚧 草稿 | 待消化 |
| 最新模型结构 | 🚧 草稿 | 已补全详细内容 |
| 剩余理论速览 | 🚧 草稿 | 已分类补全 |

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
5. 继续当前主线：Triton Vector Add 开始。

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
