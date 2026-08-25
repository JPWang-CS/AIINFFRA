# AGENTS.md — AIINFFRA Agent 启动说明

> 任何 Codex / Claude 进入本仓库时，先读本文件。
> 快速恢复上下文：先读 [HISTORY.md](./HISTORY.md)，再读本文件的文件地图和规则。

---

## 项目一句话

从昇腾 NPU 算子开发转向 NVIDIA GPU/ML 系统工程师方向。

- 当前主线：PATH B Triton 实现阶段
- 方向：Triton 为主力，CUDA 作为底层
- 最终目标：ML 系统工程师

---

## 启动读取顺序

1. [AGENTS.md](./AGENTS.md) — 本文件，规则和文件地图
2. [HISTORY.md](./HISTORY.md) — 跨电脑恢复、进度快照、最近变更
3. [PATH.md](./PATH.md) — 唯一进度权威源
4. [NOW.md](./NOW.md) — 当前焦点
5. [roadmap/ai-infra-curriculum.md](./roadmap/ai-infra-curriculum.md) — 总执行计划
6. [notes/llm/README.md](./notes/llm/README.md) — 大模型内容板块

---

## 当前主线

```text
Triton Vector Add
-> Triton MatMul
-> Triton Fused Softmax
-> Triton Flash Attention
-> Triton GQA / Fused MLP

构建能力强化：RoPE / RMSNorm / GQA / MLA / MoE / FlashAttention / PagedAttention / 量化 / 投机解码
```

代码位置：`solutions/triton/`

任务详情：

- [roadmap/ai-infra-curriculum.md](./roadmap/ai-infra-curriculum.md)
- [lessons/06-triton-intro.md](./lessons/06-triton-intro.md)
- [solutions/triton/README.md](./solutions/triton/README.md)

---

## 文件地图

| 区域 | 用途 |
|------|------|
| `PATH.md` | 唯一进度权威源 |
| `NOW.md` | 当前学什么 |
| `HISTORY.md` | 跨电脑恢复、历史记录 |
| `roadmap/` | 学习计划 |
| `roadmap/execution-system.md` | 统一学习与实验流程 |
| `roadmap/multi-node-multi-gpu.md` | 多机多卡专项路线 |
| `lessons/` | 主题课 |
| `notes/` | 知识笔记 |
| `notes/llm/` | 大模型内容聚合板块 |
| `notes/llm/operator-building.md` | 最新模型与算子构建能力路线 |
| `solutions/` | 自己写的代码 |
| `reference/` | 参考实现，不直接复制 |
| `weekly/` | 回顾周报 |
| `.claude/agents/` | Claude Agent 配置 |
| `.claude/skills/` | Claude Skills |
| `templates/` | 算子、系统和分布式实验记录模板 |
| `.codex/agents/` | Codex Agent 配置 |

---

## 核心规则

1. `PATH.md` 是唯一进度权威源。
2. `NOW.md` 决定当前焦点。
3. 不要默认修改 `PATH.md` / `NOW.md`，除非用户明确要求。
4. `notes/llm/` 是内容聚合，不是另一条学习线。
5. 先读完原理，再直接去 LeetGPU 题目编辑器从空题面写；LeetGPU 通过后必须把当次平台 `solve`/kernel 原样归档到 `solutions/`，并在对应 lesson、`PATH.md`、算子 README 建立“题目 → 代码 → 验证”索引，再做真实 GPU benchmark。只有 wrapper 或 reference，不算保存了 LeetGPU 版本。
6. 每个学习单元至少留下：正确性、性能数字、可讲清的面试口径；算子/实验固定遵循“看原理 → LeetGPU 写题并通过 → 同步本地 → 真实 GPU benchmark → 性能分析”。
7. 一次只推进一个当前主线，不并行多个大计划。
8. 用户定节奏，不强制跳级。

---

## Agent 使用建议

| 场景 | 使用 |
|------|------|
| 问下一步 | 读 `PATH/NOW/HISTORY`，参考 `roadmap/ai-infra-curriculum.md` |
| Triton 实现 | 参考 `solutions/triton/` 和 `lessons/06` |
| 代码审查 | 使用 `code-review` skill |
| 概念解释 | 使用 `concept-explain` skill，Ascend→CUDA 映射 |
| 性能分析 | 使用 `perf-analysis` skill |
| 理论线 | 使用 `theory-study` skill，产出 `notes/algorithms/` 笔记 |
| 面试 | 使用 `interview-prep` skill，参考 `roadmap/interviews.md` |
| 周报 | 使用 `weekly-report` skill |
| 跨电脑恢复 | 使用 `progress-resume` skill，读 `HISTORY.md` |
| Triton 实现指导 | 使用 `triton-guide` skill |
| 最新算子构建 | 使用 `operator-building` skill，参考 `notes/llm/operator-building.md` |

---

## 完成任务后的动作

1. 确认正确性：和 PyTorch / reference 对齐。
2. 记录性能：GFLOPS、GB/s、耗时或显存。
3. 写笔记：`notes/` 或 `notes/llm/`。
4. 生成/更新周报：`weekly/`。
5. 只有用户要求时才更新 `PATH.md` / `NOW.md`。
