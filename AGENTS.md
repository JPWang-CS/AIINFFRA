# AGENTS.md — AIINFFRA Agent 启动说明

> 任何 Codex / Claude 进入本仓库时，先读本文件。
> 快速恢复上下文：先读 [HISTORY.md](./HISTORY.md)，再读本文件的文件地图和规则。

---

## 全项目固定模型分工（强制）

本文件是 `D:\Desktop\Code\Learn\AIINFFRA` 的根 `AGENTS.md`。本节覆盖本仓库的全部目录、所有未来对话/任务以及所有 Codex、Claude 和其他 Agent；任何子目录说明都不得降低、绕过或静默改写本节规则。

1. 主 Agent 固定使用 `gpt-5.6-sol`（以下简称 Sol）。Sol 负责恢复上下文、理解目标、思考方案、拆解任务、识别风险、定义验收标准、审查执行证据、整合结果并作最终答复。
2. 每个对话开始时必须先应用本节路由规则。只要请求包含任何实际执行动作——包括检索或读取项目资料、运行命令、编辑文件、测试、benchmark、git 操作等——Sol 必须把边界清楚、可验收的执行任务委派给当前对话内的 worker subagent；worker 固定使用 `model=gpt-5.6-luna`、`reasoning_effort=max`。
3. 纯概念问答，或只需思考且无需工具和项目执行时，Sol 可以直接回答；这仍然算已应用本节路由规则。
4. Luna worker 只能修改委派时明确授权的范围；必须保留用户及其他 Agent 的并发改动，不得撤销、覆盖、重置或吞并无关修改。worker 必须执行必要验证并回报可核查证据（包括涉及的路径、命令和结果）；Sol 必须复核这些证据，承担最终判断和责任，责任不得转移给 worker。
5. 委派不得创建用户可见的新 task/thread；必须使用当前对话内的 subagent。
6. 如果 subagent 不可用，或 `gpt-5.6-luna` 与 `reasoning_effort=max` 的组合不可用，Sol 必须在继续执行前明确说明降级原因和范围，不得静默换模型或 reasoning effort。为避免阻塞，Sol 只能做维持任务推进所必需的最小本地执行，并在最终答复中说明降级及其影响。
7. 根 `AGENTS.md` 是本项目上述模型路由和协作边界的统一入口；进入本仓库的每个未来对话、任务和 Agent 都必须遵守它。

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
| `roadmap/gpu-foundations.md` | GPU 底层架构与全栈性能优化主课 |
| `roadmap/multi-node-multi-gpu.md` | 多机多卡专项路线 |
| `lessons/` | 主题课 |
| `notes/` | 知识笔记 |
| `notes/cuda/README.md` | GPU/CUDA 底层知识入口 |
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
5. 先读完原理，再直接去 LeetGPU 题目编辑器从空题面写；通过后必须把当次平台 `solve`/kernel 原样归档到 `solutions/`，并在对应 lesson、`PATH.md`、算子 README 建立“题目 → 代码 → 验证”索引。只有 wrapper 或 reference，不算保存了 LeetGPU 版本。
6. 每个可执行章节固定只有两个验收段：**LeetGPU：正确性与代码归档**、**服务器：真实性能**。不要把同一流程拆成多个重复的 Step 表；性能章节必须以前一章节通过为前置条件。
7. 一次只推进一个当前主线，不并行多个大计划。
8. 用户定节奏，不强制跳级。
9. 状态统一使用：`WIP`（编写中）、`LEETGPU_PASS`（平台通过且原始代码已归档）、`GPU_VALIDATED`（真实 GPU 已验证）、`COMPLETE`（所有归档与面试材料齐全）。未满足门槛不得写成完成。
10. 每个当前可执行单元的 lesson 必须有一张可直达的单元卡：题目入口、当前代码快照/路径、LeetGPU 状态、服务器状态、下一步；lesson 里的代码必须标明来源（本地文件 / LeetGPU 编辑器快照）和是否已同步，不能把两份 WIP 混写成同一版本。
11. GPU 硬件知识必须和算子优化绑定：写清“硬件机制 → 代码旋钮 → 预期 profiler 变化 → 实测”。MatMul、Softmax/Norm、FlashAttention、Fused MLP/GQA 是极致性能锚点，只有 `LEETGPU_PASS` 后才进入服务器 P0–P8 优化阶梯；不得用不同精度、单一有利 shape 或无证据的 autotune 数字冒充优化成果。

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
