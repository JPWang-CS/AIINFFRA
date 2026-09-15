# AGENTS.md — AIINFFRA Agent 启动说明

> 任何 Codex / Claude 进入本仓库时，先读本文件。
> 快速恢复上下文：先读 [HISTORY.md](./HISTORY.md)，再读本文件的文件地图和规则。
> Codex 主 Agent 当前使用 GPT-6 Astra；模型、自治和输出规则按 OpenAI 官方 model guidance 维护。
> 参考：[GPT-6 Astra 模型规格](https://developers.openai.com/api/docs/models/gpt-6-astra) · [Model guidance](https://developers.openai.com/api/docs/guides/latest-model)

---

## 全项目模型与协作规则（强制）

本文件是 `D:\Desktop\Code\Learn\AIINFFRA` 的根 `AGENTS.md`。本节覆盖本仓库的全部目录、所有未来对话/任务以及所有 Codex、Claude 和其他 Agent；任何子目录说明都不得降低、绕过或静默改写本节规则。

1. 项目主 Agent 默认使用 `gpt-6-astra`（以下简称 Astra）。Astra 负责恢复上下文、理解目标、方案设计、任务拆解、风险判断、验收标准、执行证据审查、结果整合和最终答复。
2. Astra 可以直接完成边界明确的只读规划/复核，例如读取必要文件片段、运行 `git status`/`git diff`、核对 worker 证据；纯概念问答直接回答。只读/分析请求不授权修改。
3. 具体实现、文件修改、测试、benchmark 以及外部/远端动作，默认委派 `gpt-5.6-luna` worker（subagent）；worker 的具体角色遵守本项目后续领域路由。`commit`/`push` 等远端写入仍需用户明确授权。
4. `reasoning_effort` 按任务动态分级：`low` 用于简单读取、检索、status/diff、链接检查和机械验证；`medium`（默认）用于常规文档或代码修改、常规测试和普通 Git 工作流；`high` 用于多文件实现、一般调试和代码审查；`xhigh` 用于复杂性能优化、疑难 bug 和重要架构设计；最困难且质量优先的任务才使用 Astra 支持的 `max`，必须说明理由，`max` 不得作为默认值。
5. 每个用户请求默认最多一个 worker（不是绝对上限）；同一任务优先通过 `send_input` 复用。只有任务可独立并行、写入范围互斥且能显著降低延迟时才增加 worker。
6. 不为验证、补引用等自然后续另开 worker，由原 worker 收尾。worker 报告保持简洁，只给结论、路径、命令/测试证据和未验证项。
7. 失败止损：连续 120–180 秒无有效进展先检查状态；不得启动重叠写范围替代 worker；最多一次有依据的重试或接管。机制不可用时，Astra 必须显式说明并做最小必要执行。
8. 不创建用户可见的新 task/thread；项目领域专用 agent 路由继续保留，通用规则只约束模型、effort、worker 数量、复用、授权和变更保护。
9. worker 只能修改委派时明确授权的范围，必须保留用户及其他 Agent 的并发改动；所有 worker 完成后返回可核查证据，由 Astra 复核并承担最终责任。
10. 根 `AGENTS.md` 是本项目统一入口；worker 机制或指定模型不可用时，按第 7 条显式说明降级原因、范围和影响。

---

## Astra 适配规则（强制）

- 用户请求带有行动意图时，在已授权范围内直接执行并持续到目标完成；不要停在“可以做”或只给计划。
- 在提出澄清问题前，先完成上下文中已经授权且必要的可逆、只读和修复动作；只有当答案会实质改变结果，或动作不可逆/破坏性/需要外部确认时，才请求用户决定。
- 用户指令优先于 skill 指南；如果某个 skill 使任务暂停、要求额外许可或偏离目标，说明触发它的确切 `SKILL.md` 路径和相关指令。
- 审计当前可见的 `AGENTS.md`、skill 和 Agent 配置，发现冲突时以用户指令、本文件和当前任务范围为准；不把历史草稿或 Agent 建议当作用户已完成事实。
- 输出先给结论，再给必要证据。使用简洁、直接的中文和项目术语；只在信息确实平行、顺序明确或便于比较时使用列表/表格，避免套话、虚构的复合标签和无关展开。
- 验证强度与变更风险匹配：运行必要且有意义的检查；低影响、可逆的文档或配置改动不扩展成无关的大范围测试。

---

## 项目一句话

从昇腾 NPU 算子开发转向 NVIDIA GPU/ML 系统工程师方向。

- 当前实践动作：Triton Softmax 服务器闭环，随后进入 GPU 架构主课与完整算子路线
- 方向：从 Ascend NPU 迁移到 NVIDIA GPU；CUDA/Triton 与算子极致性能是主干
- 最终目标：GPU 高性能 LLM 算子与性能优化工程师；vLLM 是系统落地加分项

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
实践线：NPU→GPU 架构与性能工程 → 完整 GPU/LLM 算子体系 → 核心算子极致优化 → 低精度/量化 → Prefill/Decode GPU 分析 → Mini Transformer 验证 → vLLM 落地 → 多 GPU/MoE。

论文线完全单列：经典论文、关键公式与作者关键代码、最新论文持续阅读；不要求每篇实践。
```

代码位置：`solutions/triton/`

任务详情：

- [roadmap/curriculum/README.md](./roadmap/curriculum/README.md)
- [lessons/06-triton-intro.md](./lessons/06-triton-intro.md)
- [solutions/triton/README.md](./solutions/triton/README.md)

---

## 文件地图

| 区域 | 用途 |
|------|------|
| `PATH.md` | 唯一进度权威源 |
| `NOW.md` | 当前学什么 |
| `HISTORY.md` | 跨电脑恢复、历史记录 |
| `roadmap/curriculum/` | 模块化正式课程；按篇/章/节路由 |
| `roadmap/` | 专项路线与兼容入口 |
| `roadmap/execution-system.md` | 统一学习与实验流程 |
| `roadmap/gpu-foundations.md` | GPU课程兼容入口，正文路由到`roadmap/curriculum/gpu/` |
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
7. 实践线与论文线独立存在；每条线一次只推进一个当前单元，不在同一条线并行多个大计划。
8. 用户定节奏，不强制跳级。
9. 状态统一使用：`WIP`（编写中）、`LEETGPU_PASS`（平台通过且原始代码已归档）、`GPU_VALIDATED`（真实 GPU 已验证）、`COMPLETE`（所有归档与面试材料齐全）。未满足门槛不得写成完成。
10. 每个当前可执行单元的 lesson 必须有一张可直达的单元卡：题目入口、当前代码快照/路径、LeetGPU 状态、服务器状态、下一步；lesson 里的代码必须标明来源（本地文件 / LeetGPU 编辑器快照）和是否已同步，不能把两份 WIP 混写成同一版本。
11. GPU 硬件知识必须和算子优化绑定：写清“硬件机制 → 代码旋钮 → 预期 profiler 变化 → 实测”。极致性能锚点覆盖 GEMM、Reduction/Norm、Fused MLP、Prefill Attention、Decode/PagedAttention、Quantized GEMM、MoE Grouped GEMM；有平台题面时只有 `LEETGPU_PASS` 后才进入服务器完整优化阶段。不得用不同精度、单一有利 shape 或无证据的 autotune 数字冒充优化成果。
12. 主动收集用户代码及对话心得，整理并纳入适当课程正文，不要求用户逐次提醒；同时保留代码原始权威文件，引用并解释典型代码片段、性能证据和失败案例。每条材料注明是用户原话还是 Agent 整理，标明来源与验证状态；不能把误解写成正确结论，也不能捏造用户感想。讨论按日期归档，分为已确认、提案和待定，并与课程正文分离；总路由提供讨论记录链接。该规则长期有效。
13. 课程网页与发布的 Markdown 必须是纯教程：正文保留概念、技术条件、代码来源、运行命令和验证指导，不写 WIP、等待验收、Agent、本轮、旧稿待讨论、缺口或其他协作过程话语。学习状态仍以 `PATH.md` 为唯一权威源，当前焦点由 `NOW.md` 维护；编写过程和验证边界写入 `HISTORY.md` 或日期决策记录，平台成绩保留在原始实验记录。用于教学的真实性能案例可引用，但必须带测量条件，不能把未运行的示例写成实测成果。尚无纯正文的章节不得被课程站侧栏或文章列表暴露；原始旧文件继续保留。

---

## 课程连续阅读规则

- CUDA/Triton 代码先随原理写入教程，保留完整程序、环境要求、正确性检查和计时命令；真实 GPU 执行按用户学到对应课时现场完成，不作为教材补写的阻塞条件。未执行的代码不得记为 GPU 验证通过，实际运行记录仍放在进度与实验文件。
- 全课程使用直接的工程讲解：先说明计算对象和条件，再展开机制、公式、代码与例子。删除写作计划式开场、口号、反复的“真正掌握”评价与协作过程话语；技术限制、数值合同和反例保留，避免机械删除“不能”而损坏正确性。

- 本地教材的知识覆盖维护于 .codex/pdf-coverage.json，区别详细解释、部分覆盖和未融入；追加内容时核对原文与对应正文，不以资料链接或章节数量代替覆盖。原 PDF 的 SHA256 由 .codex/check-pdf-coverage.cjs 检查。该表是维护记录，不加入网页“序”、不代表用户学习进度。

- 论文与算法使用同一课程阅读器，按“主题分类 → 论文 → 小节”多级折叠收录，不另开独立笔记窗口或设置汇总序章。论文线与实践线仍独立推进；公式应解释符号、维度与推导，关键代码在当前正文内逐项对应，不能只给文件入口。

- 前台不单列“序”、学习地图或信息汇总课。按类学习的目录由顶部和侧栏提供，分类入口直接进入实际内容。仓库索引可保留作维护用途，不加入正文或前后章学习顺序。

- 每个算子在自身章节内连续展开原理、实现、正确性、基线及深入优化；不得拆出必须来回跳转的独立优化学习线。性能目录只作可选实验检查表。
- 正文不得用“本章接着某章/先去看某文件”替代必要解释。算子讲解后直接接同页实践，题目、关键代码、正确性与性能对比放在一起。
- 教程正文不链接源码文件；关键片段直接展示，原文件和精确来源保留给维护使用。运行命令与 LeetGPU 题目入口保留，不能把文件跳转替代讲解。
- PDF 不逐处标页码、章节位置；只在需要识别方法时简要说明来源，章末保留少量参考入口。精确定位与源码引用维护于 .codex/course-source-index.json，原样代码通过隐藏 source-check 校验。

## Agent 使用建议

引用旧课和用户代码时必须完成内容融合：当前教程内直接展示理解该主题所需的代码/关键片段、公式、推导和讨论结论，不能用“见某个md”替代讲解。源码文件作为维护依据，不在正文设置跳转；原样片段与教学改写明确区分，代码同步检查不得通过修改用户原文件来满足。

课程学习风格以用户指定的AIInfraGuide学习路线正文为参考：先全景与学习顺序，再按主题展开知识点、连续讲解、代码/图解、推荐资料与掌握标准。参考的是教程内页，不是宣传首页；不复制第三方长段落或未经核实的硬件说法。深度课保留完整推导与用户实践，不得为了格式统一缩成清单。网页使用清晰的文档层级，正文目录和重点标注服务阅读，不承载学习状态。

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

### 可选 external worker 路由

- Astra 无需用户点名，根据任务自动在 Luna 与 DeepSeek 中二选一：DeepSeek 用于独立、边界清楚、非敏感的长文档/代码分析或第二审查；Luna 用于原生 MCP/thread coordination、敏感上下文和远程写入。
- 选 DeepSeek 时直接调用已安装个人 Codex plugin `deepseek-subagent` 暴露的 MCP tools：`spawn_deepseek_subagent` + `wait_deepseek_subagent`；默认只启动一个 job，并等待同一个 `job_id`。逻辑名 `cc-switch/deepseek:max` 映射为 `deepseek-v4-pro + model_reasoning_effort=high`；`flash` 仅用于明确要求的低成本任务。
- DeepSeek 默认只读；只有用户明确授权具体文件修改时才使用 workspace-write。认证由 plugin 调用的本机 Codex CLI `--profile deepseek` 内部处理；API key 可以由该内部认证链读取，但不得暴露给 Astra、worker、prompt、command-line args、MCP 返回或日志，也不得让 worker 检查 secret/config 文件。
- MCP tools 不可用或 provider 调用失败一次时，视为 DeepSeek 暂不可用并回退 Luna；不做外层重试。旧全局 `cc-switch-deepseek-worker` skill/launcher 仅作为该回退路径的兼容说明。
