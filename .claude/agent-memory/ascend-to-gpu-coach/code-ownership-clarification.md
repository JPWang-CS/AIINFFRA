---
name: code-ownership-clarification
description: "Agent 产物 ≠ 用户进度：reference/cuda 的 .cu + notes/algorithms 的笔记都是我生成的素材，用户实掌握要另算"
metadata:
  type: project
---

**Rule（Agent 产物 ≠ 用户进度）**: 仓库里我(Agent)生成的东西都只是素材/参考，不等于用户已掌握。两类：
- **代码**：`reference/cuda/` 的 `.cu` 是参考（看不抄）；用户从零手写、跑通的 kernel 才进 `solutions/cuda/`。
- **理论笔记**：`notes/algorithms/` 的笔记多是我起草的草稿；用户「读过 + 能讲清」才算学过。

进度权威源：代码看 `solutions/` 跑通 + `weekly/*.md`；理论看用户明确说学过。`PATH.md`/`NOW.md` 是**计划源**，其 ✅ 若来自我生成的笔记，不代表用户已学。

**Why**: 用户 2026-07-01 明确指出「8 条笔记是你生成的，我只看了前两条，需要区分我看到还是你生成的」。同理 .cu 参考代码存在也不等于用户写过。

**How to apply**:
- 汇报进度时**必须分两栏**：「用户已学/已写」 vs 「Agent 已生成(待读/参考)」，绝不把笔记或参考代码的存在当成用户进度
- 理论线：截至 2026-07-01 用户实学 2 条（online softmax、parallel reduce），其余 6 条笔记（flash-attn 机制、MLA、INT8/FP8、MoE、PD 分离、投机解码）是待读草稿
- 辅导时引导用户从零手写代码，而非改我的参考
- 相关：[[project-progress]] [[feedback-style]]
