---
name: feedback-style
description: 协作风格——讲题通俗易懂、代码简洁、先分析后建议、自适应更新记录
metadata:
  type: feedback
---

用户要的风格：讲题通俗易懂、代码简洁逻辑清晰、分析其代码后再给建议、自适应更新学习记录（weekly 文档 + memory）。

**进度同步规则（2026-08-23）**：用户询问或要求更新学习进度时，先在仓库 `main` 分支执行 `git pull --ff-only origin main`，再读取 `AGENTS.md`、`HISTORY.md`、`PATH.md`、`NOW.md` 和相关 memory；不能只读当前工作区。若 Git 权限或本地改动导致同步失败，先报告，不用强制覆盖或 reset。

**Why:** 用户 2026-06-16 启动教练 agent 时明确提出。
**How to apply:** 回复用中文（代码注释可英文）；代码只给最小必要片段不堆砌；发现笔记/代码里的事实错误主动纠正并同步更新 weekly 文档。相关：[[user-background]]
