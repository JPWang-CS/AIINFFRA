# AIINFFRA Memory System

持久化记忆系统，用于跨 session 保持项目上下文。

## 目录结构

```
memory/
├── user/       # 用户背景、偏好、知识
├── feedback/   # 用户反馈（正确/避免的做法）
├── project/    # 项目状态、里程碑
└── reference/  # 外部资源指针
```

## Memory 类型

### user（用户信息）
记录用户角色、目标、知识水平。

**示例**：
- `background.md` - Ascend C 背景，转型目标
- `learning-style.md` - 偏好的学习方式
- `expertise.md` - 技术栈掌握情况

### feedback（反馈）
记录"这样做/不要这样做"的指导。

**示例**：
- `code-style.md` - 代码风格偏好
- `interaction.md` - 交互方式偏好

**格式**：
```markdown
---
name: feedback-name
description: 一句话总结
metadata:
  type: feedback
---

**规则**: [具体规则]
**Why**: [用户原因]
**How to apply**: [何时应用]
```

### project（项目信息）
记录项目状态、决策、进度。

**示例**：
- `progress.md` - 当前进度（权威源是 PATH.md）
- `decisions.md` - 关键决策记录
- `milestones.md` - 里程碑

### reference（外部资源）
记录外部系统的资源位置。

**示例**：
- `leetcode.md` - 相关 LeetCode 题目
- `papers.md` - 论文资源库

## 如何使用

### Agent 读取
在对话开始时，Agent 读取 `MEMORY.md` 索引。

### Agent 写入
发现值得记录的信息时：
1. 写入具体 memory 文件
2. 更新 `MEMORY.md` 索引

## MEMORY.md 索引格式

```markdown
# Memory Index

## User
- [Title](file.md) — hook

## Feedback
- [Title](file.md) — hook

## Project
- [Title](file.md) — hook

## Reference
- [Title](file.md) — hook
```

## 规则

- **不重复**：先检查是否有现有 memory
- **可更新**：信息变化时更新而非新建
- **要过期清理**：删除过时的 memory
- **不存可推导的**：代码结构、git 历史等
