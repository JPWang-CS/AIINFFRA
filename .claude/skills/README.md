# AIINFFRA Skills

可复用的技能模块，供 Agent 调用。每个 skill 是一个独立的 `.md` 文件，包含：
- **name**: 技能标识符
- **description**: 何时使用
- **content**: 技能的具体实现

## 可用 Skills

| Skill | 用途 | 调用时机 |
|-------|------|---------|
| [code-review](./code-review.md) | 代码审查（正确性/性能/最佳实践） | 用户完成代码初版时 |
| [concept-explain](./concept-explain.md) | 概念解释（Ascend→CUDA 映射） | 用户问"什么是 XX"时 |
| [next-step](./next-step.md) | 路径规划（基于 PATH/NOW） | 用户问"接下来做什么"时 |
| [perf-analysis](./perf-analysis.md) | 性能分析（瓶颈诊断+优化建议） | 用户跑完 benchmark 时 |
| [interview-prep](./interview-prep.md) | 面试准备（叙事+题库） | 用户说"准备面试"时 |
| [theory-study](./theory-study.md) | 理论学习（6 子类指导） | 用户说"这周学 XXX"时 |
| [weekly-report](./weekly-report.md) | 周报生成（基于 git） | 用户说"生成周报"时 |

## 调用方式

### 方式一：通过 Agent
Agent 内部根据场景选择合适的 skill 调用。

### 方式二：直接调用
```
用户: /skill code-review
→ 直接调用 code-review.md
```

## 添加新 Skill

创建新文件 `[name].md`，格式：
```markdown
---
name: "skill-name"
description: "一句话描述用途"
---

# Skill Name

## 用途
[何时使用]

## 工作流程
[具体步骤]

## 输出格式
[预期输出]

## 调用时机
[何时触发]
```

## 设计原则

1. **单一职责** - 每个 skill 只做一件事
2. **可组合** - 多个 skill 可以组合使用
3. **独立调用** - 不依赖特定 agent 上下文
4. **清晰触发** - 调用时机明确
