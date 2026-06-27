# Agent 架构重构建议

## 当前状态

```
.claude/agents/
├── ascend-to-gpu-coach.md  # 主教练（完整）
├── base-coach.md          # 基础教练
└── cuda-tutor.md          # CUDA 专项
```

## 建议架构

### Agent 层次结构

```
1. 主协调者（Orchestrator）
   └── ascend-to-gpu-coach（已有，需微调）

2. 专项 Agent
   ├── base-coach（已有）- 路径导航、资源指引
   ├── cuda-tutor（已有）- CUDA 算子开发
   ├── triton-guide（新增）- Triton 算子开发
   ├── theory-explainer（新增）- 理论线概念讲解
   ├── code-reviewer（新增）- 代码审查
   ├── perf-analyst（新增）- 性能分析
   └── interview-coach（新增）- 面试准备

3. 辅助 Agent
   └── weekly-reporter（新增）- 周报生成
```

## 新增 Agent 模板

### triton-guide.md
```markdown
---
name: "triton-guide"
description: "Triton 专项辅导 - 算子开发、autotuning、Triton vs CUDA"
model: opus
memory: project
---

[参考已创建的 .claude/skills/theory-study.md 中的 Triton 部分]
```

### theory-explainer.md
```markdown
---
name: "theory-explainer"
description: "理论线讲解 - 6 子类概念讲解，论文导读"
model: opus
memory: project
---

专注于理论线（GPU 优化算法/量化/注意力/架构/推理/训练）
概念讲解和论文导读。
```

### code-reviewer.md
```markdown
---
name: "code-reviewer"
description: "代码审查 Agent - 正确性/性能/最佳实践审查"
model: opus
memory: project
---

[参考 .claude/skills/code-review.md]
```

## 调用关系

```
用户请求
    ↓
ascend-to-gpu-coach（主协调）
    ↓
    ├─ 路径问题 → base-coach
    ├─ CUDA 开发 → cuda-tutor
    ├─ Triton 开发 → triton-guide
    ├─ 理论学习 → theory-explainer
    ├─ 代码审查 → code-reviewer
    ├─ 性能问题 → perf-analyst
    └─ 面试准备 → interview-coach
```

## 实施建议

### 方式一：手动创建（推荐）
1. 复制 `base-coach.md` 作为模板
2. 根据上述描述修改
3. 添加到 `agents/` 目录

### 方式二：通过 CLI 创建
```bash
# 需要特殊权限，用户需在 settings.local.json 中添加权限
```

## 与 SKILL 的关系

- **Agent**：处理复杂交互，有状态，多轮对话
- **Skill**：单一功能，无状态，可组合

Agent 内部可以调用多个 Skill：
```
cuda-tutor 调用：
- concept-explain（解释概念）
- code-review（审查代码）
- perf-analysis（分析性能）
```

## 下一步

1. 用户确认架构设计
2. 手动创建新增 Agent 文件
3. 更新 ascend-to-gpu-coach 中的协作部分
4. 测试调用流程
