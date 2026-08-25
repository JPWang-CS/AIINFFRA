---
name: "progress-resume"
description: "进度恢复技能 - 跨电脑/新会话启动时，基于 HISTORY.md、AGENTS.md、PATH.md 快速恢复上下文"
---

# Progress Resume Skill

## 用途

在切换电脑、开启新会话、或长时间未接触仓库时，先恢复当前进度和下一步，不重读整个仓库。

## 工作流程

### 1. 先读入口

```text
1. AGENTS.md       # 项目规则和文件地图
2. HISTORY.md      # 进度快照、最近变更、下一步
3. PATH.md         # 唯一进度权威源
4. NOW.md          # 当前焦点
5. roadmap/ai-infra-curriculum.md  # 总执行计划
```

### 2. 确认当前状态

```text
- 当前主线是什么？
- 已完成到哪一步？
- 待办池有哪些？
- 最近一次更新是什么时候？
- 工作区是否有未提交改动？
```

### 3. 输出恢复摘要

```text
📍 当前主线：[Triton 实现阶段]

✅ 已完成：
- [关键进度]

⏳ 待办：
- [下一步]

📄 关键文件：
- [文件链接]

🎯 建议下一步：
- [一个可执行任务]
```

## 用户明确要求更新进展时

1. 以 PATH.md 为权威源，先核对 HISTORY.md / NOW.md / 最近提交。
2. 同步 PATH.md、NOW.md、HISTORY.md，并检查相关 lesson、weekly、skill、agent memory 是否有过期状态。
3. 只把已验证的正确性、真实 GPU 型号、性能数字和代码归属写入“完成”；草稿单独标注。
4. 更新后运行 git diff --check，由用户要求时再提交和推送。

## 算子实验验收顺序

所有算子或实验必须先在 LeetGPU 完成正确性与平台验收，再上 AutoDL 等真实卡做 benchmark；LeetGPU 未通过时，不得把真实卡实验作为下一步。

## 规则

- 优先读 `HISTORY.md`，不要只靠对话记忆。
- 以 `PATH.md` 为进度权威源。
- 不要在没有用户确认时修改 `PATH.md` / `NOW.md`。
- 新会话默认从当前主线继续，不重排整个学习计划。

## 调用时机

- 用户说“换电脑了，继续”
- 用户说“看看当前进度”
- 新会话开始且需要上下文
- 用户说“下一步做什么”
