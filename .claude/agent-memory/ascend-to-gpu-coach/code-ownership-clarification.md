---
name: code-ownership-clarification
description: "仓库 .cu 文件是 Agent 参考代码，用户实际工作在 LeetGPU/4090 上手写并记录在 weekly md"
metadata:
  type: project
---

**Rule**: 仓库 `cuda-kernels/` 下的 `.cu` 文件是 Agent 写的参考实现。用户在 LeetGPU/4090 上从零手写自己的 kernel，进度记录在 `weekly/*.md` 和 `MAIN.md` 中。

**Why**: 用户的 memory 明确写明 ".cu files are Agent reference"。用户的学习方式是"自己写——仓库 .cu 是参考，从空文件开始"。

**How to apply**:
- 评估进度时以 MAIN.md + weekly/*.md 为准，不以 .cu 文件存在为准
- 辅导时引导用户从零手写，而非让用户阅读/修改 Agent 的参考代码
- 更新进度追踪时必须标注"用户已完成" vs "Agent 参考已有"
- 用户完成某个版本后，他们会在 weekly md 里记录并在 LeetGPU/4090 上跑通
