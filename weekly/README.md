# Weekly — 回顾周报

> **回顾型日志**，不是教程。每完成一个学习主题，记录实际发生了什么。
> 教程内容在 [lessons/](../lessons/)，进度在 [PATH.md](../PATH.md)，当前焦点在 [NOW.md](../NOW.md)。

## 这个文件夹放什么

拉模式下，每当你完成 [NOW.md](../NOW.md) 的一轮（学完一个主题 + 写完代码），我会写一篇回顾，记录：
- 实际做了什么、花了多久
- 卡在哪、怎么解决的
- 关键数据（GFLOPS、bandwidth、正确性）
- 对面试叙事有用的点

这些是**事后记录**，对求职叙事和复盘有用。和早期"预写未来周教程"不同——那些已经拆进 [lessons/](../lessons/) 了。

## 模板

回顾由我自动生成（你不用填）：你说"X 学完了" → 我读 `git log` 拿提交/修改时间和改动文件 → 按 [_template.md](_template.md) 的格式写一篇，命名 `YYYY-MM-DD-<主题>.md`（如 `2026-06-24-gemm-tiled.md`）。

## 已有回顾

| 日期 | 主题 | 对应 PATH |
|------|------|----------|
| [2026-06-29-week3-infra](2026-06-29-week3-infra.md) | Agent/Skills 搭建 + 内容打磨 | 全路线 |
| [2026-06-25-gemm-done](2026-06-25-gemm-done.md) | GEMM 完成 + 仓库重组 | A2 A3 |
| [2026-06-22-week2-gemm-naive](2026-06-22-week2-gemm-naive.md) | GEMM naive + tiled fp16 | A2 A3 |
| [2026-06-16-week1-cuda-basics](2026-06-16-week1-cuda-basics.md) | CUDA 基础 + Vector Add + 4090 配置 | A1 |

> 早期 week-03/04 的教程已拆进 [lessons/](../lessons/)（尚未开始学，无回顾记录）。
