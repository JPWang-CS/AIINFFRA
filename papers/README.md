# 论文资料与阅读入口

论文与算法的教学正文集中在[统一阅读器](../roadmap/curriculum/gpu/course-site/index.html?chapter=41)，按八个主题连续展开模型原理、推导、关键代码、数值例子、图解、研究结论和适用边界。既有 PDF、论文源笔记、作者实现位置与历史材料继续保留，作为正文的来源。

速读卡可以概括问题、贡献和阅读判断；正式教程按照机制需要展开，不设固定页数。每个技术主题在阅读器内讲清核心原理与实现映射，避免在 `papers/` 与 `notes/algorithms/` 重复维护两份同样的机制正文。学习状态仍以 `PATH.md` 为准。

- [论文阅读与资料维护流程](process.md)
- [论文主题教程模板](template.md)
- [2026 观察池](watchlist-2026.md)：可选的候选收集与追踪表
- 可选抓取命令（从仓库根目录运行）：`python scripts/research_watch.py --topic distributed --days 14 --write-inbox`

原有 attention、inference、training、compiler 等目录中的论文笔记继续作为来源材料，由阅读器按主题引用和解释。观察池与抓取脚本只用于发现候选，不更新 PATH/NOW，也不自动安排阅读任务。
