# 论文与算法阅读器重建记录

日期：2026-10-14

## 已确认

- 论文与算法统一进入课程站同一阅读器，按八类组织：模型基础与完整结构；注意力、序列建模与长上下文；MoE 与条件计算；训练、优化与后训练；生成、推理与检索算法；量化与模型压缩；多模态与生成模型；高效计算与系统论文。
- 模型基础从六篇连续内容开始，ID 41–46：语言建模、Attention、Transformer Block、位置编码、完整 Decoder、Transformer 家族。其后 DeepSeek 整体结构 ID 30、31、39 归入同类。
- 既有 ID 20–40 和对应 URL 保持不变；新增主题使用 ID 47–60。课程站顶部论文入口从 25 改到 41，`NOW.md` 的 MLA 入口保持不变。
- 14 篇新增专题覆盖 Mamba、训练与 Scaling、LoRA、对齐/RL、采样/搜索、检索/记忆、GPTQ、AWQ、SmoothQuant、视觉语言、Diffusion/Flow Matching、Scan/Selection、蒸馏/压缩和长上下文。核心公式在正文逐步展开，代码片段对应公式，配小规模数值例并就近放机制图。
- 图文件和预览保存在 `papers/topics/figures/`；代码语义检查在 `papers/topics/examples/paper_checks.py`。检查使用 NumPy 与标准库；可选 `--torch` 只针对 DPO 核心梯度路径。
- 同一主题只在阅读器维护完整机制讲解。速读卡与正式教程分开；不设每篇一页限制，也不要求每个模型家族都有 benchmark 表。核心论文和重要最新论文须能解释动机、推导、shape、算法、作者实现、相关质量实验与限制。
- 论文索引、流程和模板已移除重复正文要求；图解位置与推导/代码相邻。学习进度仍由 PATH/NOW 管理，本次未改这两个文件。

## 提案

暂无待批准提案。新增主题内容依照以上已确认范围填充，不代表增加学习状态。

## 来源与实现边界

课程章节继续使用各自原始来源路径，`paper-catalog.cjs` 增加显式分类和分类内 `order`，不重排 ID 数组。新专题图片使用相对于原始 Markdown 来源的链接；本地 PDF 保留原文件与 SHA256 登记。DeepSeek-V4.1 报告的训练机制案例按本地来源说明，不复制报告正文。

A 提供的完整 Transformer CPU 验证记录保存在 `papers/foundations/examples/validation-2026-10-14.txt`。它记录源代码检查、NumPy 数值校验、梯度有限差分、缓存路径和完整 PyTorch CPU 训练/生成示例。GPU 执行与性能测试未运行。

最终 build 与全站复查收据见 `papers/topics/examples/validation-2026-10-14.txt`。59章 build、内容/编辑/导航/paper-reader/离线/主题检查和优化后的全站链接审计均通过；浏览器直接以 `file://` 打开第41章，并检查浅色与暗色显示。固定旧锚点基线记录在 `.codex/paper-legacy-anchors.json`，来源提交为 `10383c57ccc62cc1c081398d1651ef851d7c4470`。14张专题 PNG 已按对应 SVG 的内在尺寸重新生成。source index 与恢复记录已更新；GPU和性能执行未进行，学习进度未改。
