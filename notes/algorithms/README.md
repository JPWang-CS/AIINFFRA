# 论文与算法阅读入口

论文线独立于实践线，围绕模型与系统机制开展阅读、公式推导和作者关键代码分析；每篇不要求实现。学习进度以 [PATH.md](../../PATH.md) 为准，当前主题以 [NOW.md](../../NOW.md) 为准。

[打开统一论文与算法阅读器](../../roadmap/curriculum/gpu/course-site/index.html?chapter=41)。阅读器在同一套侧栏、正文和搜索中按八个主题组织 41 篇论文与算法，从语言模型和完整 Transformer 原理开始，保留既有 ID 与 URL。

| 主题 | 内容范围 | 阅读器入口 |
|---|---|---|
| 模型基础与完整结构 | 语言建模、Attention、Transformer Block、位置编码、Decoder、架构家族与 DeepSeek 整体结构 | [41：语言模型基础](../../roadmap/curriculum/gpu/course-site/index.html?chapter=41) |
| 注意力、序列建模与长上下文 | GQA、MLA、DSA、Mamba、Gated DeltaNet 与长上下文 | [24：GQA](../../roadmap/curriculum/gpu/course-site/index.html?chapter=24) |
| MoE 与条件计算 | 专家路由、dispatch、专家计算与合并 | [32：MoE](../../roadmap/curriculum/gpu/course-site/index.html?chapter=32) |
| 训练、优化与后训练 | 交叉熵、优化器、Scaling、LoRA、DPO/PPO/GRPO 与后训练工作负载 | [48：训练目标与 Scaling](../../roadmap/curriculum/gpu/course-site/index.html?chapter=48) |
| 生成、推理与检索算法 | 采样、搜索、Speculative Decoding、RAG 与记忆 | [51：采样与搜索](../../roadmap/curriculum/gpu/course-site/index.html?chapter=51) |
| 量化与模型压缩 | INT8/FP8、GPTQ、AWQ、SmoothQuant、蒸馏与剪枝 | [33：INT8 与 FP8](../../roadmap/curriculum/gpu/course-site/index.html?chapter=33) |
| 多模态与生成模型 | ViT、CLIP、视觉条件生成、Diffusion 与 Flow Matching | [56：视觉语言模型](../../roadmap/curriculum/gpu/course-site/index.html?chapter=56) |
| 高效计算与系统论文 | Online Softmax、Parallel Reduce、FlashAttention、可编程注意力、PD、PagedAttention、ZeRO、Scan/Selection | [20：Online Softmax](../../roadmap/curriculum/gpu/course-site/index.html?chapter=20) |

## 阅读深度

速读建立问题、贡献、关键公式和适用范围的概览。正式精读围绕具体主题展开推导、符号与张量维度、算法数据流、对应代码、数值例子、相关图解、实验口径和局限。教程篇幅由机制所需解释决定；实验数字在支撑质量或性能结论时按原论文条件阅读，不要求每篇架构论文都提供吞吐比较。

论文源文件和历史笔记保留在 `notes/algorithms/`、`papers/` 与 `roadmap/curriculum/papers/`。站点的来源、分类和显示次序由 `roadmap/curriculum/gpu/course-site/paper-catalog.cjs` 维护，稳定锚点登记在 `.codex/course-section-anchors.json`；这些索引不记录学习进度。

最新论文可进入[观察池](../../papers/watchlist-2026.md)，筛选与可选抓取流程见 [papers/process.md](../../papers/process.md)。本地 PDF 的文件校验由课程维护脚本管理；正文只在相关机制处引用材料。
