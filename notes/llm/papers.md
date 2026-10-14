# 大模型论文学习入口

论文线与实践线独立推进，目标是理解问题、机制、公式、代码和适用条件。论文阅读不要求每篇实现；benchmark 只在解释论文主张或比较质量时按需记录。学习状态以 [PATH.md](../../PATH.md) 为准，当前论文焦点以 [NOW.md](../../NOW.md) 为准。

## 统一阅读器

[打开论文与算法阅读器](../../roadmap/curriculum/gpu/course-site/index.html?chapter=41)。阅读器与 GPU 主课共用目录、搜索和本地静态页面；从模型原理与完整 Transformer 开始，再按八个主题阅读。每篇正文结合推导、张量形状、关键算法代码、数值例子、图解、质量条件与局限。作者实现以仓库入口和代码标识符导航。

| 主题 | 阅读顺序与入口 ID |
|---|---|
| 模型基础与完整结构 | 语言模型、Attention、Transformer Block、位置编码、完整 Decoder、架构家族（41–46）；再读 DeepSeek 整体结构（30、31、39） |
| 注意力、序列建模与长上下文 | GQA、MLA、DSA、Mamba、Gated DeltaNet、长上下文（24、25、26、47、27、60） |
| MoE 与条件计算 | MoE 路由与专家（32） |
| 训练、优化与后训练 | 训练目标与 Scaling、Adam/AdamW、LoRA、对齐与强化学习、训练推理工作负载（48、37、49、50、40） |
| 生成、推理与检索算法 | 采样与搜索、Speculative Decoding、检索与记忆（51、34、52） |
| 量化与模型压缩 | INT8/FP8、GPTQ、AWQ、SmoothQuant、蒸馏与压缩（33、53–55、59） |
| 多模态与生成模型 | 视觉语言模型、Diffusion 与 Flow Matching（56、57） |
| 高效计算与系统论文 | Online Softmax、Parallel Reduce、FlashAttention、可编程注意力、PD 分离、PagedAttention、ZeRO、Scan/Selection（20–23、28、29、35、36、38、58） |

ID 与旧课程 URL 保持稳定；同一分类内的阅读顺序由目录维护，不要求 ID 连续。

## 阅读深度

速读先回答研究问题、主要贡献、核心公式和是否值得继续。正式精读再按主题展开推导、变量与维度、算法执行过程、作者关键代码映射、实验设置、结果、质量边界和失败条件。主题教程按理解需要组织篇幅，不以单页作为上限；benchmark 与性能数字服务于具体研究结论，不作为所有模型家族的统一指标。

新论文候选仍可放进[观察池](../../papers/watchlist-2026.md)。可选抓取命令与筛选流程见 [papers/process.md](../../papers/process.md)，正文结构参考 [papers/template.md](../../papers/template.md)。阅读器目录不单独维护学习进度。
