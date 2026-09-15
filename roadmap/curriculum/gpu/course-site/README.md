# GPU 与 CUDA 课程网站

入口：[课程网页](index.html?chapter=1)。网站包括GPU基础五章、[访存布局首章](index.html?chapter=7)、[并行归约、Softmax 与归一化](index.html?chapter=8)、[GEMM：从分块实现到性能优化](index.html?chapter=9)、[Activation 与 Fusion](index.html?chapter=10)、[Prefill Attention](index.html?chapter=11)、[Decode / PagedAttention](index.html?chapter=12)、[量化算子](index.html?chapter=13)、[MoE](index.html?chapter=14)和[Sampling/KV](index.html?chapter=15)，以及模型与系统四章：[模型分析](index.html?chapter=16)、[Mini Transformer](index.html?chapter=17)、[vLLM](index.html?chapter=18)、[多 GPU](index.html?chapter=19)。正文来自课程目录中的Markdown，示例代码保留在各章examples中。网站为本地静态文档，不对外发布。

## 构建

构建与导航测试建议使用Node.js 24.15或更新的24.x；也支持package.json所列的22.22.2及以上22.x或26及以上版本。从本目录执行：

~~~bash
npm ci --ignore-scripts
npm run build
npm run check
npm run test:navigation
npm run test:papers
~~~

依赖版本固定在 package.json 与 package-lock.json。构建使用 Marked 解析 Markdown、KaTeX 排版数学、highlight.js 为代码着色；生成的 HTML、样式与数学字体可以离线阅读，不从 CDN 拉取运行资源。

构建遇到无效公式、课程管理字样或源码摘录不一致会失败。check.cjs检查39篇正文（18篇主课、21篇论文与算法）的路由、静态资源与源码链接、唯一锚点、公式、搜索索引、重点提示和可复制代码一致性，并独立复算sector、bank、矩形转置、向量copy主体/尾部索引以及缓存容量。它不代替浏览器视觉检查、nvcc编译或GPU测试。

## 本地预览

从仓库根目录执行：

~~~bash
python -m http.server 8765 --bind 127.0.0.1
~~~

访问 http://127.0.0.1:8765/roadmap/curriculum/gpu/course-site/index.html?chapter=1 。

页面保留既有 chapter ID；chapter=6 为退休的“序”，旧链接转到访存与布局 chapter=7。其余 1–19 的实际章节入口不变，共 18 篇正文。右侧目录支持章节内锚点；复制按钮只复制源码，不包含行号与高亮标记。浏览器剪贴板功能需要安全上下文，localhost通常满足要求。

重点提示使用GitHub风格的引用标记：IMPORTANT用于核心结论，WARNING用于边界，TIP用于实践复盘，NOTE用于理解提示。代码围栏可使用语言后跟`{2,4-6}`指定重点行。提示与高亮只强调关键位置，不替代连续解释。

保持开发服务的仓库根目录不变，否则本地 CUDA 手册、solutions 与 notes 的相对链接无法定位。

## 链接审计记录（维护说明）

2026-09-10 的仓库级复核覆盖 125 个自有 Markdown 文件、1104 个自有链接和 34 个 Markdown hash 引用，结果为 `bad=[]`。外链初检为 94 个唯一 URL：90 个 HEAD 通过，3 个环境重定向经网页核实有效，1 个真实 404 已修复；扩展复核为 122 个唯一 URL：119 个 HEAD 通过，vLLM 旧 blog 重定向经网页核实有效，另外 2 个真实 404（ZeRO、DeepSeek V4）已按官方页面修复。该统计是主 agent 的仓库级检查结果，不等同于本地静态站的运行时 HTTP 监控。

本站 `link-audit.cjs` 不请求外网；它递归检查生成的 HTML 产物，使用完整 `file://` 路径解析本地 URL，验证 URL 编码、目标文件、禁止 raw Markdown、全部已发布章节的路由、目标 HTML 的真实 DOM 锚点、附属页目录锚点，以及 PDF 的 `#page=N` 片段。路由集合由已发布 article 生成，并与静态检查要求的 39 篇正文互相核对。外链可访问性需要另行验证；本地服务器、浏览器视觉布局、CUDA 编译和 GPU 正确性/性能不由该脚本证明。

顶部的 GPU、算子、模型与系统分区、侧栏树、正文与URL共用导航状态。侧栏按当前分区切换，每章子目录可折叠；分区上次阅读章节只保存在本次页面会话的Map中，不更新学习状态。jsdom导航测试执行真实app.js并模拟点击、历史与锚点；布局与滚动坐标被简化模拟，因此它不是浏览器截图或真实像素布局测试。

## 维护

全部已发布阅读页的小节 ID 保存在仓库 .codex/course-section-anchors.json：已有标题保留编号，新标题只追加编号；重命名标题时为新键保留原编号，不按正文位置重新编号。构建在成功后保存新增键，既有条目不要清空。旧标题的链接可通过别名解析到当前小节。

正文构建会拒绝代码围栏外孤立的单美元行、未闭合的块公式，以及 source-check 只截出一行 Python 函数签名的情况。构建通过仍需检查浏览器公式和示意图，不能代替 GPU 编译验证。

旧课引用应在当前正文中展开，而不是依赖跳转到Markdown文件。原样摘录可在代码围栏前添加HTML注释 `source-check: solutions/cuda/路径.cu`（放在 `<!-- ... -->` 中），构建会检查紧随的代码块是否原样包含于源文件。也支持相对当前章节的路径。教学改写应明确标注，不使用原样摘录标记。同步检查只读取源文件，不修改用户代码。

正文只放教程、来源、技术前提和有条件的实测案例；学习状态与编写记录在仓库维护入口管理。旧教程及原始代码不因本站接入新章而删除。

阅读风格参考AIInfraGuide学习路线正文：浅色文档导航、正文内目录、清晰的二三级标题、主题重点、推荐资料与掌握标准。详细讲解和代码不因路线页格式而删减。网页支持单章连续阅读、标题锚点、章节全文检索与前后跳转；不使用宣传式首屏。

## 论文阅读器

全站正文以概念、机制和代码为主，不显示课程编写状态。18 篇主课的开场在源 Markdown 中维护，21 篇论文的发布开场由 paper-introductions.cjs 管理；原始笔记保留。npm run test:editorial 检查已淘汰的计划式套话，不机械删除技术条件。

[论文与算法](index.html?chapter=25)与主课程共享顶部导航、侧栏、正文和检索。侧栏结构为“主题分类 → 论文 → 二三级小节”，不另设汇总序章；六类为数学与并行算法、Attention、模型架构、量化与低精度、推理系统、训练与优化。论文线不排在系统课之后，不改变学习状态。

paper-catalog.cjs 维护21篇来源与分类。发布时清理协作状态和冗余文件跳转，原笔记不覆写；DSA 使用独立整理正文。论文公式和代码预先构建，阅读时加载该篇 DOM，避免一次创建所有公式节点。旧独立页面在当前窗口跳转到相应课程文章，保留可映射的小节锚点。

paper-reader.test.cjs 检查全部论文的目录锚点、折叠、分区切换、公式与代码加载及复制。attention_checks.py 只验证 MLA 线性重排和稀疏选择的数学语义，不代表 CUDA/Triton 正确性或 GPU 性能验证。现有专题接入不表示每篇已完成新的逐条文献核验。

新增主题：[DeepSeek-V4.1-Flash](index.html?chapter=39)与[Scaling Law、强化学习与长任务推理](index.html?chapter=40)。沿用原有分类和URL，分类内前后篇按目录顺序连接，不要求数值ID相邻。CUDA同步章直接包含stream-ordered allocation示例。教材覆盖登记位于仓库 `.codex/pdf-coverage.json`，原件哈希与目标完整性用 `node .codex/check-pdf-coverage.cjs` 从仓库根目录检查；该检查不证明语义已穷尽。
