# GPU 与 CUDA 后续课程生成记录

日期：2026-09-09。

## 用户授权与内容边界

用户要求接着第一部分生成后续内容，继续依据本地 CUDA 手册、参考 AIInfraGuide，写成详细深入的工程师课程，并保留旧教程中的代码、讨论心得和实测分析。涉及实践时接好 LeetGPU 与服务器入口；网页保持纯教程，并改善阅读设计。

本次按已讨论的第一篇顺序继续：整体结构 → 执行 → 存储 → 协作 → 性能证据。没有将整个七篇提案视为已确认，没有推进用户学习完成状态。

学习状态以 PATH.md 为准，当前焦点仍由 NOW.md 维护。课程正文不出现 WIP、等待验收、Agent 编写等管理提示。真实代码、实验数字和个人疑问可以作为教学材料保留，实验条件和推断边界不能省略。

## 文件与材料映射

| 来源 | 新课程归属 |
|---|---|
| 原执行调度章、计算管线章、编译章 | [第二章](../gpu/02-cuda-execution-and-scheduling/README.md)：warp、ILP、分支、数值格式、PTX/SASS |
| 原寄存器/存储章、CUDA naive/tiled GEMM 与 Softmax | [第三章](../gpu/03-registers-and-memory-system/README.md)：活跃区间、local、sector、bank、转置 |
| 原同步章、Triton debug、FA1/FA2 的 CTA 与 Q tile 讨论 | [第四章](../gpu/04-synchronization-and-asynchronous-execution/README.md)：作用域、归约、stream/event、流水线 |
| 原资源章、benchmark章、MatMul sweep 与 Nsys P0-lite 原始记录 | [第五章](../gpu/05-performance-analysis-and-optimization/README.md)：资源预算、Roofline、计时、证据与优化 |

所有原始章节文件、lessons、solutions、notes、日志原位保留，不通过删除旧材料完成迁移。新文中的简化解释不改写原始日志。MatMul 各批次结果独立记录，不拼成同一次测量。

## 核对原则

- 本地手册13.3，698页；印刷正文页 +16 = PDF阅读器页，引用前按目录和正文核实。
- 32B sector 与32bank算例给定访问宽度和对齐条件；不把同地址广播混成bank冲突。
- 线程块、warp、逻辑数据tile与物理计算管线分开。16×16线程块的一warp跨两行。
- 部分warp归约的有效数据集合与intrinsic参与集合分开；参考例程用完整32lane参与、尾值置零。
- 矩形转置不仅交换shared索引，也交换对应block/线程的输出坐标，并检查尾块与错误返回。
- stream事件记录在slot最后一次使用之后；host buffer复用需等事件完成。
- occupancy与性能不是同一量；s2共享内存减少但耗时更长的记录不证明唯一硬件原因。

## 网站实现与验证边界

站点继续在本地使用，不对外发布。Markdown为内容源，构建输出HTML；新增固定版本marked、KaTeX与highlight.js依赖和锁文件。公式在构建时排版，字体随站点保存，不依赖访问外部CDN。章节/节锚点、搜索、源码和前后导航属于阅读界面，不维护学习完成状态。

CUDA示例的源代码审查与CPU侧索引算例检查，不等于nvcc编译或GPU运行。当前未提供远端设备执行授权和环境，本次不新增实卡性能结果。维护记录中保留该边界，不在教程页面堆管理声明。

本次不commit/push，不改PATH/NOW的学习状态。新增示例不得顶替用户的LeetGPU原始solve或计作平台通过。

## 收尾结果

- 新第二至第五章正文、7个CUDA示例及3个辅助脚本已保存；旧教材、用户代码和原始日志未删除。
- 网站构建生成5章，59处数学表达式通过KaTeX排版；本地链接、唯一锚点、搜索索引、第一章完整源码同步和管理文字检查通过。
- CPU侧独立复算sector/bank与矩形转置索引通过；资源预算脚本得到37.5%/25%两组结果，Roofline脚本得到Vector Add的1/12 FLOP/byte及GEMM的945.231 FLOP/byte。
- 代码审查修正了转置写回坐标、未初始化输入、NaN漏检、错误仅打印却返回成功、不完整warp的shuffle参与集合、FMA链比较口径和数处手册页码。
- 本地课程URL返回HTTP 200。没有进行浏览器视觉/交互测试，没有进行nvcc或真实GPU运行；上述静态与算术检查不替代它们。
- Sites规则用于长文排版与静态交付，保留本地文档站，不初始化新框架或对外发布。一个写作任务在超过约定等待窗口后已停止，主线程接手剩余明确校对；不存在重叠写入。
