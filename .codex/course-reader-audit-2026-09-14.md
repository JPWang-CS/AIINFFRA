# 课程收尾与阅读器验收

维护记录，不发布到教程正文，不代表学习完成状态。

> 这是较早一批的验收与待办快照。下文当时列出的教材缺口已在后续非实验补齐任务中处理；当前交付与验收见 [非实验课程补齐记录](course-completion-audit-2026-09-14.md)，当前剩余事项以 `pdf-coverage.json` 的 `remaining_work` 为准。

## 正文

- chapter=39：补 Base/Instruct 的评估条件、effort 与 scaffold 对照、工具版本敏感性、多智能体预算边界；未发现单独移除 CED/CSA2/FP4/Engram 的同预算数值消融，不虚构结果。
- 核对固定作者提交的压缩、索引与 FP4 路径。参考索引先全域 einsum 再 mask；inplace FP4 写回反量化数据。数学选择语义不等于物理限算量，数值量化不等于 packed KV 驻留。
- chapter=18：补 NPU/GPU 软件执行分层、Embedded-7B shape/参数/KV/分页账本，区分该模型、2.0 Flash 与 Pro；保留 TP1 功能记录，旧 TP4 静态分析不记为运行证据。
- 原 solutions、notes、外部仓库和 PDF 未改；PATH/NOW 未推进；未 commit/push。

## 源码与算式检查

- `v41_checks.py`：候选池、因果边界、反例、接受概率、Sinkhorn、effort、FP4 payload 检查通过。
- `pangu_embedded7b_ledger.py`：配置账本、零长度、整页/尾页、极大整数和 17 个非法输入通过；完整程序在本页展开并由 source-check 核对。
- 最终 build：39 篇阅读页、107 个兼容文档、39 份本地资源、1487 次公式渲染、127 处原样代码摘录。
- 全链接检查：108 HTML、5353 个本地 href/src、4160 个片段、1155 个兼容页目录项通过。该检查后仅修正公式分隔符和正式基准名称，未改变链接或标题锚点。
- PDF 原始 SHA256、覆盖区间、目标文件及纯正文词语检查通过。
- 发布清理保留正式名称 `Agent's Last Exam`，不再生成部分翻译的错误名称；构建新增单行双美元公式拒绝规则。
- 导航回归最终通过（101.414 秒）；论文阅读器回归最终通过（40.274 秒，21 篇论文/6 类）；字体锚点回归最终通过（1.451 秒）。前期失败包括过宽摘要断言、远端 README.md 入口与精简字体测试缺少 chapter-nav；分别修正断言、收敛为官方目录入口、增加可选节点保护后通过。DOM 测试不替代下面的截图。

## 真实浏览器观察

使用临时 in-app tab，没有导航用户阅读页；临时 viewport 已 reset。

| 操作 | 结果 |
|---|---|
| 桌面第一章小节跳转 | 标题定位正确，左右目录高亮，主机/设备图可读 |
| 390×844 窄屏目录 | 可打开；选择小节后关闭；页面没有整体横向溢出 |
| 窄屏结构图 | 保持图内字尺寸，容器横向滚动，并显示横向阅读提示 |
| 顶部论文分区 | 同页进入，分类/论文/小节目录保留 |
| MLA 公式 | 求和、上下标、根号正常显示 |
| 搜索 V4.1 | 原来完整目录在结果之前；修复后搜索框下直接出现结果 |
| 点击搜索结果 | 清空查询，恢复目标分类目录，正文仍在同一阅读器 |
| V4.1 新作者代码小节 | 小节定位与内联高亮代码正常 |
| PanGu 新公式 | 发现四处单行双美元未渲染；改独立分隔行后实际截图正常 |
| PanGu 完整代码 | details 展开后完整程序可见，无需源码文件跳转 |

截图保存在 `tmp/course-audit/`：`host-device-desktop.png`、`mla-formula-desktop.png`、`search-fixed.png`、`v41-author-code.png`、`mobile-diagram-hint.png`、`pangu-formulas.png`、`pangu-code-expanded.png`。

验收途中临时浏览器页发生 ERR_INSUFFICIENT_RESOURCES，HTTP 服务仍为 200；待测试进程结束后新临时页恢复并完成上述复验。没有修改系统代理、安全设置或关闭用户页。

## 边界

这是代表页面的真实操作验收，加全站结构/链接检查，不是逐段阅读全文或全 WCAG 审计。CUDA/Triton/CUTLASS 与完整模型设备实验留到对应课现场运行。CUDA 手册尚有后置专题未系统展开，覆盖表保留 11 个 not_integrated 主单元，不用章节数掩盖这些差异；其中包括 Green Contexts、Cluster Launch Control、Dynamic Parallelism、外部互操作和完整 C++ 语言支持等。CUDA Tile 高级 view/atomic 也没有在此次论文/系统案例补写中完成。

## 当时校准后的待办（已归档）

当前结构化清单以 `.codex/pdf-coverage.json` 的 `remaining_work` 为准；历史增量记录保留原来的工作范围，但不再用旧的 remaining 描述覆盖后来的正文。

### 已有正文，不重复列为未写

- Graph 的 child、conditional、device launch 教学代码与机制解释。
- GEMM swizzle 坐标、WMMA/TMA例程及固定版 CUTLASS WGMMA 集成。
- DSM histogram、multicast 官方集成，以及 FP32 Attention lookahead。
- Mini Decoder 多层校准、共享折叠、clipping、packed checkpoint 与 heldout 评估；随机未训练模型、NumPy反量化执行的边界仍保留。
- V4.1 Base/Instruct、effort/scaffold/多智能体评估对照及 Minimal Inference 核心路径；不能继续写成“全部评估协议和作者代码待补”。

### 教材未写与需要深入

1. CUDA 未系统融入的11个主单元：CUDA Python、高级API概览、功能总览、Green Contexts、Error Log、Cluster Launch Control、Extended GPU Memory、Dynamic Parallelism、Interoperability、Driver Entry Points、C++语言支持。两项概览按内容并入现有课，不新增序章。
2. cuTile 高级 view/atomic；GEMM/Attention 的作者mainloop、复杂布局和warp分工；VMM跨设备共享和allocator memory-pool IPC。
3. 真实预训练模型量化适配、真实校准/质量评估与部署格式；不能把随机Mini模型结果当成这部分完成。
4. V4.1 多模态、Engram真实哈希/门控、mHC/DSpark作者实现，以及RL/Rollout状态流和盘古2.0匹配版本的系统案例。

### 独立于教材编写的两类事项

- **现场验证**：编译、数值正确性、Profiler/指令/overlap、设备容量和端到端性能；按用户学到对应课时执行。
- **证据不可得**：未公开的组件消融、内部数据、训练/生产代码，以及历史记录缺失的版本与设备轨迹。保留限制，不虚构补全。

本次仅修正记录；三份PDF、77个主单元状态、用户学习焦点和网页内容均不因此变化。
