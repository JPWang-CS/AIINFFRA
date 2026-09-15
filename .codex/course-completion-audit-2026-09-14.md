# 非实验课程补齐与验收

## 范围

用户要求：除自己现场运行的实际实验外，补齐已列出的教材缺口、作者实现深读和阅读质量问题。没有授权新的学习成绩、真实设备结果或 Git 提交/推送。

## 内容交付

| 范围 | 正文与实现 |
|---|---|
| CUDA 编程与编译 | CUDA Python bindings/SIMT、cuTile view/atomic、C++ 执行空间/对象/设备链接、Error Log、Driver Entry Points；完整教学程序与CPU坐标检查 |
| CUDA 资源与系统接口 | Green Context、CLC、EGM、CDP2、VMM alias/fabric、pool IPC、Vulkan/GL/D3D12/NvSci；required/preferred cluster与特殊grid计数、batched copy、stream priority/connections分别进入原章节 |
| GEMM / Attention | 固定 CUTLASS/FA2/FA3 作者路径、schedule分支、warpgroup与stage、stride/causal/tail、online state和K/V/P释放边界；源码中数学维度约定逐项区分 |
| 真实模型量化流程 | 标准本地Llama选择性W4 overlay，完整校准/预检/保存加载/当前heldout评估；QKV和gate/up共享fold，其余浮点例外明确；不冒充通用引擎checkpoint或低位运行 |
| V4.1 / 系统 | 视觉、Engram真实门控、DSpark、mHC/Mega边界、训练数据/学习率/packing、rollout版本流与提交回滚、盘古2.0配置/依赖/安全定位与来源边界 |
| 连续阅读 | 移除维护型旧课/实验文件跳转，保留正文代码与实测；复杂字符流程改表，旧参考锚点兼容；不新增序章或独立优化线 |

精确来源与各组检查见 `completion-evidence-*.json`，统一路由保存在 `course-source-index.json`。

## 重要复核修正

- 手册样例不直接照抄：Numba显式边界、CuPy显式dtype、版本化batched-copy签名。
- Green Context H2D和kernel在同一non-blocking stream上排序，不依赖默认stream隐式同步。
- Binary与timeline semaphore分开讲；CLC失败不解码/不重试，fixed-block的prologue优势不反写。
- CDP child launch错误和输出sentinel可观测，正常cleanup失败不返回PASS。
- CUTLASS Auto与显式schedule分开；本文N/K与库N/K约定明确。
- 量化加载前验证全部拓扑/数组；换heldout后重新计算指标，不把旧MSE当当前评估；输入和校准隔离。
- mHC pre为1×n、residual为n×n、post为n×1；源码comb轴为source/destination，数学左乘矩阵是其转置；F不在Mega-mHC融合内部。
- Engram门控包含RMS、学习权重、signed square root和copysign零点语义；CPU示例不再用普通cosine sigmoid冒充作者路径。
- 多模态段移到架构之后，只保留一份；来源/环境维护话语与代码版本证据分离。

## 检查记录

- 新增GEMM CPU 7项、Attention CPU 8项、量化合同CPU 8项、CUDA协议CPU 8项通过。
- cuTile CPU坐标、V4.1视觉/mHC轴、Engram门控、rollout生命周期、学习率与packing检查通过。
- 课程Python文件AST扫描通过；最终文件数以最后一轮输出为准。
- CUTLASS wrapper 使用 `C:/Program Files/Git/bin/bash.exe -n` 检查通过；没有执行构建或GPU程序。
- 导航、论文阅读器、字体定位完整回归已返回通过；最终构建、链接和截图结果追加于本文末尾。

## 来源边界与实验边界

本次已列的教材写作清单完成，不代表逐函数复刻CUDA全部API或完整作者训练/生产系统。原始77个来源单元仍保留各自明确范围；不得用条目数计算“完成百分比”。

openPangu2.0匹配镜像的omni-npu0.2.0 adapter未从可访问资料取得：公开配置/部署说明与受限代码访问分别记录，不宣称实现未公开，也不根据配置猜backend。容器内只读distribution定位流程已给出；没有下载/执行服务镜像或权重，没有绕过访问限制。

真实CUDA/cuTile/Triton/CUTLASS编译、正确性、Profiler、模型质量/性能及NPU/多卡运行，由用户在对应课程现场完成。已有用户实测保持原样，不新增GPU_VALIDATED或学习完成状态。

## 最终发布验收

- 最终构建：39个阅读章节、100个可读Markdown页面、1637个公式、148处原样源码摘录检查通过。
- 80个课程Python文件AST通过；CPU测试和Bash语法检查通过，不代表CUDA编译或设备执行。
- 全量链接检查：101个HTML产物、5120个本地目标、4042个章节DOM片段、1078个文档目录目标通过。随后唯一正文修正为完整Llama程序summary后的空行，不改变链接集合。
- 内容、文风、PDF哈希与覆盖目标检查通过；PATH/NOW哈希与任务开始一致，git diff --check通过。
- 浏览器抽查了深层目录联动、论文分类、mHC公式形状、本地Llama同页实践和完整程序。修复了完整程序的Markdown/HTML块分隔，使其实际渲染为代码；增加防回归断言。长折叠代码块限定高度并支持键盘聚焦。
- 窄屏390×844的DOM边界为375px，无整页横向溢出；截图通道返回缩放异常，因此不把该截图算作窄屏视觉验收。临时viewport已恢复，测试标签已关闭。不是完整WCAG审计。
- 截图保存在tmp/course-completion；08-full-program-scroll.png是修复后桌面实拍。
- 最终导航、21篇论文/6分类阅读器、字体加载后的锚点定位回归全部通过；整站增加未渲染代码围栏检查并通过。

## 协作边界

五组互斥Luna写入，主agent核对主源、修正公式/代码并完成集成。最后一组尚未返回最终答复时已写入收尾文件；停止后保留最新改动，由主agent接管来源记录、预训练配置补充和整站验收。没有重叠worker写入，没有修改PATH/NOW、原PDF、原始solutions/reference或公共环境配置。
