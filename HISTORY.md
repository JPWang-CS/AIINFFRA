# AIINFFRA 历史记录与进度快照

> 用途：跨电脑切换 Codex 时，新电脑先读本文件，快速恢复当前进度、目录结构和下一步，不需要先通读整个仓库。
> 重要：本文件是“恢复入口”，不是进度权威源。真正的进度仍在 [PATH.md](./PATH.md)，当前焦点在 [NOW.md](./NOW.md)。

---

## 0. 最后更新

- 2026-09-17（用户已阅读完成第一章从开头到 2.4；第 3 节《完整程序：沿着一次计算走一遍》尚未阅读，下一入口定位至 `chapter=1#chapter-1-section-11`。本次只推进阅读，不新增实验或 GPU 验证。）

- 2026-09-15（用户决定从模块化课程第一篇第一章重新学习；本次将既有 CUDA/Triton Vector Add、早期 CUDA GEMM（FP32/FP16 LeetGPU 归档与 RTX4090 大 K 对照）、Triton MatMul、Softmax 实验、原始代码与测量证据融入访存/归约/GEMM对应课程章节，课程经过时直接复盘或跳过，不清零、不降级。A1 CUDA Vector Add 的 LeetGPU 技术正确性已完成，Lesson 01/weekly 还记录本地 RTX4090 `696 GB/s / 0 error`（1M FP32、256 threads/block），但仅有 Lesson 01 快照、无独立原始 `solve` 归档，PATH 保持 `WIP（平台正确性已完成，归档缺口）`，不伪称 `LEETGPU_PASS`；该结果与 Triton RTX3090 840.1/843.0 GB/s 不是同一次实验。保留 CUDA GEMM RTX4090 K=2048 naive 5033 vs tiled 3118 GFLOPS（约0.6x，K=8192方向相同）、MatMul 的 A100 LeetGPU 24.54 ms/55.3th 与 RTX3090 GPU_VALIDATED/Nsys、Softmax #5 的 0.29 ms/47.0th；未产生新的 GPU 成绩，Softmax RTX3090 服务器验证仍待补，论文线继续 MLA。）

- 2026-09-14（按“除现场实际实验外其余补齐”更新课程正文：CUDA Python/cuTile/C++/日志/Driver Entry Points，GreenCtx/CLC/EGM/CDP/interop/VMM/pool IPC和高级提交语义；GEMM/FA2/FA3作者路径；本地Llama选择性W4完整流程；V4.1视觉/Engram/DSpark/mHC、预训练配置与rollout状态流、盘古来源/部署安全边界。复核修正API样例、GreenCtx复制依赖、child错误、mHC形状与comb转置、Engram门控和量化重载指标；清理正文维护跳转、保留旧锚点。五组互斥worker与主agent集成，最后一组停止后保留已写收尾文件。覆盖表按实际正文更新，不是仅改状态；现场实卡实验与不可得源码/数据边界单列。PATH/NOW、原PDF和原始实践不变，未commit/push。详细验收见 `.codex/course-completion-audit-2026-09-14.md`。）

- 2026-09-14（修正PDF覆盖维护记录：Graph高级节点、swizzle/WGMMA固定版集成、DSM/multicast、Attention lookahead、Mini Decoder量化流程及V4.1公开评估/作者代码不再重复列为未写。剩余项分为教材未写、已有内容需深入、现场设备验证和不可获得证据；保留77个主题状态及原PDF身份，不把部分覆盖升级成全面完成。只修改后台覆盖表、验收记录与HISTORY；PATH/NOW、课程正文、网页和原始代码未改，未commit/push。复用Luna未交回产物后停止，由主agent完成最小记录修正。）

- 2026-09-14（补 V4.1 公开评估与作者 Minimal Inference 解读、昇腾/盘古案例；区分 Base/Instruct、effort/scaffold/多 Agent 对照与组件消融。固定作者源码复核发现先全域 einsum 后候选 mask、FP4 inplace 写回反量化值，正文不再把最小参考实现视为生产算量或 packed KV 占用的证明。系统章结合 Embedded-7B 实际配置、模型适配类、TP1 功能记录，区分 2.0 Flash/Pro 和旧 TP4 静态分析，内联完整 CPU 权重/KV/分页账本。真实浏览器发现搜索结果被目录挤到下方，修复搜索展示、清空/选择后恢复目录及 Markdown 摘要；窄屏结构图增加横向阅读提示。来源、PDF 覆盖与验证记录留在后台，不推进 PATH/NOW，不改原始代码/外部仓库，不新增 GPU/NPU 成绩，未 commit/push。）

- 2026-09-14（按用户要求并行补前五方向，合并六篇原章节：GEMM swizzle/WGMMA 固定版 CUTLASS 集成；独立 FP32 Attention lookahead；两层 Mini Decoder 校准/折叠/clipping/打包/加载/heldout 评估；两 CTA DSM 与固定版 multicast 作者实现；Driver/lazy loading/VMM/IPC/PDL/domains 与高级 Graph。新增代码与关键片段留在原正文，后台来源/覆盖记录更新。复核修正 checkpoint 已有文件误删风险、Driver/PTX 接口、基线隔离及完成边界。CPU 流程可运行；CUDA/Triton、CUTLASS 集成未在 GPU 验证，无性能成绩声明。PATH/NOW 和用户原始 solutions 未改，未 commit/push。Attention 教学 baseline 曾被 worker 临时追加内容后按首次读取恢复，没有事前哈希，不声称字节级验证。）

- 2026-09-14（GEMM 章补 TMA+WMMA 一槽/双槽完整实现：两份 FP16 tensor map、A/B 合计 1024 字节事务、逐槽 token 与复用时序，同页展示核心代码和可展开 host reference/guard/计时入口。CPU 双输入就绪模型由一个 Luna worker 编写，主 agent 核对手册/API并补 CUDA/正文；CUDA 未编译/未跑 GPU，不是 WGMMA、warp specialization 或性能胜出结论。当前学习状态和原始实践不变，未 commit/push。）

- 2026-09-14（GEMM 章补完整单 warp WMMA 双缓冲程序：相同 FP16 输入/FP32 输出，对比一槽与两槽 cp.async 提交顺序，显式 padding、FP64 reference、输出 guard、可选同形状 CUDA-event 计时；同页保留完整代码和调度图。新增 CPU pair-cover/地址对齐/槽位顺序三项检查通过。Luna worker 无文件与检查点后收窄仍无产物，已停止，主 agent 接管；CUDA 未编译/未跑 GPU，不声称加速或强库基线胜出。PATH/NOW、原始实践与旧实测保持不变。）

- 2026-09-14（同步章补二维 tensor map：fastest-first 维度、字节 stride、逻辑宽度与物理 padding、descriptor 传参、显式事务登记、越界补零；加入完整 CUDA 8×32 tile 导出检查和 CPU 坐标测试，同页解释 GEMM A/B 描述符映射。CUDA 未编译/未跑 GPU；无 swizzle/multicast/完整流水 GEMM 性能声明。Luna 负责独立 CPU 检查，主 agent 完成手册/API 核对、CUDA 及正文；原 PDF、原始实践、PATH/NOW 保留，未 commit/push。）

- 2026-09-11（继续补同步、GEMM、Prefill 三章：增加一维 TMA roundtrip 完整程序、两种完成边界 SVG、事务计数与 source-read wait、padding/guard 检查；GEMM 补 full/empty、代际与 phase、合法 WGMMA warpgroup 及资源账本；Attention 补 K/V/P 最后使用、online state 更新依赖和资源分析，分子乘积加括号。Luna worker 经检查点与中断收窄仍无文件产物后已停止，主 agent 接管确定范围；CUDA 无 nvcc/GPU 验证，伪代码与可运行源码区分。保留既有代码、实测记录和 PATH/NOW，未 commit/push。）

- 2026-09-11（继续补 CUDA 手册内容：执行章同页加入 CUDA SIMT/cuTile/Triton 的 tile 表达对照、块坐标与元素地址、尾块、完整 cuTile 向量加法、GEMM accumulator 和编译提示；同步章加入 CUDA Graph 参数更新、动态长度/分桶、capture 依赖与图内内存生命周期。两份 CUDA 程序和 cuTile 程序由同一 Luna worker 编写，复核修正设备尾区哨兵、逐次 replay 检查、capture 失败清理和依赖缺失处理。新增 SVG 生命周期图，保留全部旧锚点和原始用户代码；PATH/NOW 不变。正文以本地 CUDA13.3 手册为基准并核对官方 Runtime API；CUDA/cuTile 设备执行留到对应课现场完成，不记为编译/GPU 通过。浏览器连接工具报错，HTTP200、构建与结构检查不代替截图验收。）

- 2026-09-11（量化章补入校准统计、分组 RTN、SmoothQuant/W8A8、AWQ-style 缩放搜索、逐列/后缀参考/分块 GPTQ、packed INT4 CPU 计算与 Triton kernel；算法、代码和实验命令同页。新增 CPU lab 的 12 项测试及 W8A8 检查通过，GPU 程序准备好后留到用户学到对应课现场运行，未记为 GPU 通过。按作者代码核对关键变换与 Cholesky 更新；教学实现与完整作者库区分。）

- 同次全站文字清理覆盖 18 篇主课开场与 21 篇论文阅读页：删除写作计划式介绍、统一模板副标题、旧路线挂靠和反复的协作评价；保留数学、接口条件、代码及历史实验。原始 notes/solutions 未改，论文发布层清理元信息并修正 MoE 容量/单步读取量混淆与 AdamW 状态说法。小节稳定编号扩展到全部阅读页，保留重命名别名；PATH/NOW 学习焦点不变。

- 2026-09-11（补深原 GEMM 与同步章：寄存器微块、CTA/warp/指令 tile、PTX m16n8k16 布局图与坐标检查、带尾块的 FP16 WMMA 示例、环形 stage 和 cp.async 示例、TMA copy/read 完成边界；保留已有 FP32 sweep/trace，增加条件化因果分析和同页验证命令。一个 Luna worker 写入 CPU 状态机及两份 CUDA 示例，主 agent 复核并修正维度、FIFO/乱序完成、zero-fill 可观测性。CPU 新增 8 项测试与原 GEMM 账本回归通过；nvcc 不可用，没有 GPU 验证。PATH/NOW、原 solutions/reference、原 PDF 未改。）

- 同次发布检查新增独立单美元公式分隔符与仅函数签名摘录的拒绝规则；修复同步/性能章的旧分隔符、GEMM split-K 的 qquad 转义和 tiled C 的 lane 方向文字，按手册澄清退出线程不等于必然 barrier 死锁。

- GEMM 与同步章新增稳定小节编号注册，保留补写前的锚点含义；例如此前 stream-ordered allocation 的 chapter=4#chapter-4-section-13 仍指向原小节，不被中间插入内容挤走。两章新增标题追加编号，未重排其他课程。

- 2026-09-11（继续融合本地三份 PDF：同页论文目录新增 V4.1（chapter=39）与 Scaling Law/RL/长任务（chapter=40），CUDA 同步章补 stream-ordered allocation 的完整跨 stream 示例。新增两组 NumPy 教学检查，原 PDF 哈希与后台主题覆盖登记；保持 PATH/NOW、用户源码和实验记录不变。Luna worker 长时间无检查点/文件产物后停止，主 agent 接管。CPU/公式/摘录检查通过，CUDA 示例未编译、未跑 GPU。）

- 同次浏览器验收发现通用 SVG 图片规则使 KaTeX 根号高度从应有约 18.65px 缩为约 0.06px；已排除数学 SVG，浏览器重验恢复正常。styles/app 改用内容哈希查询参数避免旧缓存；直达章节即时定位，字体就绪后可校正锚点但不覆盖用户滚动。新增字体定位测试，保留原有折叠/导航行为。

- 2026-09-11（论文与算法并入同一课程阅读器：六类目录、19 篇正文，保留独立论文线；旧论文网页入口转入同页阅读。补 FA2 在线状态数值例、MLA 公式与 NumPy 对照、DSA 索引选择及复杂度边界。原笔记、用户代码和实验记录保留；NOW 仅换网页入口，PATH 与学习事实不变。）

- 2026-09-11（移除前台算子“序”：正文、搜索和侧栏不再发布该汇总页，GPU 基础后直接进入访存与布局；chapter=6 旧链接兼容转入 chapter=7，其他章节 URL 保持不变。原汇总内容保留在维护归档，原始代码与学习状态不变。）

- 2026-09-11（按“先写完、后整改”清理旧路线表述，补课程站第 16–19 章：模型分析、Mini Transformer、vLLM、多 GPU；新增模型与系统导航分区。NumPy/CPU 检查通过，GPU/服务/NCCL 未执行。学习事实与当前焦点不变，详见[模型与系统记录](./roadmap/curriculum/decisions/2026-09-11-model-and-systems.md)。）

- 2026-09-10（补入课程站第 13–15 章：量化、MoE、Sampling/KV。九类算子均有独立正文入口，新增 CPU 语义检查、基础 Triton kernel 与 GPU 验证脚本；不是完整 GPU 优化或平台通过。具体范围见[剩余算子记录](./roadmap/curriculum/decisions/2026-09-10-remaining-operators.md)。学习状态和原始代码不变。）

- 2026-09-10（已发布章节清理源码文件跳转，精确源码/PDF 定位转入 .codex/course-source-index.json；PDF 参考按章集中。Decode 补连续参考实现和页内结果比较，实践区集中题目、验证与计时命令。原始源码、文献、实验和 PATH/NOW 未改。）

- 2026-09-10（课程改为连续阅读：GEMM 去除跨章前置，访存/归约/GEMM 的密集文献入口后移，Prefill/Decode 补足就地语义。优化与算子同章展开，独立性能入口改为实验检查表；总路线同步简化。保留源码、测量记录及阅读状态，不改 PATH/NOW。）

- 2026-09-10（第一章主机—GPU 系统图与 GPU 内部资源树改为 SVG 结构图，区分任务提交、H2D/D2H 数据传输与资源归属；移除这两处字符图代码块。保持教程正文和学习进度不变。）

- 2026-09-10（正文去除以日期指代实验的开头与标题，GEMM/性能章改为源码、配置记录和原始 trace 的直接引用。Decode 补入两份 PDF 的容量预算、量化组成项、CED、CSA2 来源模型及重放边界，新增 CPU 推导检查。未改学习状态或原始实验记录。）

- 2026-09-10（分析两份新增 PDF 的融合位置与错误边界；新增课程站第 12 章 Decode/PagedAttention，含分页地址、GQA、online/split-KV、追加/COW、890 B/token 手算、CPU/host 检查及 GPU 验证脚本。性能章补请求指标与 Amdahl，Fusion 章补 Single-Pass mHC。详见[融合记录](./roadmap/curriculum/decisions/2026-09-10-pdf-integration-and-decode.md)。未修改 PATH/NOW，未新增平台或真实 GPU 成绩，未 commit/push。）

- 2026-09-10（新增[本地课程资料页](./roadmap/curriculum/materials.md)，接入《大模型推理实践》讲义与 DeepSeek-V4.1-Flash 技术报告的本地来源、封面/目录核对结果和按主题 PDF 页码入口；仅核对文件与目录，尚未逐章精读，不代表新增学习状态或已读完成。）

- 2026-09-10（将执行/存储章中的GEMM实现直接融入正文，保留原始naive与tiled代码，新增源码摘录一致性检查；不再只给旧lesson链接。见[融合记录](./roadmap/curriculum/decisions/2026-09-10-inline-gemm-integration.md)，学习状态不变。）

- 2026-09-10（修复顶部/侧栏联动并加入可折叠章节子目录；增加DOM导航测试覆盖分区记忆、折叠、锚点、搜索、历史和非法URL。实际网页内容为GPU基础五章、算子总览与访存首章，不代表整套九族正文完成；详见[导航与内容范围](./roadmap/curriculum/decisions/2026-09-10-navigation-and-content-scope.md)。未改学习状态。）

- 2026-09-10（按用户指定的AIInfraGuide学习路线正文替换阅读风格：七页统一为文档布局，加入正文目录与标题锚点，算子总览按主题组织知识点、资料与掌握标准。保留五主体、九算子族、完整推导、用户代码与实测记录，不改学习状态。详见[风格决定](./roadmap/curriculum/decisions/2026-09-10-aiinfra-reading-style.md)。）

- 2026-09-10（恢复用户确认的五主体顶层方案，完整算子保留九族并继续首章访存/布局；现有GPU课程增加重点呈现、图解和关键补缺。PATH/NOW仅统一目录口径，不修改学习完成事实。讨论与边界见[五主体与网页优化记录](./roadmap/curriculum/decisions/2026-09-10-five-pillars-and-layout.md)。）

- 2026-09-09（继续生成GPU/CUDA后续课程，按执行、存储、协作、性能证据组织新第二至第五章，原教程、讨论及实践记录原位保留；网页接入五章导航、离线公式与代码高亮。详细来源与验证边界见[本次记录](./roadmap/curriculum/decisions/2026-09-09-cuda-continuation.md)。未推进用户学习状态，新增CUDA示例不等于实卡验证。）

- 2026-09-09（第一部分全文与网页改为纯教程，清理状态和协作过程提示，保留CUDA原理、代码、手算和资料入口；构建增加管理词与源码同步检查。详情见本文第10节。PATH/NOW学习状态未变。）

- 2026-09-08（按用户确认落地[新版第一部分](./roadmap/curriculum/gpu/01-gpu-hardware-map-and-generations/README.md)：本地CUDA手册§1.1–1.3/§2.1对照、完整CUDA程序逐段解释、1D/2D地址手算、Triton代码与讨论问题回填、架构/版本边界。网页只将第一章作为本轮新正文，后续标旧稿待讨论。新增教学示例未做nvcc/实卡验证，用户阅读待验收，学习进度不变。）

- 2026-09-08（新增[课程讨论归档](./roadmap/curriculum/decisions/2026-09-08-course-discussion.md)：记录已确认的课程原则、CUDA Programming Guide 13.3/698页的章节对照要求、七篇课程顺序提案及后续回填候选；七篇仍为待讨论，不替换现有路线，本轮未推进学习状态。）

- 2026-09-04（按AIInfraGuide文档站形式重构课程网页：删除营销式Hero、统计数字、四步卡片、一页堆八章和每章重复的大型资料卡；改为顶部导航、左侧八章知识树、中间单章正文、右侧本页目录与`?chapter=N`切换。同步删除各章重复的NPU迁移/统一验收模板，并将第一章开头重排为“为什么需要GPU→系统到SM→架构/芯片/SKU→算子映射→代际演进”。）
- 2026-09-04（第一篇增加静态课程网站`roadmap/curriculum/gpu/course-site/`：从八章Markdown构建单页HTML，提供侧栏导航、搜索、阅读进度、代码复制、响应式布局和每章CUDA Programming Guide官方对照入口。网页作为主要阅读界面，Markdown继续作为正文源；未改变学习完成状态。）
- 2026-09-04（保存实践课程结构复盘并重写第一篇：移除独立NPU映射章，改为八章“硬件总图→执行调度→寄存器/Shared/存储→计算管线/Tensor Core→资源/Occupancy/Roofline→同步/异步流水→编译/PTX/SASS/库→Benchmark/Profiler/调试”。新增结构决策记录；寄存器章详细补充register file、线程私有语义、活跃区间、分配粒度、accumulator、spill到local memory、bank conflict、coalescing和Triton地址tile。全部仍为`WIP`，未记录用户完成。）
- 2026-09-04（课程术语规则补齐：专业缩写第一次出现时必须写英文全称和中文含义，后文才使用简称。第一章新增集中缩写表并补充CUDA、GPC、SM、CTA、SIMT、SFU、MMA、TMA、PTX、JIT、HBM、NUMA、P2P等定义，同时先定义kernel、grid、warp、lane和stream；该规则写入课程总编写标准。）
- 2026-09-04（按用户复核再次重构第一章：五个短课仍然割裂，现已合并为一篇约千行的《GPU架构总论》。补充Host到GPU launch、GPC/SM、CTA/warp/thread、warp调度与eligible warp、SIMT分歧、global transaction、L1/shared/L2/register/local memory、bank conflict、occupancy资源公式、Tensor Core分层tile以及Vector Add/Softmax/GEMM/Attention映射；架构演进、编译兼容和实验方法作为后续部分连续展开，不再拆文件。）
- 2026-09-04（第一章从“说明卡”重写为五课连续教材：合并原十六个短小节，按“读懂GPU名称与内部层次→架构演进的四条矛盾→编译兼容与kernel决策→实验身份和资源账本→拓扑/运行状态/带宽验收”推进。正文新增连续机制讲解、算子贯穿案例、公式、命令、实验和推导题；修正CUDA Samples 12.9起移除旧`bandwidthTest`、改用NVBandwidth的版本边界。旧碎片文件删除，入口和前后课导航同步更新；状态仍为`WIP`。）
- 2026-09-04（补齐第一篇十章导航：第二章至第十章均新增可点击的独立章节入口，不再在总目录中保留纯文本死项。未开课章节的入口页明确标注“待讨论/未开课”，并列出计划子课、代码或实验落点、验收证据及前后章路由；不将目录提纲冒充已完成正文。）
- 2026-09-04（再次统一课程口径：正式课程不突出任何GPU产品，不设置默认主卡；正文按架构机制、代码落点、性能因果和验证方法组织，具体型号只用于分支举例、论文环境和历史实测身份。第一章十六节改为通用课程，资源卡与实验基线可复制到任意目标设备；现有特定GPU性能数字仍保留在PATH、solution和历史证据中。）
- 2026-09-04（全盘复查第一篇第一章：纠正“以环境采集代替GPU知识课”的偏差，重构为“NVIDIA GPU全景、架构代际与实验基线”。新增九节知识主课，覆盖架构/芯片/产品/Compute Capability、Fermi至Blackwell、不同芯片分支、Ada/Hopper并行分支、Blackwell 10.x/12.x、PTX/SASS兼容和跨代kernel决策；原七节环境实验后移为第十至第十六节。）
- 2026-09-03（课程体系模块化大换血：顶层文件改为薄路由，正式课程进入`roadmap/curriculum/`；第一篇第一章“GPU实验环境与硬件基线”拆成七节，并新增只读环境采集脚本。旧lesson、note、solution、weekly、性能与失败证据原地保留，由新课程映射复用。第一章尚无服务器输出，仍为`WIP`；第二章尚未讨论或开课。）
- 2026-09-03（论文线完成独立化校准：不再使用“模型主干+枝干+实践挂载点”决定阅读，改为经典主干、专题序列和最新论文观察池；论文只要求阅读、公式推导和作者关键代码，核心/重要最新论文至少精读+代码，不默认实践。当前MLA，下一篇DSA；GPU第二章仍仅有课程名称与边界，尚未讨论或开课。）
- 2026-09-03（全盘替换旧学习计划：实践主干重新定位为 Ascend NPU → NVIDIA GPU 高性能 LLM 算子，GPU 架构、完整算子体系、极致性能、低精度/量化、Prefill/Decode GPU 分析为核心；Mini Transformer 是贯穿验证载体，vLLM 是真实系统落地与加分项。论文线完全单列，继续纯阅读、公式推导和作者关键代码阅读，不要求每篇实践。旧 M0–M5/W1–W12 不再作为现行路线；已有代码、LeetGPU状态、RTX 3090数字、失败实验和周报全部保留。）

- 2026-09-02（用户确认 FA2 统一笔记已阅读完成；下一理论节为 MLA（DeepSeek-V2/V3）。该状态仅代表阅读完成，不代表 Triton 实现或 LeetGPU/服务器验证完成。独立 GPU 结构课程进入待办，待编写 lesson，覆盖 GPU/GPC/TPC/SM、CTA/Warp/Thread、CUDA Core/Tensor Core、内存层级、调度/occupancy、PTX/SASS 与 Triton 映射；不改变当前 B2 → B3 算子线顺序）
- 2026-09-01（Triton Softmax LeetGPU #5 用户通过版已原样归档至 [`solutions/triton/fused_softmax.py`](./solutions/triton/fused_softmax.py)；`SuccessPublicTrace`，2026-09-01 00:37:33，0.29 ms，47.0th percentile。B2 状态更新为 `LEETGPU_PASS`；服务器二维 row-wise baseline 尚未开始，未标记 `GPU_VALIDATED`/`COMPLETE`）
- 2026-09-01（FA2 行展开澄清：`r` 是 score/output 的 Query 行，`s` 是 score 矩阵的列索引即第 `s` 个 Key/Value token，`c` 是 V/O 的特征列；分子是对 `s` 的 `exp(S_rs) * V_s` 加权求和，分母是该行 exp(score) 的归一化和。FA2 统一笔记仍约 50% WIP，未视为学完）
- 2026-08-31（按用户裁决重规划课程：Softmax 定义、稳定性、Online Softmax、Parallel Reduce 和 CUDA Softmax 视为已掌握；B2 缩为 10 分钟 CUDA → Triton 映射 → LeetGPU #5 原始 solve/kernel 归档 → RTX 3090 row-wise baseline，完成后立即 B3 FlashAttention。旧 1-pass 重写、三版 benchmark、warp-shuffle 深钻和 Softmax P0–P8 全部降为可选优化债务；Lesson08 重写为精简迁移检查点，当前仍为 `WIP`）
- 2026-08-30（补齐 B2 正式课程 [`lessons/08-triton-fused-softmax.md`](./lessons/08-triton-fused-softmax.md)：曾包含当前单元卡、稳定公式、Triton row/program 映射、mask/`other=-inf`、starter TODO、LeetGPU #5 正确性归档、服务器 row-wise benchmark、资源边界和验收清单；当时状态 `WIP`，尚无用户代码，后由 2026-08-31 重规划为迁移检查点）
- 2026-08-30（用户决定 MatMul 先阶段性收口：LeetGPU `LEETGPU_PASS`、RTX 3090 `GPU_VALIDATED` baseline 已完成，当前最佳 20.830 ms / 19,794.1 GFLOPS / `torch.mm` 80.3%；Nsight Systems P0-lite 已归档。剩余 NCU counters、PTX/SASS、spill/occupancy、多 shape 回归和完整 P0–P8 极致优化转入 GPU 优化篇，不再阻塞主线；随后切换为 B2 Triton Softmax 迁移检查点）
- 2026-08-30（Triton MatMul Nsight Systems P0-lite：完整归档 s3/s2 两份 raw log，并按 Grid=64×16、Block=256 从 trace 剔除 4 次 correctness，重算 60 次大 shape。s3 mean 21.208 ms、255 regs/thread、0.098 MB DymSMem；s2 mean 22.362 ms、255 regs/thread、0.049 MB，慢 5.44%。结论：降低 stages 虽将 shared memory 减半，但未解除 register bottleneck，pipeline 变浅反而退化；下一步缩小输出列 tile 验证 accumulator/register pressure。NCU counters 因 AutoDL 权限阻塞，证据边界保持 P0-lite）
- 2026-08-28（Triton MatMul LeetGPU `SuccessPublicTrace` 已归档：LeetGPU #02、Triton、A100-80GB、24.54 ms、55.3th percentile；最终版仅将 `tl.dot` 指定为 `input_precision='ieee'`，历史 WIP 保留默认 TF32 失败证据（4×4 最大绝对误差 0.1275177001953125）；服务器适配版已在 RTX 3090 `GPU_VALIDATED`，MatMul 单元总体 `GPU_VALIDATED` 但尚未 `COMPLETE`，下一步 Nsight Compute / P0–P8；Vector Add 原始 `solve` 归档缺口保持不变）
- 2026-08-28（保存进度：当前主线仍为 Triton MatMul；理论侧 FlashAttention-2 统一笔记 [notes/algorithms/flash-attention-2.md](./notes/algorithms/flash-attention-2.md) 用户阅读约 50%，状态仍为 WIP/🚧，未视为已读完或已掌握；下一步继续阅读统一笔记后半部分，结合公式与 Triton/CUDA 代码映射）
- 2026-08-26（进一步把硬件知识与算子优化绑定：确定 MatMul、Softmax/Norm、FlashAttention、Fused MLP/GQA 四类极致性能锚点，新增“硬件机制→代码旋钮→counter→实测”映射、P0–P8 优化阶梯、强 baseline/roof 与停止条件）
- 2026-08-26（Triton MatMul IEEE FP32 配置 sweep：RTX 3090 最佳为 BLOCK_M=128、BLOCK_N=32、BLOCK_K=256、w8、s3，22.033 ms / 18,713.5 GFLOPS，为 torch.mm 的 77.8%；128×64×128 因 shared memory 131,072 B 超过 101,376 B 上限而编译失败）
- 2026-08-26（新增 `notes/triton/matmul-performance-analysis.md`：记录 MatMul sweep 的硬件解释、资源失败原因、Nsight Compute P0、邻域搜索、L2 排布、TF32 对照和 autotune 后续顺序）
- 2026-08-26（GPU 底层架构与优化全盘纳入主计划：统一为 G0–G8 九层能力、L0–L11 实验梯和十层优化矩阵；按 B1/B2/B3/M3/M4 Just-in-Time 挂载，不改变当前 Triton MatMul 焦点）
- 2026-08-26（Triton MatMul 服务器适配版在 AutoDL RTX 3090 完成 4 组边界正确性测试；IEEE FP32 下 Triton 24.681 ms / 16,706 GFLOPS，torch.mm 17.120 ms / 24,083.3 GFLOPS，约为 69.4%；LeetGPU 页面仍无法运行，原始代码保持 WIP）
- 2026-08-26（全盘重构章节规则：统一 WIP → LEETGPU_PASS → GPU_VALIDATED → COMPLETE 状态；lesson 只保留 LeetGPU 与服务器两段；校准 A1/A3 代码产物错配、旧状态、运行环境和链接问题）
- 2026-08-26（复盘发现 MatMul lesson 内嵌的是用户本轮 LeetGPU 编辑器快照，而 `solutions/triton/matmul.py` 仍是另一份 WIP；已在章节标明来源和未同步状态，后续通过后以原始平台版本统一归档）
- 2026-08-26（全盘校准发现两处旧产物错配：A1 CUDA Vector Add 只有 Lesson 01 代码快照、没有本地归档；PATH 原 A3 的 float tiled 名称实际对应 fp16 文件，已改为 A3=fp16 已完成、A3+=float 计划项）
- 2026-08-26（按反馈把当前 MatMul LeetGPU 草稿代码快照直接放进 Lesson 06 的 5.5 章节；明确标注未通过，后续修正后继续更新快照）
- 2026-08-26（根据反馈简化 Lesson 06 MatMul 结构：删除无独立价值的“5.5 三步走”，改为“5.5 LeetGPU：正确性与代码归档”和“5.6 服务器：真实性能”两章；题目、当前代码、归档要求直接放进对应章节）
- 2026-08-26（补齐 LeetGPU 代码归档规则：学习计划必须一眼列出题目入口、通过后的原始 `solve`/kernel、本地 `solutions/` 文件和正确性/性能证据；发现 Vector Add 当前本地文件是 wrapper，LeetGPU 原始代码未单独归档，已明确标记缺失，不再把 wrapper 当作平台版本）
- 2026-08-26（用户继续编写 Triton MatMul LeetGPU 题：已写 M/K 输出 tile、FP32 accumulator、N 维归约循环和 `tl.dot` 框架；尚未通过。当前问题为 `offset/offs` 命名不一致、`tl.arange` 归约偏移写法、A/B load 边界 mask、C 的 masked store 与循环作用域；通过前不做本地同步或 AutoDL benchmark）
- 2026-08-24（全盘复盘并统一 B1 Vector Add 验收记录格式；纠正本次 AutoDL 实际卡型为 RTX 3090，并改为运行时动态记录 GPU 型号；修正 README、课程、路线图和恢复入口中的旧状态，当前焦点为 Triton MatMul）
- 2026-08-24（开始阅读 Triton MatMul；尚未创建或编写 `solutions/triton/matmul.py`，本日学习到此结束）
- 2026-08-23（AutoDL RTX 3090 完成 Triton Vector Add 真实 GPU 验收：正确性通过，Triton 840.1 GB/s，`torch.add` 843.0 GB/s；当前焦点切换到 Triton MatMul）
- 2026-08-20：理论线完成 DeepSeek-V3.2 config + KV Cache / 权重显存 / FLOPs 三笔手算；下一步进入 FA2 → MLA → DSA。
- 2026-08-20（全盘校准：算子主线固定为 Triton B1，统一执行顺序为“自己写 → LeetGPU → 真实 GPU → benchmark”；纯 CUDA kernel 后置；新增最新论文/模型资料快照 `notes/llm/updates/2026-08-20.md`，同步 PATH/NOW/课程与构建路线）
- 2026-08-20（Triton Vector Add 由用户自己完成并通过 LeetGPU；B1 剩余真实 GPU 验证与 GB/s 记录）

- 2026-08-14（确认 DeepSeek-V4：2026-04-24 预览开源、07-31 Flash 正式、08-13 V4-Pro-0813 正式；核心 = CSA + HCA（MLA 骨架）+ Lightning Indexer + mHC/Muon + MXFP4；1M ctx 下 prefill ≈ V3.2 的 27%、KV ≈ 10%；主线不切 V4，改为 V3.2 打底、V4 做增量，挂主线 A 第 3 步；新笔记 deepseek-v4.md，tracker / 架构地图 / algorithms README / NOW 同步）
- 2026-08-13（理论线重构为"主干 + 枝干 + 字典"：主干=模型主线，枝干=必要小模块按挂载点学；主线 A 第 1 步手算是热身，A5/FA1 在第 2 步注意力接续（FA2 → MLA → DSA），训练侧枝干 A1 挪到 serving 之后；新笔记：FA2、FA4/FlexAttention、GDN/Qwen3.5、DSA、SageAttention3/Kascade、优化器 Adam/AdamW【枝干 A1】）
- 2026-08-10
- 当前主线：PATH B Triton 实现（B1 vec_add 待用户自己写，matmul 下一步）
- 并行强化：最新模型与算子构建能力（GQA/MLA/MoE/FlashAttention/PagedAttention 等）
- 当前状态：A5 读码完成；Triton Vector Add 已由用户自己写完、通过 LeetGPU，并在 AutoDL RTX 3090 完成 benchmark（840.1 GB/s vs `torch.add` 843.0 GB/s）；当前进入 MatMul；softmax_1pass 仍为后置 CUDA 草稿；本机 venv 已就绪（torch CPU + triton-windows）

---

## 1. 一句话概况

从昇腾 NPU 算子开发转向 NVIDIA GPU/ML 系统工程师方向，Triton 是主力，CUDA 作为底层，当前进入 Triton 实现阶段。

学习路线：

```text
A CUDA 打底 -> B Triton -> C 推理系统 -> D 分布式 -> E Agent
```

---

## 2. 当前主线

算子线现在做 PATH B：Triton 实现；理论线并行推进主线 A（DeepSeek-V3.2 → V4 增量）。

完成顺序：

```text
1. Triton Vector Add
2. Triton MatMul
3. Triton Fused Softmax
4. Triton Flash Attention
5. Triton GQA / Fused MLP
```

代码落盘位置：`solutions/triton/`

详细任务和验收：

- [roadmap/ai-infra-curriculum.md](./roadmap/ai-infra-curriculum.md)
- [lessons/06-triton-intro.md](./lessons/06-triton-intro.md)
- [solutions/triton/README.md](./solutions/triton/README.md)

---

## 3. 已完成进度

### 算子线

| 阶段 | 状态 | 日期/说明 |
|------|:--:|------|
| A1 CUDA 基础 + Vector Add | ✅ | LeetGPU 跑通 |
| A2 GEMM naive | ✅ | 2026-06-16，LeetGPU `2_matrix_multiplication` |
| A2+ GEMM fp16 naive | ✅ | 2026-06-22 |
| A3 GEMM tiled | ✅ | LeetGPU 跑通 |
| A3+ GEMM fp16 tiled + benchmark | ✅ | 4090 实测 tiled 约 0.6x naive，L2/occupancy 结论 |
| A4 Softmax 3-pass | ✅ | 2026-07-01，LeetGPU `5_softmax` |
| A4 Softmax 2-pass fused online | ✅ | 2026-07-11，`softmax_online.cu` |
| A4 1-pass true online | 🚧 | Agent 草稿 `softmax_1pass.cu`（2026-08-09，算法模拟+编译通过），待用户重写 |
| A4 warp shuffle / benchmark | ⏳ | 待做 |
| A5 Flash Attention 读码 | ✅ | 2026-08-10，逐段注释完成，[阅读笔记](./notes/cuda/flash-attn-reading.md)，发现 2 个真实 bug |
| B1 Triton vec_add + matmul | `GPU_VALIDATED` 已阶段性收口 | Vector Add 技术验收完成（840.1 GB/s），但原始 LeetGPU `solve` 归档缺失；MatMul baseline 已完成，剩余 P0–P8 优化延期至 GPU 优化篇 |
| B2-B5 Triton 实现 | B2 当前 | B2 Triton Softmax 迁移检查点完成后立即进入 B3 FlashAttention |

### 存档：A4 Softmax 详情（2026-07-01 ~ 08-09）

> 课程：[Lesson 04](./lessons/04-softmax.md) · 周报：[2026-07-22](./weekly/2026-07-22-softmax-online.md)

| 优化 | 说明 | 状态 |
|------|------|:--:|
| 3-pass naive | findMax → countSum → normalize，~1ms | ✅ `softmax_naive.cu` 2026-07-01 |
| 2-pass fused online | 一趟出 (partial_max, partial_sum) → host merge → normalize | ✅ `softmax_online.cu` 2026-07-11 |
| true online 1-pass | per-thread K-element scan + tree reduce merge (m,s) pair → `maxSumkernel` | 🚧 Agent 草稿 `softmax_1pass.cu` 2026-08-09（算法模拟 + 编译通过），待用户重写 |
| warp shuffle reduce | `__shfl_down_sync` 替代 shared memory 归约 | ⏳ 暂缓（用户决定） |
| benchmark 对比 | 3-pass vs online vs warp shuffle → ncu 分析带宽 | ⏳ 暂缓（用户决定） |

**2026-07-10 实践要点**：

- per-thread online scan：逐元素维护 `(m, s)` pair，公式 `m_new=max(m,val), s=s·exp(m-m_new)+exp(val-m_new)`
- tree reduce merge 公式：`s_new = s_a·exp(m_a-m_new) + s_b·exp(m_b-m_new)`，满足交换律+结合律
- 哨兵 NaN：两个空线程 merge 时 `-inf - (-inf) = NaN` → `if (m_a == -INFINITY)` 跳过
- `__syncthreads()`：同 block 所有线程必须全部到达，否则死锁；不能提前 `return`
- Device 指针：kernel 写入的 device 指针不能在 host 直接读，必须 `cudaMemcpy`；`cudaMalloc` 用 `cudaFree`
- 性能陷阱：normalize 必须多 block 并行，单线程串行 N 个 `expf` 会崩到 60ms
- LeetGPU 通过方案：3-pass（`findMax_kernel` + `countSum_kernel` + `softmax_kernel`）~1ms baseline

LeetGPU `5_softmax` 贴 `solve()` 提交；服务器 `KERNEL=xxx.cu ./run.sh` 测精度 + 带宽（harness → `solutions/cuda/softmax/main.cu` + `run.sh`）。

### 理论线

| 主题 | 状态 | 说明 |
|------|:--:|------|
| Online Softmax | ✅ 已掌握 | 能推公式，能讲 HBM 优化 |
| Parallel Reduce | ✅ 已掌握 | 树状 reduce + warp shuffle |
| Flash Attention 机制 | ✅ FA1 已消化；FA2 阅读完成 | FA1 于 2026-08-10 经 A5 读码 + 问答消化；FA2 统一笔记于 2026-09-02 阅读完成，但尚未进行 Triton 实现或 LeetGPU/服务器验证；FA3 待补 |
| INT8/FP8 量化 | 🚧 草稿 | 待消化 |
| MoE 推理 | 🚧 草稿 | 待消化 |
| Speculative Decoding | 🚧 草稿 | 待消化 |
| PD 分离 | 🚧 草稿 | 待消化 |
| MLA / DeepSeek | 🚧（DeepSeek-V3.2 config + 三笔手算 ✅ 2026-08-20；MLA / DSA 待学） | FA2 统一笔记已于 2026-09-02 阅读完成；下一理论节为 MLA（DeepSeek-V2/V3） |
| 最新模型结构 | 🚧 草稿 | 已补全详细内容 |
| 剩余理论速览 | 🚧 草稿 | 已分类补全 |
| FA2 / FA4 / GDN / DSA / SageAttention3 | FA2 阅读完成；其余 🚧 草稿 | FA2 统一笔记 2026-09-02 阅读完成；FA3/FA4/GDN/DSA/SageAttention3 随主线步骤消化，不单独排队 |
| 优化器 Adam/AdamW | 🚧 草稿 2026-08-13 | 枝干 A1 第 1 段（主线 A serving 之后） |
| DeepSeek-V4（CSA+HCA） | 🚧 草稿 2026-08-14 | 主线 A 第 3 步：CSA/HCA → mHC/Muon → MXFP4/混合精度 |

---

## 4. 大模型板块

`notes/llm/` 是内容聚合板块，不独立维护进度。

| 文件 | 内容 |
|------|------|
| [notes/llm/README.md](./notes/llm/README.md) | 板块入口和 PATH 映射 |
| [notes/llm/architectures.md](./notes/llm/architectures.md) | 模型结构 |
| [notes/llm/inference-systems.md](./notes/llm/inference-systems.md) | 推理系统 |
| [notes/llm/training-systems.md](./notes/llm/training-systems.md) | 训练系统 |
| [notes/llm/interview.md](./notes/llm/interview.md) | 面试 |
| [notes/llm/papers.md](./notes/llm/papers.md) | 论文 |
| [notes/llm/operator-building.md](./notes/llm/operator-building.md) | 最新模型与算子构建路线 |

---

## 5. 学习计划结构

所有学习计划挂在 `roadmap/` 下，当前总入口：

```text
roadmap/ai-infra-curriculum.md   # 总执行计划 M0-M5
roadmap/vllm.md                  # 推理系统源码深挖
roadmap/distributed.md           # 分布式训练
roadmap/agents.md                # Agent 实验室
roadmap/interviews.md            # 面试
roadmap/leetgpu-ladder.md        # 可选 CUDA 深钻
```

---

## 6. 重要约束

- `PATH.md` 是唯一进度权威源。
- `NOW.md` 决定当前学什么。
- `notes/llm/` 是内容聚合，不是另一条学习线。
- 算子线写代码，理论线写笔记。
- 代码从空文件开始写，参考实现只用于对照。
- 每个学习单元至少要有：正确性、性能数字、可讲清的面试口径。
- 除非用户明确要求，不要修改 `NOW.md` 和 `PATH.md`。

---

## 7. 最近变更记录

| 日期 | 内容 |
|------|------|
| 2026-08-24 | 全盘复盘 B1 进度，统一 Vector Add 验收格式并清理旧的“benchmark 待做”记录；当前焦点为 Triton MatMul |
| 2026-08-23 | Triton Vector Add 在 AutoDL RTX 3090 完成真实 GPU 正确性与带宽 benchmark；当前进入 MatMul |
| 2026-08-14 | V4 资料入库（发布线 / CSA+HCA / mHC/Muon / MXFP4 / MegaMoE / TileLang / 磁盘 KV）；tracker、架构地图、algorithms README、NOW 同步；新增 [deepseek-v4.md](./notes/algorithms/deepseek-v4.md) 草稿 |
| 2026-08-10 | NOW.md 瘦身：已完成单元移入 HISTORY 存档，NOW 只留当前焦点 + 历史跳转 |
| 2026-08-10 | 校准 B 线流程：自己写 → LeetGPU → 真实 GPU → 性能分析 |
| 2026-08-10 | A5 读码完成（笔记 + 2 个 bug）；`vector_add.py` 为 Agent 草稿待用户重写；本机 venv 搭好 |
| 2026-08-08 | 修复 `lessons/02/03/05` 中公式写在代码块内的问题，公式恢复正常渲染 |
| 2026-08-09 | A4 1-pass true online：Agent 起草 `softmax_1pass.cu`（算法模拟+编译通过），待用户重写；跳过三版 benchmark |
| 2026-08-06 | 补全所有学习计划，当前主线切到 Triton 实现 |
| 2026-08-06 | 新增 AGENTS.md、progress-resume/triton-guide skill，优化 coach agent |
| 2026-08-06 | 新增最新模型与算子构建能力路线，接入 M2.5 |
| 2026-08-06 | 新增 `notes/llm/` 大模型内容板块 |
| 2026-08-06 | 新增 `solutions/triton/` Triton 代码落盘入口 |
| 2026-08-03 | 新增 PATH 执行参考，补模型结构与理论速览 |
| 2026-07-22 | Softmax 2-pass fused 记录与 A5 准备 |
| 2026-07-11 | 完成 `softmax_online.cu` |
| 2026-07-01 | Softmax 3-pass naive LeetGPU 跑通 |
| 2026-06-22 | GEMM fp16 naive/tiled 跑通 |
| 2026-06-16 | GEMM naive 跑通 |

---

## 8. 新电脑恢复步骤

1. 读本文件，恢复上下文。
2. 读 [roadmap/ai-infra-curriculum.md](./roadmap/ai-infra-curriculum.md) 当前主线。
3. 读 [NOW.md](./NOW.md) 和 [PATH.md](./PATH.md) 确认最新状态。
4. 检查 `git status` 和 `git diff`，确认是否有未提交改动。
5. 算子线从当前 [NOW.md](./NOW.md) 的 Triton MatMul 开始；理论线按主线 A（DeepSeek-V3.2 → V4 增量）。

---

## 9. 关键文件速查

| 目的 | 文件 |
|------|------|
| 当前学什么 | [NOW.md](./NOW.md) |
| 权威进度 | [PATH.md](./PATH.md) |
| 总学习计划 | [roadmap/ai-infra-curriculum.md](./roadmap/ai-infra-curriculum.md) |
| Triton 课程 | [lessons/06-triton-intro.md](./lessons/06-triton-intro.md) |
| Triton 代码位置 | [solutions/triton/](./solutions/triton/) |
| 大模型板块 | [notes/llm/README.md](./notes/llm/README.md) |
| 理论线 | [notes/algorithms/README.md](./notes/algorithms/README.md) |
| 面试 | [roadmap/interviews.md](./roadmap/interviews.md) |

## 10. 2026-09-09 课程正文与发布边界

- GPU 第一章正文与 `roadmap/curriculum/gpu/README.md` 改为教程口径：移除状态、协作过程、旧稿提示、用户对话痕迹和本机运行叙述；保留技术条件、源码来源、运行命令、LeetGPU/服务器验证指导。
- 第一章第 6 节改为“概念辨析与常见问题”，继续使用现有 CUDA Vector Add、Triton Vector Add 和 MatMul 片段；完整 CUDA 示例注释只描述代码用途，代码块与 `.cu` 文件仍由构建检查一致。
- 课程站构建器只发布第一章，不再生成后续章节占位文章、旧稿侧栏、WIP footer 或讨论链接；构建检查 HTML 可见管理词泄漏，并检查 `__global__` 以 `<code>` 形式渲染。
- 本次没有修改 `PATH.md`、`NOW.md`、`solutions/`，没有运行真实 GPU；浏览器连接失败，因此只进行 HTTP、生成内容、语法和源文件一致性检查，不宣称视觉验收。
