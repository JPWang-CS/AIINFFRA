# 2026-09-25 全课程阅读校订

## 范围与完成口径

本轮针对课程阅读器的全部 39 篇正文：18 篇 GPU、算子、模型与系统课程，21 篇论文/算法专题；同时完整审阅两份面试资料。附属页面另做发布、链接与导航回归，不把自动检查说成逐篇精读。论文专题全文走读与原论文核对也分开记录：本轮没有声称完整读完资料库中每一份原始 PDF。

此前浅层语言校订见 [第一阶段记录](reader-language-review-2026-09-25.md)。本记录追加全文阅读与技术核对，不倒改此前的覆盖范围。

用户既有实验、原始 solve/kernel、阅读进度及真实性能记录保留。本轮教材修订不代表用户已学完、LeetGPU 通过或 GPU 实验完成。没有获得本轮 commit/push 授权。

## 工作依赖与写集

| 工作项 | 写集 | 证据与验收 |
|---|---|---|
| 论文 A | Online Softmax、Reduce、FA1/FA2、GQA、MLA 六篇源正文 | [逐篇阅读记录](full-reader-review-2026-09-25-papers-a.md)，公式/代码差异复核 |
| 论文 B | DSA、GDN、FA4、Sage3/Kascade、V3.2、V4、V4.1 七篇 | [逐篇阅读记录](full-reader-review-2026-09-25-papers-b.md)，原论文/配置来源，发布层可见性 |
| 论文 C | MoE、量化、投机解码、PD 分离、PagedAttention、优化器、ZeRO、推理负载八篇 | [逐篇阅读记录](full-reader-review-2026-09-25-papers-c.md)，算法与系统语义复核 |
| 主课 GPU | GPU 目录五篇课程 README | [全文记录](full-reader-review-2026-09-25-gpu.md)及 source-check 回归 |
| 主课算子 | operators 目录九篇课程 README | [全文记录](full-reader-review-2026-09-25-operators.md)、历史实验与验收边界保留 |
| 主课模型/系统 | model-analysis、mini-transformer、systems、multi-gpu 四篇 README | [全文记录](full-reader-review-2026-09-25-systems.md)、状态与端到端论证 |
| 面试与集成 | roadmap/interviews.md、notes/llm/interview.md、发布脚本/介绍、HISTORY、本记录 | 招聘与题型来源、公式算例、全站构建和实际浏览器 |

各组互斥写入。先完成论文，再复用工作代理深入主课；主代理负责面试资料、跨篇矛盾、发布层和验收。返工只处理发现的问题，不重新调查已确认的部分。

## 阅读方法

按章节顺序读至文件末尾，检查新术语是否先定义、公式符号与张量维度是否明确、代码是否真正实现所述算法、结论是否能由前文推出。保留必要推导与失败案例，不以删除内容或统一模板代替校订。已有清楚段落允许不改，但阅读记录应说明覆盖和判断。

技术事实优先核对原论文、作者实现及官方文档。新架构的报告、模型配置、参考实现和生产引擎版本不混用；性能数字记录硬件、工作负载和比较对象，未满足条件时不推广。原论文正文核对范围见各组记录。

## 招聘与求职问题来源

核对日期为 2026-09-25。公开页面可能变化，以下是准备方向样本，不是市场统计或个人资格判断：

- [NVIDIA CUTLASS Kernels](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Software-Engineer--CUTLASS-Kernels-_JR2018988)：矩阵算子、体系结构、DSL/C++、实现和性能分析。
- [NVIDIA AI Inference Performance](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Software-Engineer---AI-Inference-Performance_JR2024262)：模型到服务的性能路径与可复现基准。
- [NVIDIA Inference Systems 新毕业生岗位](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Software-Engineer--AI-Inference-Systems---New-College-Graduate-2026_JR2015076)：计算机基础、推理引擎和工程实现；用于和高级岗位区分要求层次。
- [腾讯高性能算子优化](https://careers.tencent.com/jobdesc.html?postId=2072509101644103680)：CUDA/CUTLASS/Triton、通信与计算重叠、端到端性能。
- [Inferact Kernel Engineering](https://jobs.ashbyhq.com/inferact/384d9db8-c712-4caa-8091-444b4189e161)：kernel、低精度、工具与开源引擎。
- [Anthropic GPU Performance](https://job-boards.greenhouse.io/anthropic/jobs/4926227008)：算子、框架、分布式与生产负载。
- [Tilde Kernel Engineer](https://jobs.ashbyhq.com/tilderesearch/bc4e4071-cf64-4460-8265-b1e5a603d6b8)：算法、实现与技术表达。
- [FriendliAI GPU Kernel](https://jobs.ashbyhq.com/friendliai/cd4dd23b-cf94-46ec-afc3-f84037dca735/)：低精度、数值与跨设备 kernel；作为补充对照，未新增无依据的岗位频率结论。

[牛客个人 NPU 面经](https://www.nowcoder.com/feed/main/detail/57033fe8898b4e85b97c48d8b2dcfb28)和 [Reddit GPU screening 个人记录](https://www.reddit.com/r/qualcomm/comments/1sebf82/qualcomm_finished_gpu_screening_round_how_to/)只用于观察手写基础算子、线程/缓存和项目解释这类题型。技术答案另由官方材料与自行推导支撑，不复制面经中的未经证实结论。

面试资料新增两组实际推理问题：

- GPU：地址与 sector、tile/资源退化、归约尾部、越界定位、profiler 干扰、浮点路径、自定义算子注册、异步资源生命周期、张量并行和 NPU 经历表述。
- LLM：GQA/GiB 与分页、在线 Softmax 合并、MLA 联合缓存、投机采样接受率、量化反而变慢、吞吐/P99 冲突和 ZeRO 峰值显存。

官方方法核对包括 [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)、[Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)、[CUDA kernel/内存模型](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)、[PyTorch custom op](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)，以及两份面试资料中的算法原论文链接。

## 主代理复核与返工

- GDN 首轮改稿把预测误差取自衰减前状态，与原论文式 (8) 不等价；要求改为先衰减状态再计算误差，并纠正 Value 维度及混合注意力的复杂度表。记录这一返工，不能仅接受代理自报。
- 发布器会删除规划段，也会覆盖论文引言；新增实质问题不能放在被过滤的 H2 下。逐篇检查源文修订是否进入最终网页。
- 性能倍数、线性复杂度和总容量/每步流量是重点复核项；删除无依据的普遍化说法，不删除既有真实实验。
- GQA 原稿的 SuperGLUE 表与原论文不符；主代理查阅原论文 §3/Table 1 后改为 T5-XXL 生成任务结果，明确 TPUv4、batch 与并行设置，不当作 GPU 对照实验。源标题改名沿用原阅读器编号。
- MLA 初稿补入跨块重标定，但累加表达式仍保留 token 维而没有归约；主代理改为矩阵乘累加并解释维度。FA1 教学代码另修正 max 返回值、广播、缩放和非整除尾块；在线 Softmax 摘要表不再把 FlashAttention 写成固定一次读取。
- 论文 B 初稿仍遗留 NVFP4 尾数、矩阵操作数、V3.2 hidden state 与容量单位错误，返工后按原论文与算术检查修正。GDN 数学转义中出现的控制字符也已清理，并加入自动拒绝规则。
- 论文 C 初稿的投机采样、量化与分片生命周期仍有问题，已要求返工：完整分布与候选概率分离，区分接受数和前进量；量化零输入与累加缩放顺序；反向前参数可用性及峰值显存。未仅凭初稿“完成”报告验收。
- 新增 reader-semantics.test.cjs，验证面试问题没有被发布过滤器吞掉，以及 GQA/MLA 容量、页尾、在线合并、GDN 更新和拒绝采样算例；它是定点回归，不是全文正确性证明。

## 验收

CPU 检查（项目 .venv/Scripts/python.exe）：

- papers/examples 中 attention_checks.py、v41_checks.py、workload_checks.py、v41_tensor_walkthrough.py、v41_training_contracts.py、v41_engram_state.py、workload_lifecycle.py 均退出 0。
- model-analysis/examples/registration_math_checks.py，算子 02/04 的 cpu_semantics.py、05 的 cpu_attention_checks.py、06 的 cpu_paged_decode_checks.py 均退出 0。
- 量化 quantization_lab.py 的 --demo 与默认 12 项测试通过；--help 仅检查命令入口，不计为正确性测试。
- CPU Torch 比较 GQA 分组 einsum 与 repeat_interleave 稠密参考，H/G 为 4/2、6/1、8/8 时最大差均为 0。临时验证命令曾因下标书写错误失败一次，修正验证命令后通过；不是仓库代码失败。
- 直接提取 FA1 正文单头 forward Python 围栏，在 CPU Torch 执行 N=1/31/32/33/65、d=8/16 的十组输入，与稠密参考对照通过；最大绝对差不超过 8.345e-7。
- 直接提取 INT8 正文教学围栏，在 CPU Torch 检查全零 scale，以及 256 项整数内积在 FP32 缩放后再转 FP16 的溢出回归，退出 0。

网页冻结后的验收：

- npm run build 退出 0：39 篇正文、102 篇附属 Markdown 页面、42 项资源、1780 处公式、171 段原始源码摘录。
- node check.cjs --content-only、node editorial-audit.cjs、node reader-semantics.test.cjs、node offline-static.test.cjs 均退出 0。
- node navigation.test.cjs、node paper-reader.test.cjs 均退出 0，覆盖分区、路由、折叠、历史、搜索、21 篇论文加载、公式/代码与复制。
- 浏览器通过 HTTP 实际抽查 GQA 结果表的旧书签、GDN 公式与伪码、面试页的投机采样算例；目录能跳转，文字、表格、公式正常显示。不是逐屏截图全站。
- git diff --check 通过。PATH.md、NOW.md、solutions/、reference/ 和 downloads/ 无本轮修改。

- 冻结后 node link-audit.cjs 退出 0：103 HTML、5365 个本地 href/src 目标、4150 个目标片段、1126 个附属文档目录片段。外站链接提供来源入口，此检查没有逐一请求外站。
- node theme.test.cjs 与 node font-anchor.test.cjs 均退出 0，覆盖主题及字体就绪后的锚点稳定、用户滚动取消与导航取消。
- 浏览器最后恢复到第二章 SIMT（chapter=2#chapter-2-section-5）；没有标记新的阅读完成。HTTP 页面可访问，file:// 只做离线结构检查，未声称本轮完成本地文件模式视觉验收。

任务图全部收尾：三组论文与三组主课审阅产物已接收；存在的初稿错误已返工并在上文记录。未保留冲突写集，没有后台继续修改正文的工作代理。各组记录中的早期判断不覆盖本页最终验收结论。

## 完成边界

39 篇正文及两份面试资料已完成逐篇阅读与校订；保留章节已有的严谨段落，不为增加差异而重写。副页、原始参考源码与所有原 PDF 不在逐篇全文重读的声明范围内。原论文核对集中在正文引用的定义、公式、配置和实验结论，所读材料与限制分别记录。

未做新的 CUDA/Triton 编译、LeetGPU 提交、GPU/NPU benchmark、分布式训练或真实服务压测。CPU 数学检查与网页测试不能替代这些验证；公开招聘、模型配置和软件文档仍可能更新。本轮未 commit/push。
