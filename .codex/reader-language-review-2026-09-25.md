# 课程书面表达与面试学习复核

## 范围与依据

本轮从远端 `a7c27f3` 继续修订。对象是已有 NPU 算子开发经验、正在学习 NVIDIA GPU 并准备面试的读者。审阅范围为 GPU 基础五章、算子九章、模型与系统四章，以及两份面试资料；论文阅读器的独立正文未列入本轮逐段校订范围。

阅读检查关注四个问题：概念是否先定义再使用，例子能否由前文推导，性能解释是否说明条件，以及读者能否将机制、取舍和验证方法连贯地复述。对已有结构进行局部校订，保留代码、公式、历史实验和阅读断点。

## 主要阅读障碍与处理原则

| 阅读位置 | 障碍 | 校订方向 |
|---|---|---|
| SIMT、谓词与独立线程调度 | 源码中的双路径求值与指令谓词混在一起；寄存器交换和内存同步的说明过于笼统 | 用条件执行、结果选择、数据可见性分别解释代码；明确参与线程范围 |
| 访存、归约与 GEMM | 部分长句同时堆叠布局、资源、计数器和反例 | 先说明地址或数据流，再解释资源代价，最后给出可验证的性能判断 |
| 融合、Attention、量化与 MoE | 多个英文术语连续出现，局部优化与整体收益之间缺少连接 | 首次引入术语时交代含义；围绕中间结果、缓存和通信建立因果关系 |
| 模型与系统 | 测量范围、缓存状态和通信角色缺少阅读语境 | 区分核函数、模型层、前向与请求，沿执行顺序解释各项成本 |
| 面试资料 | 重复的限制性提醒较多，技术说明容易变成零散检查项 | 保留适用条件，组织成可复述的机制与验证过程，示范表述不作为个人实绩 |

## 技术核对

- [NVIDIA CUDA Best Practices：Branch Predication](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#branch-predication)：谓词为假的线程不执行对应指令的结果写入，也不为该指令计算地址或读取操作数；这与源码先算两个值再选择的表达方式不同。
- [NVIDIA Volta Tuning Guide：Independent Thread Scheduling](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html#independent-thread-scheduling)：区分带同步的 warp 寄存器交换指令与通过共享/全局内存交换数据时的同步要求。
- [Triton Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)：核对行覆盖、二次幂补齐和编译资源的区别。归约章节进一步修正 `D=32` 时仍有空闲 lane 的过度概括，解释 `D=257` 选择 512 是本例的最小二次幂策略，并区分一次加载整行与 CUDA 跨步循环。

## 验证记录

- 与本次拉取后的 Git HEAD 对照，20 份正文的 365 个代码围栏、243 个块级公式以及原标题序列均保留。原始 `solutions/`、`reference/`、下载资料和 `PATH.md` / `NOW.md` 未修改。
- `npm run build` 通过：39 篇文章、102 篇附属 Markdown 页面、42 项资源、1734 处公式、171 段原始源码摘录。
- `node check.cjs --content-only`、`node editorial-audit.cjs`、`node offline-static.test.cjs` 通过；最终文字修订后已重建并复验。
- `node navigation.test.cjs` 通过：分区、侧栏、折叠、旧锚点、前进后退、搜索与直接路由。后续收尾仅修改段落，未改路由或脚本。
- `node link-audit.cjs` 通过：103 个 HTML、5278 个本地目标、4069 个片段链接和 1102 个附属文档目录链接。此检查不请求外站。
- 浏览器通过 HTTP 抽查第二章 SIMT 小节、归约章节的形状映射小节，以及面试页的 Prefill/Decode 段落；标题、正文、代码与公式正常显示，旧书签仍定位至对应小节。浏览器抽查不等于逐屏检查全站，也没有重新验证本地 `file://` 打开行为。
- `git diff --check` 通过。未执行新的 CUDA/Triton 编译、LeetGPU 提交或 GPU benchmark；没有新增或改写性能成绩。本轮未 commit/push。

GPU 基础、算子及模型/面试材料按互斥写集委派校订，主代理复审技术边界并完成书面语收尾、构建与浏览器检查。少量会误导读者的新句子已在验收时修正，包括 draft/target 缓存的区别、rank 的定义和形状到线程映射的适用条件；未接受与已有证据不符的完成或性能声明。
