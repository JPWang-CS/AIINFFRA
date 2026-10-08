# NOW — 现在做什么

> 这里只保留两个当前焦点：实践线一个单元、论文线一个单元。完整地图看 [PATH.md](./PATH.md)，执行细则看 [roadmap/ai-infra-curriculum.md](./roadmap/ai-infra-curriculum.md)，历史证据看 [HISTORY.md](./HISTORY.md)。

---

## 课程网页

- [静态课程入口](./roadmap/curriculum/gpu/course-site/index.html)
- [当前实践：第一篇第二章《CUDA 执行模型与指令调度》→ 4.《Block 波次与尾部利用率》](./roadmap/curriculum/gpu/course-site/index.html?chapter=2#chapter-2-section-8)
- [当前论文：MLA（低维缓存与权重吸收）](./roadmap/curriculum/gpu/course-site/index.html?chapter=25)

课程网页是预生成的静态文件。在文件管理器中双击 `roadmap\curriculum\gpu\course-site\index.html`，用浏览器阅读；不需要启动本机服务。若 Codex 内点击 Markdown 链接只显示 HTML 源码，请用文件管理器打开同一文件。

## 实践线：从模块化课程第一篇第一章重新开始（WIP）

当前沿第一篇课程继续学习。第一章已读完，第二章第 1 节与第 2 节（含 2.1–2.3）及第 3 节《SIMT 与分支执行》已读；用户会自行跳过已经掌握的内容，既有实验不清零。PATH/HISTORY 中已有的 `LEETGPU_PASS`、`GPU_VALIDATED`、原始代码与性能证据继续有效，经过对应章节时直接复盘或跳过；遇到验收缺口、环境变化需复测或明确的后续优化，再按对应流程补齐。

- 当前课：[第一篇第二章：CUDA 执行模型与指令调度 → 4.《Block 波次与尾部利用率》](./roadmap/curriculum/gpu/course-site/index.html?chapter=2#chapter-2-section-8)
- 课程总入口：[GPU 硬件与性能基础](./roadmap/curriculum/gpu/README.md)
- Softmax 服务器真实性能验证仍是历史未完成项，但不再作为当前入口；后续在对应算子章按验收补齐

- 2026-10-08 已阅读第一篇第一章、第二章第 1 节与第 2 节（含 2.1–2.3），并完成第 3 节《SIMT 与分支执行》。下一入口为第 4 节《Block 波次与尾部利用率》：`chapter=2#chapter-2-section-8`。当前课程状态保持 `WIP`，本次仅推进阅读，未新增 GPU 验证事实。

本次学习分析：讨论了行主序 GEMM 中 A[M,N]、B[N,K] 的地址展开，四条独立 FMA 累加链对 ILP 与求和顺序的影响，以及大数吃小数、FP32/FP64、树形归约、Kahan 与 FMA 的精度边界。从本轮提问看，你已经开始把调度改写与数值代价联系起来。接下来可练习用一段代码分别说明执行依赖、求和顺序和舍入误差，再用参考结果检验判断。

下一步学习建议：先只读第 4 节，手算假设 4 个 SM、每个 SM 同时驻留 2 个 block、各 block 耗时相同的情形，分别求 grid=8/9/16/17 时的波数与末波 block 数；明确区分 block 波次尾部与 warp 分歧，暂时不需要跑 GPU。再读第 5 节，连接 input、accumulator、output dtype 与 FMA 精度。可用以下问题复述第 3 节：

- 一个分支谓词下，warp 中哪些 lane 仍活动？谓词执行与真正分支在代价上如何区分？
- warp 内寄存器交换解决什么问题，为什么不能替代 block 级内存同步？
- 分支分歧、谓词执行和越界 mask 分别会怎样影响有效工作量与可观测性能？

MatMul 已阶段性收口为 RTX 3090 `GPU_VALIDATED` baseline；已有数据不清零，NCU、PTX/SASS、spill/occupancy、低精度和多 shape 将在 GEMM 章内继续深化。

GPU主课入口：[GPU硬件与性能基础](./roadmap/curriculum/gpu/README.md)，按整体结构、执行、存储、协作和性能证据五章组织。教材生成不代表用户阅读或服务器实验完成。[完整算子总览](./roadmap/curriculum/operators/README.md)与首章供后续衔接。

---

## 论文线：MLA（DeepSeek-V2/V3）

状态：`WIP`。论文线与实践线同级，当前只做纯阅读、公式推导和关键代码阅读，不要求现在实现 kernel。

- 当前笔记：[MLA（DeepSeek-V2/V3）](./notes/algorithms/mla-deepseek.md)
- 本轮必须回答：latent KV compression 的张量与存储路径、低秩投影公式、Decode 时 KV cache 的取舍、与 MHA/GQA/FA 的关系、关键实现代码如何落到 GPU memory/layout
- 论文卡出口：能不看材料讲动机、关键公式、数据流、代码路径和边界；达到可独立讲解后再进入下一篇
- 上一节：[FlashAttention-2 统一笔记](./notes/algorithms/flash-attention-2.md) 已于 2026-09-02 阅读完成，但未实现、未做 LeetGPU/服务器验证

---

## 只在需要时打开

- 总地图：[PATH.md](./PATH.md)
- GPU架构课程：[第一篇路由](./roadmap/curriculum/gpu/README.md)
- 完整执行计划：[roadmap/ai-infra-curriculum.md](./roadmap/ai-infra-curriculum.md)
- 统一验收流程：[roadmap/execution-system.md](./roadmap/execution-system.md)
- Triton 调试：[Lesson 07](./lessons/07-triton-debugging.md)

官方资料入口：[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) · [Triton](https://triton-lang.org/main/) · [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) · [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/)
