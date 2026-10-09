# NOW — 现在做什么

> 这里只保留两个当前焦点：实践线一个单元、论文线一个单元。完整地图看 [PATH.md](./PATH.md)，执行细则看 [roadmap/ai-infra-curriculum.md](./roadmap/ai-infra-curriculum.md)，历史证据看 [HISTORY.md](./HISTORY.md)。

---

## 课程网页

- [静态课程入口](./roadmap/curriculum/gpu/course-site/index.html)
- [当前实践：第一篇第二章《CUDA 执行模型与指令调度》→ 6.《依赖链、分支与执行调度实验》](./roadmap/curriculum/gpu/course-site/index.html?chapter=2#chapter-2-section-12)
- [当前论文：MLA（低维缓存与权重吸收）](./roadmap/curriculum/gpu/course-site/index.html?chapter=25)

课程网页是预生成的静态文件。在文件管理器中双击 `roadmap\curriculum\gpu\course-site\index.html`，用浏览器阅读；不需要启动本机服务。若 Codex 内点击 Markdown 链接只显示 HTML 源码，请用文件管理器打开同一文件。

## 实践线：第一篇第二章，执行调度实验（WIP）

第一章已读完，第二章第 1–5 节（含子节）已读。第 6 节《依赖链、分支与执行调度实验》已在 RTX 3090 完成默认构建与 `sm_86` native 的小/大 shape 实测及小 shape memcheck，设备正确性记为 `GPU_VALIDATED`；当前焦点为“检查最终 SASS 与多轮计时”，课程整体仍为 `WIP`。既有实验和性能证据保持有效。

- 当前课：[第一篇第二章：CUDA 执行模型与指令调度 → 6.《依赖链、分支与执行调度实验》](./roadmap/curriculum/gpu/course-site/index.html?chapter=2#chapter-2-section-12)
- 课程总入口：[GPU 硬件与性能基础](./roadmap/curriculum/gpu/README.md)
- Softmax 服务器真实性能验证仍是历史未完成项，但不再作为当前入口；后续在对应算子章按验收补齐

- 2026-10-09 已完成第 6 节默认构建与 `sm_86` native 用户复测。四个 kernel、尾块、各自 CPU reference、计时边界和源码 FMA 吞吐的讲解见[RTX 3090 实验笔记](./notes/cuda/execution-and-scheduling-rtx3090-2026-10-09.md)；默认构建原始日志见[实验原始日志](./notes/cuda/logs/2026-10-09-execution-and-scheduling-rtx3090.txt)，sm86 原始日志见[sm86 原始日志](./notes/cuda/logs/2026-10-09-execution-and-scheduling-rtx3090-sm86.txt)，讨论记录见[CUDA 执行模型讨论（2026-10-09）](./roadmap/curriculum/decisions/2026-10-09-cuda-execution-discussion.md)。下一步检查最终 SASS 与多轮计时。
- 当前新增的 `--branch-probe` 仍是 `WIP`：代码已进入同一 `.cu` 和服务器同路径脚本，下一步由服务器 `git pull` 后编译运行；本机只有真实 nvcc 编译与 CPU self-test，未宣称 GPU probe 通过。

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
