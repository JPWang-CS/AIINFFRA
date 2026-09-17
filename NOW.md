# NOW — 现在做什么

> 这里只保留两个当前焦点：实践线一个单元、论文线一个单元。完整地图看 [PATH.md](./PATH.md)，执行细则看 [roadmap/ai-infra-curriculum.md](./roadmap/ai-infra-curriculum.md)，历史证据看 [HISTORY.md](./HISTORY.md)。

---

## 课程网页

- [当前实践：第一篇第一章《从 CUDA 程序看 GPU 的整体结构》→ 5.《架构、芯片、产品与软件版本》](http://127.0.0.1:8765/roadmap/curriculum/gpu/course-site/index.html?chapter=1#chapter-1-section-23)
- [当前论文：MLA（低维缓存与权重吸收）](http://127.0.0.1:8765/roadmap/curriculum/gpu/course-site/index.html?chapter=25)

本机服务入口：若一条打不开，从仓库根目录运行 `python -m http.server 8765 --bind 127.0.0.1`。

## 实践线：从模块化课程第一篇第一章重新开始（WIP）

当前从《从 CUDA 程序看 GPU 的整体结构》开始学习。用户会自行跳过已经掌握的内容；这次回到第一篇只改变课程入口，不清零既有实验。PATH/HISTORY 中已有的 `LEETGPU_PASS`、`GPU_VALIDATED`、原始代码与性能证据继续有效，经过对应章节时直接复盘或跳过；遇到验收缺口、环境变化需复测或明确的后续优化，再按对应流程补齐。

- 当前课：[第一篇第一章：从 CUDA 程序看 GPU 的整体结构 → 5.《架构、芯片、产品与软件版本》](http://127.0.0.1:8765/roadmap/curriculum/gpu/course-site/index.html?chapter=1#chapter-1-section-23)
- 课程总入口：[GPU 硬件与性能基础](./roadmap/curriculum/gpu/README.md)
- Softmax 服务器真实性能验证仍是历史未完成项，但不再作为当前入口；后续在对应算子章按验收补齐

- 2026-09-17 已阅读完成第一章从开头到 4.5；第 5 节《架构、芯片、产品与软件版本》尚未阅读，下一入口为该节。当前课程状态保持 `WIP`，本次仅推进阅读。

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
