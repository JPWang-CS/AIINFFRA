# NOW — 现在做什么

> 这里只保留两个当前焦点：实践线一个单元、论文线一个单元。完整地图看 [PATH.md](./PATH.md)，执行细则看 [roadmap/ai-infra-curriculum.md](./roadmap/ai-infra-curriculum.md)，历史证据看 [HISTORY.md](./HISTORY.md)。

---

## 课程网页

- [静态课程入口](./roadmap/curriculum/gpu/course-site/index.html)
- [当前实践：访存与布局算子 → 服务器：真实性能](./roadmap/curriculum/gpu/course-site/index.html?chapter=7#chapter-7-section-28)
- [当前论文：MLA（低维缓存与权重吸收）](./roadmap/curriculum/gpu/course-site/index.html?chapter=25)

课程网页是预生成的静态文件。在文件管理器中双击 `roadmap\curriculum\gpu\course-site\index.html`，用浏览器阅读；不需要启动本机服务。若 Codex 内点击 Markdown 链接只显示 HTML 源码，请用文件管理器打开同一文件。

## 实践线：Triton Matrix Transpose V2 服务器验证（WIP）

Triton Matrix Transpose V2 的 LeetGPU 原始代码已归档，状态为 `LEETGPU_PASS`；`SuccessPublicTrace` 为 2026/10/13 23:21:59，耗时 `0.33 ms`，`50.5th` percentile。当前唯一实践焦点是服务器正确性与真实性能验证，尚未提供 GPU 型号、shape、dtype、重复次数、耗时或有效带宽，因此不推进到 `GPU_VALIDATED`。下一步先补服务器测试与计时入口，再验证 V2。

- 当前课：[访存与布局算子 → 服务器：真实性能](./roadmap/curriculum/gpu/course-site/index.html?chapter=7#chapter-7-section-28)
- 平台代码：[transpose_v2_leetgpu.py](./solutions/triton/transpose_v2_leetgpu.py)
- 单元卡：[Triton Matrix Transpose V2](./notes/triton/transpose-v2-2026-10-13.md)
- 原始平台结果：[2026-10-13-transpose-v2-leetgpu.txt](./notes/triton/logs/2026-10-13-transpose-v2-leetgpu.txt)
- 题目入口：[LeetGPU Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose)
- 课程中的平台源码示例：[第 6.1 节](./roadmap/curriculum/gpu/course-site/index.html?chapter=7#chapter-7-section-18)；平台验收段：[LeetGPU：正确性与代码归档](./roadmap/curriculum/gpu/course-site/index.html?chapter=7#chapter-7-section-27)
- 课程总入口：[GPU 硬件与性能基础](./roadmap/curriculum/gpu/README.md)
- Softmax 服务器真实性能验证仍是历史未完成项，但不再作为当前入口；后续在对应算子章按验收补齐

- 第一篇 GPU 基础课程的阅读与实验入口保留为[第二章《CUDA 执行模型与指令调度》→ 6.《依赖链、分支与执行调度实验》](./roadmap/curriculum/gpu/course-site/index.html?chapter=2#chapter-2-section-12)，不是当前实践入口。第一章已读完，第二章第 1–5 节（含子节）已读；阅读断点不变。
- branch-probe 的 split/uniform 中位数比（`steps=1/32/256`）为 `1.009132/1.500624/1.929890`；这是已完成实验复盘，不是当前转置任务。原始服务器输出见[RTX 3090 branch-probe 日志](./notes/cuda/logs/2026-10-10-branch-probe-rtx3090.txt)，代码到结果的说明见[RTX 3090 执行调度实验记录](./notes/cuda/execution-and-scheduling-rtx3090-2026-10-09.md)。原始日志的 min/max 字段受统计打印 bug 影响，不作为有效极值。本机 nvcc 静态 SASS 不能替代服务器实际二进制核对；无需重复已通过的 probe。

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
