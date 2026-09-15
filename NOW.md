# NOW — 现在做什么

> 这里只保留两个当前焦点：实践线一个单元、论文线一个单元。完整地图看 [PATH.md](./PATH.md)，执行细则看 [roadmap/ai-infra-curriculum.md](./roadmap/ai-infra-curriculum.md)，历史证据看 [HISTORY.md](./HISTORY.md)。

---

## 课程网页

- [主课程](http://127.0.0.1:8765/roadmap/curriculum/gpu/course-site/index.html?chapter=1)
- [论文与算法网页](http://127.0.0.1:8765/roadmap/curriculum/gpu/course-site/index.html?chapter=25)

本机服务入口：若一条打不开，从仓库根目录运行 `python -m http.server 8765 --bind 127.0.0.1`。

## 实践线：Triton Softmax 迁移检查点

状态：`LEETGPU_PASS`。Softmax 定义、数值稳定性、Online Softmax、Parallel Reduce 和 CUDA Softmax 已掌握，本单元不重学原理。

```text
10 分钟 CUDA → Triton 映射
→ RTX 3090 二维 row-wise correctness
→ RTX 3090 strong baseline（ms / 相对 torch.softmax / effective GB/s）
→ 证据归档
→ 按需要阅读 GPU 基础与对应算子；优化在该算子内继续
```

- 当前课：[Lesson 08 — Triton Softmax 迁移实战 / 检查点](./lessons/08-triton-fused-softmax.md)
- 题目入口：[LeetGPU Softmax #5](https://leetgpu.com/challenges/softmax)
- 原始通过版：[solutions/triton/fused_softmax.py](./solutions/triton/fused_softmax.py)
- LeetGPU 证据：`SuccessPublicTrace`，2026-09-01 00:37:33，0.29 ms，47.0th percentile
- 服务器状态：未开始；只有 RTX 3090 row-wise 正确性和性能数字齐全后，才升为 `GPU_VALIDATED`
- 当前不做：旧 CUDA `softmax_1pass` 重写和重复三版 benchmark；Softmax/Norm 的系统性能优化并未取消，在对应算子章的实践与优化部分继续

MatMul 已阶段性收口为 RTX 3090 `GPU_VALIDATED` baseline；已有数据不清零，NCU、PTX/SASS、spill/occupancy、低精度和多 shape 将在 GEMM 章内继续深化。

GPU主课入口：[GPU硬件与性能基础](./roadmap/curriculum/gpu/README.md)，按整体结构、执行、存储、协作和性能证据五章组织。教材生成不代表用户阅读或服务器实验完成；当前Softmax实践和MLA论文焦点保持不变。[完整算子总览](./roadmap/curriculum/operators/README.md)与首章供后续衔接。

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
