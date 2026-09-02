# NOW — 现在做什么

> 进来先看这。算子线只推进一个当前单元；完整地图 → [PATH.md](./PATH.md) · 密集课表 → [roadmap/ai-infra-curriculum.md](./roadmap/ai-infra-curriculum.md) · 历史学习 → [HISTORY.md](./HISTORY.md)

---

## 🔧 算子线：当前唯一动作

**B2 Triton Softmax 迁移检查点 · `LEETGPU_PASS`**

Softmax 定义、数值稳定性、Online Softmax、Parallel Reduce 和 CUDA Softmax 已掌握，本单元不重学原理。现在只做：

```text
10 分钟 CUDA → Triton 映射
→ LeetGPU #5 从题面写 Triton
→ 原始 solve/kernel 原样归档
→ RTX 3090 row-wise baseline
→ 立即进入 B3 FlashAttention
```

- 当前课：[Lesson 08 — Triton Softmax 迁移实战 / 检查点](./lessons/08-triton-fused-softmax.md)
- 题目入口：[LeetGPU Softmax #5](https://leetgpu.com/challenges/softmax)
- 当前代码：[solutions/triton/fused_softmax.py](solutions/triton/fused_softmax.py)，LeetGPU 原始通过版已归档
- LeetGPU 结果：`SuccessPublicTrace`，2026-09-01 00:37:33，0.29 ms，47.0th percentile
- 当前服务器状态：未开始；下一步做二维 row-wise softmax 的 RTX 3090 正确性与 baseline
- 调试入口：[Lesson 07 — Triton Debugging](./lessons/07-triton-debugging.md)

**状态门槛**：当前 `LEETGPU_PASS`；RTX 3090 row-wise 正确性和 ms/GB/s 证据齐全才是 `GPU_VALIDATED`。服务器未开始，没有这些证据不写 `COMPLETE`。

**明确不阻塞当前动作**：旧 CUDA `softmax_1pass` 用户重写、三版 benchmark、warp-shuffle 深钻，以及 Softmax P0–P8 极致优化，全部放入 PATH 的可选优化债务池。

**MatMul**：已阶段性收口为 RTX 3090 `GPU_VALIDATED` baseline；剩余深钻留在 [GPU 优化篇](./roadmap/gpu-foundations.md#matmul-优化债务池-deferred-backlog)，不回头插入 B2。

- 待办：独立 GPU 结构课程（GPU/GPC/TPC/SM、CTA/Warp/Thread、CUDA Core/Tensor Core、内存层级、调度/occupancy、PTX/SASS、Triton 映射），待编写 lesson；不改变当前 B2 → B3 算子线顺序。

### 官方入口

- Triton：[Triton language API](https://triton-lang.org/main/python-api/triton.language.html)
- GPU架构：[CUDA Programming Guide — Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html) · [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- NVIDIA Ampere：[Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
- NVIDIA Hopper：[Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)

---

## 🧠 理论线

**主线 A：DeepSeek-V3.2 第 2 步 — 注意力（MLA → DSA）**

FA1 机制、online 更新公式和 merge 结合律已掌握；FA2 理论阅读已完成（2026-09-02）。当前进入 MLA（DeepSeek-V2/V3），重点是 latent KV compression、低秩投影、KV cache 取舍；不改变算子线先完成 B2 再进 B3 的顺序。

- 当前笔记：[MLA（DeepSeek-V2/V3）](./notes/algorithms/mla-deepseek.md)
- 下一理论节：DSA（DeepSeek-V3.2）；算子线仍先完成 B2，再进入 B3 Triton FlashAttention，不回头补 Softmax 旧债务

---

## 📚 已完成 / 历史入口

| 单元 | 状态 | 详情 |
|---|---|---|
| A1-A4 CUDA 算子线 | A4 知识 ✅；旧实现债务 ⭐ | [Lesson 04](./lessons/04-softmax.md) · [HISTORY.md](./HISTORY.md) |
| A5 Flash Attn 读码 | ✅ 2026-08-10 | [阅读笔记](./notes/cuda/flash-attn-reading.md) |
| FA2 理论阅读 | ✅ 2026-09-02 | [上一节：FlashAttention-2 统一笔记](./notes/algorithms/flash-attention-2.md) |
| B1 Triton Vector Add / MatMul | 阶段性收口 | [Lesson 06](./lessons/06-triton-intro.md) · [PATH.md](./PATH.md) |
| B2 Triton Softmax | `LEETGPU_PASS` 当前；服务器待做 | [Lesson 08](./lessons/08-triton-fused-softmax.md) · [PATH B2](./PATH.md) · [代码](./solutions/triton/fused_softmax.py) |

*想换方向或调节奏，直接说。*
