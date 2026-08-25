# NOW — 现在做什么

> 进来先看这。两条线并列，各有"现在 + 接下来"。完整地图 → [PATH.md](./PATH.md) · 密集课表 → [roadmap/ai-infra-curriculum.md](./roadmap/ai-infra-curriculum.md) · 历史学习 → [HISTORY.md](./HISTORY.md)

---

## 🔧 算子线（动手）

**现在 · B1 — Triton MatMul（从空文件开始）**

> **课程**：[Lesson 06 — Triton 入门](./lessons/06-triton-intro.md)  
> **代码**：[solutions/triton/matmul.py](./solutions/triton/matmul.py) — tiled GEMM，`tl.dot` + K 循环（待创建）
> **前置已满足**：A5 Flash Attn 读码 ✅（[阅读笔记](./notes/cuda/flash-attn-reading.md)）

**已完成**：
- Triton Vector Add 已由你自己完成并通过 LeetGPU（2026-08-20）
- AutoDL RTX 3090 正确性通过；Triton 840.1 GB/s，`torch.add` 843.0 GB/s（2026-08-23）

**执行纪律**：所有新算子先走 **看完原理 → 去 LeetGPU 题目编辑器写题并通过 → 同步本地 solutions → AutoDL 实际 GPU benchmark → 性能分析**。记录必须包含本次实际 GPU 型号；CUDA 纯 kernel / warp shuffle / 手写 FlashAttention 后置到 Triton 主线完成后，不插队。

**接下来**：先看完 Part 5 原理，直接去 [LeetGPU Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication)（#02）写 MatMul；提交通过后同步本地，再上真实卡记录 GFLOPS。

**今日收尾（2026-08-24）**：已开始阅读 MatMul 部分；尚未创建或编写 `solutions/triton/matmul.py`。

---

## 🧠 理论线（理解）

**现在 · 主线 A：DeepSeek-V3.2 第 2 步 — 注意力（FA2 → MLA → DSA）**

当前任务：阅读 [FlashAttention-2 笔记](./notes/algorithms/flash-attention-2.md)，重点搞清 FA1 → FA2 的 work partitioning、同步/非矩阵乘开销、Q/K/V tile 分配；完成一页总结后，再进入 MLA → DSA。

第 1 步是热身：三笔账给第 2 步的 MLA/DSA 和第 3 步的 V4 增量提供数字基础。A5 读完的 FlashAttention 在第 2 步接续（FA2 → MLA → DSA）；训练侧枝干 A1（FP8 训练 → [优化器](./notes/algorithms/optimizers-adam.md) → ZeRO/FSDP）挪到 serving 之后，不插队。

**路线**：第 1 步热身（手算）→ 第 2 步注意力（FA2 → MLA → DSA）→ 枝干 A2（KV 量化）→ 第 3 步 V4 增量（CSA/HCA → mHC/Muon → FP4）→ 第 4 步 MoE → 枝干 A3（权重量化）→ 第 5 步 MTP → 第 6 步 serving（含 V4 磁盘 KV / TileLang）→ 枝干 A1（训练侧）→ 主线 B（Qwen3.5，含枝干 B1 Mamba/SSM）。最新论文与资料快照见 [2026-08-20 update](./notes/llm/updates/2026-08-20.md)。完整路由：[algorithms README](./notes/algorithms/README.md)

**已完成**
- ✅ FA1 机制（2026-08-10，A5 读码消化）：滚动公式 + tiling 为什么省 HBM
- ✅ online 更新公式（2026-07-10）：为什么比 3-pass 省 3× HBM 读写
- ✅ merge 公式满足交换律 + 结合律，可用于 tree reduce

**备注**：V4 已确认存在——2026-04-24 预览开源（Pro ~1.6T / ~49B active · Flash 284B / ~13B active），07-31 Flash 正式，08-13 V4-Pro-0813 正式。主线不切 V4：V3.2 是完整开源基线，V4 的 27%/10% 需要 V3.2 做分母；V4 作为增量挂在第 2 步注意力之后（CSA/HCA → mHC/Muon → FP4），详见 [deepseek-v4.md](./notes/algorithms/deepseek-v4.md)。2026-08-13 新草稿（FA2 / GDN / DSA / FA4 / SageAttention3-Kascade）仍随主线步骤消化，不单独排队。

## 📚 已完成 / 历史

| 单元 | 状态 | 详情/跳转 |
|------|------|------|
| A1-A4 CUDA 算子线 | ✅ | [HISTORY.md 存档](./HISTORY.md)（含 A4 三版 softmax 细节）· [周报 07-22](./weekly/2026-07-22-softmax-online.md) |
| A5 Flash Attn 读码 | ✅ 2026-08-10 | [阅读笔记](./notes/cuda/flash-attn-reading.md)（3 个 `__syncthreads` + 2 个真实 bug） |
| 理论线已掌握 | online softmax · parallel reduce · FA1 机制 | [algorithms README](./notes/algorithms/README.md) |
| B1 当前 | Vector Add 已自己写完、通过 LeetGPU，并完成 AutoDL RTX 3090 benchmark（840.1 GB/s）；MatMul 阅读中 | [vector_add.py](./solutions/triton/vector_add.py) · [MatMul 参考](./reference/triton/matmul/matmul.py) |

---

*想换方向或调节奏，直接说。*
