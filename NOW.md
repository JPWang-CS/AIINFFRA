# NOW — 现在做什么

> 进来先看这。两条线并列，各有"现在 + 接下来"。完整地图 → [PATH.md](./PATH.md) · 密集课表 → [roadmap/ai-infra-curriculum.md](./roadmap/ai-infra-curriculum.md) · 历史学习 → [HISTORY.md](./HISTORY.md)

---

## 🔧 算子线（动手）

**现在 · B2 — Triton Fused Softmax**

> **当前课程**：[Lesson 08 — Triton Fused Softmax](./lessons/08-triton-fused-softmax.md)
> **当前代码**：尚未创建；先在 LeetGPU #5 从题目模板编写，通过后归档到 `solutions/triton/fused_softmax.py`
> **LeetGPU 最终代码**：[solutions/triton/matmul_leetgpu.py](./solutions/triton/matmul_leetgpu.py) — 原始 `solve`/kernel 已归档，`LEETGPU_PASS`；SuccessPublicTrace：A100-80GB，2026-08-28 22:23:16，24.54 ms，55.3th percentile
> **历史 WIP**：[solutions/triton/matmul_leetgpu_wip.py](./solutions/triton/matmul_leetgpu_wip.py) — 默认 TF32 精度失败快照；4×4 case 最大绝对误差 `0.1275177001953125`
> **服务器代码**：[solutions/triton/matmul.py](./solutions/triton/matmul.py) — 独立验证版，RTX 3090 `GPU_VALIDATED`；当前 baseline 最佳 20.830 ms / 19,794.1 GFLOPS，`torch.mm` 80.3%
> **前置已满足**：A5 Flash Attn 读码 ✅（[阅读笔记](./notes/cuda/flash-attn-reading.md)）

**已完成**：
- Triton Vector Add 已由你自己完成并通过 LeetGPU（2026-08-20）
- AutoDL RTX 3090 正确性通过；Triton 840.1 GB/s，`torch.add` 843.0 GB/s（2026-08-23）；原始 LeetGPU `solve` 归档缺口保持不变
- Triton MatMul LeetGPU 通过并完成原始代码归档；IEEE 输入精度版本通过，历史默认 TF32 版本保留为失败案例

**执行纪律**：所有新算子只分两章：**LeetGPU 正确性与代码归档 → 服务器真实性能**。记录必须包含题目、原始 `solve`、本地代码、实际 GPU 型号和 benchmark 数字；CUDA 纯 kernel / warp shuffle / 手写 FlashAttention 后置到 Triton 主线完成后，不插队。

**当前状态**：B2 `WIP`，尚无用户代码。MatMul 已阶段性收口为 `GPU_VALIDATED` baseline，剩余优化延期至 [GPU 优化篇](./roadmap/gpu-foundations.md#matmul-优化债务池-deferred-backlog)。

**接下来**：先读 [Lesson 08](./lessons/08-triton-fused-softmax.md) Part 0–4，确认稳定公式、program mapping、mask/`other=-inf` 和 starter TODO；然后打开 [LeetGPU Softmax](https://leetgpu.com/challenges/softmax)（#5）从平台模板写第一版。

**今日进度（2026-08-30）**：MatMul baseline 收口，剩余优化转 GPU 优化篇；创建 B2 Triton Fused Softmax 正式课程，当前从 Lesson 08 Part 0 开始。

---

## 🧠 理论线（理解）

**现在 · 主线 A：DeepSeek-V3.2 第 2 步 — 注意力（FA2 → MLA → DSA）**

当前任务：继续阅读 [FlashAttention-2 统一笔记](./notes/algorithms/flash-attention-2.md)，用户已阅读约 50%，状态仍为 🚧 WIP；重点搞清 FA1 → FA2 的 work partitioning、同步/非矩阵乘开销、Q/K/V tile 分配。

**接下来**：继续阅读统一笔记后半部分，结合公式与 Triton/CUDA 代码映射；完成并消化后再进入 MLA → DSA。

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
| B1 已阶段性收口 | Vector Add 已通过 LeetGPU 并完成 AutoDL RTX 3090 benchmark（840.1 GB/s），但原始 LeetGPU `solve` 尚未单独归档；MatMul LeetGPU 已 `LEETGPU_PASS`，服务器版已 `GPU_VALIDATED`，剩余极致优化延期至 GPU 优化篇 | [Lesson 06](./lessons/06-triton-intro.md) · [性能分析](./notes/triton/matmul-performance-analysis.md) · [LeetGPU 最终版](./solutions/triton/matmul_leetgpu.py) · [历史 WIP](./solutions/triton/matmul_leetgpu_wip.py) · [服务器版](./solutions/triton/matmul.py) |
| B2 当前 | Triton Fused Softmax：课程已创建，代码尚未开始 | [Lesson 08](./lessons/08-triton-fused-softmax.md) · [PATH B2](./PATH.md) · [GPU 优化篇](./roadmap/gpu-foundations.md) |

---

*想换方向或调节奏，直接说。*
