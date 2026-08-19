# NOW — 现在做什么

> 进来先看这。两条线并列，各有"现在 + 接下来"。完整地图 → [PATH.md](./PATH.md) · 密集课表 → [roadmap/ai-infra-curriculum.md](./roadmap/ai-infra-curriculum.md) · 历史学习 → [HISTORY.md](./HISTORY.md)

---

## 🔧 算子线（动手）

**现在 · B1 — Triton Vector Add（自己写）→ MatMul**

> **课程**：[Lesson 06 — Triton 入门](./lessons/06-triton-intro.md)  
> **代码**：[solutions/triton/vector_add.py](./solutions/triton/vector_add.py) — 1D element-wise + mask 尾块  
> **前置已满足**：A5 Flash Attn 读码 ✅（[阅读笔记](./notes/cuda/flash-attn-reading.md)）

**已完成**：
- Triton Vector Add 已由你自己完成并在 LeetGPU 跑通（2026-08-20）；真实 GPU benchmark 尚未完成

**执行纪律**：所有新算子先走 **自己从空文件写 → LeetGPU 跑通 → 服务器真实 GPU 跑通 → 性能分析**。CUDA 纯 kernel / warp shuffle / 手写 FlashAttention 后置到 Triton 主线完成后，不插队。

**接下来**：服务器真实 GPU 跑通 → 记录 GB/s → 再写 MatMul（`tl.dot` + K 循环）

---

## 🧠 理论线（理解）

**现在 · 主线 A：DeepSeek-V3.2 第 1 步 — config + 手算**

任务：先读 [DeepSeek-V3.2 手算工作纸](./notes/algorithms/deepseek-v32-handcalc.md) §2 前置——KV cache 是什么、在 DeepSeek 里 MLA/DSA/V4 怎么用——再算三笔账：KV cache（128K 单请求 ≈9GB）、权重显存（BF16 ≈1.37TB）、前向 FLOPs（4K prefill ≈303 TFLOP）；每个数写一句"所以需要 XX"。算完对照答案，过 → 第 2 步注意力（FA2 → MLA → DSA），再进第 3 步 V4 增量（CSA/HCA → mHC/Muon → FP4）。

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
| B1 当前 | Vector Add 已自己写完并通过 LeetGPU；真实 GPU benchmark 待做 | [vector_add.py](./solutions/triton/vector_add.py) · [softmax_1pass.cu](./solutions/cuda/softmax/softmax_1pass.cu) |

---

*想换方向或调节奏，直接说。*
