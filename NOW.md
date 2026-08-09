# NOW — 现在做什么

> 进来先看这。两条线并列，各有"现在 + 接下来"。完整地图 → [PATH.md](./PATH.md) · 密集课表 → [roadmap/ai-infra-curriculum.md](./roadmap/ai-infra-curriculum.md) · 历史学习 → [HISTORY.md](./HISTORY.md)

---

## 🔧 算子线（动手）

**现在 · B1 — Triton Vector Add（自己写）→ MatMul**

> **课程**：[Lesson 06 — Triton 入门](./lessons/06-triton-intro.md)  
> **代码**：[solutions/triton/vector_add.py](./solutions/triton/vector_add.py) — 1D element-wise + mask 尾块  
> **前置已满足**：A5 Flash Attn 读码 ✅（[阅读笔记](./notes/cuda/flash-attn-reading.md)）

**已完成**：
- `vector_add.py` 是 Agent 代写的草稿（2026-08-10，CPU 解释器验证过）——按仓库规则"代码自己从空文件写"，**不算完成**

**接下来**：你自己从空文件写 vec_add（Lesson 06 Part 2）→ **LeetGPU 跑通 → 服务器真实 GPU 跑通 → 性能分析** → 标记 ✅ → 再写 MatMul（`tl.dot` + K 循环）

---

## 🧠 理论线（理解）

**现在 · 模型结构与最新模型**
和 A5 读码配对——Flash Attention 1 的机制（`(m, l, acc)` 滚动 + scale 修正 + tile 流转）已消化，其余草稿仍待逐条深钻。

- ✅ Flash Attention 1 机制：经 A5 读码 + 问答消化（2026-08-10），能讲清滚动公式和 tiling 为什么省 HBM
- ✅ 能推一遍 online 更新公式，能讲清"为什么比 3-pass 省 3× HBM 读写"（2026-07-10）
- ✅ merge 公式 `s_new = s_a·exp(m_a-m_new) + s_b·exp(m_b-m_new)` 满足交换律+结合律 → 可用于 tree reduce

**接下来**：模型结构与最新模型（[最新模型与结构](notes/algorithms/latest-model-architectures.md) + [剩余理论速览](notes/algorithms/remaining-theory-primer.md)）→ GQA/MLA/MoE → Mamba/SSM → 剩余理论逐条深钻

## 📚 已完成 / 历史

| 单元 | 状态 | 详情/跳转 |
|------|------|------|
| A1-A4 CUDA 算子线 | ✅ | [HISTORY.md 存档](./HISTORY.md)（含 A4 三版 softmax 细节）· [周报 07-22](./weekly/2026-07-22-softmax-online.md) |
| A5 Flash Attn 读码 | ✅ 2026-08-10 | [阅读笔记](./notes/cuda/flash-attn-reading.md)（3 个 `__syncthreads` + 2 个真实 bug） |
| 理论线已掌握 | online softmax · parallel reduce · FA1 机制 | [algorithms README](./notes/algorithms/README.md) |
| B1 草稿（不算完成） | Agent 代写，待你自己重写 | [vector_add.py](./solutions/triton/vector_add.py) · [softmax_1pass.cu](./solutions/cuda/softmax/softmax_1pass.cu) |

---

*想换方向或调节奏，直接说。*
