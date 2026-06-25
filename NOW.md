# NOW — 现在做什么

> 进来先看这。两条线并列，各有"现在 + 接下来"。完整地图 → [PATH.md](./PATH.md)

---

## 🔧 算子线（动手）

**现在 · A4 — Softmax**
GEMM（naive + tiled）已全部完成。这一步写 Softmax：naive 3-pass → online 1-pass → warp shuffle reduce。

- 读：[Lesson 04 — Softmax](./lessons/04-softmax.md) · 深入 [warp-and-sync §4](./notes/cuda/warp-and-sync.md)
- 理论提前铺好了：[online softmax](./notes/algorithms/online-softmax.md) · [parallel reduce](./notes/algorithms/parallel-reduce.md)
- 自己写：`softmax_naive`（3-pass），LeetGPU `5_softmax` 跑通
- 验收：
  - [ ] 结果正确（max abs err < 1e-5）
  - [ ] 理解 max trick 为什么必须
  - [ ] 能解释 warp shuffle reduce 的 `__shfl_down_sync` 语义
- 跑通后：改成 online 1-pass，对比吞吐

**接下来**：A5 读 Flash Attn CUDA → B1 Triton 入门

---

## 🧠 理论线（理解）

**现在 · online softmax（第一条）**
和 A4 Softmax 天然配对——边写 softmax 代码，边理解它底层的算法原理。

- 读：[notes/algorithms/online-softmax.md](./notes/algorithms/online-softmax.md)（15 分钟）
- 关键推导：为什么单 pass 等价于 3-pass → `m_new = max(m_old, max(new))` + `correction = exp(m_old - m_new)`
- 产出：[ ] 能推一遍 online 更新公式，能讲清"为什么比 3-pass 省 3× HBM 读写"

**接下来**：parallel reduce → Flash Attention 机制 → INT8/FP8 量化 → GQA → MLA

---

## ✅ 刚完成

- 算子线 A3/A3+: GEMM fp16 naive + tiled LeetGPU 跑通 ✅（2026-06-22）
- 理论线: online softmax + parallel reduce 学完

---

*想换方向或调节奏，直接说。*
