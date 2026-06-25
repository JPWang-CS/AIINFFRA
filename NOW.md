# NOW — 现在做什么

> 进来先看这。两条线并列,各有"现在 + 接下来"。完整地图/还剩多少 → [PATH.md](./PATH.md)
> 这是学习进度的当前快照,不绑日历。每推进一步我就更新这里。

---

## 🔧 算子线（动手）

**现在 · A3 — Tiled GEMM**
naive GEMM（A2）跑通了,瓶颈是 memory-bound。这一步用 shared memory tiling 提速。

- 读：[Lesson 03 — GEMM Tiled](./lessons/03-gemm-tiled.md) · 深入 [memory-model §3.3](./notes/cuda/memory-model.md)
- 自己写：`gemm_tiled`，TILE=32，从空文件开始（别抄 [reference](./reference/cuda/gemm/gemm.cu)）
- 验收：
  - [ ] 结果对（max abs err < 1e-3）
  - [ ] GFLOPS 比 naive ≥ 5×
  - [ ] 能讲清两个 `__syncthreads` 各防什么竞争
- 跑通后存进 [solutions/cuda/](./solutions/cuda/)

**接下来**：A4 Softmax → A5 读 Flash Attn → B1 Triton 入门

---

## 📐 理论线（理解）

**现在 · online softmax**
单 pass 增量算 max+sum,是 Flash Attention 的心脏。给 A4/A5 铺路。

- 学：为什么单 pass 等价于 3-pass，推导更新公式
- 产出：一页笔记 → [notes/algorithms/](./notes/algorithms/)（GPU优化算法类）

**接下来**：parallel reduce → Norm 的 reduce 模式 → AWQ（量化）

---

## 完成一步后

跟我说"X 学完了/卡在 Y"，我会：
1. 更新 [PATH.md](./PATH.md) 进度
2. 按你的 git 提交/修改时间，自动写一篇 [weekly/](./weekly/) 回顾
3. 把这步的"接下来"提上来，重写这份 NOW

*想换方向或调节奏，直接说。*
