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

**现在 · GQA（Grouped Query Attention）**
online softmax 和 parallel reduce 已学完。下一步理解 GQA——现代 LLM 推理的标配注意力机制。

- 读：[papers/attention/gqa.md](./papers/attention/gqa.md)（已写） · 补充 [notes/algorithms/mla-deepseek.md](./notes/algorithms/mla-deepseek.md)（MLA 是 GQA 的延伸）
- 关键数字：G=8 是甜点，KV cache 节省 87.5%（LLaMA 2 70B：64 heads → 8 KV heads）
- 产出：能讲清 MHA→MQA→GQA→MLA 的演进线 + 各模型的 G 值

**接下来**：MLA（DeepSeek-V2/V3）→ AWQ 量化 → continuous batching

---

## ✅ 刚完成

- 算子线 A3/A3+: GEMM fp16 naive + tiled LeetGPU 跑通 ✅（2026-06-22）
- 理论线: online softmax + parallel reduce 学完

---

*想换方向或调节奏，直接说。*
