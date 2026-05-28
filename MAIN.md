# MAIN — 算子优化主线

> 核心思路：5 个算子，每个从 naive → 极致优化，作为面试核心竞争力
> 每个阶段只聚焦一件事

---

## 当前状态

**Week 1** (2026-05-29 ~ 06-04) | 主题：**跑通环境 + GEMM v0**

---

## 算子优化路线（5 个算子走到底）

```
算子           Week    优化阶梯
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GEMM          1-3     naive → tiled → vec4 → double buffering → tensor core
Softmax       4-5     naive → online → warp reduce → fused w/ matmul
RMSNorm       6       cross-warp reduce → fused residual
Flash Attn    7-9     tiled → online softmax → causal → multi-head
GQA Attn      10      从 Flash Attn 扩展到 grouped query
```

每个算子的产出：
- **可运行的 .cu 文件**（每个版本一个 kernel 函数，同文件对比）
- **Nsight profiling 数据**（带宽利用率、occupancy、计算吞吐）
- **优化笔记**（为什么这一步比上一步快？瓶颈从什么变成了什么？）

---

## 本周 (Week 1)

**只做一件事**：写出 GEMM 的 naive + tiled 版本，能跑、算对、会计时。

- [ ] 找台 GPU（AutoDL T4 即可），搭好 CUDA + nvcc + ncu 环境
- [ ] 跑通 `cuda-kernels/gemm/gemm.cu`（naive + tiled 两个 kernel）
- [ ] 搞懂每个概念：grid/block/thread 索引、shared memory、`__syncthreads`
- [ ] 对比两个 kernel 的耗时和 GFLOPS

---

## 优化阶梯速查（以 GEMM 为例）

| 版本 | 技术 | 预期提升 | 瓶颈在哪 |
|------|------|:---:|------|
| v0 naive | 每线程算一个输出，全从 global memory 读 | 1× | 访存延迟 |
| v1 tiled | shared memory 分块，每个 block 算 TILE×TILE | ~10× | shared memory bank conflict |
| v2 vec4 | float4 向量化加载，128-bit aligned | ~1.3× | 计算吞吐 |
| v3 double buffer | 两个 shared memory buffer，copy 和 compute 重叠 | ~1.2× | 指令发射 |
| v4 tensor core | mma.sync，FP16 in → FP32 accumulate | ~3× | Tensor core 利用率 |

**关键**：每一步都要用 Nsight 看 bottleneck 从 memory → compute → instruction 的转移。

---

## 里程碑

- [ ] Week 3 — GEMM 全版本完成，tensor core 版本达到硬件峰值的 70%+
- [ ] Week 5 — Softmax + online 优化，理解 warp shuffle
- [ ] Week 6 — RMSNorm，理解跨 warp reduce
- [ ] Week 9 — Flash Attention 完整实现，tiling + online softmax
- [ ] Week 10 — GQA Attention，从 Flash Attn 扩展

---

## 快捷链接

| 需求 | 文件 |
|------|------|
| GEMM 起点 | [cuda-kernels/gemm/gemm.cu](./cuda-kernels/gemm/gemm.cu) |
| Softmax 起点 | [cuda-kernels/softmax/softmax.cu](./cuda-kernels/softmax/softmax.cu) |
| RMSNorm 起点 | [cuda-kernels/layernorm/layernorm.cu](./cuda-kernels/layernorm/layernorm.cu) |
| Flash Attn 起点 | [cuda-kernels/flash_attention/flash_attn.cu](./cuda-kernels/flash_attention/flash_attn.cu) |
| GPU 架构对比 | [cuda-kernels/notes/gpu-architecture.md](./cuda-kernels/notes/gpu-architecture.md) |
| 论文索引 | [papers/README.md](./papers/README.md) |
| 面试大纲 | [interviews/README.md](./interviews/README.md) |

---

## 规则

1. **一个算子优化完了再开下一个**，不并行
2. **每个版本的改进必须能用一句话说清楚**（写进代码注释）
3. **有 GPU 的时候跑 profiling**，没 GPU 的时候看 PTX 静态分析
4. **每周日 git commit**：`git commit -m "week-X: <做了什么优化，多少 GFLOPS>"`

---

*最后更新: 2026-05-29 · Week 1 · GEMM v0*
