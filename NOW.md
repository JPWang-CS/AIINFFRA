# NOW — 现在做什么

> 进来先看这。两条线并列，各有"现在 + 接下来"。完整地图 → [PATH.md](./PATH.md)

---

## 🔧 算子线（动手）

**现在 · A5 — 读 Flash Attention CUDA**

> **课程**：[Lesson 05 — 读懂 Flash Attn CUDA](./lessons/05-flash-attn-reading.md)  
> **核心代码**：[reference/cuda/flash_attention/flash_attn.cu](./reference/cuda/flash_attention/flash_attn.cu) — 单头 causal, Br=Bc=32  
> **前置已满足**：3-pass softmax ✅ / online softmax ✅ / shared memory tiling ✅

**阅读路线**（今天）：
1. 机制回顾：[notes/algorithms/flash-attention-mechanism.md](./notes/algorithms/flash-attention-mechanism.md)（5 分钟，已经读过可跳过）
2. 逐段读 CUDA 代码，对照 Lesson 05 Part 3 的注释
3. 检验清单：标注每个 `__syncthreads` 的作用、解释 online 更新公式、理解 tiling 策略

| 概念 | 你已会 | 在 flash_attn.cu 里找 |
|------|:--:|------|
| shared memory 分配 | ✅ GEMM tiled | Q_tile/K_tile/V_tile + m_i/l_i（第 32-54 行）|
| per-thread online scan | ✅ maxSumkernel | `m_new = fmaxf(m, score)` + rescale（第 113-123 行）|
| shared mem tree reduce | ✅ softmax | 这里没用！因为每个 thread 管一行 Q，online 扫全部 K |
| `__syncthreads()` | ✅ | 第 47、94、125 行 |
| register 累加器 | ✅ GEMM | `acc[128]` 全程寄存器（第 67 行）|
| 越界补哨兵 | ✅ | 条件判断：`if (global_row < N)` |

**接下来**：B1 Triton 入门（Lesson 06）→ B2 Triton GEMM → B3 Triton Flash Attn

---

## 📦 已完成 · A4 — Softmax（主线交付 · 优化延伸暂挂）

>   **课程**：[Lesson 04 — Softmax](./lessons/04-softmax.md)（含优化建议 Part 6）  
>   **基础代码**：[softmax_naive.cu](./solutions/cuda/softmax/softmax_naive.cu) — 3-pass 已跑通（2026-07-01），在此之上改

| 优化 | 说明 | 状态 |
|------|------|:--:|
| 3-pass naive | findMax → countSum → normalize，~1ms | ✅ `softmax_naive.cu` 2026-07-01 |
| true online | per-thread K-element scan + tree reduce merge (m,s) pair → `maxSumkernel` | ✅ LeetGPU 已实现 2026-07-10 |
| ~~fuse max+sum~~ | 已被 true online 替代（含在一个 kernel 里了） | ➖ |
| warp shuffle reduce | `__shfl_down_sync` 替代 shared memory 归约 | ⏳ |
| benchmark 对比 | 3-pass vs online vs warp shuffle → ncu 分析带宽 | ⏳ |

> **2026-07-10 实践要点**：
> - **per-thread online scan 实现正确**：逐元素维护 `(m, s)` pair，公式 `m_new=max(m,val), s=s·exp(m-m_new)+exp(val-m_new)`
> - **tree reduce merge 公式**：`s_new = s_a·exp(m_a-m_new) + s_b·exp(m_b-m_new)` 满足交换律+结合律
> - **哨兵 NaN 问题**：两个空线程 merge 时 `-inf - (-inf) = NaN` → 需要 `if (m_a == -INFINITY)` 跳过
> - **`__syncthreads()` 规则**：同 block 所有 256 个线程必须全部到达，否则死锁；不能提前 `return`
> - **Device 指针**：kernel 写入的 device 指针不能在 host 直接读，必须 `cudaMemcpy`；`cudaMalloc` 的用 `cudaFree`，不能 `free()`
> - **性能陷阱**：normalize 步骤必须多 block 并行，单线程串行 N 个 `expf` 直接崩到 60ms
> - **`maxSumkernel` 是正确设计**：block 内出 (partial_max, partial_sum) → host merge / 单 block merge → 多 block normalize
> - **当前 LeetGPU 通过方案**：3-pass（`findMax_kernel` + `countSum_kernel` + `softmax_kernel`）~1ms，作为 baseline

- **LeetGPU** `5_softmax`：贴 `solve()` 直接提交，在线判题
- **服务器**：`KERNEL=xxx.cu ./run.sh` 本地测精度 + 带宽，harness → [main.cu](./solutions/cuda/softmax/main.cu) + [run.sh](./solutions/cuda/softmax/run.sh)

**接下来**：warp shuffle 替代 shared memory → benchmark 三版对比 → A5 读 Flash Attn CUDA

---

## 🧠 理论线（理解）

**现在 · online softmax**
和 A4 Softmax 天然配对——代码已写 `maxSumkernel`，边写边理解底层算法原理。

- ✅ 能推一遍 online 更新公式，能讲清"为什么比 3-pass 省 3× HBM 读写"（2026-07-10）
- ✅ merge 公式 `s_new = s_a·exp(m_a-m_new) + s_b·exp(m_b-m_new)` 满足交换律+结合律 → 可用于 tree reduce

**接下来**：parallel reduce → Flash Attention 机制 → INT8/FP8 量化 → GQA → MLA

---

## ✅ 刚完成

- 算子线 A4: Softmax 3-pass naive LeetGPU `5_softmax` 跑通 ✅（2026-07-01）
- 算子线 A3/A3+: GEMM fp16 naive + tiled LeetGPU 跑通 ✅（2026-06-22）
- 理论线: online softmax + parallel reduce 学完

---

*想换方向或调节奏，直接说。*
