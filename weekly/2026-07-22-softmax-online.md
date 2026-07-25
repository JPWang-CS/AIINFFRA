# Softmax 优化深挖 + A5 准备 — 2026-07-01 ~ 2026-07-22

> 算子线 · 对应 PATH 的 **A4（优化阶段）+ A5（准备）**　·　上次回顾 [week3-infra](2026-06-29-week3-infra.md)

## 做了什么

三周跨度，节奏偏慢但有真产出。主线全压在 **A4 Softmax 优化深挖** 上，A5 只开了头。

### A4 · Softmax 优化（主战场）
- **7/1** — `softmax_naive.cu`（3-pass：`findMax → countSum → softmax`，跨 block host 归约）跑通 LeetGPU `5_softmax`，作为 baseline。
- **7/2** — 本地/服务器 harness 落地：[`solutions/cuda/softmax/main.cu`](../solutions/cuda/softmax/main.cu)（造数据 → 调 `solve` → 对比 CPU 双精度参考 → 报误差/吞吐/带宽）+ `run.sh` + [README](../solutions/cuda/softmax/README.md)。LeetGPU 挂了或要看真实带宽时本地复现。
- **7/11** — [`softmax_online.cu`](../solutions/cuda/softmax/softmax_online.cu)：**2-pass 融合版**。`online_kernel` 一趟出 `(partial_max, partial_sum)`，host 上用 online 修正公式 merge，再 `normalize_kernel` 收尾。3 kernel → 2 kernel，HBM 读 3× → 2×。
- **7/22** — [`softmax_opt.cu`](../solutions/cuda/softmax/softmax_opt.cu)：**warp shuffle 优化版的工作起点（占位）**。文件头写明优化路线 ① online → ② warp shuffle，但**当前代码体还是 3-pass**，等 `__shfl_down_sync` 改造。

### 理论线 · online softmax（配对 A4）
- ✅ 能手推 online 更新公式 `m_new=max(m,val), s=s·exp(m-m_new)+exp(val-m_new)`
- ✅ merge 公式 `s_new = s_a·exp(m_a-m_new) + s_b·exp(m_b-m_new)` 满足交换律+结合律 → 可上 tree reduce
- ✅ 能讲清"为什么比 3-pass 省 3× HBM 读写"

### A5 · Flash Attn 读码（只开个头）
- NOW.md 把焦点切到 A5，[lessons/05-flash-attn-reading.md](../lessons/05-flash-attn-reading.md) 已就位
- [notes/algorithms/flash-attention-mechanism.md](../notes/algorithms/flash-attention-mechanism.md) 微调（7/22 commit）
- ⚠️ **实际 CUDA 读码还没开始** — `reference/cuda/flash_attention/` 未动，lessons/05 未改

### 顺带
- 7/22 把 Claude 的 agent/skill 体系移植一份到 `.codex/agents/*.toml`（codex 端镜像）

## 关键数据

| 维度 | 数字 |
|------|:----:|
| 提交数（6/29 之后） | 5 |
| 新增 CUDA 文件 | 3（`softmax_naive` / `softmax_online` / `softmax_opt`）+ harness 2 |
| LeetGPU 通过 | `5_softmax` baseline ✅ |
| baseline 性能 | 3-pass ~1ms（N=百万级） |
| HBM 读次数 | 3-pass 3× → online 2× |
| 理论笔记真掌握 | online softmax ✅（用户实学，非 Agent 草稿） |

> 注：`softmax_online.cu` 与 `softmax_opt.cu` 两版**未在本地/服务器跑过对比 benchmark**（harness 已就位，但 `KERNEL=... ./run.sh` 的横向数据还没出）。

## 卡点 / 怎么解决的

1. **block 间 sum 怎么合**：每个 block 的 `partial_sum` 是相对自己的 `block_max` 算的，直接相加会错。
   **解法**：host merge 时乘修正因子 `partial_sum[i] × exp(partial_max[i] − global_max)`，把各 block 的求和基准拉到同一个 `global_max` 上。这就是 online softmax 的 merge 公式落地。

2. **越界线程污染 sum**：尾 block 线程数 < 256，越界 `val` 若置 0 会把 `exp(0)=1` 加进 sum。
   **解法**：越界置 `-INFINITY`，`exp(-INF)=0` 不污染；sum 阶段显式 `idx < N ? ... : 0.0f`。

3. **(记档，未在仓库落地)**：NOW.md 记的 "true online per-thread scan + tree reduce merge `(m,s)` pair → `maxSumkernel`" 是**比 `softmax_online.cu` 更激进的 1-pass 写法**（Flash 的心脏）。当前仓库里只有 2-pass（block 内先 reduce max 再 reduce sum），**真正的 per-thread 同步维护 `(m,s)` 的 1-pass 版本还没提交** —— 是下一步要补的硬骨头。

## 面试可用点

- **"我把 3-pass softmax 压成 2-pass"** — 能讲清每次融合省的是一次全量 HBM 读 + 一次 kernel launch，并能推 online merge 修正公式。
- **跨 block 归约的工程套路** — block 内 shared-mem tree reduce 出 partial → D2H → host 串行 merge → 再 launch normalize。能说清"为什么跨 block 归约必须回 host"（无 block 间同步原语），以及为什么这是后续 warp shuffle + 单 block / atomic 改造的动机。
- **算法和工程配对学** — softmax 算子和 online softmax 理论同步推，能现场推公式不是只会调 API。

## 产出物

- [x] `solutions/cuda/softmax/softmax_naive.cu`（3-pass baseline，LeetGPU 过）
- [x] `solutions/cuda/softmax/main.cu` + `run.sh` + `README.md`（本地 harness）
- [x] `solutions/cuda/softmax/softmax_online.cu`（2-pass fused）
- [x] `solutions/cuda/softmax/softmax_opt.cu`（warp shuffle 工作起点，占位）
- [x] 理论线：online softmax 真掌握（公式 + 省存原理）

## 下一步（接 NOW）

1. **warp shuffle reduce** — 在 `softmax_opt.cu` 上把 shared-mem tree reduce 换成 `__shfl_down_sync`，跑 `KERNEL=softmax_opt.cu ./run.sh` 出横向数据
2. **三版 benchmark** — naive 3-pass / online 2-pass / warp shuffle，对有效带宽，ncu 看瓶颈
3. **补 1-pass true online**（per-thread `(m,s)` scan，Flash 写法）— 当前 NOW 描述但仓库没有的那版
4. **A5 真正开读** — 逐段读 `flash_attn.cu`，标每个 `__syncthreads`，再切 B1 Triton
