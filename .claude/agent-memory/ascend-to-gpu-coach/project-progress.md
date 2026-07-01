---
name: project-progress
description: 进度——A4 Softmax naive 已过(2026-07-01)，现挖 softmax 优化；A5 Flash Attn 读码 next。权威源 PATH.md
metadata:
  type: project
---

**进度以仓库 `PATH.md` 为唯一权威源**（拉模式：用户问→读 PATH→指 NOW；完成→更新 PATH + 自动写 weekly 回顾）。本条只记 memory 该留的非显然事实。

两条平行路径（不是旧的 5 算子阶梯）：**算子线**（A CUDA→B Triton→C 推理…，动手写代码）+ **理论线**（每周一条算法/理论，产出 notes/algorithms 笔记）。方向 Triton-first，CUDA 只到 B 级，不深钻 tensor core（除非用户要或面试需要）。详见 [[user-background]]。

**当前（2026-07-01）**：
- 算子线：A1–A4 全过。A4 Softmax 3-pass naive 刚在 LeetGPU `5_softmax` 跑通，现进优化阶段（fuse max+sum → online 1-pass → warp shuffle reduce → 吞吐对比）。下一站 A5 读 Flash Attn CUDA → B1 Triton。
- 理论线：**区分「我生成」vs「用户学过」**——8 条笔记是我(Agent)起草的草稿，不等于用户掌握。用户实际学过 2 条：online softmax + parallel reduce（NOW.md 前两条）。其余 6 条（flash-attn 机制、MLA、INT8/FP8、MoE、PD 分离、投机解码）是待读草稿。里程碑 ≥12 指用户真正掌握数，不是笔记数。见 [[code-ownership-clarification]]。

**留存的事实纠正**（A2 GEMM naive，LeetGPU `2_matrix_multiplication`，FP32，C=A×B，2D grid 16×16，`threadIdx.x→k`）：
- 写回用 `=` 非 `+=`（每线程独占输出，无需累加）
- 该 kernel 的 B 访问其实是连续合并访问（之前笔记误判 uncoalesced）；真瓶颈是数据复用率低（A 读 K 次、B 读 M 次）→ memory-bound

**A3+ 反直觉结果（好面试素材）**：4090 实测 fp16 tiled GEMM 只有 naive 的 ~0.6x（K=2048/8192）。原因：naive 的重复访存已被 L2 cache 吃掉 + tiled 版 occupancy 受限。结论"tiling 不总是赢，要看 L2 命中和 occupancy"。

**环境**：本机无 nvcc（`which nvcc` 无果），CUDA 在 LeetGPU 浏览器端跑；`solve` 入口签名不可改，参数已是 device pointer。本地有 4090 时用 `reference/cuda/CMakeLists.txt` 编译对照。

**How to apply:** profiling 数据让用户从 LeetGPU/4090 拿，别开本地 nvcc。不要按旧阶梯"不跳级"推进——用户定节奏，卡住帮看、要完整代码就给。相关：[[feedback-style]] [[code-ownership-clarification]]
