---
name: project-progress
description: 进度——A2 GEMM naive 已过(2026-06-16)，现 A3 tiled；拉模式两条线，权威进度看仓库 PATH.md
metadata:
  type: project
---

**进度以仓库 `PATH.md` 为唯一权威源**（拉模式：用户问→读 PATH→指 NOW；完成→更新 PATH + 自动写 weekly 回顾）。本条只记 memory 该留的非显然事实。

两条平行路径（不是旧的 5 算子阶梯）：**算子线**（A CUDA→B Triton→C 推理…，动手写代码）+ **理论线**（每周一条算法/理论，产出 notes/algorithms 笔记）。方向 Triton-first，CUDA 只到 B 级，不深钻 tensor core（除非用户要或面试需要）。详见 [[user-background]]。

**当前（2026-06-24）**：算子线在 A3 tiled GEMM；理论线起步，第一条 online softmax。

**留存的事实纠正**（A2 GEMM naive，LeetGPU `2_matrix_multiplication`，FP32，C=A×B，2D grid 16×16，`threadIdx.x→k`）：
- 写回用 `=` 非 `+=`（每线程独占输出，无需累加）
- 该 kernel 的 B 访问其实是连续合并访问（之前笔记误判 uncoalesced）；真瓶颈是数据复用率低（A 读 K 次、B 读 M 次）→ memory-bound

**环境**：本机无 nvcc（`which nvcc` 无果），CUDA 在 LeetGPU 浏览器端跑；`solve` 入口签名不可改，参数已是 device pointer。本地有 4090 时用 `reference/cuda/CMakeLists.txt` 编译对照。

**How to apply:** profiling 数据让用户从 LeetGPU 拿，别开本地 nvcc。不要按旧阶梯"不跳级"推进——用户定节奏，卡住帮看、要完整代码就给。相关：[[feedback-style]]
