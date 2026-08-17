---
name: project-progress
description: 进度——A5 读码完成(2026-08-10)，B1 Triton vec_add 为 Agent 草稿待用户重写；理论线主线 A DeepSeek-V3.2→V4。权威源 PATH.md
metadata:
  type: project
---

**进度以仓库 `PATH.md` 为唯一权威源**（拉模式：用户问→读 PATH→指 NOW；完成→更新 PATH + 自动写 weekly 回顾）。本条只记 memory 该留的非显然事实。

两条平行路径：**算子线**（A CUDA→B Triton→C 推理→D 分布式→E Agent，动手写代码）+ **理论线**（模型驱动主干：DeepSeek-V3.2 / Qwen3.5，概念作枝干挂载）。方向 Triton-first，CUDA 只到 B 级，不深钻 tensor core（除非用户要或面试需要）。详见 [[user-background]]。

**当前（2026-08-14）**：
- 算子线：A1-A3 ✅；A4 3-pass/2-pass ✅，1-pass true online 为 Agent 草稿 `softmax_1pass.cu`（2026-08-09，算法模拟+编译通过），待用户重写；warp shuffle/三版 benchmark 暂缓。A5 Flash Attn 读码 ✅（2026-08-10，阅读笔记 `notes/cuda/flash-attn-reading.md`，发现 2 个真实 bug）。B1 Triton vec_add 为 Agent 草稿 `solutions/triton/vector_add.py`（CPU 解释器验证过），按仓库规则待用户从空文件重写；matmul 待做。
- 理论线：用户已掌握 online softmax + parallel reduce + FA1 机制。Agent 草稿：FA2、FA4/FlexAttention、GDN/Qwen3.5、DSA、SageAttention3/Kascade、DeepSeek-V4、优化器 Adam/AdamW。主线 A：DeepSeek-V3.2 第 1 步 config+手算 → 第 2 步注意力（FA2 → MLA → DSA）→ 第 3 步 V4 增量（CSA/HCA → mHC/Muon → FP4）。
- 计划：`roadmap/ai-infra-curriculum.md` 是全阶段执行参考，M2 Triton 为当前算子主线，M2.5 最新模型/算子构建为并行强化。
- 内容：`notes/llm/` 是大模型内容聚合，`notes/llm/operator-building.md` 是构建能力路线；`HISTORY.md` 是跨电脑恢复入口。

**留存的事实纠正**（A2 GEMM naive，LeetGPU `2_matrix_multiplication`，FP32，C=A×B，2D grid 16×16，`threadIdx.x→k`）：
- 写回用 `=` 非 `+=`（每线程独占输出，无需累加）
- 该 kernel 的 B 访问其实是连续合并访问（之前笔记误判 uncoalesced）；真瓶颈是数据复用率低（A 读 K 次、B 读 M 次）→ memory-bound

**A3+ 反直觉结果（好面试素材）**：4090 实测 fp16 tiled GEMM 只有 naive 的 ~0.6x（K=2048/8192）。原因：naive 的重复访存已被 L2 cache 吃掉 + tiled 版 occupancy 受限。结论"tiling 不总是赢，要看 L2 命中和 occupancy"。

**环境**：本机 venv 已就绪（torch CPU + triton-windows）；无 nvcc 时 CUDA 在 LeetGPU 浏览器端跑；`solve` 入口签名不可改，参数已是 device pointer。本地有 4090 时用 `reference/cuda/CMakeLists.txt` 编译对照。

**How to apply:** profiling 数据让用户从 LeetGPU/4090 拿，别开本地 nvcc。不要按旧阶梯"不跳级"推进——用户定节奏，卡住帮看、要完整代码就给。Agent 代写的代码不算完成，必须用户自己重写并跑通。相关：[[feedback-style]] [[code-ownership-clarification]]