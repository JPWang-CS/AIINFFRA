# Eager、torch.compile 与 CUDA Graph（2026-10-16）

## 已确认

- 第 4 章第 10 节使用同一条 FP32 pointwise 链解释 Eager、`torch.compile` 和 CUDA Graph；三种路径的输入、数学运算与输出合同相同。
- 编译对照使用 `backend="inductor"`、`fullgraph=True` 与 `options={"triton.cudagraphs": False}`，分离编译融合和 Inductor 自动 CUDA Graph 的影响。`fullgraph` 是 Dynamo 捕获约束，不承诺设备侧只有一个 kernel。
- 显式 CUDA Graph 捕获原始 Eager 函数，固定输入/输出地址，逐轮复制输入并 replay；教程解释异步返回、buffer 生命周期、静态 shape/dtype/device 检查与舍入容差。
- 教学只提供正确性路径。模型与多算子阶段再开展端到端性能和 profiler 实验。

## 依据

- PyTorch `torch.compile` 官方 API：<https://docs.pytorch.org/docs/stable/generated/torch.compile>
- PyTorch 融合调优指南：<https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html>
- PyTorch CUDA Graph：<https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>
- PyTorch CUDA Graph Trees：<https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html>
- NVIDIA CUDA Graphs：<https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html>

## 验证边界

本机 `.venv` 只安装 `2.7.1+cpu`，可验证 CPU 数学与 shape，不具备 CUDA Graph 运行条件。`--help`、CPU 检查、Python AST、课程构建、正文源码摘录、正文检查、编辑检查、导航模拟、链接审计与离线资源/DOM检查通过。离线入口检查允许有效的章节级链接，并验证小节锚点存在且属于当前章节；缺失章节和跨章 fragment 会被拒绝。编译 trace、实际 kernel 数量、GPU 正确性和性能结论需在后续目标设备实验中核实；未做浏览器视觉检查。
