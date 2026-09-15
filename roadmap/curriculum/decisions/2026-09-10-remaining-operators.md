# 剩余算子主题的正文与验证范围

## 内容落地

新增量化、MoE、Sampling/KV 三章，课程站编号为 13、14、15。三章沿用页内公式、代码片段、实践与性能对比的组织，不建立独立优化学习线，不在正文链接源码文件。模型级分析和系统课程没有在本次扩写为完整正文。

量化包含舍入/饱和/zero point、scale 粒度与地址、INT4 打包、整数点积修正、低精度计算路径、SmoothQuant/AWQ/GPTQ 的机制与误差例子。MoE 包含 router、重排、grouped 计算、combine、容量/读取区别和负载分析。Sampling 包含 Top-K/Top-P、CDF、RNG 合同、KV 暂存/提交、投机接受与剩余分布。

## 题面核对

AlphaGPU/leetgpu-challenges 核对的 commit 为 dff541e5856ddaee4c93e23c0d7489c125fd6ea3。

- #64 是 FP32 block-scale multiplication，不是 packed INT4 解包。
- #32 最终输出 INT8，容差为零，公开参考经过 FP32 matmul 与 rounding，不能只按理想 INT32 点积推断逐位一致。
- #67 对 top-k logits 做 Softmax，输出 INT32 编号，不包含专家 GEMM。
- #29 返回排序的 top-k 值，不返回 token id。
- #60 使用 PyTorch seed 与 multinomial；同分布的 CPU inverse-CDF 不保证相同 seed 下返回相同 token。

具体源码、文献及题面 URL 保存在 .codex/course-source-index.json，前台只提供实际题目入口和简短文献参考。

## 实现与验收边界

标准库 CPU 检查覆盖量化、INT4 全部双元素组合及尾部、零点修正、分组/逆映射、空专家、合并、CDF 阈值与投机概率恢复。三个 GPU 基线分别是 FP32 block scale、小 k gating、两阶段 argmax。它们不是完整量化 GEMM、完整 GPU MoE 或 stochastic Top-P kernel。

GPU 验证脚本提供同语义参考、多形状检查、独立的 benchmark-shape correctness，以及明确包含分配/launch 的 wrapper 时间。本地缺少 PyTorch/Triton，GPU 脚本返回 SKIP；没有声称编译、GPU correctness、LeetGPU 提交通过或性能测量完成。

子代理此前因额度不可用停止，本次由主 agent 直接实现并验证，没有新建替代 worker。原始 solutions、reference、PDF 和 PATH/NOW 不变，没有 commit/push。
