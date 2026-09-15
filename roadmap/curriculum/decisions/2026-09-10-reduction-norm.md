# Reduction 与 Norm 章节落地记录

日期：2026-09-10

## 范围

新增 `roadmap/curriculum/operators/02-reduction-and-norm/` 教学章节和五个示例文件；更新算子总览、第 1 章导航以及静态课程站点的构建、检查、链接审计和导航回归。未修改 `PATH.md`、`NOW.md`、`HISTORY.md`、`AGENTS.md`、`solutions/` 旧代码，未 commit/push。

## 内容决策

- 章节固定为 9 个实质大节：`[R,D]` 语义、CUDA 两级归约、FP32 row sum、已有 1D Triton Softmax、row-wise Triton Softmax、RMSNorm/LayerNorm、shape 映射、算法 bytes/roofline、实验与两个验收段。
- CUDA row sum 采用 thread 私有累加 → warp shuffle → shared warp partial → warp 0 二次 shuffle。`BLOCK_SIZE` 只接受 128/256；尾部通过 0 贡献处理；barrier 前不提前退出；FULL shuffle mask 仅用于完整 warp 执行路径。
- `row_sum.cu` 的 launcher 拒绝空 shape、空指针、无效 block size 和 `rows*cols > INT32_MAX`，round-up 使用 `long long`，正常路径返回 `cudaGetLastError()`。
- Triton 示例要求 contiguous、CUDA、同 device、同 dtype（float16/bfloat16/float32）、正有限 eps 和正 shape，并限制元素总数在有符号 int32 范围内。Softmax 的 finite 检查是可选 preflight，默认关闭，避免将额外 GPU 归约和同步混入计时。
- LayerNorm padding 采用 `tl.where(mask, values - mean, 0.0)`，分母仍为逻辑 `D`；用 `[3,4,5]` 补到 4 的正确 `2/3` 与错误 `6` 作为反例。Welford 先处理 `nA==0`/`nB==0`，再进行 delta 合并。
- LayerNorm backward 只写正确的二维公式：`dx` 的统计量沿每行列轴归约，`dgamma/dbeta` 沿行轴归约到 `[D]`；示例代码仍只实现 forward，不扩展为训练实现。

## 源码与题面边界

正文通过 `source-check` 原样嵌入 `examples/row_sum.cu` 的 kernel/launcher、`rowwise_softmax.py`、`rowwise_norms.py` 和 `solutions/triton/fused_softmax.py` 的三个连续片段。旧 `solutions/cuda/softmax/softmax_online.cu` 只用于说明 kernel/host merge 机制；其 `threadsPerGrid` 笔误使文件不能被描述为可直接编译或已验证，旧文件保持不动。

Triton 用户归档的三阶段是 GPU partial → GPU merge → GPU normalize；旧 CUDA online 才是 GPU partial → host merge → normalize。两者都要按两次输入读取、一次输出写入以及 scratch/launch 等边界计流量，不能把历史 README 中的 2N 说法直接带入新章。

LeetGPU 题面按当前平台优先，并用 AlphaGPU 公开题面核对：#4 Reduction 是 FP32 一维 sum，#50 RMS Normalization 使用 scalar gamma/beta，#113 Layer Normalization 使用 `[N,C]`、`weight[C]`、`bias[C]` 和 biased `1/C` variance。新章无 bias、gamma `[D]` 的教学 RMSNorm 不能冒充 #50 提交版；没有写入任何平台成绩。

## 验证证据与边界

已执行并通过：

- `python -B .../examples/cpu_semantics.py`：row-tail coverage、Softmax partial merge、Welford 空状态、LayerNorm padding/偏置反例和非均匀 gamma 的 `dx` 中心差分。
- 对 `examples/*.py` 使用 `ast.parse`：无 pyc 生成。
- `git diff --check`：通过。
- `npm run build`：8 chapters、105 readable Markdown pages、39 local assets、648 formulas、11 source excerpts。
- `npm run check`：8 页、公式、重点提示、源码、搜索数据、锚点和地址算术检查通过；并调用 link audit。
- `npm run test:navigation`：顶部/侧栏分区、章节折叠、子目录锚点、搜索、历史回退/前进、非法 chapter 路由和第 8 页路径回归。
- `node link-audit.cjs`：生成 HTML 的本地 href/src、编码路径、目标文件、raw Markdown 排除、chapter 1–8、DOM fragment 和附属文档目录检查。

本轮没有 CUDA 编译、GPU 执行、真实带宽、寄存器、occupancy 或 Nsight counter 证据；这些留到 LeetGPU 题面正确性与原始代码归档之后的服务器验收段。源码和 CPU 测试通过不改变学习进度状态。
