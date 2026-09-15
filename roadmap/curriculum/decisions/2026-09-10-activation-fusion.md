# 2026-09-10：Activation 与 Fusion 第四章及网页第 10 页

## 范围与落地

本次只新增/修改以下范围：

- `roadmap/curriculum/operators/04-activation-and-fusion/`：正文、CPU 语义检查、Triton 示例和 CUDA 未来验证脚本；
- `roadmap/curriculum/operators/README.md`：第四章入口；
- `roadmap/curriculum/operators/03-gemm/README.md`：只增加下一章导航；
- `roadmap/curriculum/gpu/course-site/{build.cjs,check.cjs,link-audit.cjs,navigation.test.cjs,README.md}` 与构建产物：追加 chapter 10，保留旧 1–9 路由及测试覆盖；
- 本决策记录。

未修改 `PATH.md`、`NOW.md`、`HISTORY.md`、`AGENTS.md`、旧 `solutions/` 代码及其他既有脏改动；不 commit/push。

## 数学、模型和公开题面依据

正文固定 dense MLP 符号：`X[T,H]`、`W_gate/W_up[H,I]`、`G=XW_gate`、`U=XW_up`、`Z=SiLU(G)*U`、`Y=ZW_down`，并区分非 gated GELU MLP、GELU erf exact、GELU tanh approximation、SiLU 和 SwiGLU。Hugging Face LlamaMLP 的官方实现使用 `down_proj(act_fn(gate_proj(x)) * up_proj(x))`，因此 gate/up 角色按权重语义固定，不能照抄旧 notes 中颠倒角色的文字：[官方源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)。

已核实的免费公开题面为：

- [#52 Sigmoid Linear Unit](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/easy/52_silu)：FP32 `input[N]`/`output[N]`，`rtol=atol=1e-5`；
- [#54 Swish-Gated Linear Unit](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/easy/54_swiglu)：FP32 一维偶数 `input[N]`，前半 gate、后半 up，输出 `[N/2]`，`solve(input, output, N)`，`rtol=1e-5, atol=1e-4`；
- [#83 Fused Residual Add and RMS Norm](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/83_fused_residual_add_rms_norm)：先 `z=x+residual` 再 RMSNorm，单输出且不改 residual，`rtol=atol=1e-5`；
- [#84 SwiGLU MLP Block](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/84_swiglu_mlp_block)：FP32 `x[M,d_model]`、三组权重和 `output[M,d_model]`，`rtol=atol=1e-4`。

题面链接使用已核实的 GitHub challenge 路径和 [LeetGPU 题库入口](https://leetgpu.com/challenges)，没有编造题目 slug。#54 的一维全局前后半切法与多行 `[T,2I]` packed 不等价；#84 的完整 MLP 也不等于本章点算子已经完成。

## Reference 与源代码边界

`reference/cuda/include/activations.cuh` 确有 `relu/sigmoid/silu/gelu/swiglu/geglu`，本章通过 `source-check` 原样嵌入其中的 SiLU/GELU/SwiGLU 片段，但把它明确标记为 reference，不归因于用户实践。该 header 的 GELU 函数体是 tanh approximation，而注释“same as PyTorch's default”不准确；PyTorch `F.gelu` 默认 `approximate='none'` 是 erf exact。旧 `.cuh` 文件保持不变。

为使该源代码片段可审计，`build.cjs` 的 source-check 扩展名白名单和静态资源白名单一起最小增加 `.cuh`，仍保留仓库根边界、`solutions|reference` 路径边界和原文包含检查。

旧用户 GEMM `solutions/cuda/gemm/naive_float.cu` 只用于讲解寄存器 accumulator 生命周期与 GEMM epilogue 插入点，不被写成用户已经实现过 SwiGLU 或完成过 MLP 实测。正文同时保留已有 `cudaEvent` / launch-gap 的测量边界说明。

## Triton 实现合同

`examples/fused_swiglu.py` 的 `fused_swiglu_kernel`：

- 输入是两个独立、同 shape/device/dtype 的 contiguous 2-D `[T,I]` buffer，dtype 为 FP16/BF16/FP32；不接受 packed 非 contiguous view；
- 每个 program 处理一段元素，不等于一个 CUDA thread；offset 和 `pid*BLOCK` 显式转 `tl.int64`；tail 通过 `tl.load/store(..., mask=mask)` 保护；
- load 后转 FP32，使用 `exp(-abs(gate))` 稳定表达 sigmoid，最终只 storecast 一次；有限且合理幅度输入是 kernel 合同，NaN/Inf 不被假定安全；
- `numel==0` 返回预分配 output，不发射 zero-grid；`out` 与两输入做完整连续 byte-range overlap 检查，避免错位别名在不同 program 间覆盖输入；
- BLOCK 限定为 `128/256/512/1024/2048`，调用显式固定 `num_warps=4`，与 benchmark 输出一致。

packed `[T,2I]` 的 `[gate|up]` 地址是 `r*2I+i` 与 `r*2I+I+i`；`T>1` 时左右切片带 row stride `2I`，不是本 wrapper 的 contiguous 合同。任何 `.contiguous()` 转换都必须作为独立成本报告，不能放进 benchmark 热循环隐去。

## 性能账本与公平比较

对已经给定的 `G/U`，`n=T*I`、元素大小 `s` 时 separate SiLU + multiply 是 `5ns`，fused pointwise 是 `3ns`，节省 `2ns`；`5/3` 仅是忽略计算与 launch 且有效带宽相同的逻辑流量比，不是全局速度上限。三 kernel `sigmoid + mul + mul` 是 `8ns`。FP16 `T=4096,I=11008` 的具体账本为：`5ns=450887680B`、`3ns=270532608B`、节省 `180355072B`，只涵盖已给定 G/U 的 pointwise 链。

更高层的中间激活账本（排除 X/weights/Y，同 `s`）为：独立 projection + SiLU + mul + down 读 Z 为 `8ns`；单独 fused SwiGLU 后为 `6ns`；只有 projection producer 真实成对产出 Z 的专门 epilogue 才可能到 `2ns`。`W_gu=[W_g|W_u]` 一次 GEMM 仍是 `4THI` FLOPs；普通 column tile 可能将 G/U 分到不同 CTA，不能自动得到 paired epilogue。完整 MLP 现场重算 Z 会带来不同 down 输出块之间的重复计算；让一个 CTA 保持全部 H 累加器又会增加寄存器压力，不能把三个 GEMM 写进一 kernel 当成免费消除中间张量。

FP32 benchmark 使用预分配 buffer 和 `torch.ops.aten.silu.out(G,out=S)` + `torch.mul(S,U,out=Z)` 两-kernel split baseline；如果目标 Torch 没有 `aten.silu.out` overload，脚本显式报错，不退化成三 kernel。FP16/BF16 只做 FP32 reference 后最终 cast 的 correctness，或另立中间 cast/bytes 口径，不与 FP32 fused/split 性能表混合。

## CUDA 13.3 本地 PDF 对照

正文使用本地 PDF 的直接页链接：

- [PDF 第 67 页（印刷第 51 页）](../../../../downloads/cuda-programming-guide.pdf#page=67)：§2.3.3.3 寄存器由编译器管理，寄存器上限可能导致 spill；§2.3.3.4 local memory 是 thread-local 作用域，不等于低延迟物理片上存储；
- [PDF 第 655 页（印刷第 639 页）](../../../../downloads/cuda-programming-guide.pdf#page=655)：intrinsics 的误差来自测试，不是精度保证；
- [PDF 第 656 页（印刷第 640 页）](../../../../downloads/cuda-programming-guide.pdf#page=656)：§5.5.9.3 `--use_fast_math` 会把部分 API 替换为 intrinsic，精度和特殊值语义可能改变。

## 验证边界

### 最终收尾记录

原 worker 的任务被中断，协调工具返回 `not_found`；读取任务确认 `interrupted / notLoaded` 后，由主 Agent 接管剩余修改，没有启动重叠写入者。

收尾修正了丢失反斜线的公式、稳定 sigmoid 临时变量命名、Residual Norm 的显式缩放公式、多行 packed 的错位解释，并把新 kernel 与 reference GELU 函数改成原样 `source-check` 摘录。GPU 验证脚本在计时前检查实际 shape 的 fused/split/reference，单独检查输出有限性，细分 dtype 容差，打印 Triton 版本、num_warps 和两种流量口径。

最终实际通过：

- `cpu_semantics.py`：激活导数差分、packed 索引、字节数和尾部 mask。
- 三份 Python 文件的 `ast.parse`，未生成 pyc。
- `npm run build`：10 个主页面、107 个附属阅读页、42 个资源、766 个公式、17 个源码摘录。
- `npm run check`：108 个 HTML 产物、5106 个本地 href/src、3754 个锚点引用检查通过。
- `npm run test:navigation`：保留旧章节并增加第 10 页折叠、搜索、前后跳转和直接锚点回归。
- 第 10 页 HTTP 200，页面包含 `chapter-10`。

没有 CUDA 编译、GPU 执行或新平台成绩；原始 solutions 与 reference 代码保持不变，未改变学习焦点。

已落地并可在无 CUDA 的本机做的检查：Python AST、纯标准库 `cpu_semantics.py` 的公式/导数/packing/bytes/index mask、Markdown KaTeX/source-check、站点 build/check/link-audit/navigation test。CPU 差分覆盖 SiLU、GELU exact、SwiGLU 对 gate/up 的两个偏导；ReLU 零点只检查左右约定，不做中央差分。

`validate_fusion.py` 的 CUDA 路径将来才执行：先 finite preflight，再按 dtype 容差 `assert_close(equal_nan=False)`；benchmark 预分配且打印 device/software/shape/numel/dtype/BLOCK/num_warps，CUDA Event 结果包含 Python launch gap，不称纯 kernel。真实 kernel 编译、GPU correctness、LeetGPU 原始 `solve` 归档、Nsight 的寄存器/spill/occupancy/L2/DRAM 与性能数字均不由本次静态检查代替。
