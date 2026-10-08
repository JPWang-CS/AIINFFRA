# 2026-09-25 算子主线全文走读与校订记录

本轮按 `rg --files roadmap/curriculum/operators` 确认九个目标 README，并逐文件从首行读到 EOF；只修改九个 README 与本记录。没有修改 `PATH.md`、`NOW.md`、solutions、source-check/reference、生成 HTML、目录构建代码或 `paper-catalog.cjs`，没有 commit/push。

## 读者顺序与总体判断

阅读顺序为：访存与布局 → 归约/Softmax/Norm → GEMM → Activation/Fusion → Prefill Attention → Decode/PagedAttention → 量化 → MoE → Sampling/KV。九章整体已经形成“语义/shape → 地址与数据流 → 代码 → 资源/性能原因 → 题面与实测边界”的主线；本轮主要补仍会卡住初学者的排错入口，并保留历史实验的真实条件。

## 分文件 EOF 覆盖、障碍与改进

### 01-memory-and-layout（当前 428 行）

完整覆盖：copy/view/transpose/permute；shape、stride、storage offset、byte address；CUDA/Triton copy；float4 对齐与 tail；coalescing/transaction/cache/shared；真实矩形 transpose、shared tile、padding/bank；gather/scatter/embedding/packing；benchmark 工作集与 Vector Add 历史证据；LeetGPU 与服务器命令。

真实障碍：读者容易把 `data_ptr()` 与底层 storage base 混加，把 view 时间当 materialize 时间；也容易将 `N%4==0` 当作 float4 合法条件。补充“输出正确但布局仍错”的三种排错场景，明确检查指针语义、stride、起始偏移和每行对齐。原有 32B transaction、padding 不等于速度、热工作集不等于 DRAM 峰值等段落足够严谨，未重复改写。

### 02-reduction-and-norm（当前 628 行）

完整覆盖：`[R,D]` 语义与归约轴；CUDA register/shuffle/shared；D=1000 row sum；partial 合并；Triton row-wise Softmax；RMSNorm/LayerNorm mask、Welford、forward/backward 区别；短/中/大行映射；算法 bytes/cache/roofline；实验、平台题与服务器记录。

真实障碍：平台一维 Reduction/RMSNorm 题与教学二维 row-wise kernel 的边界容易被混为同一通过；非整除行的 padding lane 若被当作真实样本会破坏 Welford；性能问题常被误诊为 `num_warps`。新增三种排错场景，分别追踪 shape 合同、空状态/biased variance 和编译资源。没有改 source-check 代码或已有公式。

### 03-gemm（当前 2099 行）

完整覆盖：A[M,N]B[N,K]符号与官方惯例映射；尾块三类 mask；naive CUDA/Triton 所有权；shared/register tile、barrier、coalescing、swizzle；算术强度与资源；SIMT/WMMA/PTX/Tensor Core 精度；WMMA 尾块；cp.async、ring buffer、stage、warp specialization、WGMMA、TMA；persistent/split-K/sliced-K；batch/grouped/stride/fusion；实验扫描、平台题、历史 4090/A100 证据和服务器命令。

真实障碍：章节很长但前置合同已连续；剩余最大风险是读者把“能跑”误当“等价”，或把 stage/tile/AI 变化直接当性能因果。新增三种排错场景，要求从 mask、最后读者/phase、资源与计时边界定位，不增加新的性能数字。保留已有失败实验与不同容差边界。

### 04-activation-and-fusion（当前 570 行）

完整覆盖：SwiGLU dense 形状；Gated/non-gated 公式；小例子；ReLU/GELU/SiLU 与 IEEE 边界；独立与 packed layout；Triton pointwise；pointwise/epilogue/完整 MLP 融合边界；FLOPs/bytes/launch；两个 GEMM 到完整 MLP；QK Norm/RoPE；旧实现、mHC、平台题和服务器实验。

真实障碍：主要是门控分支、packed `[T,2I]` 与一维题面、norm 归约依赖和 epilogue 边界。正文已有具体反例、地址公式、FP16 中间精度和平台差异；本轮不再插入模板式问题，避免稀释已充分的推理链。

### 05-prefill-attention（当前 631 行）

完整覆盖：`[B,H,S,D]` 数学与地址；dense/Flash 存储边界；online `(m,l,U)` 递推；Triton baseline；causal/GQA/varlen；FA1/FA2/FA3 work tile、warp/warpgroup、TMA、stride、mask、资源和 lookahead；精度与作者源诊断；题面、CPU/GPU correctness、服务器 benchmark。

真实障碍：读者容易把 online state 的数学等价误读成异步执行已经发生，或把教学 kernel/作者 FA2/FA3 形状合同混为一谈。正文已有明确“源码独立性不证明 SASS/重叠”、K/V/P 最后读者、causal offset 和容差边界；本轮不新增重复段落。FA3/Hopper 说法保留为固定 commit 的 source-check，不外推到其他架构。

### 06-decode-paged-attention（当前 500 行）

完整覆盖：单 Query shape/GQA；逻辑 token 到物理 page；容量/padding/HBM 流量；online Softmax/split-KV；页尾、空段、append、COW、stream ownership；CED/CSA2/Bounded Replay/FP4；paged kernel、连续参考、平台题与服务器实验。

真实障碍：最容易混淆的是逻辑可见集合、物理页容量、GQA head 复用和 profiler traffic；CED/CSA2 的 cache owner 与 index owner 也不能压成一个层编号。正文已通过公式、表格和 kernel shape 把这些边界拆开，未再添加泛化问题。

### 07-quantized-operators（当前 2017 行）

完整覆盖：仿射量化、舍入/zero point/clipping；per-tensor/token/channel/group/block；INT4 nibble/scale 寻址；W8A8/W4A16；packed tile Triton；FP8/FP4/MXFP4/NVFP4；指令/SASS 证据；校准、SmoothQuant、AWQ、GPTQ、Cholesky、延迟更新；Mini Decoder、Llama overlay、性能与实践。

真实障碍：量化存储位宽、反量化顺序、scale 粒度、整数累加与真实 MMA 经常被一条“低位更快”口号替代。新增三个排错场景，分别针对奇数 nibble、zero-point 交叉项和“文件变小但 kernel 不快”。保留 CPU 误差/heldout 数字与真实 GPU 证据为空的边界。

### 08-moe（当前 255 行）

完整覆盖：固定 k/不丢弃的路由合同；Top-K/Softmax 与模型 router 差异；重排、offset、逆映射；专家 SwiGLU、权重位置、combine 冲突；权重驻留与本步读取；Engram 对照；Triton router、Grouped GEMM、负载不均、EP；平台题和服务器记录。

真实障碍：原文已明确固定 k、不丢弃、不含 padding/共享专家；剩余卡点是总任务数相同不等于路由相同、`k/E` 不等于实际权重读取比例、原子合并顺序影响数值。新增三个排错场景，要求保留 `(token,slot)` 身份、`M_e` 分布和通信边界。

### 09-sampling-kv（当前 273 行）

完整覆盖：`[B,V]`/`[B,T,V]` shape；温度、Greedy、Top-K/Top-P 与跨阈值 token；两阶段 argmax 与平局；Top-P 选择边界；inverse-CDF/RNG/dynamic batching；append/page/ownership；draft/target cache、投机提交/回滚、residual 分布、因果 mask、MTP；平台题和服务器实验。

真实障碍：输出 token 数和已物化 KV 数不是同一计数器；draft cache 与 target cache 不能互换；验证 logits 有一位 shift，不能拿“已经看到 draft_j 的 row”验证 draft_j。正文已有带 shape 的投机例子和两阶段状态机，本轮保留并检查这些边界，没有再堆同模板问题。

## 一手来源与技术核对

- CUDA 访存、shared、同步、WMMA/TMA/WGMMA：仓库已有固定 [CUDA Programming Guide 13.3](../../downloads/cuda-programming-guide.pdf) 与 NVIDIA PTX/CUTLASS 链接；本轮没有以不确定的架构细节替换正文。
- Triton dot/fused-attention/Grouped GEMM：正文已有官方 Triton 教程链接；lookahead 段保留“普通 `tl.load` 不证明异步重叠”的限定。
- PagedAttention/stream/event：正文链接 CUDA Runtime API，并将普通 page table 与 CSA2/Bounded Replay 分开。
- 量化：正文区分编码、scale、反量化与 SASS；没有将 FP4/INT4 文件格式当作原生矩阵指令证据。

## 发布与验证边界

- 仅运行了本写集限定的 `git -c core.autocrlf=false -c core.whitespace=cr-at-eol diff --check`；未构建网站，未运行 CUDA/Triton、source-check 或生成站点测试。
- `paper-catalog.cjs` 未修改。它对 chapter 30 仍有针对旧 `## 7. 做完之后` 的 prepare 替换；V3.2 新增题目位于 `## 7. 求职核对题` 并将原收束段移到 `## 8`，主代理应在发布层移除过时替换，避免未来重新写掉正文。
- 本轮对九章的“完整阅读”不等于验证所有代码可编译或 GPU 结果；疑似技术错误只在正文已有公式/代码/官方来源能直接支撑时修订，其余保留为待现场验证项。
