# 模型分析与系统四章全文走读记录（2026-09-25）

本轮按四个 README 的正文、公式、表格、代码与图示引用顺序分块读取至 EOF，重点检查“前提 → 定义 → 公式 → 代码/数据流 → 排错与验证”。只修改指定四个 README 与本记录；保留 source-check 代码、原始实验、PATH/NOW 与现有状态，不构建网页。

## roadmap/curriculum/model-analysis/README.md

- 总行数：151；已读至 EOF。
- 完整覆盖：测量边界；Prefill/Decode；矩阵形状与 FLOPs；batch 对权重/KV 的影响；KV 显存与活跃对象；时间线因果；ledger CPU 实践；参考与导航。
- 观察 1：线性层参数量 `2D²+2DHkv d+3DI`、Prefill 因果 Attention `4BHq d S(S+1)/2`、Decode `4BHq d S` 与 KV payload 公式前提清楚，未发现需要改的维度/单位错误。
- 观察 2：`D=4096,Hq=32,Hkv=8,d=128,L=32,S=4096,b=2` 的 512 MiB/4 GiB KV 例子可复算；文本明确区分逻辑 FLOPs、实际指令、算法 bytes 与 profiler traffic。
- 观察 3：实践表与时间线段落已经把 kernel/layer/model/request 边界、关键路径、分配/转换/通信成本和 profiler 证据分开，未强行追加模板问答。
- 修改：本轮无必要正文修改。
- 主要 primary 来源：NVIDIA [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/)、[Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)、CUDA Programming Guide（正文现有链接）。
- 未确认项：未执行本地 ledger 或 GPU profiler；示例只作静态算术与方法说明。

## roadmap/curriculum/model-analysis/mini-transformer/README.md

- 总行数：622；已读至 EOF。
- 完整覆盖：block/残差；RoPE 绝对位置；GQA/cache/mask；NumPy 前向；full/token/chunk 对照；算子替换；custom op/FakeTensor/autograd/opcheck/compile；模型替换与 benchmark；参考导航。
- 观察 1：RoPE 的 `past…past+S-1`、GQA head repeat、cache 追加和因果 mask 的历史位置边界说明连贯；明确指出 repeat/concat 是教学路径，不是生产优化。
- 观察 2：RMSNorm 公式及梯度推导一致，mask 补零但分母仍用真实 D 的边界清楚；custom op、FakeTensor、opcheck、fullgraph 与端到端三路径检查的职责分开。
- 观察 3：正文替换示例把 `MiniTransformer.forward` 的参数名误写为 `norm_fn`，而前文接口是 `norm`；已修为 `norm=norm_sum`。另检查 source-check `registered_rmsnorm.py` 的 `_check_opcheck`：`torch.randn((1,3)).t()` 形成 `[3,1]` 且 stride `(1,3)`；大小为 1 的维度可能仍满足 PyTorch contiguous 判定，因此未将其认定为确定错误，也未修改 source-check，建议主代理在可用 PyTorch 环境现场确认。
- 修改位置/类型：仅修正正文示例参数名；未改 source-check 代码、公式、测试状态或历史结果。
- 主要 primary 来源：[PyTorch custom operators](https://docs.pytorch.org/docs/stable/torch.compiler_custom_ops.html)、[custom Triton kernel tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)、Triton/NumPy 语义（正文现有参考）。
- 未确认项：缺 PyTorch/Triton/GPU 环境，未运行完整脚本、opcheck、梯度或 benchmark。

## roadmap/curriculum/systems/README.md

- 总行数：534；已读至 EOF。
- 完整覆盖：请求到设备执行；token budget/continuous batching；KV 容量、prefix cache、offload/PD；backend 核实；vLLM 本地服务与 bench；收益判定；请求 shape→ModelRunner→Attention backend；Embedded-7B 与 openPangu 2.0 区分；权重/KV ledger；模型源码/backend/运行记录证据边界；参考导航。
- 观察 1：scheduler 伪码明确是教学抽象，token budget 与请求数、Prefill/Decode 状态和输出归属分开；TTFT/TPOT、冷/热 cache、随机压力与业务质量边界清楚。
- 观察 2：Embedded-7B 的 4096/32/8/128 shape、GQA 复用、KV page 公式和 8,030,887,936 参数/14.958694 GiB/139,264 bytes per token 算术可复核；CPU ledger 明确不构成设备性能证据。
- 观察 3：openPangu Flash/Pro 的规模、专家数、MLA/DSA/SWA、版本 checkout、容器与共享主机风险被明确隔离；模型类调用 `Attention` 不能推断具体 Flash/Paged backend，要求日志/profiler 证据。
- 修改：本轮无必要正文修改。
- 主要 primary 来源：vLLM [serve CLI](https://docs.vllm.ai/en/latest/cli/serve/)、[bench serve](https://docs.vllm.ai/en/latest/cli/bench/serve/)、[automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)，以及正文固定版本模型卡/部署链接。
- 未确认项：未启动 vLLM、未读取外部容器/模型权重、未执行本地 ledger 或请求 benchmark。

## roadmap/curriculum/systems/multi-gpu/README.md

- 总行数：175；已读至 EOF。
- 完整覆盖：DP/TP/PP/EP/CP；两次线性层 TP 公式与 CPU reference；KV cache 不等比缩小；collective 语义与 ring 等效带宽；EP/PP/CP；重叠条件；双 GPU 实践；并行方案选择；参考导航。
- 观察 1：TP 第一层列切、第二层行切的 `Z=sum_r Z_r` 推导清楚，明确不能把完整 partial 输出拼接；bias 位置、Norm 跨切分维度和 SwiGLU 双分支边界也已说明。
- 观察 2：AllGather/ReduceScatter/AllReduce 的输出语义、ring 等效带宽分母、shape/dtype/rank 顺序和 collective 顺序限制写得准确；不把等效带宽当物理链路实测。
- 观察 3：EP 负载偏斜、PP 空泡公式、CP 的在线 softmax 状态合并和 overlap 的依赖/生命周期均有具体排错方向；实践脚本明确需要两张 GPU，单卡不伪造通信成绩。
- 修改：本轮无必要正文修改。
- 主要 primary 来源：NVIDIA [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)、[Megatron parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)、CUDA Programming Guide（正文现有参考）。
- 未确认项：未运行 `torchrun`/NCCL、多 GPU 拓扑或通信 benchmark。

## 验证与交付边界

- 四篇 README 均已从首行完整读到 EOF；标题/锚点、source-check 代码、公式、原始实验与状态均保留。
- 正文唯一修改是 Mini Transformer 替换示例的 `norm_fn` → `norm`；source-check 疑似 contiguous 断言错误已告知主代理但未改动。
- 最终需由主代理对四个 README 及本记录运行限定路径 `git diff --check`；本轮未构建网页、未运行模型/GPU/NCCL benchmark。
