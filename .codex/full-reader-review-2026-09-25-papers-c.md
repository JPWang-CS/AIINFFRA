# 论文组 C 全文复核记录（2026-09-25）

本记录包含初稿与返工证据。原稿中“标题集合未改”的说明只适用于早期阶段；最终已重命名部分标题、合并无依据的性能表，并由主代理维护旧锚点。单层分页伪码的缓存计数和量化边界另经主代理收尾。最终状态与测试见 [总验收记录](full-reader-review-2026-09-25.md)。

范围：以下八篇正文均按行分块读至 EOF，包含公式、伪码、表格和章节末尾；本记录只覆盖论文组 C，不修改 `paper-catalog.cjs`、面试材料、主课、原始源码、状态文件或生成站点。主标题均保持原样；两个小节标题改为：`papers/inference/paged-attention.md` 的 `### 面试必考题` → `### 求职追问`，以及 `notes/algorithms/optimizers-adam.md` 的 `### 5. 显存账（训练必考）` → `### 5. 训练状态显存核算`。行数为本轮修改后的正文行数。

## `notes/algorithms/moe-inference.md`（111 行）

- 已读全部章节：问题定义、Router/Top-K、load imbalance、EP/AllToAll、专家权重管理、取舍表、Ascend 对应、求职追问、参考。
- 主要问题与有效内容：原文正确抓住稀疏激活、专家分片和 dispatch/combine 三条主线；但把激活参数少直接等同于更快，把 KV cache 与专家激活混列，并用未区分 top-k、本地/远端桶的 AllToAll 数字下结论。保留 Router 伪码和 DeepSeek/Mixtral 作为结构例子。
- 修改位置：开头重新定义 dense/MoE 与推理数据流；表格将 KV 改为由层数/head 配置决定；负载不均改为专家桶、GEMM 和尾部等待的因果说明；AllToAll 改为 `2×T×K×D×b` 的全跨 rank 上界并解释本地合并、打包和 trace；offload、共享专家和求职追问改为条件化表述。
- Primary URLs 与核对内容：
  - https://arxiv.org/abs/2401.04088：Mixtral 每层 8 个 FFN expert、每 token 选 2 个、总/激活参数的论文定义。
  - https://arxiv.org/abs/2412.19437：DeepSeek-V3 总/激活参数、DeepSeekMoE、辅助损失无关负载均衡和共享设计的论文摘要。
  - https://github.com/deepseek-ai/DeepSeek-V3：作者代码入口，未据此推断当前 serving backend。
- 剩余未确认项：不同 vLLM/DeepSpeed 版本的 expert batching、capacity/overflow 语义和 AllToAll 融合布局仍需按目标 checkout 与 profiler 核对；后续复核已删除无来源的 DeepSeek expert speculation 归因。

## `notes/algorithms/quantization-int8-fp8.md`（137 行）

- 已读全部章节：问题定义、格式表、INT8 对称/非对称公式、粒度、FP8、性能/精度、Ascend 对应、代码伪码、扩展主题。
- 主要问题与有效内容：原文的对称/非对称公式、scale 粒度和 FP8 E4M3/E5M2 介绍可用；但把格式、算法、训练/推理和 kernel 混在一起，并给出 A100/H100 固定 TFLOPS、tokens/s、精度百分点和普适硬件结论。保留并修正了教学伪码，使激活也先转成 INT8、累加后乘 `scale_x*scale_w`。
- 修改位置：开头增加“格式—算法—运行时 kernel”三层；补 PTQ/QAT 定义；FP8 补 scale 更新策略；删除固定性能/精度表，改为同模型、shape、dtype、版本和请求边界的测量链；跨设备段改为核对舍入、饱和、布局和指令；新增 PTQ/QAT、INT8/FP8 kernel 的求职追问。
- Primary URLs 与核对内容：
  - https://arxiv.org/abs/2306.00978：AWQ 用激活统计保护显著通道、等价缩放且不依赖反向重构。
  - https://arxiv.org/abs/2210.17323：GPTQ 的近似二阶、逐层训练后权重量化定位。
  - https://arxiv.org/abs/2211.10438：SmoothQuant 的等价缩放、W8A8 PTQ 和其论文条件下的结果。
  - https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html：FP8 recipe、amax history、E4M3/E5M2 混合用途。
  - https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html：矩阵乘 FLOPs、tile、Tensor Core 对齐与 shape 依赖。
- 剩余未确认项：目标 GPU（含 Ada/Hopper 等代际）、TensorRT-LLM/vLLM 版本的实际 FP8/INT8 kernel、scale layout 和校准集质量仍需现场验证；后续复核已删除相对 FP16 的伪精度倍数，补上零张量 scale、`[-127,127]` 对称裁剪、FP32 还原乘法和非 native INT8 伪码说明。

## `notes/algorithms/speculative-decoding.md`（131 行）

- 已读全部章节：逐 token 背景、Draft-Verify、拒绝采样伪码、两组加速公式、取舍表、Tree Attention、Medusa、Self-Speculative、Ascend 对应、求职追问、参考。
- 主要问题与有效内容：原文正确指出草稿/目标两阶段和并行验证；但 `p/q` 未截断、残差未归一化，两个加速公式口径不一致，且把接受率和固定倍数当作通用结论。保留变体结构，重写为前缀条件接受概率与每轮成本。
- 修改位置：伪码改为 `min(1,p/q)` 与归一化 `max(0,p-q)`；用 `E[L_acc]=Σ_i Π_j p_j` 和草稿/验证/提交成本的 token throughput proxy 替换固定倍数；表格改为按场景观察变量；新增分布保持、k 选择和高接受率仍变慢的求职追问。
- Primary URLs 与核对内容：
  - https://arxiv.org/abs/2211.17192：原论文的并行验证与不改变目标分布的采样方法。
  - https://arxiv.org/abs/2305.09781：SpecInfer 树状候选/Tree Attention 变体入口。
  - https://arxiv.org/abs/2401.10774：Medusa 多 draft head 变体入口。
- 剩余未确认项：具体引擎的 bonus token、KV 回滚、batch 调度和 draft model API 仍需按实现版本核对；后续复核已补齐向量残差采样、目标 logits 对齐和 `1+E[L_acc]` 进度口径。

## `notes/algorithms/pd-disaggregation.md`（126 行）

- 已读全部章节：Prefill/Decode 对比、P/D 物理分离流程、KV 传输例子、DistServe 数表、硬件选型、chunked prefill、Ascend 对应、求职追问、参考。
- 主要问题与有效内容：原文保留了阶段隔离、KV 迁移和 chunked prefill 三个核心对象；但把 Prefill/Decode 绝对化为 compute-bound/memory-bound，把一次链路算术当成普遍“非瓶颈”，并给出固定时延、倍数和成本比例。改为以 shape、batch、链路有效带宽、打包/同步和 TTFT/TPOT/SLO 的因果链描述。
- 修改位置：阶段表与干扰机制；KV payload/有效带宽段；性能表替换为复现实验边界；硬件选型改为目标约束；chunked prefill 和三条求职追问改为可验证条件。
- Primary URLs 与核对内容：
  - https://arxiv.org/abs/2401.09670：DistServe 的 TTFT/TPOT 约束、P/D 资源与并行策略共同优化、带宽感知放置。
  - https://arxiv.org/abs/2311.18677：Splitwise 主题入口，核对 P/D 阶段异构资源动机。
  - https://arxiv.org/abs/2308.16369：SARATHI 的 chunked-prefill 与 decode-maximal batching 机制。
  - https://arxiv.org/abs/2407.00079：Mooncake P/D 传输系统入口，未据摘要外推当前实现。
- 剩余未确认项：具体 RDMA/RoCE/NVLink 传输布局、KV 压缩、版本参数和服务编排仍需目标集群实测；后续复核已移除 TBT/TPOT 混淆、固定 L40S 建议和无来源 Ascend/MindIE/CloudMatrix 断言。

## `notes/algorithms/optimizers-adam.md`（193 行）

- 已读全部章节：SGD、Adam EMA/偏差修正、两步手算、AdamW、显存核算、低显存变体、ZeRO/FSDP、取舍表、Ascend 对应、求职追问。
- 主要问题与有效内容：Adam 公式和两步算例保留；原文把“每参数 16B”、AdamW 默认、ZeRO-1 通信和训练阶段固定成普遍事实，并把 optimizer state、参数、梯度生命周期混在一起。修改为精度/实现条件下的核算，补状态生命周期和 FSDP full-shard 语境。
- 修改位置：开头改为更新规则/状态容量/通信三层；AdamW 段明确衰减不进入 m/v；显存段限定 16B 假设；ZeRO 段改为按阶段和 collective 说明；表格及求职回答删除固定倍数，新增可复述更新链。
- Primary URLs 与核对内容：
  - https://arxiv.org/abs/1412.6980：Adam 的一阶/二阶矩与偏差修正原始论文入口。
  - https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html：AdamW weight decay 不累积进 momentum/variance 的实现定义。
  - https://docs.pytorch.org/docs/main/fsdp.html：FULL_SHARD 的 all-gather、reshard、reduce-scatter 和 optimizer state 语义。
- 剩余未确认项：低精度 optimizer state、foreach/fused/capturable 实现的额外峰值和目标训练框架的 bucket overlap 需按版本/profile 核对；后续复核已明确 v 是 raw second moment，并移除 FP8 master dtype 的未经核对断言。

## `papers/inference/paged-attention.md`（444 行）

- 已读全部章节：摘要与动机、KV 碎片、虚拟内存类比、block table/manager、Attention kernel、Prefill/Decode 伪码、copy-on-write、block size、性能结果、vLLM 数据结构与 CUDA 伪码。
- 主要问题与有效内容：逻辑 block→物理 block、引用计数、前缀共享和 copy-on-write 主线有效；原文把 MHA 的 KV 公式写成通用 hidden_dim，把显存利用率、block size、vLLM 默认和吞吐表当成跨版本事实，Decode 伪码还把新 KV 写成对 attention output 的投影。
- 修改位置：一句话和动机；KV 公式限定 MHA/GQA；页表查找语境；Prefill/Decode 伪码的 KV 生命周期；block size 与性能段改为实验条件；新增独立 H2 求职追问，避免被出版清理过滤。
- Primary URLs 与核对内容：
  - https://arxiv.org/abs/2309.06180：论文摘要确认动态 KV、碎片/重复、near-zero waste、共享与论文条件下 2–4× 结果。
  - https://github.com/vllm-project/vllm：实现入口；未将历史 `block_size=16` 当作当前默认。
- 剩余未确认项：当前 vLLM 版本的 block manager、paged kernel layout、prefix cache/COW 细节和不同 KV dtype 需按 checkout 核对；后续复核已清理历史默认 block size、固定性能表、旧版本断言和伪码的 KV 投影生命周期。

## `papers/training/zero-paper.md`（408 行）

- 已读全部章节：摘要/动机、DP 冗余、Adam FP16 状态、ZeRO-1/2/3、代码伪码、容量/通信、数据流、TP 对比、论文性能表、FSDP 用法、Ascend 对应、求职题、offload。
- 主要问题与有效内容：阶段递进、按需参数聚合、reduce-scatter、FSDP 代码入口有效；原文把 ZeRO-3/FSDP 等同、固定显存倍数和通信倍数，并将 ZeRO-2 静态字节算错为 2.xB/参数。已按参数/梯度/state 精度和临时对象重写。
- 修改位置：一句话和动机；阶段表；ZeRO-1/2 容量/通信；ZeRO-3 生命周期和通信；论文性能表改为复现实验边界；FSDP 和求职题改为 full-shard 语义；offload 删除固定百分比/倍数。
- Primary URLs 与核对内容：
  - https://arxiv.org/abs/1910.02054：ZeRO 原论文摘要、分片目标和大规模实验声明。
  - https://docs.pytorch.org/docs/main/fsdp.html：FSDP process group、FULL_SHARD、SHARD_GRAD_OP、all-gather/reduce-scatter 语义。
  - https://github.com/microsoft/DeepSpeed：DeepSpeed 官方实现入口。
- 剩余未确认项：具体 DeepSpeed ZeRO stage、offload、bucket、overlap、optimizer state dtype 和 FSDP 版本行为需在目标训练栈测量；后续复核已补足 ZeRO-3 反向前再次 all-gather、prefetch/buffer 峰值和 FSDP 与 ZeRO 语义限定。

## `roadmap/curriculum/papers/inference-workloads.md`（426 行）

- 已读全部章节：训练/推理分类、Scaling Law 推导与代码、MoE 边界、生命周期成本、RL/old logp/clipping、rollout 状态机、长尾调度、前缀共享、权重同步/KV 生命周期、共驻/分离、工具依赖图、KV offload、adaptive speculative、总加速公式、问题清单。
- 主要问题与有效内容：公式、状态表和 CPU 检查边界较完整；主要障碍是 Prefill/Decode 与训练语境进入太快、MoE 预算对象未拆开、裁剪段用训诫句、求职复述缺少“指标如何闭环”。
- 修改位置：开头定义训练/rollout/在线服务的指标轴；Scaling Law 说明 κ 的近似边界；MoE 预算拆为路由/GEMM/通信/放置；PPO clipping 改为正向机制解释；新增独立“求职追问：把训练—推理循环说完整”H2。
- Primary URLs 与核对内容：
  - https://arxiv.org/abs/2203.15556：Chinchilla 在固定计算预算下联合缩放模型和 token 的实验结论，支持“比例依数据/模型族而定”的限定。
  - https://arxiv.org/abs/2412.19437：DeepSeek-V3 的 MoE 激活规模和训练/后训练上下文，作为 MoE 边界的补充来源。
- 剩余未确认项：RL rollout 的具体框架版本、工具调用 loss mask、权重发布协议和异步 KV owner/version 需结合目标系统源码与 trace。

## 总体 diff-check 与风险

- 八篇全部覆盖至 EOF；标题集合未改，未改 `paper-catalog.cjs`、面试文件、主课、原始源码、状态文件或生成站点。
- `git diff --check` 应作为交付前检查；代码/伪码围栏和公式均需逐篇复核，尤其 ZeRO/PagedAttention 的伪码是教学抽象，不代表 DeepSpeed/vLLM 当前内部 API。
- 本轮没有运行 GPU/NPU benchmark、真实服务压测或分布式训练；所有性能/容量数字均改为条件化推导或来源限定，不能替代目标环境实测。
