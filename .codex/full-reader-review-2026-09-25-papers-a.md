# 论文组 A 全文走读记录（2026-09-25）

此记录是组内初稿审阅证据，不是最终验收声明。主代理随后修正了 GQA 无来源的 SuperGLUE 表、MLA 累加遗漏 token 归约、FA1 Python 广播/尾块和 Online Softmax 摘要表；原始行数与部分“保留”判断因此已被后续修订取代。最终内容、返工与 CPU 执行结果见 [总验收记录](full-reader-review-2026-09-25.md)。

本记录只覆盖本轮指定写集；正文均按文件顺序分块读取至 EOF，再按“前提 → 定义 → 公式 → 代码/数据流 → 理解检验”复核。发布源由 `roadmap/curriculum/gpu/course-site/paper-catalog.cjs` 确认：20–25 直接发布对应 notes/papers 文件；`prepare()` 会替换首段介绍、过滤指定规划段、清理状态/智能体字样，并在 23、25 追加教学段。因此本记录以源 Markdown 为审阅对象，不把生成网页当作正文证据。

## notes/algorithms/online-softmax.md

- 总行数：204；已完整读至 EOF。
- 实际观察：原文把 standalone online softmax 的 2 次输入扫描与 FlashAttention 的分块 IO 简化成“真正的 1 趟”，容易把在线状态融合误解成每个 Q/K/V 元素只访问一次；性能表中的 FlashAttention `N` 读、`0` 写也不代表一般实现的 HBM 事务。
- 实际观察：原文“实际提速 ~1.5–2×”没有绑定 shape、精度、GPU、计时范围或本地实测，不能保留为一般结论。
- 实际观察：多线程 merge 段已给出修正因子，但跨 block 的 atomic 只写成可选路径，缺少复合 `(m,S)` 状态的合并协议约束。
- 修改位置/类型：开头读写模型加前提；FlashAttention 段改为 Q/K/V tile、片上状态和复用的书面数据流；性能表改为条件化口径；删除无证据固定倍数；补充跨 block atomic 的协议限制；改写昇腾类比与面试追问。
- 主要 primary 来源：Milakov & Gimelshein，[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)，核对在线 normalizer 的分块/修正思想；Dao et al.，[FlashAttention](https://arxiv.org/abs/2205.14135)，核对 IO-aware tiling、避免物化完整 attention 矩阵的机制。
- 未确认项：未在本机 GPU 上重新统计具体 HBM bytes 或 standalone/FlashAttention 时间；文中不保留固定性能倍数。

## notes/algorithms/parallel-reduce.md

- 总行数：112；已完整读至 EOF。
- 实际观察：树归约代码默认 `BLOCK_SIZE` 为 2 的幂且输入已覆盖，原文未说明尾部条件；顺序归约段也未明确每轮 shared-memory 读写的同步边界。
- 实际观察：warp shuffle 被写成“最快/延迟最低”，且 `0xffffffff` 未注明要求完整 warp 参与；部分有效 lane、源 lane 不活跃时不能沿用该 mask。
- 实际观察：0.8×、0.3× 的性能表没有实验条件，无法作为课程事实；浮点加法非严格结合也未在开头说明。
- 修改位置/类型：补充归约可组合性与浮点舍入限制；注明边界/2 的幂前提；明确 shuffle mask 与源 lane 约束；将性能数字改为空待测表；重写跨 block 与昇腾类比。
- 主要 primary 来源：NVIDIA，[CUDA C++ Programming Guide 5.4.6.6](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)，核对 `__shfl_sync` mask、参与线程与同步约束；NVIDIA，[Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)，核对 warp shuffle 适用范围与寄存器交换。
- 未确认项：没有在目标设备上比较三种实现的实际延迟；性能栏留“待测”。

## notes/algorithms/flash-attention-mechanism.md

- 总行数：161；已完整读至 EOF。
- 实际观察：原文把 causal mask 的收益固定写成计算量和访存减半，忽略非整除边界与 tile mask；PagedAttention page 与 attention tile 也被写成一一对应。
- 实际观察：A100 带宽、100/20/400 ms、5×/10×、SRAM/HBM 带宽比等数字没有 shape、精度、实现和计时条件；不能作为课程事实。
- 实际观察：recomputation 的方向正确，但“重算比存储+读取更快”需要绑定具体反向实现和硬件瓶颈；`m/l` dtype 也不应写成所有实现固定 FP32。
- 修改位置/类型：收紧 FlashAttention 的 IO 数据流和“1 趟”表述；将无条件性能数字改为按环境实测；补充 causal tile、PagedAttention page/tile、重算和状态 dtype 的限制；保留可复算容量示例与原论文链接。
- 主要 primary 来源：Dao et al.，[FlashAttention](https://arxiv.org/abs/2205.14135)，核对 IO-aware tiling、exact attention 与不物化中间矩阵；作者仓库 [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)，作为实现边界和版本对照。
- 未确认项：未在本机 GPU 重测延迟、HBM bytes 或 causal mask 实际节省比例。

## notes/algorithms/flash-attention-2.md

- 总行数：706；已完整读至 EOF。
- 实际观察：公式、手算例和 `alpha` 同步缩放 `$\ell/U$` 的推导一致；教学 Triton/CUDA 代码已明确不代表论文级 kernel，应保留这一边界。
- 实际观察：Triton `grid`、causal tile、尾块 mask 和 warp 分工均有教学映射，但原文部分句子容易把映射当作完整 FA2 实现；FP32 状态是常见策略而非逻辑必需。
- 实际观察：论文报告的性能数字已有硬件/shape/dtype 条件限定；末尾“七题不看笔记”等元话语不利于连续书面阅读。
- 修改位置/类型：删除顶部 Agent 状态元话语；在 Q-tile、Triton grid、store、FP32 状态和 causal mask 处补充适用范围；修正 FA2 状态记账的书面表达；将自检收束改为证据导向说明，未改 source-check 代码。
- 主要 primary 来源：Dao et al.，[FlashAttention-2](https://arxiv.org/abs/2307.08691)，核对减少 non-matmul FLOPs、沿序列增加 block 并行、warp 分工和论文指标；Triton，[Fused Attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)，对照教学接口语义。
- 未确认项：仓库教学 Triton/CUDA 未在本轮编译或 GPU 上验证；论文性能数字不作为本仓库成绩。

## notes/algorithms/mla-deepseek.md

- 总行数：1049；已完整读至 EOF。
- 实际观察：DeepSeek-V2 的低秩 KV、decoupled RoPE、`W_QK/W_OV` 吸收和 KV 元素数手算主线连贯；随机小 shape naive/absorbed 等价检查应保留。
- 实际观察：第 12 节伪代码把每个 tile 的 `online_softmax(scores)` 直接累加 latent output，按字面实现会漏掉跨 tile 的 running max、分母和旧状态重标定。
- 实际观察：原文把“只保存 c_KV”写得过于绝对，完整 cache 还包含共享的 RoPE key；Q compression 主要影响当前计算/训练 activation，不是直接减少历史 KV cache。
- 修改位置/类型：收紧 content cache 限定；将“KV Cache 账本”改为书面标题；重写第 12 节伪代码为显式 `(m,l,latent_acc)` 更新并补禁止逐 tile softmax 直接相加的说明；保留原有公式、代码检查和状态记录。
- 主要 primary 来源：DeepSeek-V2，[arXiv:2405.04434](https://arxiv.org/abs/2405.04434)，核对 MLA 公式、decoupled RoPE、Appendix C 吸收与 cache 维度；作者仓库，[deepseek-ai/DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2)，核对公开项目对 MLA 与推理实现的定位。
- 未确认项：本轮未运行示例代码或目标 GPU kernel；权重吸收后的浮点误差只保留源文中的随机小 shape 检查口径。

## papers/attention/gqa.md

- 总行数：202；已完整读至 EOF。
- 实际观察：原文 MQA 的 `1.6%` 是特定 H 下的数，不是通用比例；GQA Cache 相对 MHA 应按 `G/H` 推导，并要求 query/KV head 数整除映射。
- 实际观察：LLaMA 2 70B 容量例子混用 MB/GB 与 MiB/GiB；修正后为 320 KiB/token、2.5 MiB/token、640/80 GiB 的纯 Cache 算术示例。
- 实际观察：PyTorch `einsum` 复用了同一 `s` 标签，未区分 query/key 序列维度；Triton “唯一核心改动”、8× 并发和 G=8 甜点值均过度泛化。
- 修改位置/类型：修正 GQA 定义/比例、容量单位、`einsum` 标签和 head-divisibility 前提；将固定吞吐/并发结论改为条件化；把求职追问改为 Cache 手算、uptraining 前提和 kernel stride/page 排查。
- 主要 primary 来源：Ainslie et al.，[GQA](https://arxiv.org/abs/2305.13245)，核对 GQA 定义、5% uptraining、mean-pooling 比较和质量结果；Meta，[Llama 2 model card](https://github.com/meta-llama/llama/blob/main/MODEL_CARD.md) 与 [Llama 2 paper](https://arxiv.org/abs/2307.09288)，核对 70B 使用 GQA；作者仓库，[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)，核对 GQA API 的 head 数整除约束。
- 未确认项：未在本机安装 flash-attn 或 GPU 上测 GQA/PagedAttention 吞吐；性能与并发只保留机制条件。
