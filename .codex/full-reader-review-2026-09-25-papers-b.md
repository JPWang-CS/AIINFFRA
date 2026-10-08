# 2026-09-25 论文组 B 全文走读与校订记录

本记录含初稿及返工阶段的判断；GDN 数学转义、部分摘要表、书面表达与发布可见性又经主代理收尾。下文行数不是最终生成文件的固定值，最终事实与测试以 [总验收记录](full-reader-review-2026-09-25.md)为准。

本记录对应论文/算法专题组 B。初稿曾覆盖七个正文文件；本次返工写集仅为 `notes/algorithms/fa4-flexattention.md`、`notes/algorithms/attention-2026-sage3-kascade.md`、`notes/algorithms/deepseek-v32-handcalc.md` 与本记录。没有修改 `paper-catalog.cjs`、source-check 代码、原始 solutions、PATH/NOW 或生成站点，也没有 commit/push。

返工说明：本记录对应的第一版校订未通过主验收，复核发现 FA4 硬件表述、SageAttention3 FP4 数据流、Kascade/DSA 泛化语句以及 V3.2 cache/单位口径仍有错误。本轮仅返工 `fa4-flexattention.md`、`attention-2026-sage3-kascade.md`、`deepseek-v32-handcalc.md` 与本记录；其余文件不在本轮写集。

## 阅读路由

`roadmap/curriculum/gpu/course-site/paper-catalog.cjs` 的实际路由为：

- #26 DSA：源文本 `notes/algorithms/dsa-sparse-attention.md`，正文由 `roadmap/curriculum/papers/dsa.md` override；本轮完整阅读 override 正文。
- #27–31：分别读取 `notes/algorithms/gdn-linear-attention.md`、`fa4-flexattention.md`、`attention-2026-sage3-kascade.md`、`deepseek-v32-handcalc.md`、`deepseek-v4.md`。
- #39：读取 `roadmap/curriculum/papers/deepseek-v41.md`。

catalog 的 `prepare()` 还会替换标题介绍、删除状态/计划块、过滤部分章节并清理源码链接；本轮因此以源 README/notes 为完整阅读对象，没有把生成 HTML 当作权威正文。

发布层提醒：`prepare()` 会过滤名为 `## 与我何干` 的整段；本轮新增的 GDN、FA4、Sage 求职核对题已移到独立 H2，避免随过滤丢失。V3.2 的新增题目位于 `## 7. 求职核对题`，而当前 catalog 仍有针对旧 `## 7. 做完之后` 的替换规则，主代理应在发布层清理该过时规则；本轮不改 catalog。

## 完整阅读覆盖与具体问题

### `roadmap/curriculum/papers/dsa.md`（修改后 77 行，原 71 行）

完整阅读：引言、索引器与主 Attention 的双分数定义、selected gather/重新归一化、复杂度、未选 KV 生命周期、代码核对清单、CPU 检查与参考阅读。

具体问题与处理：

1. 原文把 `O(Td)→O(kd)` 放在主分支，容易被读成整条 DSA 路径的复杂度；改为明确 indexer 仍可能扫描全历史，并拆出 Top-K、gather 与主 Attention。
2. 原文只说“选中的 KV”，没有把 V3.2 实现的 MLA latent/KV entry 语境放在第一段；补充“选择对象是 entry，不等于先展开完整 K/V”。
3. 原有数学例子、mask 广播、source-check 与“只选部分不等价”的边界已足够清楚，未改其公式或代码；新增三个可现场推导的 shape/复杂度问题。

一手核对：DeepSeek-V3.2 论文 [arXiv:2512.02556](https://arxiv.org/abs/2512.02556) §2.1–2.3；官方 [DeepSeek-V3.2-Exp inference config](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/config_671B_v3.2.json)；官方 [inference repo](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)。未将论文 benchmark 或 H800 成本图表外推为本地性能。

### `notes/algorithms/gdn-linear-attention.md`（修改后 84 行，原 77 行）

完整阅读：问题定义、线性注意力状态、Gated DeltaNet 改进、Qwen3.5 混合层排布、取舍表、学习挂接和配套链接。

具体问题与处理：

1. 原递推写成 `gate×S + v⊗k`，丢掉了 delta rule 的预测误差；随后发现即使写成误差形式也必须先衰减状态。现按原论文 Eq. 8 改为 `S_decay=alpha*S`、`e=v-S_decay@k`、`S←S_decay+beta*e⊗k`，并明确 `q/k:[d_k]`、`v:[d_v]`、`S:[d_v,d_k]`。
2. 原文“没有传统 KV cache”会误伤混合架构；改为仅线性层不保存逐 token 历史，full-attention 层仍有自己的 KV cache；表格也不再把整网笼统写成近 O(N) 或把 full attention 写成无条件质量兜底。
3. 原文把 vLLM/SGLang/ONNX Runtime 都写成已专门支持，缺少统一一手依据；改为以 Transformers 官方实现的可选 `causal_conv1d`/`fla` 快速路径为例，并要求按 checkpoint/config 核对 serving 支持。

一手核对：Gated Delta Networks 原论文 [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)；[Qwen3.5 官方 config](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json)（60 层、`full_attention_interval=4`、512 experts、top-10）；[Transformers Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)。未把社区复现的内部投影宽度当作官方所有尺寸的共同配置。

### `notes/algorithms/fa4-flexattention.md`（返工后约 70 行）

完整阅读：问题、FlexAttention 的 `score_mod`/`block_mask`、FA4 backend、性能/硬件取舍、学习挂接。

具体问题与处理：

1. 原标题栏把 FA4 写成无条件“2026-03 正式发布”，改为 PyTorch 2026-03-05 官方博客公布的 backend 集成，避免把博客、wheel 默认值和所有设备可用性混成一件事。
2. 原文写“2026-07 PR 默认后端”但没有稳定 primary URL，删除该无据细节；保留博客给出的 Hopper/Blackwell 和 1.2–3.2× compute-bound 口径，并提醒版本/能力变化。
3. 修正硬件表：FA2 覆盖较广，但 FA3 作者实现面向 Hopper，不能写成“Ampere+（FA2/3）”；FA4 的 Blackwell 路径涉及 TCGEN05/TMEM 与新的异步管线。
4. 删除“只有块级稀疏才真正省 HBM”“CPU/任何 torch 都能写”这类绝对句，改为区分元素级规则、block mask、数学 reference、可用 Triton backend 与 FA4 目标设备。

一手核对：[PyTorch FlexAttention + FA4 官方博客](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)、[PyTorch 2.11 release blog](https://pytorch.org/blog/pytorch-2-11-release-blog/)、[FA4 upstream repository](https://github.com/Dao-AILab/flash-attention)、[FlexAttention docs](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html)。未声称任意 `score_mod` 都达到特化 kernel 性能。

### `notes/algorithms/attention-2026-sage3-kascade.md`（返工后约 105 行）

完整阅读：两条路线的动机、SageAttention3 FP4、Kascade anchor/reuse、取舍表、全景关系和本地学习挂接。

具体问题与处理：

1. 原文将两项都归为“2026 新优化”；改为 SageAttention3 2025-05、Kascade 2025-12，并把论文摘要/仓库的具体 baseline 与硬件写清。
2. 原文“2–5×”过度概括，改为 SageAttention3 摘要报告的 RTX 5090、1038 TOPS、约 5× 对最快 FlashAttention；Kascade 保留 H100 上最高 4.1× decode/2.2× prefill 且注明是 attention 口径。
3. 按 SageAttention3 §3 重写数据流：NVFP4 E2M1、1×16/E4M3 scale，Q/K/P/V 两个 FP4MM，P 的两级 scale；删除“Q/K 带宽直接 4×”“2–3 bit 尾数”“V 始终高精度”等错误句。
4. 明确 4-bit payload 不等于 HBM traffic/时间 4×，长序列不必然带宽受限；Kascade 的 10% 是论文实验点，anchor/head mapping 需按具体模型选择，不能写成任意模型即插即用；DSA 不概括为每层每头独立动态选择。

一手核对：[SageAttention3 arXiv](https://arxiv.org/abs/2505.11594)、[Kascade arXiv](https://arxiv.org/abs/2512.16391)、[Kascade 官方仓库](https://github.com/microsoft/kascade)。未把实验分支写成稳定 vLLM 默认路径，也未把 attention speedup 写成整模型吞吐。

### `notes/algorithms/deepseek-v32-handcalc.md`（返工后约 220 行）

完整阅读：模型追踪表、KV cache 前置、MHA/GQA/MLA/DSA/V4 对照、KV/权重/FLOPs 三笔账、对照答案和学习顺序。

具体问题与处理：

1. 原文把 685B、vLLM 671B、37B 激活和 V3.2 config 混成一个确定配置；改为官方 inference 文件名 671B、激活参数不由该文件单独证明，并明确所有量级的来源边界。
2. 原文用 BF16 MHA 示例和 FP8/MLA 实际路径时没有分层；改为 MHA 只是统一精度教学对照，MLA payload 需再核对 dtype/scale/layout。
3. 改正 K/V 依赖：它们来自每层 contextual hidden state，不是 token embedding；缓存复用依赖模型权重、位置策略、前缀状态和版本固定，无 cache 重算的是前置层计算而非只重做投影。
4. 统一容量单位并给出精确教学结果：MHA 488 GiB/523.986 GB，GQA 30.5 GiB/32.749 GB，MLA 教学假设 8.578125 GiB/9.210691584 GB；删除“9GB 所以 DSA 降容量”的因果。参数教学口径统一为 671e9，37e9 只作为 FLOPs 示例；删除 MTP-3 与 MoE 混列。
5. FLOPs 段改为工作量代理，不能单独决定 memory-bound/compute-bound；`## 8. 做完之后`保留锚点但改为书面总结，去掉学习口号。

一手核对：[官方 671B inference config](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/config_671B_v3.2.json)、[官方 V3.2-Exp README](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/README.md)、[V3.2 paper](https://arxiv.org/abs/2512.02556)。未把教学假设的 685B/37B 继续当作同一官方 config 的精确部署数字。

### `notes/algorithms/deepseek-v4.md`（修改后 267 行，原 265 行）

完整阅读：版本线、V3.2→V4差异、CSA/HCA、层排布、RoPE/sink/output projection、mHC、Muon、MXFP4/FP8/KV、训练后训练、推理系统、数字表、验证入口与参考阅读。

具体问题与处理：

1. 原文仍称 V4 官方 config “目前公开不完整”并列举多个未固定 serving 支持，改为承认官方模型卡、API 与 checkpoint/实现已经公开，但不同发布线仍需按版本核对。
2. V4-Flash 参数从 284B 修正为官方模型卡列出的 285B；V3.2/V4 其它比例继续保留为报告口径，不自行推导成硬件性能。
3. 原有 CSA/HCA、mHC、Muon、量化和训练细节已完整阅读；未在缺少逐组件同预算消融时声称单个组件造成全部能力提升。末尾补入模型卡/API/serving recipe 一手入口。

一手核对：[DeepSeek V4 model card PDF](https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf)、[DeepSeek API updates](https://api-docs.deepseek.com/updates/)、[vLLM V4-Flash recipe](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4-Flash.yaml)。未把二手媒体对 CSA/HCA 的解释当作唯一配置来源。

### `roadmap/curriculum/papers/deepseek-v41.md`（修改后 1090 行，原 1088 行）

完整阅读：参数口径与视觉张量、CED、CSA2/层次索引、890 B/token payload、Engram hash/门控/预取、DSpark、Single-Pass mHC 与 Mega API、Sinkhorn、训练数据与 packing、跨层共享训练责任、persistent KV/replay、后训练/GRPO/OPD、effort、评测阅读方法、作者最小推理实现和 CPU 检查问题。

具体问题与处理：

1. 正文已有大量 shape、生命周期和“报告数字不等于实测”的边界，保留这些实质内容；本轮只在开头补充报告版本、checkpoint、最小实现和部署策略的证据分层，避免读者把四种口径混成一项。
2. 逐段核对了 CED 的 prefill/replay 代理、CSA2 Full/Reindex/Reuse、候选池 16384 与最终 Top-K 512、890 B/token 的 FP4 payload 计算、DSpark 接受率、mHC comb source/destination 转置、Sinkhorn √n 与训练样本版本语义；公式和 source-check 未改。
3. 未把作者最小实现中的 `inplace=True` FP4 数值变换写成物理 packed FP4 cache，也未把综合 benchmark 写成 CSA2/Engram/CED 单组件因果证据；保留原文已有的限制说明。

一手核对：[DeepSeek-V4.1-Flash arXiv:2609.19969](https://arxiv.org/abs/2609.19969)（2026-09-17 v1，552B backbone、16B decode/8B prefill、890 B/token、45T tokens）、[官方 HF checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)、[固定版本 inference](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277/inference)。未将社区部署 recipe、未运行的本地服务或 CPU 检查当作作者 GPU 性能证明。

## 未验证项与风险

- 本轮未改 `paper-catalog.cjs`，未构建网站；仅在指定七篇源正文和本审阅记录上做 diff 检查。
- 未运行 source-check Python；代码块未改，但完整 GPU 编译、作者 checkpoint 推理、FA4/GDN/Kascade 运行环境仍未在本机验证。
- V4/V4.1 属于当前日期附近的快速演进资料；报告、模型卡、仓库 commit 和 serving recipe 可能继续变化，后续应保留固定 URL/commit 与发布日期。
- 本轮返工仍未构建网站；需由主代理确认 prepare 后三篇短文的题目/章节是否按预期保留，并在发布层清理 chapter 30 对旧 `## 7. 做完之后` 的替换。
