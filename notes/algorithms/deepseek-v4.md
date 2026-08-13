# DeepSeek-V4：CSA + HCA（主线 A 增量笔记）

> 挂靠：主线 A（DeepSeek-V3.2 → V4，第 2 步注意力之后做增量）
> 状态：🚧 Agent 草稿（2026-08-14），待随主线 A 第 3 步消化，不计入已学
> 先决条件：先消化 [FA1 机制](flash-attention-mechanism.md)、[MLA](mla-deepseek.md)、[DSA](dsa-sparse-attention.md)——这篇是"V3.2 打底 + V4 增量"，不是从零科普。

## 0. 这篇笔记回答什么问题

之前主线一直在 V3.2，被问到"没有 ds 不是有 V4 吗"——V4 确实存在，而且已经开源。这篇把 V4 讲清楚，同时说清一个学习判断：

**为什么主线不直接切到 V4？**

1. V3.2 是完整的开源基线：config 字段齐全、手算工作纸已建、推理栈支持成熟（vLLM / TRT-LLM / SGLang / TileRT 都有 sparse MLA）。V4 的官方 config 目前公开不完整，直接上手会缺一块。
2. V4 的省账要用 V3.2 当分母：官方说"1M 上下文下 V4-Pro 的 prefill FLOPs ≈ V3.2 的 27%、KV ≈ 10%"。不先算懂 V3.2，这个 27%/10% 就没有意义。
3. V4 不是全新架构，是 V3.2 的下一代：它保留了 MLA 的低秩 latent 骨架，把 DSA 的"选 token"升级成"先压缩、再稀疏"。顺序应该是 FA2 → MLA → DSA → 再看 V4 怎么改。

所以安排：主线 A 第 1、2 步照旧走 V3.2，第 3 步做 **V4 增量**（CSA/HCA → mHC/Muon → FP4 精度）。这篇就是第 3 步的笔记。

## 1. 发布线（先分清版本，别把三个日期混成一条新闻）

| 日期 | 事件 | 内容 |
|---|---|---|
| 2026-04-24 | V4 系列预览发布 + 开源 | V4-Pro（约 1.6T 总参 / 约 49B 激活，另有一说 1.5T）、V4-Flash（284B / 约 13B 激活）；原生 1M 上下文 |
| 2026-07-31 | V4-Flash-0731 正式版 | Flash 线转正 |
| 2026-08-13 | V4-Pro-0813 正式版 | 架构/参数与预览版一致（fingerprint `fp_v4pro_20260812`），变化全在后训练：DeepSWE 12.8→62.7、NL2Repo 38.5→61.5、DSBench-Hard 67.2；API 新增 Responses + Anthropic 协议；推理三档 none/high/max；官方预告近期涨价 |

两个值得记住的判断：

- **预览版和正式版架构一样**：正式版晚三个月，涨的是后训练能力，不是结构。学架构以 04-24 的技术内容为准，学能力以 08-13 的榜单为准。
- **参数口径有出入**：1.6T 和 1.5T 都有人用，说明"总参"这个数本身就依赖计数口径（是否含 embedding、是否含 MTP）。学习时记"约 1.6T / 49B active"就够，别在精确值上较劲。

## 2. V3.2 还差什么，V4 补了什么

V3.2 已经有两层省法：

- MLA：KV 不按 head 存，压成低维 latent，KV cache 从几百 GB 降到个位数 GB。
- DSA：KV 虽然省了，但 attention 计算仍要跟所有历史 token 比一遍；DSA 用 indexer 只选 top-k（K=2048）个 key 做完整 attention，把 O(L²) 降到 O(L·K)。

但到 1M 上下文，V3.2 还有两个问题：

1. **粒度还是 token**：DSA 的 top-k 在原始 token 上做，1M 个历史 token 光做索引、维护 indexer 的 KV 也不便宜；而且每个请求的 KV 即使 MLA 压缩后也还是线性涨。
2. **稀疏选错就丢信息**：top-k 是近似，万一选漏了关键内容，没有兜底。

V4 的思路一句话：**先把 token 压缩成"压缩条目"，再在压缩条目上做稠密和稀疏两层注意力**。存得比 MLA 更少，算得比 DSA 更少，同时用稠密层兜底召回。

官方给的账（1M 上下文，对比 V3.2）：

```text
V4-Pro：   prefill FLOPs ≈ V3.2 的 27%    KV cache ≈ V3.2 的 10%
V4-Flash： prefill FLOPs ≈ V3.2 的 10%    KV cache ≈ V3.2 的 7%
V4-Pro vs BF16 GQA-8 基线：KV ≈ 2%
```

注意这是"同上下文、同任务"下的相对数，不是 V4 比 V3.2 快 4 倍的意思——V4 参数量更大，单 token 成本反而更低，这是"规模变大但成本不线性变贵"的关键证据。

## 3. CSA：压缩稀疏注意力（核心零件）

### 3.1 一个 CSA 层干两件事：压缩 + 稀疏选择

先看名字：CSA = Compressed Sparse Attention。它把"注意力"拆成两个阶段：

**阶段一：压缩（Compress）**

- 粒度 m=4：每 4 个连续 token 压成 1 个 compressed entry。
- 窗口 2m：压缩不是 4 个独立一组，而是取长度 2m=8 的窗口、前后重叠（overlapped），窗口内做 joint softmax 加权求和。
- 形状变化：`[seq, d_model] → [seq/4, d_model]`（近似看；实际带 overlap，条目数略多）。

为什么重叠？单个 4-token 硬切会丢失边界上下文；重叠窗口让每个压缩条目都看到邻居信息，压缩质量更高。

**阶段二：稀疏选择（Sparse attend）**

- Lightning Indexer：一个轻量打分器，对全部压缩条目打分，选 top-k（Pro k=1024，Flash k=512）。
- 核心 attention 是 **shared-KV 多 query**：所有 head 共用同一组压缩 KV（MLA 风格低秩 latent 的延续），但每个 head 的 Q 各自算。
- 形状：`[num_queries, d] × [k, d]`，而不是 `[num_queries, 1M] × [1M, d]`。

### 3.2 为什么这样省（形状账）

1M 上下文，先压成约 250K 个压缩条目（÷4），再只对 top-k=1024 个做注意力：

```text
DSA（V3.2）：  对 1M 个原始 token 选 top-k，KV 条目 1M 级别
CSA（V4）：    对 250K 个压缩条目选 top-k=1024，KV 条目降到 250K 级别
注意力计算：  queries × (k + 滑窗) 而不是 queries × 全历史
```

省的是两笔：KV cache 存储（条目少 4 倍 + 低精度，见第 9 节）和 prefill 的 QK^T/PV 计算（只对选中的条目算）。

### 3.3 代价与取舍（面试要能说清）

- **压缩有损**：4 个 token 压成 1 个向量，细节必然丢。所以 V4 不能只有 CSA，必须有 HCA + 滑窗兜底（下一节）。
- **top-k 是近似**：indexer 的质量决定召回。官方给的数据：indexer 的 QK 全程 FP4、打分量化到 BF16，top-k 快 2 倍，KV recall 99.7%。这个"快 2 倍 + 99.7% 召回"就是取舍的量化表达。
- **多了一组算子**：压缩、解压、indexer 都是新 kernel，不是白省。省的是显存和 attention 主计算，多的是压缩管线的工程复杂度。
- **和 DSA 的区别要讲清**：DSA 在原始 token 上选 top-k（粒度 token，K=2048）；CSA 先压缩再选（粒度 compressed entry，k=1024）。CSA 的 KV 更小、预填充更省，但需要压缩质量做前提。

## 4. HCA：压缩稠密注意力（兜底）

HCA = Compressed Dense Attention（官方叫法里 HCA 的"稠密"指对压缩条目全量做）。

- 粒度 m'=128：每 128 个 token 压 1 个条目（比 CSA 粗得多）。
- 没有 indexer、没有 overlap：对**所有**压缩条目做 dense attention，不选。
- 作用：保证"每段历史至少被整体看过一次"。CSA 负责省、HCA 负责召回下限——即使 indexer 选漏了某段细节，HCA 也能拿到那段的整体信息。

这和之前见过的两个思想同源：

- DSA 的 anchor 层：少数层做 exact，其余层复用索引。
- Kascade：anchor 层算 exact top-k，跨层复用。

V4 的差别是"HCA 全层都有"：前 2 层纯 HCA，之后 CSA/HCA 交替（详见第 5 节），每一层都有稠密分支，而不是只在 anchor 层。

## 5. 层排布与滑窗（config 里怎么读）

- 每层保留最近 128 个**未压缩 token** 的滑窗分支（`sliding_window=128`）：当前位置附近用原始 token 精确注意力，细节不依赖压缩。
- V4-Pro：61 层 + MTP；前 2 层纯 HCA，之后 CSA/HCA 交替，最后一层不压缩。
- V4-Flash：43 层，同一种排布模式。
- 官方把每层压缩粒度写成 `compress_ratios` 数组（V4-Pro 形如 `[128, 128, 4, 128, 4, ..., 4, 0]`）：128=HCA 层、4=CSA 层、0=不压缩层。

学习时怎么读这个数组：不要背，先看懂三点——

1. 开头 2 个 128：输入附近信息密度最高，先用稠密把上下文"灌"进去。
2. 中间 4/128 交替：CSA 层省算力，HCA 层保召回，交替避免信息只走一条稀疏路。
3. 末尾 0：最后一层对原始 token 注意力，输出前把精度补回来。

## 6. 三个注意力细节（在 config 里能看到的新字段）

### 6.1 partial RoPE（只旋最后 64 维）

- `qk_rope_head_dim=64`：Q/K 只有最后 64 维做旋转位置编码，其余维度不带位置信息。
- 输出端用负位置逆旋转：把绝对位置去掉、只留相对位置差，对位置偏移更鲁棒。
- Infra 影响：K 的 RoPE 维和非 RoPE 维精度需求不同 → 这是 KV cache 混合精度（第 9 节）的直接原因。

### 6.2 attention sink（可学习 logit）

- 每个 head 一个可学习标量，直接加到 attention score 上。
- softmax 行和可以 < 1：注意力不需要"用完"所有概率质量，模型可以学会对不相关内容不给权重。
- 这是把"attention sink token"从工程技巧变成可学习参数；长上下文下更稳定。

### 6.3 grouped output projection（分组输出投影）

- 128 heads × head_dim 512；分成 16 组 × 8 heads。
- 组内先压到 bottleneck（`o_lora_rank=1024`）再投影回 hidden。
- 参数账：约 184M 替代原来约 470M 的平铺输出投影——输出投影是 Decode 阶段的大头权重之一，这里直接砍掉一半以上。

config 快照里能看到的 V4 新字段（与 V3.2 的 `kv_lora_rank` 不兼容，别混用）：

```text
q_lora_rank=1024    head_dim=512    qk_rope_head_dim=64
o_groups=8          o_lora_rank=1024    sliding_window=128
compress_ratios=[128,128,4,128,4,...,4,0]
```

## 7. mHC：流形约束超连接（残差流改造）

mHC = manifold-constrained HyperConnectivity。

- 普通残差是一条直路 `x + attn(x)`；mHC 是 4 条并行残差流（超连接），每层输出由多条流融合。
- 关键约束：每条流的 B 矩阵被约束为**双随机矩阵**（20 步 Sinkhorn-Knopp 迭代），谱范数 ≤ 1。
- 作用：防止深模型 + 长序列下残差累积爆炸，让 61 层（Flash 43 层）和 1M 上下文能稳定训练。
- 工程开销：约 1F1B（一个前向 + 一个后向）的 6.7%——是训练侧的投资，推理时几乎不额外花钱。

面试口径：mHC 解决"深 + 长"的数值稳定性，Muon 解决优化器效率和显存，两个一起让 V4 敢把上下文拉到 1M。

## 8. 优化器：Muon（替代 AdamW 的大矩阵部分）

V4 训练不再只用 AdamW：

- **大矩阵参数**（attention/FFN 的权重矩阵）用 Muon：动量方向做正交化（Newton-Schulz 混合迭代：前 8 步系数 `(3.4445, -4.7750, 2.0315)`，后 2 步 `(2, -1.5, 0.5)`），再按 RMS 缩放到 γ=0.18。
- **小参数**（embedding、lm_head、mHC bias、RMSNorm）仍用 AdamW。

为什么这么做（和 [Adam/AdamW 显存账](optimizers-adam.md) 连起来看）：

- AdamW 对每个参数维护 m、v 两份状态，大矩阵参数多时显存可观（约 2 倍参数量）。
- Muon 只维护一份动量，更新方向用正交化替代逐元素缩放；对矩阵参数来说，正交方向比逐元素缩放更符合"矩阵优化"的几何。
- 显存之外还有工程收益：ZeRO 兼容性靠 knapsack 分桶解决——把参数按桶打包，既保持 Muon 需要的全局正交结构，又能做 ZeRO 的切分。

学习顺序：先在第 7 步（serving 后）补 AdamW 的显存账，现在只需理解"Muon = 动量 + 正交化 + 一次缩放，替代 AdamW 的逐元素二阶矩"。

## 9. 精度：MXFP4 权重 + FP8 非专家 + 混合 KV

V4 的量化不是单一方案，是按"哪里敏感"分层：

| 部分 | 精度 | 原因 |
|---|---|---|
| MoE 专家权重 | MXFP4（E2M1，微缩放格式），量化感知训练 | 专家权重是显存和带宽大头，压到 0.5 字节/参数 |
| 非专家权重 | FP8 | 需要保精度的地方不用 4 bit |
| Lightning Indexer 的 QK | 全程 FP4 | 打分本身是近似操作，低精度换速度 |
| indexer 打分输出 | 量化到 BF16 | 保住 top-k 选择质量（KV recall 99.7%） |
| KV cache | 混合：RoPE 维 BF16、其余 FP8 | RoPE 维对位置误差敏感，非 RoPE 维可以更激进 |

要点：

- MXFP4 的 E2M1 = 2 bit 指数 + 1 bit 尾数 + 微缩放因子，不是简单 4 bit 定点。学的时候把它归到 [数值格式笔记](quantization-int8-fp8.md) 的延伸，先理解"微缩放"是给一小块共享一个 scale。
- 量化感知训练（QAT）意味着 FP4 不是推理时才压的，是训练时就带着量化误差练的——这和 AWQ/GPTQ 这种"训练后量化"是两条路线。

## 10. 训练与后训练（为什么正式版晚三个月）

预训练的关键设计：

- **32T+ tokens，序列 4K → 16K → 64K → 1M 渐进**：先短后长，避免 1M 上下文直接训不稳。
- **前 1T tokens 纯 dense attention，之后再切 sparse**：先学会"满注意力"再学"省注意力"，稀疏路线有个好老师。
- **Anticipatory Routing**：用 t−Δt 的 hidden state 算路由、θt 算 feature；loss spike 时启用，缓解 MoE/稀疏训练的路由抖动。
- **SwiGLU Clamping**：激活裁剪到 [−10, 10]，配合 mHC 控数值范围。

超参对照（训练成本的口径）：

```text
V4-Pro：   61 层，hidden 7168，384 experts / 6 active，峰值 LR 2.0e-4
V4-Flash： 43 层，hidden 4096，256 experts / 6 active，峰值 LR 2.7e-4
```

后训练：OPD（专家模型分域培养 + 蒸馏合并）：

1. 分域培养：每个专家模型在自己的领域 SFT + GRPO（领域特化）。
2. 多教师蒸馏：用全词表逆 KL，把多个专家模型的知识合并回主模型。
3. 工程细节：教师权重 offload 到分布式存储、只缓存最后一层 hidden state、数据按 teacher 排序——都是在"显存放不下多个教师"前提下的省钱做法。

这条对学习的意义：V4-Pro-0813 架构没变、benchmark 大涨，说明**后训练已经是能力增量的大头**。学模型不能只看架构，还要看数据/训练配方。

## 11. 推理系统侧：V4 的工程闭环

V4 是"架构省 + 工程省"一起上的，只学注意力不学工程会漏一半：

- **MegaMoE**：细粒度专家波次调度，通信与计算重叠。通用推理加速 1.5–1.73×，RL/agent 场景 1.96×；已在 DeepGEMM 开源，NVIDIA + 昇腾双平台验证。
- **TileLang DSL + DeepGEMM**：TileLang 替代手写 CUDA，DeepGEMM 替代 cuBLAS；Host Codegen 把 kernel 调用开销压到 <1μs；Z3 SMT 辅助验证；kernel 设计追求 batch-invariant + 确定性（同输入同输出，不依赖 batch 形状）。
- **磁盘 KV cache**：压缩 KV 落盘，前缀命中直接跳过 prefill——1M 上下文的前缀复用是磁盘 KV 的典型场景。
- **SWA KV 三档策略**：Full / Periodic / Zero——滑动窗口的 KV 是全部保留、周期性保留、还是全丢，三档换显存。

和主线 C（推理系统）的挂接：vLLM sparse MLA、PD 分离、PagedAttention 学完，再用 MegaMoE/TileLang/磁盘 KV 对照"V4 怎么把省下的资源变成吞吐"。

## 12. 数字账（记住这组数）

| 指标 | V3.2 | V4-Pro | V4-Flash |
|---|---|---|---|
| 总参数 | ~685B | ~1.6T（另一口径 1.5T） | 284B |
| 激活参数 | 37B | ~49B | ~13B |
| 层数 | ~61 | 61 + MTP | 43 |
| 注意力 | MLA + DSA | CSA + HCA（MLA 骨架） | 同左 |
| KV 粒度 | token（top-k 2048） | compressed entry（top-k 1024） | compressed entry（top-k 512） |
| 1M ctx KV cache | 基准 100% | ~10% | ~7% |
| 1M ctx prefill FLOPs | 基准 100% | ~27% | ~10% |
| vs BF16 GQA8 KV | — | ~2% | ~2% |

读法：**参数量大了，但单位 token 的存和算都小了**。面试讲 V4 先给这组数，再讲 CSA/HCA 怎么实现。

## 13. 与 V3.2 的关系 & 学习挂载点

- **不要把 V3.2 的手算公式直接套到 V4**：V4 没有 `kv_lora_rank`，改用 `compress_ratios`；`q_lora_rank=1024`、`head_dim=512`、`qk_rope_head_dim=64` 和 V3.2 的 512+64 是两套口径。等主线走到这一步，重开一张 V4 手算工作纸。
- 主线 A 挂载顺序：第 1 步 V3.2 手算 → 第 2 步注意力（FA2 → MLA → DSA）→ **第 3 步本篇（V4 增量）** → 第 4 步 MoE（对照 MegaMoE）→ 第 5 步 MTP → 第 6 步 serving（对照磁盘 KV/TileLang）→ 枝干 A1（训练侧，Muon 已提前见过）。
- 主线 B（Qwen3.5 GDN）和 V4 的对比点留到第 3 步：GDN 用 recurrent state 换线性长上下文，V4 用压缩+稀疏换线性长上下文，两者对 serving 的影响不同（state 管理 vs top-k 索引）。

## 14. 验证入口（等学到这里再深挖）

- HF Transformers docs: DeepSeekV4（config 字段）
- llama.cpp PR #24162（社区实现，能看 kernel 怎么落地）
- vLLM recipe: DeepSeek-V4-Pro（serving 配置和显存估算）
- SGLang Day 0 博客（推理栈支持状态）
- Megatron-LM issue #4468（Muon / FP4 QAT 的训练侧讨论）

## 15. 参考

- 心智观察所：掀开 DeepSeek-V4 的技术账本（news.ifeng.com/c/8sibRo38L6M）
- 腾讯云 deePhub：DeepSeek-V4 深度解读（百万上下文工程细节）
- Sebastian Raschka：CSA and HCA（llm-architecture-gallery）
- idlemachines：DeepSeek V4 from the inside
- 21 经济：V4-Pro 正式版发布细节
- LMSYS：SGLang Day 0