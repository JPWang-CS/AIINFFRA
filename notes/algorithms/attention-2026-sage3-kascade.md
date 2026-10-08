# 2026 注意力新变体：SageAttention3（FP4）与 Kascade（anchor 稀疏）

> 注意力演进类 · 两条新路线：量化注意力 + 训练免稀疏注意力 · 论文整理
> 挂靠：主线 A 注意力实现侧 · 量化字典 · 面试全景

---

## 解决了什么问题

FlashAttention 已经把“避免物化 N×N 中间矩阵”推进到很高水平。SageAttention3（2025-05）与 Kascade（2025-12）分别从量化和结构复用继续压长序列推理的成本：

1. **降低注意力矩阵乘的操作数精度**：SageAttention3 用 NVFP4 微缩放，把 Q、K、注意力块 P 和 V 都送入两个 FP4 矩阵乘；它不是只量化 Q/K 的缓存技巧
2. **只给重要 token 花算力**：跨层复用 top-k 索引，只算最相关的历史（Kascade）

两者都主要改 attention 的执行路径，但收益依赖量化误差、候选覆盖、块密度、硬件和工作负载；“更便宜”需要在相同质量与端到端口径下验证。

---

## SageAttention3：FP4 量化注意力

### 核心思路

- 把 Q、K、P、V 的 tile 量化到 **NVFP4（E2M1；1×16 量化组，每组使用 E4M3 scale）**，由 FP4MM 同时接收 FP4 数据和对应 scale
- 用 Blackwell 的 FP4 Tensor Core 做 `QK^T` 和 `P V`；`P` 还使用两级缩放：先按行把 online-softmax 的 P 调整到 E4M3 更易表示的范围，再做第二层 FP4 微缩放
- 裸数据编码从 BF16 到 4-bit 只说明 payload 的位宽变化，不等于 HBM traffic 或 kernel 时间也按 4 倍变化；scale、读写合并、softmax、转换和工作集都会参与。长序列也不必然是带宽瓶颈
- 论文还探索了 8-bit 训练（SageBwd），训练侧仍是开放问题

### 关键数据与取舍

- 论文摘要报告在 RTX 5090 上达到 1038 TOPS、相对当时最快 FlashAttention 约 5×；这不是跨 GPU、跨 shape 的统一 2–5× 结论
- 依赖 Blackwell FP4 硬件；4090（Ada）没有 FP4 tensor core，只能学原理
- 论文把推理路径描述为 plug-and-play，但是否能直接替换仍取决于模型 dtype、量化校准、kernel 和硬件；不要把论文实现自动等同于任意 Hugging Face checkpoint

取舍：

- NVFP4 的 E2M1 是 1 个符号位、2 个指数位和 1 个尾数位；精度与范围依赖每个 1×16 block 的 E4M3 scale，不能把它写成“2–3 bit 尾数”
- 量化误差对长上下文/检索类任务更敏感，上线前必须评测
- 和 FP8 时代一样的问题：量化注意力最终是"默认选项"还是"可选优化"，取决于精度-速度权衡

---

## Kascade：训练免的跨层稀疏注意力

### 核心思路

作者方法依赖在实验模型中观察到的两个性质：

1. 部分注意力分布的概率质量集中在较少历史位置。
2. 某些层和 head 之间的重要位置存在可利用的相似性。

做法：

```text
在少数 anchor layer 上：算 exact top-k（head-aware，每个 head 独立选）
在中间 reuse layer 上：直接复用 anchor 的索引，只对这 k 个 key 做 attention
```

- anchor 层不是随便选的：用动态规划在开发集上挑"跨层相似度最大"的层组合
- 训练免：论文方法不要求重新训练，但 anchor 层集合、head mapping、候选策略仍需针对具体 checkpoint 和开发集选择；不能把“training-free”写成“任何模型无需校准即可套用”
- kernel 做 tile 级操作（tile_size=32），目前主要支持 fp16，vLLM 集成在 experimental 分支

### 关键数据与取舍

- 论文与官方仓库在 `top-k=10%` 的配置下报告相对 FlashAttention-3 baseline 的最高 4.1× decode、2.2× prefill；这是 attention 测量口径，不是整模型吞吐
- anchor 层越多越准但越贵；DP 选层是部署前要做一次的工作
- 稀疏率越高精度可能下降；10% 是论文报告的一组实验点，不是对所有模型和任务的通用推荐值

取舍：

- **和 DSA 对比**：DSA 的公开 V3.2 实现由 indexer 产生选择，并且其 mask/共享粒度要按实现核对；不能把它概括成“每层每头都独立动态选”。Kascade 则在 anchor 层计算选择、在 reuse 层复用经过模型选择的索引，二者是“重新评分/选择”与“跨层复用”的对照
- 跨层复用假设"相邻层高权重 key 稳定"，如果模型不满足这个性质（深层语义变化大），需要更多 anchor 层兜底

---

## 与主线的关系：2026 注意力优化全景

```text
可编程：FlexAttention + FA4      → score/mask 规则与块稀疏元数据进入后端
量化：  SageAttention3           → Q/K/P/V 的 NVFP4 矩阵乘与两级 P scale
稀疏：  DSA（indexer 选择）       → 选择集合改变主 Attention 访问，cache 仍按实际保存计
稀疏：  Kascade（跨层复用）       → anchor 选择与 reuse 层复用，需按模型选择 mapping
```

回答“FlashAttention 之后还能怎么优化”时，先分别说清改变的是 score 规则、operand 精度、访问集合还是跨层索引复用，再给出论文指定硬件和 baseline 的数字；不要把四个方向压成一张无条件的加速表。

## 与我何干

- **理论线**：SageAttention3 挂"量化"子类，Kascade 挂"注意力演进"子类——正好把你已有的 INT8/FP8 知识和 FA 知识接上
- **硬件限制**：SageAttention3 的原生 FP4 路径需要对应硬件；Kascade 的运行支持还取决于内核和软件版本，不能仅凭论文在 H100 上测量就判定其他设备不可运行。
- **C 阶段**：可以阅读官方仓库的 experimental vLLM integration branch；仓库明确说明该分支和 paged kernels 仍会变化，不能把它写成主线 vLLM 的稳定默认路径

## 求职核对题

- SageAttention3 的 FP4 量化降低了哪些 operand 的表示与搬运成本？为什么“缓存是 FP4”不等于“硬件一定执行原生 FP4 MMA”？
- Kascade 的 anchor 层为何需要开发集上的层选择和 head mapping？如果候选池漏掉深层真正需要的 token，后续 Top-K 能否补回？
- 报告中的 4.1×/2.2×比较了哪一个 attention baseline、硬件和 top-k 比例？怎样设计 ablation 才能把 index reuse 与 kernel tile 优化分开？

---

*配套：[SageAttention3 原论文](https://arxiv.org/abs/2505.11594) · [Kascade 原论文](https://arxiv.org/abs/2512.16391) · [Kascade 官方仓库](https://github.com/microsoft/kascade) · [DSA 稀疏注意力](dsa-sparse-attention.md) · [FA4/FlexAttention](fa4-flexattention.md) · [量化 INT8/FP8](quantization-int8-fp8.md)*
