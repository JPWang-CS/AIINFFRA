# DeepSeek Sparse Attention（DSA）与 GLM-5

> 模型架构类 · 可学习 indexer + top-k 稀疏注意力 · DeepSeek-V3.2 / GLM-5 生产验证
> 挂靠：主线 A DeepSeek-V3.2 第 2 步 · 主线 B GLM-5 · C 阶段 sparse MLA

---

## 解决了什么问题

- 长上下文 attention 是 O(L²)：128K 上下文里，每个 query 要和 128K 个 key 算点积，但绝大多数分数接近 0——大部分计算和访存都在算"无关紧要"的历史
- 已有稀疏方案（局部窗口、固定全局 token、哈希近似）要么有损，要么需要专门重新训练
- DSA 的思路：**用轻量 indexer 找出最相关的 top-k 个历史 token，只对它们做完整 attention**，并声称可以做到无损（indexer 学得好，被丢掉的历史本来就是无关的）

## 核心思路

### 1. Lightning Indexer：先打分，再选 top-k

每个 query 不是直接和全部历史做 attention，而是先过一个小型打分网络：

```text
query q
  → 打分：score(q, k_j) 对全部历史 key 计算相关性分数
  → 选择：取分数最高的 K 个 key（K = 2048）
  → 完整 attention：只在这 K 个 KV 上做标准的 QK^T / softmax / PV
```

- indexer 本身要轻：打分网络如果太重，省的 attention 开销又花回去了
- indexer 也要维护自己的 KV（给历史 token 打分用），有独立的轻量 cache
- top-k 选择是动态的：每层、每头选出的 key 都不一样，所以不能靠固定窗口跳过

### 2. 复杂度账

```text
完整 attention:   O(L²)   —— 每个 query 遍历全部历史
DSA:              O(L·K)  —— 每个 query 只遍历 K 个

128K 上下文、K=2048：L/K = 64
→ 计算和 HBM 读取的量级都省约 64x（理想情况）
```

注意这不是白拿的：

- top-k 选择本身有开销：打分（O(L) 小矩阵乘）+ 选择/排序（在硬件上要做 top-k kernel）
- K 是每层的上限，实际开销还要看 indexer 质量和每头稀疏度

### 3. 和 MLA 组合（DeepSeek-V3.2）

DSA 不是单独出现的，DeepSeek-V3.2 是"三件套"：

```text
MLA        → KV cache 存储变小（低秩 latent + 解耦 RoPE）
DSA        → 只读取/计算 top-k 的 KV，长上下文 O(L²) → O(L·K)
DeepSeekMoE → 激活参数少，权重显存大
MTP-3      → 推测解码，decode 每步多产出几个 token
```

- DeepSeek-V3.2：~685B 总参 / 37B 激活（vLLM recipe 口径写 671B，不同来源略有出入）
- 推理侧因此出现了专门的实现：vLLM 的 sparse MLA（FLASHMLA_SPARSE backend）、TRT-LLM 的 MTP-3 + sparse MLA、TileRT 的 `fp8_ds_mla` KV cache dtype

### 4. GLM-5：直接复用 DSA

- 智谱 GLM-5 在代码层面**复用 DeepSeek 的 DSA 注意力实现**（架构级复用，类似当年大家都用 GQA）
- GLM-5：78 层、256 experts、每 token 激活 8 个、激活参数约 44B、稀疏度约 5.9%（DeepSeek-V3.2 约 5.4%）
- 意义：稀疏注意力成为"标准件"后，模型之间互相复用实现，就像标准 attention 一样——infra 工程师只需适配一次 kernel，多个模型都能吃

## 关键数据与取舍

| 项目 | 数值 |
|---|---|
| DSA top-k | K = 2048（DeepSeek-V3.2）|
| 复杂度 | O(L²) → O(L·K) |
| DeepSeek-V3.2 | ~685B total / 37B active，MLA + DeepSeekMoE + DSA + MTP-3 |
| GLM-5 | 78 层、256 experts、8 active、~44B active、稀疏度 ~5.9% |
| 128K 上下文 | 量级省 ~64x（L/K=64，理想值）|

取舍：

- **indexer 决定稀疏质量**：选错 key 就是硬伤；DeepSeek 报告里强调 top-k 检索结果对 RL 稳定性很关键
- **动态 top-k 难优化**：每层每头都不同，无法整块跳过，kernel 要做 top-k 选择 + gather/select，比 causal 的块稀疏难写得多
- **indexer 有额外开销**：训练和推理都要维护 indexer 的 KV，打分和 top-k 的算力/带宽成本要计入总账

## 与我何干

- **理论线**：把 DSA 和 [Kascade](attention-2026-sage3-kascade.md) 对比着记——一个靠可学习 indexer（要训练），一个靠跨层结构复用（免训练）
- **C 阶段**：vLLM / SGLang / TRT-LLM / TileRT 都专门支持 V3.2/GLM-5（sparse MLA、fp8_ds_mla、MTP-3），这是 2026 年模型服务的实际工作内容，不是论文概念
- **[面试]** 口径："DeepSeek-V3.2 长上下文为什么便宜？" → MLA 把每 token KV 变小 + DSA 把参与计算的 key 从全部历史降到 top-k（K=2048），两者叠加

---

*配套：[MLA（DeepSeek）](mla-deepseek.md) · [GDN 线性注意力](gdn-linear-attention.md) · [模型追踪表](model-tracker.md)*