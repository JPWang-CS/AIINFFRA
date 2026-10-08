# DeepSeek-V3.2 手算工作纸（主线 A 第 1 步）

> 目的：用一组明确的教学假设核对 KV 容量、权重 payload 和线性层 FLOPs；每笔账都注明它能回答什么、不能回答什么。
> 方法：先写 shape、dtype 和计数口径，再代入公式；不要从一个数字直接推出部署方案或瓶颈。
> 说明：本页把官方 V3.2 inference config、论文和教学假设分开。官方仓库文件名为 `config_671B_v3.2.json`；总参数、激活参数和权重精度若未在同一来源中同时给出，只能标为量级或假设，不能混写成精确部署账。

---

## 1. 先认识输入（模型追踪表 DeepSeek 行）

| 配置项 | 量级 | 是什么 |
|---|---|---|
| 总参数 | 本页统一采用教学假设 `671e9` | 全部权重的参数量；先固定来源再换算 payload |
| 激活参数 | 本页把 `37e9` 只作为 FLOPs 示例假设 | 每个 token 的实际路径，需按专家和共享层定义 |
| 层数 | 61（官方 inference config） | 每层有注意力与 FFN，但前三层为 dense 配置 |
| KV 压缩维度 | kv_lora_rank=512 + qk_rope_head_dim=64 | MLA 每 token 每层真正要存多少个数 |
| 注意力 | MLA + DSA | 压缩 KV + 只算 top-k（K=2048） |
| MoE | 官方 inference config 的 256 routed experts、8 active experts | 这是专家路由配置；MTP/投机解码另算，不与 MoE 参数表混列 |

## 2. 前置：KV cache 是什么，DeepSeek 怎么用

计算缓存容量之前，需要先明确哪些张量会被后续生成步骤重复使用，以及这些张量的形状。

### 2.1 一次 attention 要算什么

Transformer 每层注意力对每个 token 算三份向量：

```text
Q（Query）：  当前 token "想找什么"
K（Key）：    历史 token "我有什么"（可被匹配的标签）
V（Value）：  历史 token "我能提供什么"（实际内容）
```

一个 query 对历史所有 token 算分数：

```text
score = softmax(Q · Kᵀ / √d)
输出  = score · V
```

注意：score 完全由 Q 和 K 生成——Q·Kᵀ 点积就是打分；V 不参与打分，只负责最后按分数加权。生成时 score 要现算，因为 Q 是新的，但 K 来自缓存。

举例：当前 token 是"猫"，它要跟前面所有词比对。Q 负责提问，每个历史词的 K 负责被匹配，V 负责提供"一旦匹配上就带走的内容"。

### 2.2 为什么推理时要缓存 K/V

生成是逐 token 来的：第 1000 个 token 要跟 1~999 全部比一遍。对固定的模型权重、位置编码策略、前缀和前向状态，第 1000 步之前的 K/V 可以复用；但它们不是 token embedding 直接产生的常量，而是各层由该 token 的 contextual hidden state 投影得到。改变模型版本、位置策略、前缀状态或中间表示，旧 K/V 就不能无条件沿用。

为什么可以复用？因果注意力保证，在模型权重、位置和前缀状态不变时，token j 的 contextual hidden state 不会被后来的 token 反向改变。prefill 一次前向算出 prompt 各层的 K/V；decode 每步只为新 token 计算各层 K/V 并追加。若没有 cache，就必须重新执行产生这些 hidden state 的前置层，代价不只是重新做一次 K/V 投影。

两个选择：

- 每一步重新执行前缀，会重复前置层和注意力计算；增长规律还取决于统计单步还是整个生成过程，不能只用 K/V 投影概括。
- 保存已算出的历史表示，使后续步骤复用它们。这就是 KV cache 的基本取舍：用存储避免重复计算。

所以 KV cache 一句话定义：**推理时把每个历史 token 每层的 K、V 向量存在显存里，供后续 token 的 attention 重复使用。**

补充三个容易混的点：

- **“追加”不是数值相加**：KV cache 是沿序列维度拼接，`cache_K[t]=K_t`，`cache_V[t]=V_t`，缓存从 `[K1]` 一路长到 `[K1,K2,...,Kt]`。
- **K/V 只算一次**：在固定模型、位置策略和前缀状态下，第 i 个 token 在某层的 contextual hidden state 确定后，$K_i=H_iW_K$、$V_i=H_iW_V$ 可缓存；它们不是只由 token embedding 决定。位置编码、层归一化、前缀缓存和权重版本变化都可能使旧缓存失效。
- **Q 不缓存**：每一步 attention 只用当前 token 的 Q 去查全部历史 K/V；旧 token 的 Q 在后续步不再被使用，缓存 Q 只会白占显存。
### 2.3 KV cache 有多大：先看形状，再算字节

每个 token 经过每一层，都会产生一组 K/V。存多少，取决于"每层有几个 KV head、每个 head 多长、K 和 V 各一份"：

```text
每 token 每层 KV 元素数 = 2 × KV head 数 × head_dim
```

以下统一采用 MHA 教学假设：每层 128 个 KV head、`head_dim=128`、61 层、128K 上下文、BF16、无分页/共享前缀/副本。它用于校对单位，不是 V3.2 的实际 cache contract：

```text
每 token 每层 = 2 × 128 × 128 = 32768 个数 × 2 字节 = 65,536 B
61 层 → 3,997,696 B / token
128K 上下文 → 523,986,010,112 B = 488 GiB = 523.986 GB（单个请求）
```

在不共享前缀的上述假设下，KV 容量随上下文长度和并发请求数增长，因此可能成为长上下文部署的约束。实际系统还可能共享前缀、分片或量化缓存；是否首先受 KV 限制，需要与权重及临时空间一起计算。

顺带解释 GQA：在同一 BF16/MHA 教学口径下，把 KV head 从 128 降到 8，逻辑 KV payload 除以 16：

```text
2 × 8 × 128 × 2 × 61 × 131072
= 32,749,125,632 B = 30.5 GiB = 32.749 GB
```

多个 query head 共享一组 K/V，但实际物理读取还取决于 kernel 是否复用已加载的 KV；不能从 head 数直接推出端到端时间或质量变化。

### 2.4 DeepSeek 怎么用：MLA → DSA → V4

**MLA（V2/V3/V3.2）**：不存完整的 K/V，而是先压成低维 latent（`kv_lora_rank=512`）和位置分支（`qk_rope_head_dim=64`），attention 时按实现重建需要的表示。缓存里的有效 payload 可按 576 个数做教学估算，但实际 dtype、scale 和布局要以 checkpoint/inference kernel 为准；这不是把任意 MLA 都简化成“只存一条向量”。

**DSA（V3.2）**：KV 虽然压小了，但 attention 还是要把当前 token 跟全部历史比一遍。DSA 用 indexer 先粗筛，只对 top-k（K=2048）个位置做完整 attention——这是"算得少"的省法。注意：KV cache 本身依然要存（存 MLA 压缩版，才能随时取任何历史位置），省的是 attention 的计算量。

**V4（CSA/HCA）**：压缩条目和稀疏选择改变了历史表示与访问粒度。具体压缩比、层排布和 cache payload 不能直接从 V3.2 的 `512+64` 套出，应以对应 V4 技术文档和 checkpoint 为准（详见 [deepseek-v4.md](deepseek-v4.md)）。

### 2.5 三个数字先分清（别混）

```text
KV cache 大小 = 每 token 每层存几个数 × 层数 × 上下文 × 每个数几个字节
                ↑ MLA 决定存几个数        ↑ 长度/并发 决定多少份   ↑ 量化决定字节
```

MLA 改变每个历史位置保存的内容，数值格式改变元素字节数，DSA 改变主 Attention 选取的位置。V4 还改变压缩条目的组织，因此需要另列缓存公式，而不能只替换一个 Top-K 参数。
## 3. 第一笔账：KV cache 多大

**为什么有这个量**：推理时要给每个历史 token 保存后续 Attention 需要的表示。这里的第一笔账只算一个明确的逻辑 payload，不把分页、scale、副本和 allocator 混进来。

**公式**：

```text
每 token 每层教学 payload = (kv_lora_rank + rope_dim) × 每元素字节
单请求教学 payload = 每 token 每层字节 × 独立保存层数 × 上下文长度
```

**代入**：

```text
每 token 每层 = (512 + 64) × 2 字节 = 1,152 B
61 层 → 70,272 B/token
128K 上下文 → 9,210,691,584 B = 8.578125 GiB = 9.210691584 GB
```

**对照**（同口径 128K）：

```text
MHA（128 heads × 128 dim）:  488 GiB / 523.986 GB
GQA（8 KV heads）:            30.5 GiB / 32.749 GB
MLA（512+64、BF16教学假设）:  8.578125 GiB / 9.210691584 GB
```

这些数字是容量账，不是“当前请求实际从 HBM 读取多少”。DSA 可能减少本次主 Attention 访问的 entry，但不会自动删除未选 KV，也不会把保存容量从 9.210691584 GB 变成更小；只有改变 cache contract、量化或真正 eviction 才改变容量账。

## 4. 第二笔账：权重显存多大

**为什么有这个量**：部署首先要容纳可被请求访问的权重。MoE 的“激活参数”只描述本次 token 经过的计算路径，不等于所有 expert 权重都必须完整驻留同一张卡；分片、量化、卸载和副本会改变实际放置。下面仅作 BF16/FP8/FP4 payload 的数量级练习。

**公式**：

```text
权重显存 = 总参数 × 每参数占用字节
BF16/FP16: 2 字节/参数    FP8: 1 字节/参数    FP4: 0.5 字节/参数
```

**代入（统一采用本页教学假设 671e9 参数）**：

```text
BF16: 671e9 × 2 ≈ 1.342 TB
FP8:  671e9 × 1 ≈ 671 GB
FP4:  671e9 × 0.5 ≈ 335.5 GB
```

**对到硬件**（80GB 卡）：

```text
BF16 → 仅按 80GB payload 粗算约 17 张
FP8  → 仅按 80GB payload 粗算约 9 张
```

参数总量和精度只给出 payload 下限；卡数还取决于分片、可用显存、KV、激活和通信缓冲。上面的张数是向上取整的容量练习，不是部署建议。

## 5. 第三笔账：一次前向多少 FLOPs

**为什么有这个量**：FLOPs 便于估计矩阵乘的工作量，但瓶颈还取决于权重/激活读取、Attention、通信、batch、设备和实现。

**公式**：

```text
prefill FLOPs ≈ 2 × token 数 × 激活参数
decode 每 token FLOPs ≈ 2 × 激活参数
```

**代入**：

```text
prefill 4096 token: 2 × 4096 × 37e9 ≈ 303 TFLOP（线性层示例）
decode 每 token:     2 × 37e9 ≈ 74 GFLOP（线性层示例）
```

**配合权重看瓶颈**：

```text
  decode 每 token：
  计算量可用激活参数做粗估，但每步实际读取的权重受专家集合、权重驻留和 cache 复用影响
  → 是否 memory-bound 要结合 batch、带宽和 profiler 判断
prefill：
  303 TFLOP 只是线性层粗估，不含 Attention、通信与调度
  → 是否适合 PD 分离要按 shape、batch 和端到端时间测量
```

## 6. 对照答案

| 手算项 | 答案（教学口径） | 读法 |
|---|---|---|
| KV cache（128K 单请求） | 8.578125 GiB / 9.210691584 GB（MLA BF16 教学假设） | 容量按 cache contract 计；DSA 改主 Attention 访问，KV 量化另算 |
| 权重显存（671e9、BF16 教学假设） | ≈ 1.342 TB | 再结合分片、精度、KV 与通信规划 |
| Prefill FLOPs（4096 token，线性层粗估） | ≈ 303 TFLOP | 还要加入 Attention、batch 和实测时间 |
| Decode 每 token（线性层粗估） | ≈ 74 GFLOP | 权重读取量按实际专家访问和放置测量 |

## 7. 求职核对题

1. 为什么同一份 `kv_lora_rank=512` 不能直接推导出所有实现的物理 KV bytes？请列出 dtype、scale、layout 和副本四个变量。
2. 给定 61 层、128K 和 `512+64` 的教学 payload，如何分别算单请求逻辑容量与实际分片显存？
3. 为什么 `2×激活参数` 只是 GEMM FLOPs 代理，不能单独证明 Decode 是 memory-bound 或 Prefill 必须做 PD 分离？

## 8. 做完之后

把三笔账重新写一遍时，先在纸上标出参数口径、dtype、层数、上下文长度和是否共享；再把“容量”“本次访问量”“FLOPs”和“实测时间”分成四列。这样才能继续把 V3.2 的 MLA/DSA 与后续 Attention、量化和 serving 章节连接起来，而不会把一个教学假设当成部署结论。

---

参考：[DeepSeek-V3.2 论文](https://arxiv.org/abs/2512.02556) · [官方 inference config](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/config_671B_v3.2.json) · [MLA 公式与代码](mla-deepseek.md)。
