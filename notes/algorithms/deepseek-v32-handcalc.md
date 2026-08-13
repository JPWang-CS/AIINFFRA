# DeepSeek-V3.2 手算工作纸（主线 A 第 1 步）

> 目的：用 DeepSeek-V3.2 的配置数字算出三笔账，理解这个模型"为什么这么设计"。
> 方法：不用跑代码，拿数字套公式，每个结果写一句"所以需要 XX"。
> 说明：以下数字按 DeepSeek-V3 系公开量级估算，精确值以 HF config 为准；目标是数量级感，不是精确财报。

---

## 1. 先认识输入（模型追踪表 DeepSeek 行）

| 配置项 | 量级 | 是什么 |
|---|---|---|
| 总参数 | ~685B（vLLM recipe 写 671B，口径略差） | 全部权重的大小，决定"要占多少显存" |
| 激活参数 | 37B | 每个 token 实际参与计算的参数（MoE 特性） |
| 层数 | ~61（V3 系） | 每层都有注意力 + FFN |
| KV 压缩维度 | kv_lora_rank=512 + qk_rope_head_dim=64 | MLA 每 token 每层真正要存多少个数 |
| 注意力 | MLA + DSA | 压缩 KV + 只算 top-k（K=2048） |
| MoE | DeepSeekMoE + MTP-3 | 稀疏专家 + 投机解码 |

## 2. 前置：KV cache 是什么，DeepSeek 怎么用

动手算账之前，先搞清"KV cache"这四个字在模型里到底指什么。不然公式背下来也讲不清。

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

生成是逐 token 来的：第 1000 个 token 要跟 1~999 全部比一遍。第 1000 个 token 的 Q 是新的，但 1~999 的 K、V 早就算过，而且**永远不会变**——K/V 只依赖前面的 token，不依赖后面的。

为什么"早就算过"？因果注意力保证：token j 的表示只由它自己和它之前的 token 决定，后面来了新 token 也不会反向改变它。prompt 阶段（prefill）一次前向就把整段 prompt 的 K/V 全算出来；之后每生成一个新 token，就在它自己的那次前向里算出 K/V、追加到缓存末尾。所以第 1000 个 token 来时，1~999 的 K/V 要么是 prefill 算好的（prompt 部分），要么是生成时各自算好存下的（已生成的 token）。

两个选择：

- 每来一个新 token，把 1~999 的 K/V 全部重算一遍 → 计算量随序列长度平方增长，长上下文根本跑不动。
- 第一次算完就存进显存，后面直接用 → 这就是 **KV cache**：用显存换计算。

所以 KV cache 一句话定义：**推理时把每个历史 token 每层的 K、V 向量存在显存里，供后续 token 的 attention 重复使用。**

### 2.3 KV cache 有多大：先看形状，再算字节

每个 token 经过每一层，都会产生一组 K/V。存多少，取决于"每层有几个 KV head、每个 head 多长、K 和 V 各一份"：

```text
每 token 每层 KV 元素数 = 2 × KV head 数 × head_dim
```

以 MHA（每层 128 个 KV head，head_dim=128，BF16）为例：

```text
每 token 每层 = 2 × 128 × 128 = 32768 个数 × 2 字节 = 64 KB
61 层 → 64 KB × 61 ≈ 3.9 MB / token
128K 上下文 → 3.9 MB × 131072 ≈ 512 GB（单个请求！）
```

这就是为什么 KV cache 是长上下文第一瓶颈：它和上下文长度**线性增长**，而且每个请求独占一份，请求一多立刻爆显存。

顺带解释 GQA：它就是"减少 KV head 数"来省缓存——KV head 从 128 降到 8，KV cache 直接除以 16。代价是多个 query head 要共用同一组 K/V，表达能力变弱。

### 2.4 DeepSeek 怎么用：MLA → DSA → V4

**MLA（V2/V3/V3.2）**：不存完整的 K/V，而是先压成一个低维 latent（kv_lora_rank=512）+ 一小段位置编码（qk_rope_head_dim=64），attention 时再投影回完整形状。缓存里每个 token 每层只存 512+64=576 个数，而不是 MHA 的 32768 个——这是"存得少"的第一层省法。

**DSA（V3.2）**：KV 虽然压小了，但 attention 还是要把当前 token 跟全部历史比一遍。DSA 用 indexer 先粗筛，只对 top-k（K=2048）个位置做完整 attention——这是"算得少"的省法。注意：KV cache 本身依然要存（存 MLA 压缩版，才能随时取任何历史位置），省的是 attention 的计算量。

**V4（CSA/HCA）**：连"存什么"的粒度都改了——每 4 个 token 先压成 1 个 compressed entry，缓存按压缩条目存，1M 上下文下 KV 只有 V3.2 的 10%。这是"存得更少 + 算得更少"的下一代（详见 [deepseek-v4.md](deepseek-v4.md)，主线 A 第 3 步）。

### 2.5 三个数字先分清（别混）

```text
KV cache 大小 = 每 token 每层存几个数 × 层数 × 上下文 × 每个数几个字节
                ↑ MLA 决定存几个数        ↑ 长度/并发 决定多少份   ↑ 量化决定字节
```

后面三个杠杆都能看懂了：MLA 砍"存几个数"、FP8/BF16 砍"每个数几个字节"、DSA/V4 砍"attention 算几个"。
## 3. 第一笔账：KV cache 多大

**为什么有这个量**：推理时要给每个历史 token 存 K/V，供后面的 token 查询（K/V 是什么、为什么缓存 → §2 前置）。存的越少，长上下文越便宜。

**公式**：

```text
每 token 每层 KV 字节 = (kv_lora_rank + rope_dim) × 每元素字节
单请求 KV = 每 token 每层字节 × 层数 × 上下文长度
```

**代入**：

```text
每 token 每层 = (512 + 64) × 2 字节 = 1152 字节 ≈ 1.1 KB
61 层 → 1152 × 61 ≈ 70,272 字节/token ≈ 68.6 KB/token
128K 上下文 → 70,272 × 131,072 ≈ 9.2 GB（单个请求）
```

**对照**（同口径 128K）：

```text
MHA（128 heads × 128 dim）:  ≈ 512 GB
GQA（8 KV heads）:            ≈ 32 GB
MLA（V3.2）:                  ≈ 9 GB
```

**一句话结论**：MLA 把 KV 从几百 GB 压到个位数 GB，但 9GB/请求仍然不便宜 → 所以还要 DSA 只算 top-k、KV 量化（fp8_ds_mla）。

## 4. 第二笔账：权重显存多大

**为什么有这个量**：所有层所有参数都要驻留显存。MoE 很特殊：虽然每个 token 只激活 37B，但**全部 expert 权重都要放**，按总参 685B 算。

**公式**：

```text
权重显存 = 总参数 × 每参数占用字节
BF16/FP16: 2 字节/参数    FP8: 1 字节/参数    FP4: 0.5 字节/参数
```

**代入**：

```text
BF16: 685e9 × 2 ≈ 1.37 TB
FP8:  685e9 × 1 ≈ 685 GB
FP4:  685e9 × 0.5 ≈ 343 GB
```

**对到硬件**（80GB 卡）：

```text
BF16 → 需要 ~18 张（仅权重，不含 KV/激活）
FP8  → 需要 ~9 张
```

**一句话结论**：权重 1.37TB 决定"必须多卡 + EP + 量化"；量化到 FP8 直接少一半卡，这就是 FP8 在生产里是默认选项的原因。

## 5. 第三笔账：一次前向多少 FLOPs

**为什么有这个量**：FLOPs 决定"算多久"，也决定是计算瓶颈还是带宽瓶颈。

**公式**：

```text
prefill FLOPs ≈ 2 × token 数 × 激活参数
decode 每 token FLOPs ≈ 2 × 激活参数
```

**代入**：

```text
prefill 4096 token: 2 × 4096 × 37e9 ≈ 303 TFLOP
decode 每 token:     2 × 37e9 ≈ 74 GFLOP
```

**配合权重看瓶颈**：

```text
decode 每 token：
  计算只有 74 GFLOP，但必须从 HBM 读 ~1.37TB 权重（BF16）
  → 算力过剩、带宽不足 → memory-bound
  → 所以量化 + wideEP（多卡分摊权重读取）是必选项
prefill：
  303 TFLOP 是纯计算量 → compute-bound
  → 所以 prefill 和 decode 要分开（PD 分离），各自配不同的机器
```

## 6. 对照答案

| 手算项 | 答案（量级） | 一句话结论 |
|---|---|---|
| KV cache（128K 单请求） | ≈ 9 GB | MLA 压得狠，但还要 DSA/KV 量化 |
| 权重显存（BF16） | ≈ 1.37 TB | 必须多卡 + EP + FP8/FP4 |
| Prefill FLOPs（4096 token） | ≈ 303 TFLOP | prefill 计算密集 → PD 分离 |
| Decode 每 token | ≈ 74 GFLOP + 读 1.37TB | memory-bound → 量化 + wideEP |

## 7. 做完之后

1. 三个数都能自己从公式推出来，且每行"一句话结论"能讲出口
2. 回到 [模型追踪表](model-tracker.md)，在 DeepSeek-V3.2 行补一句你的数字（比如 "128K KV ≈9GB"）
3. 进 主线 A 第 2 步注意力（接 A5/FA1：FA2 → MLA → DSA）；训练侧枝干 A1（FP8 训练 → 优化器 → ZeRO/FSDP）到 serving 之后才学，不插队

---

*挂靠：主线 A 第 1 步 · 前置：[模型追踪表](model-tracker.md) · [架构地图 §11](latest-model-architectures.md)*