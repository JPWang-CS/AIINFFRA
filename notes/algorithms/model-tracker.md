# 模型追踪表：最新模型与结构

> 用途：快速记录当前值得学的模型家族、关键结构、学习状态和可验证入口。
> 规则：闭源模型只记录公开事实；表格随新模型发布更新；每个模型至少要能讲清“结构 -> 推理影响”。
> 状态：🟢 已读/可讲 · 🟡 已了解 · ⚪ 待学 · 🔴 公开信息有限

---

## 学习路径

```text
先会读 config -> 先精读 1 个 Dense 模型 + 1 个 MoE 模型
-> 再补 MLA/SSM/混合架构 -> 最后用推理系统串起来
```

推荐先学：

1. LLaMA 系列：GQA + SwiGLU + RoPE + RMSNorm，最完整的 Dense 样本。
2. DeepSeek-V2/V3.2/V4：MLA → DSA → CSA/HCA，当前最重要的结构考点（V4 详见 [deepseek-v4.md](deepseek-v4.md)）。
3. Qwen 系列：Dense/MoE 路线都有，适合量化、vLLM 部署实验。
4. Mamba/Jamba：理解 Attention 之外的替代路线。

---

## 模型家族表

| 家族 | 版本/关注点 | 关键结构 | KV cache | MoE | 学习状态 | 入口 |
|------|------------|---------|----------|:--:|:--:|------|
| LLaMA | 3.1/3.2/3.3、4 代 | GQA、RoPE、SwiGLU、RMSNorm；4 代有 MoE/多模态 | GQA | 部分 | 🟡 | [latest-model-architectures](latest-model-architectures.md) |
| Qwen | 2.5、3 代 | GQA、SwiGLU、RoPE；Dense/MoE 都有 | GQA | 部分 | ⚪ | [latest-model-architectures](latest-model-architectures.md) |
| Qwen | 3.5（397B-A17B） | GDN×3 + Full Attention×1 混合、512+1 experts、top-10、RoPE、RMSNorm | GDN 层无 KV（recurrent state）；full attention 层 GQA | ✅ | 🟡 | [GDN](gdn-linear-attention.md) |
| DeepSeek | V2/V3/R1 | MLA、DeepSeekMoE、MTP、FP8 训练 | MLA | ✅ | 🟡 | [MLA](mla-deepseek.md) · [MoE](moe-inference.md) |
| DeepSeek | V3.2（~685B / 37B active） | MLA + DSA + DeepSeekMoE + MTP-3 | MLA + DSA indexer | ✅ | 🟡 | [DSA](dsa-sparse-attention.md) · [MLA](mla-deepseek.md) |
| DeepSeek | V4（Pro ~1.6T / ~49B active · Flash 284B / ~13B active） | CSA + HCA（MLA 骨架）+ Lightning Indexer top-k + 滑窗；MoE 权重 MXFP4、Muon、mHC | CSA 压缩 KV（1M ctx 下 ≈ V3.2 的 10%） | ✅ | ⚪ | [DeepSeek-V4](deepseek-v4.md) |
| GLM | 5 | DSA（复用 DeepSeek 实现）、256 experts、8 active、~44B active | DSA indexer | ✅ | 🟡 | [DSA](dsa-sparse-attention.md) |
| Mistral/Mixtral | 7B、8x7B、8x22B | GQA、Sliding Window、MoE | GQA | ✅ | 🟡 | [latest-model-architectures](latest-model-architectures.md) |
| GPT/o-series | GPT-4o、o1、o3 | Decoder-only；o-series 推理时思考；公开细节少 | 未公开 | 未公开 | 🔴 | [latest-model-architectures](latest-model-architectures.md) |
| Claude | 3.x/4 代 | 未完整公开；Transformer + 对齐 | 未公开 | 未公开 | 🔴 | [latest-model-architectures](latest-model-architectures.md) |
| Gemini | 1.5/2.x | 多模态 Transformer；规模化 MoE/TPU 路线 | 未公开 | 大概率 | 🔴 | [latest-model-architectures](latest-model-architectures.md) |
| Mamba/Jamba | Mamba-2、Jamba | SSM / SSM+Attention 混合 | 无传统 KV | 不一定 | 🟡 | [latest-model-architectures](latest-model-architectures.md) · [remaining-theory-primer](remaining-theory-primer.md) |
| RWKV / xLSTM | RWKV-6、xLSTM | 线性注意力 / LSTM 类结构 | 小状态 | 无 | ⚪ | [remaining-theory-primer](remaining-theory-primer.md) |

---

## 每个模型要能回答的问题

1. 这个模型是 Dense 还是 MoE？
2. Attention 用什么：MHA / MQA / GQA / MLA？
3. Norm 和位置编码是什么？
4. KV cache 大概多大？长上下文会怎样？
5. 推理瓶颈是权重、KV、还是路由通信？
6. 有没有我能在本地跑通的开源权重？
7. 它和上一代模型比，结构上改了什么？

---

## 结构观察清单

- `num_key_value_heads != num_attention_heads` -> GQA/MQA
- `num_experts` / `num_experts_per_tok` -> MoE
- `rope_theta` / `rope_scaling` -> RoPE 和长上下文策略
- `intermediate_size` 很大 -> FFN 权重占比高
- `tie_word_embeddings` -> embedding/lm_head 是否共享
- 论文/代码里出现 `latent` -> 可能是 MLA 或状态压缩

---

## 更新记录

- 2026-08-03：建立模型追踪表，先记录主流家族和结构观察点。
- 2026-08-13：新增 Qwen3.5（GDN 混合注意力）、DeepSeek-V3.2（MLA+DSA+MTP-3）、GLM-5（DSA 架构复用）。
- 2026-08-14：新增 DeepSeek-V4 行（CSA+HCA、Muon、MXFP4），资料入库；入口 [deepseek-v4.md](deepseek-v4.md)。
