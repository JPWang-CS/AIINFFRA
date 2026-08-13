# Algorithms — 理论线（模型驱动主线 + 字典）

> 主线决定"学什么"，字典按需查"某个概念是什么"。不再按概念平行推进——概念只有挂在模型/系统上才有位置。
> 状态约定：`🚧 草稿/速览` 表示 Agent 已生成、用户还没消化，不计入"已学"。
> 当前学哪条由 [NOW.md](../../NOW.md) 指定，全貌和进度在 [PATH.md](../../PATH.md) 理论线。

## 为什么是"主干 + 枝干 + 字典"

按概念分类（注意力一列、量化一列）容易越学越散。主流做法是**跟模型/系统学**：拆一个开源模型，注意力、MoE、KV、投机解码、serving 全部自然出现。

- 两条主线 = 两个骨架模型：DeepSeek-V3.2 → V4（推理系统全家桶，V3.2 打底、V4 做增量）+ Qwen3.5（混合注意力 + 长上下文）
- 字典 = 概念笔记，按 6 子类归档，每篇开头带 `> 挂靠：` 说明属于哪条主线/哪个系统
- 主干 = 模型主线，决定顺序；枝干 = 必要小模块，挂在主干的具体步骤上，挂上就按顺序学、不推迟、不跳过（例：训练侧枝干 A1 挂在主线 A 第 7 步，serving 之后，不插队）
- 铁律：**主干决定顺序，枝干是主干的必要延伸**；字典只做概念速查，不是学习路径。

## 笔记规则

一条一页，固定四段 + 挂靠：

> **解决什么问题 → 核心思路 → 关键数据/取舍 → 与我何干**，开头一行 `> 挂靠：<属于哪条主线/哪个系统/哪个阶段>`

## 📌 和 papers/ 的边界（硬规矩）

俩都是"读+理论"，容易混。规矩：

> **单位是"一篇论文" → [papers/](../../papers/)；单位是"一个技术/概念" → 这里。**

- **非论文技术**（online softmax、parallel reduce、continuous batching）→ 只在这里。散在多篇论文/博客/代码里的，本就没有单篇归属。
- **有标志性论文的技术**（AWQ、GQA、ZeRO）→ 两边各一份，角度不同：
  - `papers/xxx.md`：这篇论文讲了什么（精读，带 arxiv）
  - 这里 `xxx.md`：机制 + 我会怎么实现/用，`[[链]]`到论文
- 判断口诀：**"读论文" ≠ "会实现"**。前者进 papers，后者进这里。

---

## 🧭 路由（怎么用这份 README）

| 你现在在哪 | 去哪 |
|---|---|
| 刚进来 / 不知道学什么 | 看 [NOW.md](../../NOW.md) 当前焦点 → 走对应主线 |
| 主线学完一步 | 回到主线，走下一步 |
| 主线里遇到不懂的概念 | 点步骤里的字典链接，或到下方字典按子类查 |
| 想知道某个模型的状态 | [模型追踪表](model-tracker.md) |
| 想知道全景/进度 | [PATH.md](../../PATH.md) 理论线 |

---

## 🚀 主线 A：DeepSeek-V3.2 → V4（从 config 到 serving）

> 目标：拆一个生产级 MoE 模型，把"注意力/KV/量化/投机/系统"一条线串完。

1. **config + 手算（热身）**：KV cache、权重显存、激活 FLOPs → [手算工作纸](deepseek-v32-handcalc.md)；目的：给第 2 步的 MLA/DSA 和第 3 步的 V4 增量提供数字基础（工作纸 §2 先讲清 KV cache 是什么再算账）
2. **注意力（接 A5 / FA1）**：[FA2 机制](flash-attention-2.md) → MLA（[mla-deepseek.md](mla-deepseek.md)）→ DSA（[dsa-sparse-attention.md](dsa-sparse-attention.md)）
   - 🪵 枝干 A2（KV 侧，必学）：GQA/MLA 对比 + KV 量化 / SmoothQuant（[速览](remaining-theory-primer.md)）
3. **V4 增量（V3.2 打底之上，2026-08-13 正式版）**：CSA/HCA 混合注意力（[deepseek-v4.md](deepseek-v4.md)）→ mHC + [Muon 优化器](optimizers-adam.md) → MXFP4/混合精度；和 V3.2 的 DSA 对比，把 1M 上下文 27%/10% 的账算出来
4. **MoE**：路由、显存、EP 通信（[moe-inference.md](moe-inference.md)）；对照 V4 的 MegaMoE 波次调度
   - 🪵 枝干 A3（权重侧，必学）：AWQ / GPTQ / FP8 / MXFP4（[速览](remaining-theory-primer.md)）
5. **投机解码**：MTP（[speculative-decoding.md](speculative-decoding.md)）
6. **serving（C 阶段）**：vLLM sparse MLA / TRT-LLM MTP-3 / TileRT；V4 侧：磁盘 KV + 三档 SWA + TileLang/DeepGEMM → 相关 [PD 分离](pd-disaggregation.md)
7. **横向扩展 · 枝干 A1（训练侧，必学）**：DeepSeek-V3 的 FP8 训练 → [优化器 Adam/AdamW](optimizers-adam.md)（显存账；Muon 已在第 3 步顺带看过）→ [ZeRO/FSDP](remaining-theory-primer.md) → TP/PP/EP 概念

## 🚀 主线 B：Qwen3.5（混合注意力 + 长上下文）

> 目标：理解"线性注意力 + full attention"为什么是 2026 架构模板。

1. **混合结构**：GDN 线性注意力（[gdn-linear-attention.md](gdn-linear-attention.md)）+ 3:1 成本账
   - 🪵 枝干 B1（同族扩展，必学）：Mamba / SSM 家族（[速览](remaining-theory-primer.md) + [架构地图 §6](latest-model-architectures.md)）
2. **MoE**：512+1 experts、top-10，激活 17B 的服务影响
3. **长上下文对比**：GDN 的 recurrent state vs DSA 的 top-k KV（[dsa-sparse-attention.md](dsa-sparse-attention.md)）
4. **serving（C 阶段）**：SGLang / vLLM 对 Qwen3.5 state 的管理

> 两条主线共用字典：注意力实现侧（FA2/FA4/SageAttention3/Kascade）先挂在主线 A 的注意力步骤上，消化完再单独回看。

---

# 字典（6 子类，按需查）

## GPU 优化算法
| 主题 | 挂靠 | 状态 |
|------|------|:--:|
| [online softmax](online-softmax.md)（Flash 的心脏） | FA/Flash 全线 | ✅ |
| [parallel reduce / prefix sum](parallel-reduce.md) | FA/Norm/softmax | ✅ |
| Norm 的 reduce 模式（LayerNorm/RMSNorm，[速览](remaining-theory-primer.md)，料→[reference](../../reference/cuda/layernorm/layernorm.cu)） | 任意模型每层 | 🚧 速览 |
| work partitioning（Flash 2 的思路） | 主线 A 前置 | 🚧 [速览](remaining-theory-primer.md) |

## 量化
| 主题 | 有论文? | 挂靠 | 状态 |
|------|:--:|------|:--:|
| [数值格式 INT8 / FP8](quantization-int8-fp8.md) | — | C4 量化通路 | 🚧 草稿 |
| AWQ | ✔ 两边写 | C4 / 主线 A FP8 | 🚧 [速览](remaining-theory-primer.md) |
| GPTQ | ✔ 两边写 | C4 | 🚧 [速览](remaining-theory-primer.md) |
| SmoothQuant / KV Cache 量化 | 部分 | 主线 A KV | 🚧 [速览](remaining-theory-primer.md) |
| [SageAttention3（FP4 量化注意力）](attention-2026-sage3-kascade.md) | ✔ | 主线 A 注意力实现侧 | 🚧 草稿 2026-08-13 |

## 注意力演进
| 主题 | 有论文? | 挂靠 | 状态 |
|------|:--:|------|:--:|
| MHA→MQA→GQA→MLA | ✔ [GQA](../../papers/attention/gqa.md) | 主线 A 第 2 步 | 🚧 [速览](remaining-theory-primer.md) |
| [Flash Attention 机制](flash-attention-mechanism.md) | ✔ [FA1](../../papers/attention/flash-attention.md) · [FA2](../../papers/attention/flash-attention-2.md) | 主线 A 前置 · B3 | ✅ FA1 已消化（2026-08-10 A5）；[FA2](flash-attention-2.md) 🚧 草稿 2026-08-13；FA3 待补 |
| [FA4 / FlexAttention](fa4-flexattention.md) | ✔ | 主线 A/B 注意力实现侧 | 🚧 草稿 2026-08-13 |
| [DSA（DeepSeek-V3.2 / GLM-5）](dsa-sparse-attention.md) | ✔ | 主线 A 第 2 步 · 主线 B 第 3 步 | 🚧 草稿 2026-08-13 |
| [SageAttention3 / Kascade](attention-2026-sage3-kascade.md) | ✔ SageAttention3 · Kascade | 主线 A 注意力实现侧 | 🚧 草稿 2026-08-13 |
| [MLA（DeepSeek-V2/V3）](mla-deepseek.md) | ✔ DeepSeek-V2 | 主线 A 第 2 步 | 🚧 草稿 |
| 线性注意力 / [GDN（Qwen3.5）](gdn-linear-attention.md) / Ring Attention | ✔ | 主线 B 第 1/3 步 | 🚧 [速览](remaining-theory-primer.md) + GDN 草稿 2026-08-13 |

## 模型架构
| 主题 | 挂靠 | 状态 |
|------|------|:--:|
| [最新模型架构地图](latest-model-architectures.md)（LLaMA/Qwen/DeepSeek/GPT/Claude/Gemini/MoE/SSM） | 所有主线第一步 | 🚧 草稿 |
| [模型追踪表](model-tracker.md) | 所有主线的状态索引 | 🚧 草稿 |
| [MoE 推理挑战](moe-inference.md) | 主线 A 第 4 步 | 🚧 草稿 |
| [DeepSeek-V4（CSA + HCA）](deepseek-v4.md) | 主线 A 第 3 步（V3.2 打底、V4 增量） | 🚧 草稿 2026-08-14 |
| Mamba / SSM | 主线 B 第 1 步（GDN 同族） | 🚧 [最新模型与结构](latest-model-architectures.md) |

## 推理系统技术
| 主题 | 挂靠 | 状态 |
|------|------|:--:|
| continuous batching | C3 调度 | 🚧 [速览](remaining-theory-primer.md) |
| chunked prefill / [PD 分离](pd-disaggregation.md) | 主线 A 第 5 步 · C 阶段 | 🚧 草稿 |
| [投机解码 speculative decoding](speculative-decoding.md) | 主线 A 第 4 步 | 🚧 草稿 |
| RadixAttention | C2/C3 | 🚧 [速览](remaining-theory-primer.md) |

## 训练 / 并行
| 主题 | 有论文? | 挂靠 | 状态 |
|------|:--:|------|:--:|
| [优化器：Adam / AdamW 与显存账](optimizers-adam.md) | 论文散（Adam/AdamW/ZeRO） | 🪵 枝干 A1 第 1 段（主线 A 第 7 步后；Muon 见第 3 步） | 🚧 草稿 2026-08-13 |
| ZeRO / FSDP | ✔ [已有](../../papers/training/zero-paper.md) | 🪵 枝干 A1 第 2 段 | 🚧 [速览](remaining-theory-primer.md) |
| TP / PP / EP 通信 | ✔ Megatron | 🪵 枝干 A1 第 3 段（概念） | 🚧 [速览](remaining-theory-primer.md) |

> 看到新东西随时加一行，但新概念必须先挂到某条主线上，不挂不学。

## 笔记索引（按掌握度更新）

- ✅ 用户已掌握：**[Online Softmax](online-softmax.md)**、**[Parallel Reduce](parallel-reduce.md)**、**FA1 机制（2026-08-10 A5）**
- 🚧 Agent 草稿，待消化：**[Flash Attention 机制](flash-attention-mechanism.md)**、**[FA2](flash-attention-2.md)**、**[FA4/FlexAttention](fa4-flexattention.md)**、**[GDN（Qwen3.5）](gdn-linear-attention.md)**、**[DSA](dsa-sparse-attention.md)**、**[DeepSeek-V4（CSA+HCA）](deepseek-v4.md)**、**[SageAttention3/Kascade](attention-2026-sage3-kascade.md)**、**[优化器 Adam/AdamW](optimizers-adam.md)**、**[INT8 / FP8 量化基础](quantization-int8-fp8.md)**、**[MoE 推理挑战](moe-inference.md)**、**[Speculative Decoding](speculative-decoding.md)**、**[PD 分离](pd-disaggregation.md)**、**[MLA（DeepSeek）](mla-deepseek.md)**、**[最新模型与结构](latest-model-architectures.md)**、**[剩余理论主题速览](remaining-theory-primer.md)**、**[模型追踪表](model-tracker.md)**