# Algorithms — 理论线

> 每周一条算法/理论。不只是写代码——补原理、量化、新架构、系统设计。
> 状态约定：`🚧 草稿/速览` 表示 Agent 已生成、用户还没消化，不计入“已学”。
> 当前学哪条由 [NOW.md](../../NOW.md) 指定，全貌和进度在 [PATH.md](../../PATH.md) 理论线。

## 为什么有这条线

算子优化（算子线）是手上功夫，但面试和实战还要"知道业界在用什么、为什么"。这条线补**广度和前沿**：量化怎么做、新注意力变体、推理系统的新技巧。每条学完写一页，积累成自己的"业界地图"。

## 笔记规则

一条一页，固定四段：**解决什么问题 → 核心思路 → 关键数据/取舍 → 与我何干**。

## 📌 和 papers/ 的边界（硬规矩）

俩都是"读+理论"，容易混。规矩：

> **单位是"一篇论文" → [papers/](../../papers/)；单位是"一个技术/概念" → 这里。**

- **非论文技术**（online softmax、parallel reduce、continuous batching）→ 只在这里。散在多篇论文/博客/代码里的，本就没有单篇归属。
- **有标志性论文的技术**（AWQ、GQA、ZeRO）→ 两边各一份，角度不同：
  - `papers/xxx.md`：这篇论文讲了什么（精读，带 arxiv）
  - 这里 `xxx.md`：机制 + 我会怎么实现/用，`[[链]]`到论文
- 判断口诀：**"读论文" ≠ "会实现"**。前者进 papers，后者进这里。

---

## 怎么用（配合 NOW / PATH）

- [NOW.md](../../NOW.md) 决定当前学哪条；[PATH.md](../../PATH.md) 是唯一进度源。
- 本目录按 PATH 理论线 6 子类组织，新增主题先进 README，再写独立笔记。
- `🚧 草稿` 不更新 PATH 状态；只有你能讲清并完成最小验证后，才由教练把对应项标为 ✅。
- 不在这里另建“新路线”，所有内容都挂回 PATH 的算子线/理论线。
- 大模型相关内容的聚合板块见 [notes/llm/README.md](../llm/README.md)，学习过程仍由 PATH 执行参考推进。

## 主题池（6 子类，跟业界动态走）

### GPU 优化算法
| 主题 | 状态 |
|------|:--:|
| [online softmax](online-softmax.md)（Flash 的心脏） | ✅ |
| [parallel reduce / prefix sum](parallel-reduce.md) | ✅ |
| Norm 的 reduce 模式（LayerNorm/RMSNorm，[速览](remaining-theory-primer.md)，料→[reference](../../reference/cuda/layernorm/layernorm.cu)） | 🚧 速览 |
| work partitioning（Flash 2 的思路） | 🚧 [速览](remaining-theory-primer.md) |

### 量化
| 主题 | 有论文? | 状态 |
|------|:--:|:--:|
| [数值格式 INT8 / FP8](quantization-int8-fp8.md) | — | 🚧 草稿 |
| AWQ | ✔ 两边写 | 🚧 [速览](remaining-theory-primer.md) |
| GPTQ | ✔ 两边写 | 🚧 [速览](remaining-theory-primer.md) |
| SmoothQuant / KV Cache 量化 | 部分 | 🚧 [速览](remaining-theory-primer.md) |

### 注意力演进
| 主题 | 有论文? | 状态 |
|------|:--:|:--:|
| MHA→MQA→GQA→MLA | ✔ [GQA](../../papers/attention/gqa.md) | 🚧 [速览](remaining-theory-primer.md) |
| [Flash Attention 机制](flash-attention-mechanism.md) | ✔ [FA1](../../papers/attention/flash-attention.md) · [FA2](../../papers/attention/flash-attention-2.md) | ✅ FA1 已消化（2026-08-10 A5）；FA2/3 待补 |
| [MLA（DeepSeek-V2/V3）](mla-deepseek.md) | ✔ DeepSeek-V2 | 🚧 草稿 |
| 线性注意力 / Ring Attention | ✔ | 🚧 [速览](remaining-theory-primer.md) |

### 模型架构
| 主题 | 状态 |
|------|:--:|
| [最新模型架构地图](latest-model-architectures.md)（LLaMA/Qwen/DeepSeek/GPT/Claude/Gemini/MoE/SSM） | 🚧 草稿 |
| [模型追踪表](model-tracker.md) | 🚧 草稿 |
| [MoE 推理挑战](moe-inference.md) | 🚧 草稿 |
| Mamba / SSM | 🚧 [最新模型与结构](latest-model-architectures.md) |

### 推理系统技术
| 主题 | 状态 |
|------|:--:|
| continuous batching | 🚧 [速览](remaining-theory-primer.md) |
| chunked prefill / [PD 分离](pd-disaggregation.md) | 🚧 草稿 |
| [投机解码 speculative decoding](speculative-decoding.md) | 🚧 草稿 |
| RadixAttention | 🚧 [速览](remaining-theory-primer.md) |

### 训练 / 并行
| 主题 | 有论文? | 状态 |
|------|:--:|:--:|
| ZeRO / FSDP | ✔ [已有](../../papers/training/zero-paper.md) | 🚧 [速览](remaining-theory-primer.md) |
| TP / PP / EP 通信 | ✔ Megatron | 🚧 [速览](remaining-theory-primer.md) |

> 看到新东西随时加一行。

## 笔记索引（按掌握度更新）

- ✅ 用户已掌握：**[Online Softmax](online-softmax.md)**、**[Parallel Reduce](parallel-reduce.md)**
- 🚧 Agent 草稿，待消化：**[Flash Attention 机制](flash-attention-mechanism.md)**、**[INT8 / FP8 量化基础](quantization-int8-fp8.md)**、**[MoE 推理挑战](moe-inference.md)**、**[Speculative Decoding](speculative-decoding.md)**、**[PD 分离](pd-disaggregation.md)**、**[MLA（DeepSeek）](mla-deepseek.md)**、**[最新模型与结构](latest-model-architectures.md)**、**[剩余理论主题速览](remaining-theory-primer.md)**、**[模型追踪表](model-tracker.md)**
