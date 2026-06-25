# Algorithms — 理论线

> 每周一条算法/理论。不只是写代码——补原理、量化、新架构、系统设计。
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

## 主题池（6 子类，跟业界动态走）

### GPU 优化算法
| 主题 | 状态 |
|------|:--:|
| [online softmax](online-softmax.md)（Flash 的心脏） | ✅ |
| [parallel reduce / prefix sum](parallel-reduce.md) | ✅ |
| Norm 的 reduce 模式（LayerNorm/RMSNorm，料→[reference](../../reference/cuda/layernorm/layernorm.cu)） | ⏳ |
| work partitioning（Flash 2 的思路） | ⏳ |

### 量化
| 主题 | 有论文? | 状态 |
|------|:--:|:--:|
| [数值格式 INT8 / FP8](quantization-int8-fp8.md) | — | ✅ |
| AWQ | ✔ 两边写 | ⏳ |
| GPTQ | ✔ 两边写 | ⏳ |
| SmoothQuant / KV Cache 量化 | 部分 | ⏳ |

### 注意力演进
| 主题 | 有论文? | 状态 |
|------|:--:|:--:|
| MHA→MQA→GQA→MLA | ✔ GQA | ⏳ |
| [Flash Attention 机制](flash-attention-mechanism.md) | ✔ [已有](../../papers/attention/flash-attention.md) | ✅ |
| 线性注意力 / Ring Attention | ✔ | ⏳ |

### 模型架构
| 主题 | 状态 |
|------|:--:|
| MoE（原理 + 推理挑战） | ⏳ |
| Mamba / SSM | ⏳ |

### 推理系统技术
| 主题 | 状态 |
|------|:--:|
| continuous batching | ⏳ |
| chunked prefill / PD 分离 | ⏳ |
| 投机解码 speculative decoding | ⏳ |
| RadixAttention | ⏳ |

### 训练 / 并行
| 主题 | 有论文? | 状态 |
|------|:--:|:--:|
| ZeRO / FSDP | ✔ [已有](../../papers/training/zero-paper.md) | ⏳ |
| TP / PP / EP 通信 | ✔ Megatron | ⏳ |

> 看到新东西随时加一行。

## 已学笔记（按学习顺序）

1. **[Online Softmax](online-softmax.md)** — 单趟增量 softmax，Flash Attn 的心脏（给 A4/A5 铺路）
2. **[Parallel Reduce](parallel-reduce.md)** — GPU 并行归约模式，树状 reduce + warp shuffle（A4 Softmax 要用）
3. **[Flash Attention 机制](flash-attention-mechanism.md)** — IO-aware tiling + online softmax，A5 读代码前必看
4. **[INT8 / FP8 量化基础](quantization-int8-fp8.md)** — 数值格式对比、对称/非对称量化、性能数据（C 线推理系统铺路）
