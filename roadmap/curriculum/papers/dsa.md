# DSA：索引选择与稀疏 Attention

长上下文中，除了保存 KV，还要决定一个 Query 读取多少历史位置。DSA（DeepSeek Sparse Attention）引入轻量 indexer，先选择位置，再对选中的 KV 条目执行主要 Attention。它与 MLA 的目标不同：MLA 改变缓存表示，DSA 改变本次主要计算访问的集合；在 V3.2 的实现里，选择对象是 MLA 的 latent/KV entry，而不是把完整 K/V 重新展开后再做一次无条件扫描。

## 1. 索引器选择的是位置

设历史长度为 T，indexer 为当前 Query 产生一组相关性分数 u_t。选中集合为

$$
\mathcal I=\operatorname{TopK}(u,k).
$$

主要 Attention 不直接把 u 当作最终权重，而是对选中的 K 重新计算自己的 score：

$$
s_t=\frac{q^\top k_t}{\sqrt d},\qquad
O=\frac{\sum_{t\in\mathcal I}\bigl[e^{s_t}v_t\bigr]}
{\sum_{t\in\mathcal I}e^{s_t}}.
$$

这里有两种分数：indexer 的 u 用于筛选，主 Attention 的 s 用于加权。把两者混为一谈，会误读训练目标、cache 和 kernel 数据流。

选中索引不一定每个 Attention head 都独立生成。DeepSeek-V3.2-Exp 的公开推理实现把所选位置的 mask 广播到主 Attention head 维；不能概括为“每头各选一组”。是否跨层复用又是另一个设计维度。

## 2. 代码中的 gather 与重新归一化

下面给定已经选好的位置，只演示主 Attention，不实现训练过的 indexer：

<!-- source-check: examples/attention_checks.py -->
~~~python
def selected_attention(scores, values, selected):
    ids=np.asarray(selected,dtype=np.int64)
    if ids.ndim!=1 or len(ids)==0 or len(set(ids.tolist()))!=len(ids):
        raise ValueError("nonempty unique selection required")
    if np.any(ids<0) or np.any(ids>=len(scores)):
        raise ValueError("selection outside cache")
    return softmax(scores[ids])@values[ids]
~~~

Softmax 的分母也变成所选集合的指数和。不能先算全部历史的概率，再直接删掉未选位置而不重新归一化。

例如 scores=[0,1,2]、V=[0,1,9]。只选择后两个位置，会得到不同于完整三个位置的结果。模型任务质量可能接近原方案，但这不等于对任意输入与 dense Attention 数学等价。

## 3. 复杂度要把 indexer 算进去

对单个 Query，主要 Attention 从访问 T 个位置缩为 k 个，若 head 维度固定，主分支的点积与加权和可由 O(Td) 降为 O(kd)。但 indexer 仍可能扫描整个可见历史，另有 Top-K、索引读取和 gather 成本；“主分支变稀疏”不等于整条路径只剩 k 次读写。

对整段 Prefill，轻量 indexer 的全范围评分也可能保留二次项。因此不能把整个 DSA 路径无条件写成 O(Tk)，更不能把 T/k 当作端到端加速倍数。应分别计数 indexer、选择、主 Attention、cache 读写和布局转换。

## 4. 没选中的 KV 为什么通常还要保存

当前 Query 没选中某个位置，不代表后续 Query 永远不需要它。动态选择通常仍需保留可被未来访问的历史状态，不能在一次 Top-K 后随意删除未选项。

MLA 让这些历史状态以较小 latent 存储；DSA 让主 Attention 少读取一部分。量化、层间复用、滑动窗口和持久缓存管理可以进一步改变容量或访问，但它们各自有额外假设。

## 5. 阅读代码时核对四项

- indexer 的输入、打分维度、cache 和训练目标是什么；
- 返回的索引形状怎样广播到主 Attention；
- 选中的 K/V 如何寻址，是否先完整解压或物化；
- 稀疏选择后使用什么分母，尾部、causal 和重复索引怎样处理。

## 6. 求职时应能现场核对的三个问题

1. 如果 `selected` 恰好覆盖全部历史，为什么重新计算的 Softmax 应与 dense Attention 一致？如果只保留一部分位置，为什么必须在所选集合内重新归一化？
2. indexer 仍扫描全历史时，prefill 的复杂度账应怎样拆成 indexer、Top-K、gather 和主 Attention，而不是直接写成 `O(Tk)`？
3. 如果一个索引 mask 广播到多个 Attention head，哪些张量仍按 head 独立计算，哪些缓存可以共享？回答必须同时给出 shape 和物理读取边界。

数学检查同时验证“选中全部位置等于 dense”与“只选部分通常不等价”：

~~~bash
python roadmap/curriculum/papers/examples/attention_checks.py
~~~

## 参考阅读

[DeepSeek-V3.2 论文](https://arxiv.org/abs/2512.02556) · [DeepSeek-V3.2-Exp 官方实现](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) · [V3.2 inference config](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/config_671B_v3.2.json)
