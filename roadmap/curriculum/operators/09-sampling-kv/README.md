# 第九章 Sampling 与 KV 辅助算子：选择、随机性和状态更新

模型输出的 logits 是词表中各 token 的未归一化分数，经过选择或随机采样后才能得到输出 token。对包含 B 个请求、词表大小为 V 的批次，单步采样通常处理 `[B,V]`；同时验证 T 个候选位置时，可以按 `[B,T,V]` 组织对应分布。生成引擎还要检查结束条件，并更新请求、随机数与 KV 状态。这些步骤频繁执行，核函数启动、同步和数据搬运都可能影响 Decode 时间。

候选集合、采样分布和缓存提交位置分别检查：Top-K/Top-P 决定可选 token，随机采样决定如何选择，提交协议决定哪些位置已经对下一步可见。后文始终区分两个计数器：输出序列已经追加了多少 token，以及这些 token 中有多少已经完成 forward 并写入 KV。

## 1. 从 logits 到 token

设一个请求的词表 logits 为 z[V]，温度 τ>0。概率为

$$
p_i=\frac{\exp((z_i-m)/\tau)}
{\sum_{j=0}^{V-1}\exp((z_j-m)/\tau)},\qquad
m=\max_jz_j.
$$

减去最大值防止大正数取指数时溢出，不改变概率。正温度不改变 logits 的排序，但会改变概率集中程度，因此影响 Top-P 的保留集合。τ=0 通常按单独的 greedy 规则处理，不应该直接代入除法。

Greedy 只需要 argmax，不需要先算整行 Softmax。Top-K 选择 k 个最高分 token；Top-P 则保留按概率降序排列后，累计质量首次达到阈值 ρ 的最短前缀。二者不同：ρ=0.9 不代表保留词表的 90%，同一个阈值在不同分布上可能保留一个，也可能保留很多 token。

### 必须包含跨过阈值的那个 token

若排序后的概率为 [.5,.3,.2]，ρ=.7，应保留前两个 token，因为 .5 不够，.5+.3 才达到阈值。保留后的概率要重新归一化，变成 [.625,.375]。

$$
J=\min\left\{j:\sum_{u=0}^{j}p_{\pi_u}\geq\rho\right\},
\qquad
\widetilde p_{\pi_u}=
\frac{p_{\pi_u}}{\sum_{v=0}^{J}p_{\pi_v}}\quad(0\leq u\leq J).
$$

如果只保留“累计和小于阈值”的元素，就会错误丢掉跨阈值元素。ρ=1 应保留完整支持集合，不能因为累计浮点误差提前少留一项。本章规定有限 logits、0<ρ≤1；全 mask 或没有可用 token 时，应按服务约定报错或处理 EOS，不能让 NaN 概率进入采样。

<!-- source-check: examples/semantics.py -->
~~~python
def nucleus(logits, threshold, temperature=1.0):
    if not logits or not 0 < threshold <= 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("nonempty logits, 0 < p <= 1, positive temperature required")
    if not all(math.isfinite(x) for x in logits):
        raise ValueError("finite logits required")
    ids = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
    maximum = logits[ids[0]]
    mass = [math.exp((logits[i]-maximum)/temperature) for i in ids]
    total = sum(mass)
    selected, weights, cumulative = [], [], 0.0
    for token, value in zip(ids, mass):
        selected.append(token)
        weights.append(value)
        cumulative += value / total
        if threshold < 1 and cumulative >= threshold:
            break
    normalizer = sum(weights)
    return selected, [x/normalizer for x in weights]
~~~

该参考用固定的 token id 平局规则，方便对照；它是 CPU 语义实现，不是高性能 GPU 排序。mask、温度、重复惩罚和过滤的顺序都属于接口合同，交换这些步骤可能改变结果。

## 2. GPU 选择：局部候选怎样合并

### argmax 也要规定平局

把一行拆成多个 chunk，每个 chunk 保存最大值和原始 token id。最终比较这些二元组：分数更大优先，分数相同时 token id 更小优先。这个合并规则与分块方式无关；不能只保存分数，最后再猜它来自哪个位置。

下面的两阶段实现让多个 program 扫描同一词表，再归约 partial。它接受有限 FP32 输入，输出 INT64 token id；基础包装限定词表不超过 2²⁰。

<!-- source-check: examples/sampling_gpu.py -->
~~~python
@triton.jit
def argmax_partial(X, Values, Ids, V: tl.constexpr, CHUNKS: tl.constexpr,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    token = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * V + token, mask=token < V, other=-float("inf"))
    best = tl.max(x, axis=0)
    chosen = tl.min(tl.where((token < V) & (x == best), token, 2147483647), axis=0)
    tl.store(Values + row * CHUNKS + chunk, best)
    tl.store(Ids + row * CHUNKS + chunk, chosen)
~~~

<!-- source-check: examples/sampling_gpu.py -->
~~~python
@triton.jit
def argmax_finish(Values, Ids, Out, CHUNKS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK)
    value = tl.load(Values + row * CHUNKS + c, mask=c < CHUNKS, other=-float("inf"))
    token = tl.load(Ids + row * CHUNKS + c, mask=c < CHUNKS, other=2147483647)
    best = tl.max(value, axis=0)
    chosen = tl.min(tl.where((c < CHUNKS) & (value == best), token, 2147483647), axis=0)
    tl.store(Out + row, chosen)
~~~

第一次的 grid 是 [B,ceil(V/BLOCK)]，第二次每个请求一个 program。padding 的值为 -inf、编号为最大整数，因此不会胜过有限有效值。两次 launch 都在同一个 CUDA stream，partial 写入先于最终读取；跨 stream 使用必须显式建立依赖。

多阶段增加了并行度，也增加 partial 存储与一次 launch。短行可能一阶段更快，长行或小 batch 才可能从拆分中受益。不要把“两个 kernel”或“更多 CTA”单独当作优化结论。

### Top-K 与 Top-P 不能都退化成整行排序

求 Top-K 时，局部每段保留 k 个候选再全局合并是一个正确起点：若某值在自己的段内已被 k 个值超过，它不可能进入全局 Top-K。平局顺序必须在局部与全局一致。候选量约为 chunk 数乘 k，k 很大时这个临时集合也会很大。

Top-P 除了候选排序，还需要累计概率质量。固定保留一个很小的 Top-K 再做 Top-P，只有在能够确认这些候选已经覆盖阈值质量时才正确；否则会改变目标分布。全量排序易于验证但代价较高，阈值搜索或分层选择可以降低部分排序成本，却通常需要额外扫描、归约与边界处理。

## 3. 随机性：同分布不等于同一个 token

累积分布采样先生成 u∈[0,1)，再找累计概率第一次严格大于 u 的位置。对 [.625,.375]，u=.624 选第一个，u=.625 选第二个。实现中的 <、≤ 和随机数端点范围必须一致。

<!-- source-check: examples/semantics.py -->
~~~python
def inverse_cdf(ids, weights, uniform):
    if not ids or len(ids) != len(weights) or not 0 <= uniform < 1:
        raise ValueError("valid support and uniform in [0,1) required")
    if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights),1.0):
        raise ValueError("probabilities must be normalized")
    cumulative = 0.0
    for token, weight in zip(ids, weights):
        cumulative += weight
        if uniform < cumulative:
            return token
    return next(token for token, weight in reversed(list(zip(ids, weights))) if weight > 0)
~~~

最后的回退只处理舍入造成的尾部差异，不允许把零概率 token 当作正常兜底。验证时先给定同一个 u 检查映射，再检查随机数生成器；否则一次 token 不同，无法判断问题来自排序、概率还是 RNG。

RNG（Random Number Generator，随机数生成器）的 seed 只是初始条件。算法、counter 映射、每次消费多少随机数、采样方法和执行顺序不同，都可能在同 seed 下产生不同 token。CPU 的 inverse-CDF 与 GPU 的 multinomial 可以符合相同分布，而不逐次返回相同结果。

动态 batching 时，应把随机状态与稳定的请求身份绑定。请求从 batch 第 3 行移动到第 0 行，不应无意中继承另一个请求的 RNG 状态。确定性复现还要同时固定过滤、平局规则、精度和随机数消费方式。

> [!IMPORTANT] 随机算子的三层验证
> 先验证候选集合和概率，再固定随机数验证 token 映射，最后检查统计分布与复现合同。仅比较几个随机 token，既不能证明正确，也不能定位错误。

> **题面卡：在进入 KV 状态前先写采样子算子**
>
> [Top K Selection #29](https://leetgpu.com/challenges/top-k-selection) 只输出降序 value，不是 token id；[Top-p Sampling #60](https://leetgpu.com/challenges/top-p-sampling) 验证 `logits[V] + p + seed → sampled_token`，平台 reference 使用固定 seed 与 `torch.multinomial`。两题都不覆盖本章 RNG counter 绑定、动态 batching、KV append 或 EOS 状态；从空题面写完后，继续下面的 cache/投机语义。

## 4. KV 辅助操作：写位置和提交状态

采样得到 token 只说明逻辑输出序列的下一项，不等于对应的 K/V 已经存在。进入本节后，先沿 `request_id → logical_position → page/slot` 的地址链确认写入位置，再讨论 batch 重排、共享前缀和投机候选的提交状态。

### append 与 batch 重排

新 K/V 必须写到请求的逻辑位置，而不是当前 batch 行号对应的任意连续槽位。分页缓存需要 page table 或 slot mapping；行压缩、请求退出和 batch 重排后，位置、长度、请求 id 与 RNG 状态必须一起更新。

一个写入任务可以表示为 (request_id,logical_position,kv_head,channel)。映射到物理槽位后，应检查有效范围和写所有权。两个有效写任务意外落在同一槽位时，普通 store 不会自动帮你决定哪个结果正确；原子操作也不能修复错误的 ownership。

共享前缀页是只读共享状态，追加前需要按分配器协议保证私有页；写完成与下一次 Attention 读取之间需要 stream 顺序或 event。EOS（End of Sequence，序列结束符）出现后，哪些 token 计入输出、哪些 KV 继续保留，也必须按服务协议处理，不能只修改一个 length 数字。

### 投机解码需要“暂存”与“提交”

定义已经稳定的 KV 长度为 L，又暂存了 k 个候选 token 对应的 KV。若只接受前 a 个候选，则稳定长度推进为 L+a；其余物理数据可以暂时留在 allocation 中，但后续 mask 和长度不能把它们作为有效上下文读取。

<!-- source-check: examples/semantics.py -->
~~~python
def commit_speculation(cache, prefix_length, draft_count, accepted):
    if not 0 <= accepted <= draft_count or not 0 <= prefix_length <= len(cache)-draft_count:
        raise ValueError("invalid speculative cache span")
    # Logical truncation: rejected values may remain physically allocated.
    return cache[:prefix_length+accepted]
~~~

这里用切片表示逻辑提交。真实实现通常更新元数据并回收不再需要的页，不必为“删除候选”把所有内存清零；关键是无效尾部不可见。拒绝后新采样的 replacement token 与被拒绝的候选不是同一个 token，不能沿用后者的 KV。新 token 的 KV 仍需按实际模型执行路径生成。

用一个带 shape 的小流程固定这条边界：`B=1,V=4,T=3`，稳定 KV 长度 `L=5`，draft token 为 `[1,2,0]`，`draft_probs/target_probs` 的形状都是 `[1,3,4]`，`uniform_samples` 是 `[1,4]`。若位置 0 接受、位置 1 拒绝并从 residual 分布采得 replacement token 3，逻辑输出序列的待执行后缀是 `[1,3]`，位置 2 的 draft token 不再提交；这两个 token 的状态要分阶段记录：

1. **只提交已验证的 draft KV**：`draft_0=1` 位于逻辑位置 5，page size `P_T=4` 时物理页/偏移为 `(1,1)`；KV 有效长度从 5 变为 6。replacement `3` 只是下一步要执行的 token，虽然输出前缀已经知道 `[1,3]`，它的 KV 尚未物化，不能让 Attention 读取位置 6。
2. **执行 replacement 的 target forward 并 append**：replacement 位于位置 6，地址为 `(1,2)`；只有写入完成且 metadata 更新后，KV 有效长度才从 6 变为 7。位置 7 的 draft KV 仍无效。若三个 draft 候选全接受，长度先推进到 8，再用 `uniform_samples[0,3]` 从最后一个 target 分布采 bonus token；bonus token 也要经过自己的 model forward/append，不能当作已经存在的 draft KV。

这个实例把 `[B,T,V]` 的概率验证、连续前缀规则、预测 logits 的一位 shift 和 `[page,offset]` 的两阶段提交放在同一条数据流上；输出 token 数与已物化 KV 数是两个不同的计数器。

## 5. 投机验证为何能保持目标分布

投机验证需要同时保持输出分布与缓存状态正确。草稿模型（draft）提出候选，目标模型（target）验证后接受其中一个连续前缀；两个模型分别保留与已提交上下文相符的缓存，不能互相替代。拒绝后采样出的纠正 token 在后续作为输入时，才计算并追加自己的 KV。下面先说明接受概率，再讨论与这一状态更新过程对应的性能条件。

设目标分布为 p，草稿分布为 q，草稿采样得到 token x。经典随机投机验证以

$$
\alpha(x)=\min\left(1,\frac{p(x)}{q(x)}\right)
$$

接受它。q(x)=0 的 token 不会由 q 采到，所以不能在实现中无条件对所有位置做这个除法。发生拒绝时，replacement 从剩余分布中采样：

$$
r(x)=\frac{[p(x)-q(x)]_+}{\sum_y[p(y)-q(y)]_+}.
$$

为什么不是拒绝后简单再从 p 采一次？草稿接受路径已经贡献了 min(p(x),q(x)) 的概率质量。如果再直接加上完整 p，就会重复分配。令 A=Σ_x min(p(x),q(x))，则

$$
\min(p(x),q(x))+(1-A)r(x)=p(x).
$$

例如 p=[.6,.3,.1]、q=[.2,.5,.3]，接受质量为 [.2,.3,.1]，总接受率 .6；拒绝时的剩余分布为 [1,0,0]，补上 .4 后恰好恢复 p。若 p=q，拒绝概率为零，不应继续计算一个 0/0 的剩余分布。

多 token 验证必须沿同一已接受前缀解释这些条件概率，只接受连续前缀，不能跳过拒绝位置后继续使用后续候选的 KV。温度与过滤后的目标分布也要一致；这套分布保持推导不自动覆盖近似量化、不同模型状态或任意 greedy 验证规则。

### 接受率高仍不一定快

若第 j 个位置在前面都接受的条件下，接受概率为 α_j，则前缀长度的期望为

$$
\mathbb E[A_{\mathrm{prefix}}]
=\sum_{j=1}^{k}\prod_{u=1}^{j}\alpha_u.
$$

忽略 EOS 和长度上限，带一个 replacement/bonus token 的常见算法每轮约提交 1+E[A_prefix] 个 token。速度还要除以草稿、目标验证、调度和缓存管理的时间：

$$
\mathrm{speedup}\approx
\frac{(1+\mathbb E[A_{\mathrm{prefix}}])t_{\mathrm{decode}}}
{t_{\mathrm{draft}}(k)+t_{\mathrm{verify}}(k)+t_{\mathrm{overhead}}}.
$$

增加 k 会增加可能接受的前缀，也会增加验证和暂存开销。高负载下验证还可能挤占其他请求的资源。DeepSeek-V4.1-Flash 的 DSpark 将前缀接受概率估计与引擎吞吐曲线结合选择验证长度，体现的就是这个权衡；它不是“接受率高就一直增加长度”。

### draft mask、target verification 与 MTP 的因果边界

投机解码中的两次前向有不同的 mask 合同。草稿模型生成第 `j` 个候选时，只能看到已经稳定提交的前缀和它之前已经生成的草稿 token；它不能看到同一轮中尚未生成的候选。目标模型验证长度为 `k` 的候选时，可以把“稳定前缀 + `k` 个候选”作为一个批量序列一次计算，但第 `j` 行的 causal mask 仍只允许它看稳定前缀和候选 `0..j`，不允许读取候选 `j+1..k-1`。并行验证是把多行放进一次 target forward，不是取消因果约束。

因此，若把 `draft_j` 作为拼接输入中的位置 `L+j`，目标 self-attention 的 row `L+j` 确实可以看到稳定前缀和 `draft_0..draft_j`；但这个 row 产生的是“下一个位置”的 logits，不是 `draft_j` 的预测 logits。要验证 `draft_j`，应取前一行的输出（`j=0` 时取稳定前缀最后一行；具体 offset 由接口 layout 定义），再与 `draft_j` 对齐。对应的条件可写成

$$
\mathcal A_j=\{0,\ldots,L-1\}\cup\{\text{draft}_0,\ldots,\text{draft}_{j-1}\}.
$$

具体实现把候选 token 放在输入的下一个位置，还是把对应 logits 移位到 `j`，属于张量布局合同；无论采用哪种排布，验证比较的 `p_j` 必须来自上式的条件分布，而不能误用“已经看到 `draft_j` 的 row”所产生的下一 token logits。接受第 `a` 个候选后，`draft_{a+1}` 及其后面的 KV 即使已经物理写入，也不进入下一步的有效长度；一旦在位置 `a` 拒绝，后续候选不能跳过拒绝点继续提交。树状投机或 MTP 的分支 mask 可以不是简单的下三角，但每个节点仍只能依赖其祖先和稳定前缀。

MTP（Multi-Token Prediction）和自投机也不能无条件写成“每次接收 `k` 个 token，所以加速约为 `k`”。MTP 可能复用主模型 hidden state 后由多个预测头给出候选；自投机可能使用早退层或同一模型的轻量路径；它们的 `t_draft(k)`、额外显存、校验 batch、调度和 KV 暂存协议都不同。只有在相同请求 batch、相同采样/过滤合同、target 能以一次保持因果 mask 的验证前向处理候选、并且把 draft、verify、commit、调度和同步全部计入时，下面的比值才可作为同范围估计：

$$
R(k)=
\frac{\bigl(1+\mathbb E[A_{\mathrm{prefix}}]\bigr)t_{\mathrm{base}}}
{t_{\mathrm{draft}}(k)+t_{\mathrm{verify}}(k)+t_{\mathrm{commit}}+t_{\mathrm{schedule}}}.
$$

例如 `k=3`、条件接受率为 `[0.8,0.7,0.6]` 时，
`E[A_prefix]=0.8+0.8×0.7+0.8×0.7×0.6=1.696`，期望每轮提交 `2.696` 个 token。若单 token 基线为 `1.0 ms`，而 draft、target verification、commit 和调度合计 `2.3 ms`，同一口径下的估计只有 `2.696/2.3≈1.17×`，不是 `2.696×`。这还假定接受率来自同一 batch/长度分布；prefix cache 命中、KV page 复用、EOS、请求退出和动态 batching 都会改变 `α_j` 与每轮有效工作量。若 draft 成本随 `k` 增大，或 target 验证使其他请求排队，端到端吞吐可能没有提升，甚至下降。

> **题面卡：现在写 Speculative Verification**
>
> [Speculative Decoding Verification #87](https://leetgpu.com/challenges/speculative-decoding-verification) 只接收 `draft_tokens`、`draft_probs`、`target_probs`、`uniform_samples`，验证 acceptance/residual/bonus token；它不生成 draft/target forward，不验证 causal mask 或 KV commit。完成空题面并归档后，下面的 CPU/服务器实验继续验证两次 forward 的 row shift、连续前缀和两阶段 KV 提交。

## 6. 实践：选择、采样和缓存分别验证

### LeetGPU：正确性与代码归档

平台原始提交按各自题面保存；本地 `semantics.py` 验证跨阈值 token、p=1、平局、CDF 边界、投机概率质量恢复及逻辑 KV 回退：

~~~bash
python roadmap/curriculum/operators/09-sampling-kv/examples/semantics.py
~~~

### 服务器：真实性能

先比较确定性的两阶段 argmax 与 PyTorch argmax：

~~~bash
python roadmap/curriculum/operators/validate_baselines.py sampling
python roadmap/curriculum/operators/validate_baselines.py sampling --benchmark
~~~

检查包含长度 1、非整除行、跨 1024 元素 chunk 的平局和长行。脚本报告含分配与两次 launch 的包装时间，不把它称为纯归约指令时间，也不把 argmax 成绩记作 Top-P 成绩。

Top-P 的性能实验应固定词表大小、batch、概率熵、阈值和随机性合同；比较不同选择方法时记录实际保留的候选数。完整 Decode 还要检查 host 回读、EOS 判断、batch 重排、KV 更新与 launch gap。GPU kernel 少几微秒，如果随后每步都强制同步到 CPU，端到端收益可能完全被抵消。

## 参考阅读

[CUDA Programming Guide](../../../../downloads/cuda-programming-guide.pdf) · [Speculative Decoding](https://arxiv.org/abs/2211.17192) · [DeepSeek-V4.1-Flash 技术报告](../../../../downloads/DeepSeek_V41_Tech_Report.pdf)

## 章节导航

- [上一章：MoE 算子](../08-moe/README.md)
- [下一章：模型 GPU 执行分析](../../model-analysis/README.md)
- [算子总览](../README.md)
