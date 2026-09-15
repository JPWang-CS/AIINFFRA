# 第六章 Decode / PagedAttention：从单 Query 数学到分页 KV 访问

Decode 每步处理新 Query，各请求的历史长度却不同。页表把逻辑序列映射到物理 KV 页，多组 Query head 还可能共享同一 KV head。

分页改变地址组织，Attention 的加权和仍按逻辑位置计算。实现需要把页内偏移、有效长度、在线归约以及缓存引用的生命周期对应起来。

## 单 Query 的 shape 合同与 GQA

设 batch 中第 b 个请求本步送入一个 token，计算它的 Q/K/V，再根据输出分布采样后续 token。本接口省略长度为 1 的 Query 序列轴，使用下面的三维 Q；写成 [B,Hq,1,D] 是另一种等价的接口约定，并不改变算法。

$$
Q\in\mathbb{R}^{B\times H_q\times D},\qquad
K_{\mathrm{cache}},V_{\mathrm{cache}}\in\mathbb{R}^{P\times T\times H_{kv}\times D}.
$$

P 是物理 page 数，T 是每页容纳的 token 数，Hkv 是 KV head 数。lengths[b] 给出请求 b 的逻辑长度，table[b,p] 把它的逻辑 page p 映射到物理 page。两张 cache pool 分开存放：教学 wrapper 使用独立的 cache_k 和 cache_v，避免把“一个 slot 内有 K/V”与真实的两个连续 storage contract 混在一起。table 和 lengths 是 metadata，不是 attention 数据；它们同样影响地址安全和缓存生命周期。

要求 Hq % Hkv == 0。每个 Query head 对应的 KV head 是

$$
g=\frac{H_q}{H_{kv}},\qquad
kv\_head(h_q)=\left\lfloor\frac{h_q}{g}\right\rfloor.
$$

Multi-Head Attention（MHA，多头注意力）是 Hq=Hkv；Multi-Query Attention（MQA，多查询注意力）是 Hkv=1；Grouped-Query Attention（GQA，分组查询注意力）位于两者之间。GQA 只说明多个 Query head 复用同一份 K/V；它不把一次 Decode 要扫描的逻辑 token 数从 lengths[b] 变成更短的数，也不改变每个 head 的 D。它减少的是 KV head 数及其存储需求，实际 HBM 流量还取决于 kernel 是否把同一份 K/V 在 program 内复用。

本章 baseline 的 dtype 是 FP32，D 取 {16,32,64,128}，Q/K/V 和输出都 contiguous、在同一个 CUDA device。lengths[b] 必须为正数；这个 baseline 拒绝 zero-length 请求，而不是偷偷把全 mask 输出定义为零。CPU split 模型仍测试空 split，因为 split-KV 调度可能产生空分段，空分段的 (m,l,U) 必须有稳定单位元。

## 逻辑位置到物理页：分页只改变地址路径

逻辑 token t 的地址分解为

$$
\ell=\left\lfloor\frac{t}{T}\right\rfloor,\qquad
\delta=t\bmod T,\qquad
p_{\mathrm{phys}}=table[b,\ell].
$$

然后访问

$$
K_{\mathrm{cache}}[p_{\mathrm{phys}},\delta,kv\_head,:],\quad
V_{\mathrm{cache}}[p_{\mathrm{phys}},\delta,kv\_head,:].
$$

举一个故意非连续的例子：T=4，请求 b=0 的 table[0]=[7,2]，逻辑长度为 7。逻辑 token 0,1,2,3 访问物理 page 7 的 offset 0,1,2,3；token 4,5,6 访问物理 page 2 的 offset 0,1,2。物理地址顺序是 7,7,7,7,2,2,2，不能把物理 page 编号当作逻辑顺序，也不能通过 cache[0:length] 这样的连续切片替代 page table。

分页没有改变 attention 的有效集合。下面 K[b,t,…]、V[b,t,…] 表示经页表解释后的逻辑张量，不要求实际物化成连续数组。对 Query head hq，它仍然计算

$$
s_t=\frac{Q[b,h_q,:]\cdot K[b,t,kv\_head(h_q),:]}{\sqrt D},\qquad
O[b,h_q,:]=\frac{\sum_{t=0}^{L_b-1}\bigl[e^{s_t}V[b,t,kv\_head(h_q),:]\bigr]}
{\sum_{t=0}^{L_b-1}e^{s_t}}.
$$

因此普通分页通常不会减少有效 KV 读取数量；它解决的是变长请求的物理分配、碎片和调度问题。若想减少有效读取，需要滑窗、稀疏选择、结构化压缩或其它改变可见集合的机制，不能把 page table 本身写成稀疏 attention。

> [!IMPORTANT] 三笔账分开算
> **分页管理物理地址；GQA 减少 KV head 数；稀疏选择减少本次参与 Attention 的位置。** 容量节省、算法读取量和 profiler 的 DRAM bytes 不是同一个数字。

页表的实际读取必须带双重边界：逻辑 token 先检查 t < lengths[b]，再检查逻辑 page 在 max_pages 内，加载出的物理 page 还要检查 0 <= physical < P。实现可以先构造带有候选 offset 的向量指针，只要后续 masked load/store 严格阻止无效 lane 解引用；不能在无效 logical page 上无 mask 加载 physical page ID，也不能对无效指针做无 mask 读写。无效槽位用 -1 是 metadata 的明确哨兵。

## 容量、padding 与有效 HBM 流量账本

若请求逻辑长度为 L，分配的 page 数是

$$
N_{\mathrm{page}}=\left\lceil\frac{L}{T}\right\rceil,\qquad
L_{\mathrm{allocated}}=T\left\lceil\frac{L}{T}\right\rceil.
$$

最后一页的 L_allocated-L 个槽位是 padding，不应进入 softmax 分母，也不应被当作有效 KV 读取。一个最小账本要分别列出：

| 项目 | 计数方式 | 不能省略的边界 |
|---|---|---|
| K/V 物理池容量 | 2*P*T*Hkv*D*b_elem；每个 page 是 2*T*Hkv*D*b_elem，每个 token slot 是 2*Hkv*D*b_elem | page 尾部 padding、空闲页、量化 scale |
| 有效计算 | sum_b lengths[b] * Hq * D 的 K/V 参与量 | GQA head 映射、mask、split 空段 |
| page metadata | B*max_pages*index_bytes | 页表、length、slot/ownership/event 元数据 |
| 调度与 workspace | split partial 的 m,l,U、输出和临时表 | chunk 数、归约 buffer、allocator 对齐 |

数据池的容量按物理池中的 page 数乘以 page bytes 估算；它不是实际有效内容大小，更不是全模型显存。HBM（High Bandwidth Memory，高带宽内存）流量还受片上缓存影响。若每个 KV 元素占 b_elem 字节，并假设这些数据必须从 HBM 读取、没有跨请求共享复用，每个 token、每个 KV head 的 K/V 各读一次，忽略 metadata 和事务浪费，则有条件的耗时下界为

$$
t\geq\frac{2H_{kv}D\left(\sum_b L_b\right)b_{\mathrm{elem}}}{BW_{\mathrm{HBM}}}.
$$

这只是上述冷数据或流式工作集假设下的 KV 带宽估算。若数据命中 L2，实测时间可以低于这个 HBM 估算，不意味着突破硬件带宽。忽略 softmax 和 scale，QK dot 与 PV dot 各约 2D FLOPs，因此有效数学工作量约为

$$
F\approx4H_qD\sum_b L_b,\qquad
AI_{\mathrm{ideal}}\approx
\frac{4H_qD\sum_b L_b}{2H_{kv}D(\sum_b L_b)b_{\mathrm{elem}}}
=\frac{2H_q}{H_{kv}b_{\mathrm{elem}}}.
$$

这个 AI 是基于理想 KV 下界的账本，不是实际 HBM arithmetic intensity；Q/O、page metadata、padding、重复读取、cache 命中和 split workspace 都会改变分母。实际 profiler 流量还会受 L2 命中、事务合并和并发影响。对一个 page，FP16/BF16 的 page bytes 是 2*T*Hkv*D*2，不能与单个 token slot bytes 混称。

GQA 的理论 KV 容量可按 Hkv/Hq 缩小，但不能承诺 HBM 一定获得相同倍数：同一 KV page 是否被多个 Query program 重用、page padding、metadata、allocator 碎片、split partial 和不同 head 的调度都会改变实际数字。容量账本必须与代码复用路径分开记录。

### 从 KV 字节数算到可服务的请求数

缓存预算应从张量形状计算，而不是先假设 KV 一定占用最多显存。对普通 MHA/GQA，若各层头数、维度和 dtype 相同、请求之间不共享前缀，L 层的有效 KV 数据为

$$
M_{\mathrm{KV}}=2L H_{kv}D\,b_{\mathrm{elem}}\sum_{b=0}^{B-1}S_b.
$$

式中的 2 是 K、V 两份，b_elem 才是每元素字节数。这个公式不适用于直接按同样方式计数 MLA 的联合 latent，也不含页尾、空闲页和 allocator 开销。实现如下：

<!-- source-check: examples/cache_accounting.py -->
~~~python
def kv_payload(layers, lengths, kv_heads, head_dim, element_bytes):
    if min(layers, kv_heads, head_dim, element_bytes) <= 0 or any(n < 0 for n in lengths):
        raise ValueError("invalid cache dimensions")
    return 2 * layers * sum(lengths) * kv_heads * head_dim * element_bytes
~~~

例如只算一层：8 个请求、每个 4096 个历史 token、8 个 KV head、D=128、BF16，数据量为 128 MiB。若每页 16 个 token，这组长度没有页尾浪费；长度改成 4097，则每个请求要再分配一页，其中仅一个槽位有效。共享前缀时，应按唯一物理页统计驻留容量，不能把每个请求的完整逻辑长度简单相加当作实际分配量。

整模型预算还需要权重、激活、workspace 和运行时余量：

$$
M_{\mathrm{device}}\geq
M_{\mathrm{weights}}+M_{\mathrm{KV}}+
M_{\mathrm{activations}}+M_{\mathrm{workspace}}+M_{\mathrm{reserve}}.
$$

只有各项预算满足，才进一步测试 TTFT/TPOT 能否达到要求。算得出能装下多少请求，并不代表这些请求同时运行时能满足时延目标。

权重量化与 KV 量化缩小的是不同组成项，整模型容量要分别缩放后相加。假设原权重 40 GiB、KV 20 GiB、其他 4 GiB；权重变成原来的 1/4，KV 变成 1/2，则总量为 10+10+4=24 GiB，不是把原来的 64 GiB 直接除以 8。实际还要加 scale、zero point、padding 以及反量化 workspace。

这也解释了量化优先级为什么随工作负载变化：短上下文可能以权重为主，长上下文或高并发可能以 KV 为主；如果已经受计算管线或 launch 限制，少存数据也未必带来同比例加速。判断依据是各项容量、读取量、转换成本和精度敏感性，而不是固定的“先量化谁”列表。

## Online Softmax 与 split-KV 合并

单 Query 逐块扫描 KV 时，不需要物化长度为 L 的 score/prob 向量。对已处理集合维护

$$
m=\max_t s_t,\qquad l=\sum_t e^{s_t-m},\qquad
U=\sum_t\bigl[e^{s_t-m}V_t\bigr].
$$

新 chunk 的局部状态为 (m_c,l_c,U_c)，与旧状态合并时

$$
m'=\max(m,m_c),\qquad
\alpha=e^{m-m'}\ (m=-\infty\text{ 时取 }0),\qquad
\beta=e^{m_c-m'}\ (m_c=-\infty\text{ 时取 }0),
$$

$$
l'=\alpha l+\beta l_c,\qquad
U'=\alpha U+\beta U_c,\qquad
O=\begin{cases}U'/l'&l'>0\\0&l'=0.\end{cases}
$$

实际代码用 safe_m=0 只处理全无效 tile 的数值路径，并将无效分数设为 -inf；它不是把全 mask 的数学分母改成 1。有效行的 l 一定大于零，空段或全 mask 才返回零向量。注意 alpha 来自 e^(s-m')=e^(s-m)e^(m-m')：旧状态所有未归一化质量都必须按 e^(m-m') 重标定。

一个小反例说明为什么不能直接平均局部输出。chunk A 只有一个 score=0、V=2 的 token，chunk B 只有一个 score=ln 3、V=10 的 token。合并时 m'=ln 3，alpha=1/3、beta=1，l'=1/3+1=4/3，U'=2/3+10=32/3，所以 O=8；直接平均两个局部输出得到 (2+10)/2=6，是错误的。CPU 检查把这个数值写成断言。

当一个长 KV 被拆成 C 个 chunk 时，可以让每个 program 计算一个 partial (m_c,l_c,U_c)，再由归约 program 合并。典型 grid 近似为

$$
grid=B\cdot H_q\cdot C,\qquad C=\left\lceil\frac{L}{L_{\mathrm{chunk}}}\right\rceil.
$$

这增加了 Cooperative Thread Array（CTA，协作线程数组）数量和并行度，但也增加 launch/调度、partial 写回和最终归约。每个 partial 至少包含一个标量 m、一个标量 l 和 D 个 FP32 的 U；如果按 [B,Hq,C,D] 保存，workspace 近似为 B*Hq*C*(D+2)*4 字节，还要加 metadata 和对齐。小 batch/短上下文可能被 launch 与归约成本主导；长上下文则可能用更多 CTA 隐藏 KV 带宽延迟，但寄存器、active warps、occupancy 和 L2 traffic 会互相约束。

L_chunk、num_warps、是否在一个 program 内复用 Q、partial 的布局是可测代码旋钮。预期观察量包括 kernel 数量、eligible warps、registers/thread、active warps、L2 read、DRAM bytes、launch gap 和归约 kernel 时间；只有 profiler 与多 shape 实测后才能写收益，不能用一个理论 occupancy 数字代替。

## 边界、append、COW 与 stream ownership

### 页尾、空段与全 mask

如果 T=32 而 L=33，第二个 tile 只有一个有效 token；其余 31 个槽位必须让 score 变成 -inf。仅把 K/V load 的 other 设为 0 不够，因为零向量仍会产生有限 score，并进入 softmax 分母。L=1、恰好整页、跨 BLOCK_T 的 33/65 都是必须的边界。split-KV 的 [0,0) 空段使用 (m=-inf,l=0,U=0)，与任意非空状态合并不改变结果；所有 chunk 都无效时输出定义为零并保持有限。

### 新 token 的边界

本章 wrapper 约定 lengths 是 append 之后的长度，本步输入 token 的逻辑位置为 pos=lengths[b]-1。new_k/new_v 与 q 必须来自这个输入位置，不是尚未采样出的下一个输出 token。因此

$$
page=\lfloor pos/T\rfloor,\qquad offset=pos\bmod T
$$

既覆盖普通页内追加，也覆盖刚好跨页的 pos=T-1/T。append kernel 先做 page table 与 physical page 合法性检查，再写 K/V；attention kernel 在同一 current CUDA stream 排队，所以同一 stream 内写入先于读取。跨 stream 使用时必须由调用者用 event/同步建立“append 完成且 metadata 可见”的 happens-before；不能仅靠 Python 调用顺序猜测设备执行顺序。

### Prefix sharing、COW 与 ownership

软件 KV page 是推理 runtime 的逻辑分配单位，不是 CUDA 虚拟内存页，也不是 CTA（Cooperative Thread Array，协作线程数组）或 thread block。Copy-on-Write（COW，写时复制）由软件管理：设每页 T=4，共享 page 7 中已有长度 3 的 prefix，两请求都把 table 指向 7；其中一个请求 append 到 pos=3 时，allocator 先复制 page 7 的有效内容到新 page 2，再把它自己的 table 改成 2，最后写 page 2 的 offset 3，另一个请求仍然读 page 7。新物理页不改变 token 的页内位置。硬件不会自动替 runtime 做 COW。释放或复用物理页也必须等待所有读 stream 完成。wrapper 的 preflight 只检查传入 batch 中可见的引用，拒绝 batch 内共享 append target；它不是 allocator，调用者仍须保证 batch 外请求、prefix view 和其它 stream 不引用该页，并负责全局 COW 与事件同步。

### 对照 CUDA 手册检查硬件假设

寄存器分配由编译器决定，local memory 的“local”指线程作用域，并不表示物理片上存储。本 kernel 长期保留 q 和 D 维的 u，还同时处理 K/V tile；增大 BLOCK_T 或把更多 Query head 放进一个 program，会增加活跃数据量。预期收益是减少重复 KV 读取，代价可能是 spill 与驻留 CTA 数下降。需要对照编译资源、local load/store、L2/DRAM 流量验证，不把 Python 中的变量名直接当作寄存器分配结果。

一个 CTA 内的 barrier 不能替代两个独立 kernel 或不同 stream 之间的依赖管理。因此本例把 append 与读取放在同一设备的 current stream；split-KV 的跨 CTA 合并若实现为独立 kernel，也需要明确它与 partial 写入的顺序。如果生产者与消费者处于不同 stream，应由生产者记录 event，消费者在读取前等待该 event；主机回收或复用页时还要等待最后一个读者完成。软件页表本身不提供这些同步保证。完整 API 说明可在文末资料中查阅。

## 从普通 PagedAttention 走向压缩与低精度

普通 paged decode 先解决“地址管理”这一维。模型系统还可能沿另外两维变化：改变要保存/访问的结构，或改变每个元素的编码精度。三者可以组合，但不能互相冒充：page table 不等于稀疏结构，低 bit 存储不等于原生低精度 MMA，稀疏选择也不自动意味着历史 KV 可以永久删除。

> [!WARNING] 算法变化需要不同的正确性标准
> 分页重排应保持原 Attention 数学；低精度需要误差评估；报告中的 Bounded Replay 接受近似状态，不能按“只是换存储位置”的标准验收。

DeepSeek-V4.1-Flash 将结构与精度两方面的优化结合起来。下面分别解释 CED（Causal Encoder-Decoder，因果编码器-解码器）、CSA2（Compressed Sparse Attention 2，压缩稀疏注意力第二版）和 SWA（Sliding-Window Attention，滑动窗口注意力），不把它们与普通分页寻址混为一谈。

### CED：上半层的 global KV 从哪里来

普通逐层模型要得到第 l 层的 KV，先要完成前面各层的计算，产生这一层的输入 H_l。CED 把网络划成下半部 causal encoder 和上半部 decoder，让 decoder 的 global KV 投影直接依赖 encoder 的最终表示，而不是分别等待每层自己的 hidden state。报告的式 (1) 为

$$
C_l=H_{L/2}W_l^{KV},\qquad
Z_l=H_{L/2}W_l^Z,\qquad l>L/2.
$$

这里 C_l 是 global KV 条目相关的表示，Z_l 是相应的压缩权重中间量；W_l 随层变化。因此“来自同一个 H_(L/2)”不代表所有层的投影结果都相同，更不代表直接把某一层的物理页表复制给所有 decoder 层。CED 解决的是生成这些状态需要经过多少层计算；下一节的 CSA2 再决定哪些状态跨层共用。

SWA 仍需要各层自己的局部 hidden state。若完全不计算 decoder，首个 Decode step 会缺少 decoder 的局部 KV。报告通过对末尾窗口进行 bounded replay 补齐这一部分，因而仍有 decoder 计算成本。

对一个没有前缀命中的完整 Prefill，先用“token×layer 数”作为粗略工作量代理：传统路径是 NL；encoder 处理 N 个位置，上半层只重放末尾 min(N,W) 个位置，则比例为

$$
r_{\mathrm{work}}\approx
\frac{N(L/2)+\min(N,W)(L/2)}{NL}
=\frac12+\frac{\min(N,W)}{2N}.
$$

N=4096、W=128 时，这个比例是 0.515625；N=W 时为 1。它解释了长输入场景为什么更有利，但不是实测加速比，也没有把复杂度从线性变成另一个阶。投影、Attention 的实际复杂度、不同层的计算成本、缓存命中和数据搬运都被这个代理模型省略了，端到端收益仍需测量。

### CSA2：缓存来源与索引来源是两条依赖

CSA2 将 main KV、indexer K 和 Top-K 选择的复用拆开。各层始终计算自己的 main Q 和 SWA KV；复用的是全局分支中的特定状态，不是整个 Attention 输出。

| 模式 | main KV / indexer K | Top-K 选择 | 当前层主要新增工作 |
|---|---|---|---|
| Full | 当前层产生 | 当前层产生 | KV、indexer Q/K、评分与选择、Attention |
| Reindex | 使用已有 Full 来源 | 用当前 indexer Q 重新选择 | 新查询、重新评分与选择、Attention |
| Reuse | 使用已有 Full 来源 | 使用最近与该 KV 对应的选择 | main Q、SWA KV、Attention；不再计算 indexer Q |

考虑依次为 Full、Reuse、Reindex、Reuse 的四层。它们可以共用第一层产生的 main KV，但后两层使用第三层重新计算的 Top-K。只记录“用了哪一层的 cache”是不够的，还要记录“用了哪一层的选择”。否则可能把旧索引错误地应用到另一组 KV。

下面的教学模型只追踪这两类来源，不计算作者的压缩器、indexer 或 Attention：

<!-- source-check: examples/cache_accounting.py -->
~~~python
def cache_sources(modes):
    kv_owner = index_owner = None
    result = []
    for layer, mode in enumerate(modes):
        if mode == "Full":
            kv_owner = index_owner = layer
        elif mode == "Reindex":
            if kv_owner is None:
                raise ValueError("Reindex needs an existing KV source")
            index_owner = layer
        elif mode == "Reuse":
            if kv_owner is None or index_owner is None:
                raise ValueError("Reuse needs KV and selection sources")
        else:
            raise ValueError("unknown mode")
        result.append((layer, kv_owner, index_owner))
    return result
~~~

前述四层得到 (0,0,0)、(1,0,0)、(2,0,2)、(3,0,2)：每个三元组依次是当前层、KV 来源层、选择来源层。若后面出现新的 Full，两个来源都随之更新。实现时还要管理状态生命周期、跨设备传递和并发请求隔离，这些不是一个层编号就能解决的。

### Bounded Replay：为什么窗口补算仍不等价

报告将长时间保留的 global 持久缓存与短生命周期的 SWA 缓存分开管理。global 缓存命中而 SWA 状态缺失时，并不是从一个普通物理页中读回完全相同的数据，而是重算末尾窗口。对从位置 s 开始的重放，位置 i 的局部注意力范围截断为

$$
\mathcal V_i^{\mathrm{replay}}
=\{j:\max(s,i-W+1)\leq j\leq i\}.
$$

设 W=4、s=100。完整前向中，位置 100 可见 97、98、99、100；重放中只能看到 100。即使到了位置 103，位置集合已经是 100…103，这些位置的中间 hidden/KV 也可能因更早的截断而不同。相同的最后一层可见位置，不足以证明中间状态相同。

因此报告明确接受近似状态。encoder 重放只恢复局部 SWA，命中的 global KV 不重算、不覆盖；未命中的后缀再生成自己的状态。decoder 重放产生的局部 KV 用于后续 Decode。数学一致性、输出质量和缓存命中位置敏感性需要分别检查，不能沿用普通分页重排的“结果只差浮点归约顺序”标准。

### FP4：先算条目，再算 scale

报告的 main KV 用 E2M1 编码，每 16 个 channel 配一个 E4M3 scale，平均为 4.5 bit/元素；它省略了完整 NVFP4 的第二层 global scale。indexer K 使用另一套 MXFP4 规则，SWA 仍为 FP8。读取 main KV 后先反量化，再执行 Attention，因此缓存格式与矩阵指令格式不能直接画等号。

这个模型的 global KV 容量可以按条目数复算：3 个 encoder Full 组各 compression 2，加 decoder 1 个 Full compression 1，所以 global entry 数约为 (3/2+1)S=2.5S。若 main512d 的 E2M1 每 16d 用 1 byte scale，则每 entry 为 256+32=288 B；indexerK 128d 的 MXFP4 每 32d 用 1 byte E8M0，则为 64+4=68 B。因此

$$
(288+68)\times 2.5=890\ {\rm B/token}.
$$

这是 global KV 有效数据量的手算，与该模型报告的容量口径一致；它不包含 SWA、页 padding、索引输出、allocator 或 TP 副本，不能当作全模型显存或测量结果。完整 NVFP4 的第二层 global scale 见[NVIDIA 对 NVFP4 的公开说明](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)，不能把它反推成报告 mainKV 已采用完整格式。

MLA 的容量也要按实际路径写公式。官方 DeepSeek-V3 inference/model.py 的 absorbed 路径每层保存一份共享 latent 512 和一份 RoPE key 64；每个元素若按 BF16 计为 2 字节（没有把 TP/metadata 写入这个 token 账本），因此 61 层的正确数值是

$$
61\times(512+64)\times2=70272\ {\rm B/token},
$$

不是把两项分别乘 2 后再误读成约 137 KiB/token。这个数字是数据模型示例，不是本章 page allocation 或真实全模型容量。

vLLM 的[历史 PagedAttention 设计文档](https://docs.vllm.ai/en/latest/design/paged_attention/)页面明确带有历史文档警告；它适合说明早期布局设计，不应被当作 today backend 的实现依据。当前实现应以实际 checkout、backend、metadata 和 profiler 为准。

上述容量、来源关系和重放位置可在 CPU 上独立核对：

~~~bash
python roadmap/curriculum/operators/06-decode-paged-attention/examples/cache_accounting.py
~~~

配套 CPU 检查验证容量算例、量化组成项、缓存来源更新和重放位置；这些只验证推导，不是模型性能测量。

## 分页 Attention 的实现

这个实现分成追加 KV 与 Attention 两个 kernel，按顺序提交到同一 CUDA stream。下面是 Attention 的完整核心：

<!-- source-check: examples/paged_decode_fp32.py -->
~~~python
@triton.jit
def _paged_decode_kernel(
    q_ptr, cache_k_ptr, cache_v_ptr, lengths_ptr, table_ptr, out_ptr,
    NUM_PAGES, PAGE_SIZE, MAX_PAGES, KV_HEADS,
    MAX_LEN: tl.constexpr, Q_HEADS: tl.constexpr, GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    b = pid // Q_HEADS
    hq = pid % Q_HEADS
    d = tl.arange(0, HEAD_DIM)
    q_offset = ((b * Q_HEADS + hq) * HEAD_DIM + d)
    q = tl.load(q_ptr + q_offset)
    length = tl.load(lengths_ptr + b).to(tl.int64)
    kv_head = hq // GROUP_SIZE
    m = tl.full((), float("-inf"), tl.float32)
    l = tl.zeros((), tl.float32)
    u = tl.zeros((HEAD_DIM,), tl.float32)
    scale = 1.0 / tl.sqrt(float(HEAD_DIM))

    for kv0 in tl.range(0, MAX_LEN, BLOCK_T):
        logical = kv0 + tl.arange(0, BLOCK_T)
        logical_page = logical // PAGE_SIZE
        page_offset = logical % PAGE_SIZE
        logical_valid = (logical >= 0) & (logical < length)
        table_valid = logical_valid & (logical_page < MAX_PAGES)
        physical_page = tl.load(
            table_ptr + b * MAX_PAGES + logical_page,
            mask=table_valid,
            other=0,
        ).to(tl.int64)
        valid = table_valid & (physical_page >= 0) & (physical_page < NUM_PAGES)
        token_base = (((physical_page * PAGE_SIZE + page_offset) * KV_HEADS + kv_head)
                      * HEAD_DIM)[:, None]
        cache_offset = token_base + d[None, :]
        k = tl.load(cache_k_ptr + cache_offset, mask=valid[:, None], other=0.0)
        v = tl.load(cache_v_ptr + cache_offset, mask=valid[:, None], other=0.0)
        scores = tl.sum(k * q[None, :], axis=1) * scale
        scores = tl.where(valid, scores, float("-inf"))
        tile_m = tl.max(scores, axis=0)
        new_m = tl.maximum(m, tile_m)
        safe_m = tl.where(new_m != float("-inf"), new_m, 0.0)
        alpha = tl.where(m != float("-inf"), tl.exp(m - safe_m), 0.0)
        p = tl.where(valid, tl.exp(scores - safe_m), 0.0)
        u = alpha * u + tl.sum(p[:, None] * v, axis=0)
        l = alpha * l + tl.sum(p, axis=0)
        m = new_m

    denom = tl.where(l > 0.0, l, 1.0)
    tl.store(out_ptr + q_offset, u / denom)
~~~

host launch 的 grid 是 (B*Hq,)，因此一个 program 的 pid 用整除/取模还原为 (b,hq)；它读取一行 q，沿 BLOCK_T 循环扫描该请求的历史 KV。k * q[None,:] 的形状是 [BLOCK_T,D]，tl.sum(..., axis=1) 沿 D 归约，得到每个 token 一个 score；p[:,None] * v 仍是 [BLOCK_T,D]，tl.sum(..., axis=0) 沿 T 归约成一个 D 维的 U 更新。最后只在 l>0 时除以 l，否则写零。token_base 是 [BLOCK_T,1]，d 是 [1,D]；因此 cache_offset 是 [BLOCK_T,D]。页表和 cache 的无效 lane 都经过 mask，且不从无效 logical page 无 mask 加载 physical page ID。这个 kernel 是 FP32 simple correctness baseline，不声称等同 FlashAttention-2，也不把小 M=1 的 kernel 当作 Tensor Core 性能实现。tl.range 避免把长上下文的所有 chunk 静态完全展开；实际性能仍需以目标 GPU 编译报告和 profiler 验证。

### 连续参考与分页实现怎样对比

先独立生成逻辑连续的 K/V，再把相同数据放入非连续物理页。参考实现只读取逻辑张量，分页实现读取页表与物理池，这样寻址错误不会同时出现在两条路径中。

<!-- source-check: examples/validate_paged_decode.py -->
~~~python
def dense_attention(q, k, v, lengths):
    _, q_heads, dim = q.shape
    kv_heads = k.shape[2]
    mapping = torch.arange(q_heads, device=q.device) // (q_heads // kv_heads)
    kq = k[:, :, mapping, :].permute(0, 2, 1, 3)
    vq = v[:, :, mapping, :].permute(0, 2, 1, 3)
    scores = (q[:, :, None, :] * kq).sum(-1) / dim**0.5
    positions = torch.arange(k.shape[1], device=q.device)[None, None, :]
    valid = positions < lengths[:, None, None]
    probs = torch.softmax(scores.masked_fill(~valid, float("-inf")), dim=-1)
    return (probs[..., None] * vq).sum(dim=2)
~~~

其中 mapping 把 Query head 对应到 KV head；valid 排除每个请求长度之外的位置。这个参考会物化中间张量，适合核对结果，不代表高性能实现。

准备好同一组输入后，实际比较的关键代码如下。logical_k/logical_v 包含本步输入 token 的 K/V，分页池先保存此前的 token，再由 new_k/new_v 追加本步位置：

~~~python
expected = dense_attention(q, logical_k, logical_v, lengths)
got = paged_decode(
    q, cache_k, cache_v, lengths, table,
    new_k=new_k, new_v=new_v,
)
torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)
~~~

不要只看输出形状或最后一项。先核对整张输出，再测试跨页、长短请求混合和非法写入；同一页上的公式、kernel 与参考实现可以直接逐项对照。

## 实践：写题、验证与性能对比

### LeetGPU：正确性与代码归档

从 [LeetGPU 官方题库入口](https://leetgpu.com/challenges) 搜索并核对上一章的 [#6 Softmax Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/6_softmax_attention)：题面是连续的 Q[M,d]、K[N,d]、V[N,d]、out[M,d]，solve(Q,K,V,out,M,N,d)。本章先把 M=1 作为连续长 KV 的单 Query 数学基础：它可以验证 score、softmax、PV 和长 N，但没有 page_table、ragged batch、append 或 COW。应从平台空题面完成并保存当次原始 solve/kernel，平台状态只由对应题面结果证明。

[#80 Grouped Query Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/80_grouped_query_attention) 再验证 Hq/Hkv 的 GQA 映射，但它是同长、无 batch 的多头题，不是单 Query 分页题。当前公开题库树中没有以 paged/decode 命名的对应题面；因此 page table、ragged length、split merge、append/COW 只进入本章的 CPU/GPU 扩展实验，平台题检查 head 映射，扩展实验另外检查分页和单 Query 状态。

### 服务器：真实性能

验证时覆盖 D=16/32/64/128、长度 1/4/5/33/65、MHA/GQA/MQA，以及零 Query、非法页号、输出别名和共享写入页。正确性通过后再计时；没有 CUDA 时脚本会明确跳过，不算 GPU 验证成功。

运行 CPU 与 host 检查：

~~~bash
python roadmap/curriculum/operators/06-decode-paged-attention/examples/cpu_paged_decode_checks.py
python roadmap/curriculum/operators/06-decode-paged-attention/examples/host_validation_checks.py
python roadmap/curriculum/operators/06-decode-paged-attention/examples/validate_paged_decode.py
~~~

在有 CUDA、PyTorch 和 Triton 的目标环境中，先运行 correctness，再按明确输入身份运行 tiny benchmark：

~~~bash
python roadmap/curriculum/operators/06-decode-paged-attention/examples/validate_paged_decode.py --benchmark --batch 3 --heads 4 --seq-len 9
~~~


服务器实验以前述基础正确性通过为前置。运行 correctness 后，固定 GPU、Torch/Triton 版本、B,Hq,Hkv,D,lengths,T,P,max_pages、page mapping、dtype、是否 append、输出分配和 synchronize 边界。至少分别记录：

1. paged wrapper（page-table 读取，append 是否计时）；
2. dense attention on pre-gathered logical KV；
3. gather + dense attention（把 gather、metadata、allocation 计入）。

第二项只说明连续 dense 数学 kernel 的时间，第三项才把“先恢复连续 KV”这个额外边界放进去；三者都不是天然的强 baseline。报告有效 KV bytes、分配 padding、metadata bytes、split partial bytes 和实际 profiler traffic 时，要把 GQA 的逻辑复用与物理读取分开。没有目标 GPU 的实测，就只保留正确性、编译/运行命令和待测量的硬件机制，不把示例数字写成性能结论。

本章的掌握标准是：给定 lengths/table/T/P/Hq/Hkv/D 能手算任意 token 的物理地址和容量账本；能解释 paged 为什么不改变 dense attention 数学、为什么不自动减少有效读取；能推导 chunk 的 (m,l,U) 合并和 split-KV 的额外 workspace；能指出页尾、空段、全 mask、跨页 append、COW ownership 和 stream event 的边界；最后能把 page 管理、结构压缩和精度编码分开设计并分别验证。

## 延伸资料

stream、event 与拷贝 API 的参数和同步语义可查 [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)。

## 参考阅读

[大模型推理实践](../../../../downloads/大模型推理实践.pdf) · [CUDA Programming Guide 13.3](../../../../downloads/cuda-programming-guide.pdf) · [DeepSeek-V4.1-Flash 技术报告](../../../../downloads/DeepSeek_V41_Tech_Report.pdf)。

## 章节导航

- [上一章：Prefill Attention](../05-prefill-attention/README.md)
- [下一章：量化算子](../07-quantized-operators/README.md)
- [算子总览](../README.md)
