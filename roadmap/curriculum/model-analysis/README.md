# 第一章 模型 GPU 执行分析：从单算子到 Prefill 与 Decode

模型时间包括算子、布局转换、分配、launch 和通信。单算子节省的时间能否反映到吞吐，取决于它在执行路径中的占比，以及改动是否增加了其他工作。

这里以 dense、GQA、SwiGLU 的 decoder block 建立计算和存储账本。MoE、MLA、稀疏注意力和共享缓存则按各自的数据结构调整。

## 1. 先固定测量边界

| 边界 | 包含什么 | 不能直接代表什么 |
|---|---|---|
| kernel | 一次设备计算 | 整层或请求时延 |
| layer | Norm、投影、Attention、MLP、残差等组合 | 整个模型与调度 |
| model forward | 多层、embedding、输出头 | 排队与网络返回 |
| request | 排队、准备、模型、采样、返回 | 纯 GPU 时间 |

Prefill 一次处理多个输入位置，填充各层 KV；常见自回归 Decode 每次处理一个新输入 token。Prefill 最后一个 logits 用来采样第一个输出 token，这个输出 token 的 KV 要在下一次作为输入时才生成。统计输出 token 数与已计算输入位置数时必须分开。

性能表至少记录 batch、输入/输出长度、dtype、缓存命中、是否计算所有位置的 logits、是否包含采样和分配。只写“Prefill 10 ms”不足以复现实验。

## 2. 从矩阵形状计算 FLOPs

令 batch 为 B、输入长度为 S、模型维度为 D、Query head 数为 Hq、KV head 数为 Hkv、head dimension 为 d，取 D=Hq×d。SwiGLU 中间维度为 I。

| 运算 | 输入与权重 | 输出组织 |
|---|---|---|
| Q 投影 | [BS,D] × [D,D] | [B,Hq,S,d] |
| K/V 投影 | 两次 [BS,D] × [D,Hkv d] | 各 [B,Hkv,S,d] |
| Attention 输出投影 | [BS,D] × [D,D] | [B,S,D] |
| gate/up | 两次 [BS,D] × [D,I] | 各 [B,S,I] |
| down | [BS,I] × [I,D] | [B,S,D] |

忽略 bias 和小规模归一化参数，这些矩阵共有

$$
P_{\mathrm{linear}}=2D^2+2D H_{kv}d+3DI.
$$

Prefill 线性部分约为 2BS×P_linear FLOPs。因果 Attention 有 S(S+1)/2 个有效 Query-Key 配对，QK 与 PV 合计约

$$
F_{\mathrm{attn,prefill}}\approx4BH_qd\frac{S(S+1)}2.
$$

这个数表示有用数学工作。先构造完整 S×S score 再 mask 的参考实现，可能实际计算整个方阵；分块 kernel 也有对角 tile 和 padding 开销。不能混用有用 FLOPs、实际工作量和指令计数。

Decode 若当前缓存长度为 S，线性层约为 2B×P_linear，Attention 约为 4BHqdS。L 个同构 block 乘 L；Norm、RoPE、激活、索引及采样另算。

embedding 与 LM head 不在这个 block 公式中。输出头 [D,Vocab] 每计算一个位置约有 2D×Vocab FLOPs；只计算 prompt 最后一个位置，与为所有位置返回 logprob，可能产生不同的计算和临时存储成本。

## 3. batch 对权重与 KV 的作用不同

假设 dense 权重本步各从 HBM 读取一次，并被整个 batch 充分复用，每元素占 b 字节，线性部分理想算术强度为

$$
AI_{\mathrm{weights}}\approx
\frac{2B P_{\mathrm{linear}}}{P_{\mathrm{linear}}b}=\frac{2B}{b}.
$$

BF16 的 b=2，B 从 1 增至 8，这个理想值从 1 增至 8 FLOPs/byte。真实 tile、cache 和重复加载会改变物理流量，但复用来源明确：一组权重服务更多 token。

KV 通常属于不同请求，有效数据量约为 2BSHkvdb，因此

$$
AI_{\mathrm{KV}}\approx
\frac{4BH_qdS}{2BH_{kv}dSb}=\frac{2H_q}{H_{kv}b}.
$$

B 消掉了。增大 batch 可以改善并行度，却不意味着各请求自动共用 KV。共享前缀应按唯一物理页及实际复用核算；若各 Query head 重复读取共享 KV，GQA 的容量优势也不一定同比例反映到 DRAM 流量。

> [!IMPORTANT] 分阶段，还要分算子
> Prefill 可以同时包含计算密集 GEMM、带宽敏感 Norm 和 launch 敏感小操作；Decode 也可能在大 batch 下提高 GEMM 算术强度。先逐类判断，再看关键路径。

## 4. 显存看活跃对象，而不是把中间量全部相加

无前缀共享的普通 GQA 有效 KV 数据为

$$
M_{\mathrm{KV}}=2LBSH_{kv}db.
$$

假设 L=32、D=4096、Hq=32、Hkv=8、d=128、I=11008、S=4096、b=2：B=1 时 KV 为 512 MiB，B=8 时为 4 GiB，block 权重容量却不随 batch 改变。还要另计页尾、scale、allocator 余量和可能的 TP 副本。

串行推理中，一层用完的临时张量可以释放或复用，不能把所有层的中间量直接相加当作峰值。反过来，融合前后张量可能同时活跃，异步执行可能延迟释放，CUDA Graph 与框架缓存也可能保留内存。

分别记录 allocated、reserved、KV 池容量及进程/设备观测。empty_cache 不会释放仍被引用的张量，reserved 没下降也不自动代表泄漏。

量化权重、量化 KV、减少物化影响不同组成项，应该分别缩放后相加；不要把整模型容量连续乘几个压缩比例。

下面的计算器固定了 block-only 口径：

<!-- source-check: examples/ledger.py -->
~~~python
def block_ledger(batch, sequence, layers, model_dim, query_heads, kv_heads, head_dim, intermediate, element_bytes):
    dimensions = [batch,sequence,layers,model_dim,query_heads,kv_heads,head_dim,intermediate,element_bytes]
    if any(not isinstance(x,int) or x <= 0 for x in dimensions):
        raise ValueError("positive integer dimensions required")
    if model_dim != query_heads*head_dim or query_heads % kv_heads:
        raise ValueError("inconsistent attention dimensions")
    linear_params = 2*model_dim**2 + 2*model_dim*kv_heads*head_dim + 3*model_dim*intermediate
    return {
        "linear_params_per_block": linear_params,
        "block_weight_bytes": layers*linear_params*element_bytes,
        "prefill_linear_flops": 2*batch*sequence*layers*linear_params,
        "prefill_causal_attention_flops": 4*batch*layers*query_heads*head_dim*(sequence*(sequence+1)//2),
        "decode_linear_flops": 2*batch*layers*linear_params,
        "decode_attention_flops": 4*batch*layers*query_heads*head_dim*sequence,
        "kv_payload_bytes": 2*batch*layers*sequence*kv_heads*head_dim*element_bytes,
    }
~~~

## 5. 时间线中的因果关系

先在不开 profiler 时测端到端时间，再用带层/阶段标记的时间线定位原因。Profiler 会增加开销，采集结果不能直接替代正常运行基线。

| 现象 | 核对什么 | 可检验的改动 |
|---|---|---|
| 小 kernel 之间有空隙 | Python、同步、分配、回读 | 复用 buffer、减少同步、比较图执行 |
| GEMM 时间占比高 | 实际形状、dtype、指令和资源 | 比较适用算法与 tile |
| 拷贝与 Attention 交替 | 每步 concat、重复展开 GQA | 预分配或直接维护目标布局 |
| kernel 更快而层更慢 | transpose、cast、临时张量 | 布局与消费端一起改 |
| 某设备持续等待 | 分片、负载、通信顺序 | 根据关键路径调整并行与重叠 |

Nsight Systems 看整体时序，Nsight Compute 分析选中的 kernel。寄存器很多不等于已证明 spill，低 MFU 不代表受带宽限制的任务一定实现得差。多个 stream 的区间有重叠时，也不能把所有 kernel duration 相加当作总耗时。

## 6. 实践：分开改变 batch、上下文与算子实现

~~~bash
python roadmap/curriculum/model-analysis/examples/ledger.py
~~~

数学检查验证 batch 对权重与 KV 的不同影响，以及 S=1 的因果边界。之后在可控模型中分别改变 B、S、dtype，保持其他条件一致：

| 对比组 | 固定项 | 改变项 | 要回答的问题 |
|---|---|---|---|
| batch 曲线 | 模型、序列、dtype | B | 权重复用改善多少，KV 与排队付出什么 |
| 上下文曲线 | 模型、batch、dtype | S | Attention 与缓存何时主导 |
| 算子替换 | 权重、输入、数学语义 | 一个实现 | 局部收益能否传到层和模型 |
| 低精度 | 模型与输入集 | 编码及其 kernel | 时间、显存与质量一起怎样变化 |

模型组合没有对应的 LeetGPU 题，不制造平台成绩。基础题验证局部算子，组合模型还要验证 cache、位置、残差与布局。

## 参考阅读

[CUDA Programming Guide](../../../downloads/cuda-programming-guide.pdf) · [大模型推理实践](../../../downloads/大模型推理实践.pdf) · [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/) · [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

## 章节导航

- [上一章：Sampling 与 KV 辅助算子](../operators/09-sampling-kv/README.md)
- [下一章：Mini Transformer](mini-transformer/README.md)
