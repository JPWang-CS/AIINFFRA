# 第四章 多 GPU：分片、通信与端到端收益

模型分片决定每张 GPU 保存哪些数据、执行哪些计算，以及需要交换哪些结果。增加设备可以分摊容量与工作量，也会引入通信、负载不均和流水空泡。

从可手算的矩阵分片出发，分别计算本地工作、collective 数据量和关键路径，再用通信测试与模型时间线核对。

## 1. 并行维度决定了通信含义

| 策略 | 切分对象 | 常见影响 |
|---|---|---|
| DP：Data Parallelism，数据并行 | 请求或 batch | 模型通常复制，提升整体服务能力 |
| TP：Tensor Parallelism，张量并行 | 层内矩阵/张量 | 权重与计算分片，需要聚合或交换激活 |
| PP：Pipeline Parallelism，流水线并行 | 模型层 | 阶段间传递激活，关注空泡与并发 |
| EP：Expert Parallelism，专家并行 | MoE 专家 | token dispatch/combine 与专家负载 |
| CP：Context Parallelism，上下文并行 | 序列位置 | Attention 需要跨分片交换或合并 |

这些维度可以组合，但资源分组、数据布局和 collective 顺序需要一致。它们不是“越多越好”的开关。独立 DP 实例可能提高总吞吐，却没有减少单个请求要走的模型计算。

Megatron 风格的 sequence parallelism 常与 TP 配合，在部分 Norm/Dropout 等操作中分片激活；它不能简单等同于把整个 Attention 的长上下文切开。CP 要处理跨序列分片的注意力依赖。概念相近，数据流与通信不同。

## 2. 两次线性层怎样做 TP

设 X[M,I]、第一层权重 W1[I,H]、第二层权重 W2[H,O]。把 H 平均分给 P 个 rank。第一层按输出列切分：

$$
W_1=[W_1^{(0)}\ \cdots\ W_1^{(P-1)}],\qquad
Y_r=XW_1^{(r)}.
$$

每个 Y_r 是 [M,H/P]。如果中间是逐元素激活，可以在分片上直接计算，不必先把完整 Y gather 回来。第二层按输入行切：

$$
Z_r=f(Y_r)W_2^{(r)},\qquad
Z=\sum_{r=0}^{P-1}Z_r.
$$

Z_r 是完整 [M,O] 形状的部分和，因此需要 SUM 聚合。不是把这些 Z_r 沿某个维度拼接起来。

<!-- source-check: ../examples/parallel_reference.py -->
~~~python
def tensor_parallel_mlp(x, up, down, parts):
    hidden=len(up[0])
    if parts<=0 or hidden%parts:
        raise ValueError("hidden dimension must divide evenly")
    width=hidden//parts
    partials=[]
    for rank in range(parts):
        begin,end=rank*width,(rank+1)*width
        up_shard=[row[begin:end] for row in up]
        hidden_shard=matmul(x,up_shard)
        activated=[[max(v,0) for v in row] for row in hidden_shard]
        partials.append(matmul(activated,down[begin:end]))
    total=partials[0]
    for partial in partials[1:]:
        total=add(total,partial)
    return total
~~~

这里用 ReLU 展示逐元素激活与分片兼容，不是完整 Transformer MLP。SwiGLU 的 gate/up 两条分支要按同一 H 分片，再在对应元素间相乘。

若最终 bias 在每个 Z_r 上都加一次，再 AllReduce，就会把 bias 加 P 次。正确位置通常是完成部分和聚合之后，或采用明确且等价的分摊规则。Norm 若需要跨被切分维度统计，也不能当作纯逐元素运算随意下放。

### KV Cache 不保证按 TP 数量等比例缩小

Query head 与 KV head 的划分可能不同。若 KV head 数少于 TP rank 数，某些方案会复制 KV head；不能直接用总 KV 字节除以 P 估计每卡容量。应检查本地 head 数、复制关系、dtype、页布局和分片元数据。

## 3. Collective：按输出语义选择操作

| 操作 | 结果 |
|---|---|
| AllReduce | 每个 rank 得到归约后的完整结果 |
| ReduceScatter | 归约后每个 rank 只保留一片 |
| AllGather | 每个 rank 获得各分片拼接后的结果 |
| Broadcast | 一个来源分发给其他 rank |
| All-to-All | 每个 rank 向其他 rank 发送各自对应的数据 |

AllGather 不执行求和，ReduceScatter 不是普通 scatter。shape、dtype、元素数、rank 顺序和调用顺序必须匹配；数学上想做 SUM，不代表通信库会替你识别错误分片。

### 带宽数字要说明分母

若 AllReduce 输入 payload 为 N 字节，耗时 t，可报告算法带宽 N/t。对理想 ring 模型，每个 rank 传输量常用 2(P-1)N/P 估计，于是有 ring 等效带宽：

$$
BW_{\mathrm{ring,eq}}=\frac{2(P-1)}P\frac Nt.
$$

它是模型换算，不是直接测得的某条物理链路字节数。实际 NCCL 算法、协议、拓扑和消息大小可能不同；不能拿这个数与任意网卡标称单向带宽直接比较。

小消息更受启动、同步和协议开销影响，大消息更容易显示带宽限制。模型里每步许多小 collective 与一个大 buffer 的峰值带宽，不是同一种工作负载。

## 4. EP、PP 与长上下文

### EP：token 去了哪个设备

MoE 路由产生专家编号，再把任务映射到专家所在 rank。远端任务需要 dispatch，计算后返回 combine。若一个 token 选择多个同 rank 专家，是否共用一次输入传输取决于打包方式，不能固定认为网络字节一定等于 T×k×D×dtype。

先保存每个专家 M_e 和每个 rank 的任务量，再分析偏斜。路由更集中可能改善某个 GEMM 的尺寸，却让其他设备等待；均匀随机路由不能代表真实模型负载。

### PP：分层以后还有空泡

理想、等时长、仅 forward 的 P 阶段流水线，处理 m 个微批时，一个简单利用率代理为

$$
\eta\approx\frac{m}{m+P-1}.
$$

它忽略通信和阶段失衡，也不是训练 1F1B 调度的完整公式。单个自回归请求每一步依赖前一步输出，不能任意制造独立微批填满流水线；增加并发又会占用更多 KV 并影响时延。

### CP：注意力仍然要看到正确上下文

沿序列分片后，某 Query 可能需要别的 rank 保存的 K/V。可以交换 K/V，也可以分块计算后合并归一化状态；直接平均局部 Attention 输出通常错误。应合并各段的最大值、指数和与未归一化输出，再完成最终归一化。

因果分片还可能造成负载不均：早期 Query 可见位置少，后期多。如何分配序列位置、怎样重叠通信和计算，要同时保持位置与 mask 的正确性。

## 5. 通信与计算重叠不是无条件相加收益

理想完全重叠下，一段时间可能接近 max(T_compute,T_comm)，串行则接近两者相加；真实情况取决于依赖图。通信读取的输入必须已经写好，计算读取的远端数据必须已经到达，释放 buffer 前必须确保消费者结束。

不同 stream 只提供可能的并发，不保证不同硬件资源互不竞争。通信 kernel、GEMM、拷贝与缓存可能争用带宽、SM 或链路；只看两条时间线重叠面积，不足以证明端到端受益。

先保证 collective 顺序、shape 和 buffer 生命周期正确，再尝试分块与重叠。一个 rank 提前走入下一轮 collective，另一个 rank 仍停在不同调用，可能表现为挂起，而不是普通数值误差。

## 6. 实践：数学分片与真实通信分开检查

先在 CPU 上验证列/行分片、激活位置和 bias：

~~~bash
python roadmap/curriculum/systems/examples/parallel_reference.py
~~~

然后在拥有两张可见 GPU 的个人环境中检查拓扑与 collective：

~~~bash
nvidia-smi -L
nvidia-smi topo -m
torchrun --standalone --nproc-per-node=2 \
  roadmap/curriculum/systems/examples/allreduce_probe.py --elements 262144
~~~

脚本把 rank r 的数据设为 r+1，AllReduce SUM 后应为 P(P+1)/2。通过后使用零 buffer 测量重复通信，避免反复求和导致数值爆炸。关键顺序是：

~~~python
dist.all_reduce(data)
torch.testing.assert_close(
    data, torch.full_like(data, world*(world+1)/2), rtol=0, atol=0
)
data.zero_()
dist.barrier()
start.record()
for _ in range(20):
    dist.all_reduce(data)
end.record()
end.synchronize()
~~~

各 rank 上报 event 时间，再取最大值；输出包含设备、dtype、rank 数和消息大小。这个窗口包含调用发射间隙，不是单条链路的纯传输时间。先测试小规模与多个消息大小，再移到模型中的实际 collective 形状。

通信实验需要至少两张可见 GPU、相应依赖和 torchrun 启动的进程。单卡算子题检查局部运算，多 GPU 实验检查分片、通信和进程间的数据一致性。

## 7. 最终如何选择并行方案

先满足权重、KV、workspace 的容量，再比较同一工作负载下的时延和吞吐。模型放得下时，更多独立 DP 实例可能更适合总吞吐；单请求或单层容量受限时再考虑 TP/PP；MoE 看 EP 的路由与通信；长上下文看 CP 的数据交换。

最终报告必须同时保留每卡形状、容量、计算时间、通信、空闲与端到端指标。多卡更快不能只用 FLOPs 除卡数解释，多卡更慢也不能一概归因于网络。

## 参考阅读

[NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) · [Megatron 并行策略](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) · [CUDA Programming Guide](../../../../downloads/cuda-programming-guide.pdf)

## 章节导航

- [上一章：vLLM 应用](../README.md)
- [课程总览](../../README.md)
