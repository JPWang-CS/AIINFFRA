# 第五章 Prefill Attention：从 Softmax 加权和到在线分块

Prefill Attention 同时处理多个 Query。显式保存 [B,H,S,S] 的 score/probability 矩阵，会让中间存储随序列长度平方增长。在线 Softmax 分块处理 K/V，只保留局部分数和可合并的归一化状态。

下面实现 forward：Q/K/V 为连续、同形状的 [B,H,S,D]，S>0，D 取 16、32、64 或 128；输出独立分配，支持 noncausal 和同起点 causal。dropout、backward、GQA、varlen 和自定义 mask 不在这个接口中。

## 1. 先把问题写成带 batch、head 的矩阵

本章用 `H` 表示 attention head 数，用 `d_model` 表示模型 hidden size。常见关系是 `d_model=H*D`，其中 `D` 是每个 head 的 head dimension。对一个 batch `b`、query head `h`、query 位置 `r`，基础 attention 是

$$
s_{r,t}=\frac{Q_{b,h,r,:}\cdot K_{b,h,t,:}}{\sqrt D},
\qquad
P_{r,t}=\frac{\exp(s_{r,t})}{\sum_{u=0}^{S-1}\exp(s_{r,u})},
$$

$$
O_{b,h,r,:}=\sum_{t=0}^{S-1}P_{r,t}V_{b,h,t,:}
=\frac{\sum_t\left(\exp(s_{r,t})V_{b,h,t,:}\right)}{\sum_t\exp(s_{r,t})}.
$$

这里最后一个式子的分子是向量加权和，不是 `exp(score*V)`；`t` 是 key/value 的 sequence 索引，`s_{r,t}` 是 score，不是位置变量，输出特征索引用 `c`。noncausal 情况每个 query 都看完整 K/V；同长 causal 情况只允许 `t≤r`。若把 Q、K、V 写成 `[B,H,S,D]`，连续 row-major 的 head 基址为

$$
base(b,h)=((bH+h)S)D,\qquad
offset(b,h,s,d)=base(b,h)+sD+d.
$$

这条地址式是本实现修复 batch/head 偏移的关键。实际模型常先得到 `[B,S,H,D]`，再 transpose 成 `[B,H,S,D]` view；这个 view 通常不是 contiguous。本 baseline 明确拒绝这种 stride view，调用方若先 `.contiguous()`，必须把布局转换的额外读写、workspace 和时间单列，不能把复制后的 kernel 时间写成 attention 本身的成本。旧教学 reference 只按 `[B,S,D]` 的注释声明 shape，却没有把 batch 放进 grid；它不能被当成当前四维 baseline。

## 2. 普通 attention 的代价与 Flash 的边界

朴素实现先算 `QK^T`，得到每个 batch/head 一张 `S×S` score 矩阵，再做 mask、softmax、`PV`。score 和 probability 如果以 dtype `s` 存储，各自大约需要 `BHS^2s` 字节；它们是否同时存在要按实际代码计数，不能因为最终输出是 FP32 就声称所有中间都自动是 FP32，也不能把 FP16/BF16 的 bytes 混进 FP32 表。

Flash 风格实现改变的是存储路径：Q 取一个 tile 留在寄存器或片上存储，K/V 沿 sequence 方向分 tile 读取，直接更新输出状态，不物化完整 `S×S` score/prob。计算量仍是二次的。忽略 cache 命中、尾 mask 和无效槽位，只按有效元素估一个 Q 驻留、扫描 KV 的逻辑读写量：Q 和 O 各读/写一次，K/V 被每个 Q tile 扫描一次，因此

$$
Bytes_{logical}\approx 2B H S D s\left(1+\left\lceil\frac{S}{B_Q}\right\rceil\right).
$$

这里的前一个 `1` 汇总 Q/O 各一次，后一个系数汇总 K/V 的重复扫描；它不是 profiler 的 DRAM traffic，也不是“全局 QKV 各读一次”。Flash 策略仍会随着 Q tile 数增加而重读 K/V。对 noncausal `[B,H,S,D]`，一次 QK dot 有 `2BHS^2D` FLOPs，一次 PV dot 也有 `2BHS^2D` FLOPs，合计

$$
FLOPs_{noncausal}=4BHS^2D.
$$

causal 的有用 token 对数是 `S(S+1)/2`，所以有用数学工作量为

$$
FLOPs_{causal,useful}=2BH S(S+1)D.
$$

但本章 kernel 的教学循环仍扫描每个 KV tile，只在 score 上屏蔽 future；边界 tile 和无效 lane 仍会执行部分指令。因此这个 unmasked-valid-domain FLOPs 不能当作实际指令量。若只按 `BQ=16,BKV=32` 的 padded tile 估算，QK+PV 的工作量是 `4BH ceil(S/BQ)BQ ceil(S/BKV)BKV D`；这只是 tile work estimate，不是硬件实测指令量。causal 也不能据此声称执行量或 traffic 自动减半：当前实现没有跳过 future tile，只是让无效 score 不进入状态。是否命中 L2、是否由更高层 cache 或融合路径缓解，要用 profiler 检查。

## 3. Online Softmax 的三个状态

对某一行，把已经处理的 key 集合记作 `A`，维护未归一化的

$$
m=\max_{t\in A}s_t,\qquad
l=\sum_{t\in A}e^{s_t-m},\qquad
U=\sum_{t\in A}\left[e^{s_t-m}V_t\right].
$$

新来一个 tile `T`，先得到 `m_T=max_{t∈T}s_t`，再合并

$$
m'=\max(m,m_T),
$$

$$
\alpha=\begin{cases}e^{m-m'}&m\ne-\infty\\0&m=-\infty\end{cases},\qquad
p_t=\begin{cases}e^{s_t-m'}&t\text{ 有效}\\0&t\text{ 被 mask}
\end{cases},
$$

$$
U'=\alpha U+\sum_{t\in T}\left[p_tV_t\right],\qquad
l'=\alpha l+\sum_{t\in T}p_t,\qquad m\leftarrow m'.
$$

最终输出是 `U / where(l > 0, l, 1)`。这个分支不是数学 softmax 的新定义，而是为全 masked 的虚拟 query 行提供安全输出；本章把这类行写成零。对同长 causal 方形输入，真实 query 行至少能看到自身或首 key；全 mask 保护主要服务 padding query。若 API 允许用户传入一个真实有效行的全 mask，必须另行规定输出零，本实现没有自定义 mask 参数，不能假称支持这种扩展。

完整向量手算如下。第一 tile 的 scores 是 `[0,ln 2]`，`V_0=[1,10]`、`V_1=[3,20]`，所以 `m=ln 2`、`l=1.5`、`U=[3.5,25]`。第二 tile 只有一个有效 key，score 是 `ln 4`、`V_2=[5,30]`，于是 `alpha=exp(ln2-ln4)=0.5`，`m_new=ln4`，`l_new=0.5*1.5+1=1.75`，`U_new=0.5*[3.5,25]+[5,30]=[6.75,42.5]`。最终 `O=[6.75/1.75,42.5/1.75]=[27/7,170/7]`。直接用权重 `[1,2,4]/7` 计算也得到同一结果。`U` 是向量而 `l` 是标量；每个 tile 只在合并时 rescale 一次，避免保存整行概率和反复归一化。

实现中 `tl.maximum(m, block_m)` 是逐 query 行的向量运算；`tl.max(scores, axis=1)` 才是对一个 `[BLOCK_Q,BLOCK_KV]` score tile 沿 key 轴做归约。前者不能替代后者，后者也不是把整个 query 向量压成一个标量。当此前 `m=-inf` 且当前 tile 也全 masked 时，`m_new=-inf`，代码用 `safe_m=0` 并让 `alpha/p` 为0，避免 `-inf-(-inf)` NaN；若此前已有有效 key、当前 tile 全 masked，则 `m_new=m`、`alpha=1`、`p=0`，旧的 `U/l` 原样保留。

## 4. Triton baseline：Q 驻留、KV 扫描和尾块

本实现的 grid 是

~~~python
grid = (ceil_div(S, BLOCK_Q), B * H)
~~~

一个 program 固定一个 `(b,h)` 和一个 Q 行 tile。它只把 Q tile load 一次，然后循环 `kv0=0,32,64,...`：K/V 以 `BLOCK_KV=32` load，QK dot 之后立刻 score mask，再做 online merge。`BLOCK_Q=16`、`BLOCK_KV=32` 是教学选择，不是调优结果。Q、K、V、O 的指针都以 `base=((b*H+h)*S)*D` 开始；query tile 的无效行只参与安全状态，不 store。

下面是 `examples/prefill_fp32.py` 中的完整核心 kernel，而不是伪代码。注意三个层次：`pid_q` 先转成 int64 再参与地址，`HEAD_DIM` 和 `SCALE` 是编译期参数，`S` 仍是运行时 sequence 长度；`tl.max(scores,axis=1)` 是 key 轴归约，`tl.dot(p,v)` 是 PV 的向量更新。Q/K/V 的 load mask 只保证越界指针安全，真正阻止 padding key 进入 softmax 的是 `valid_score` 后的 `-inf`。

<!-- source-check: examples/prefill_fp32.py -->
~~~python
@triton.jit
def _prefill_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B, H, S, D,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_q = tl.program_id(0).to(tl.int64)
    pid_bh = tl.program_id(1)
    q0 = pid_q * BLOCK_Q
    q_abs = q0 + tl.arange(0, BLOCK_Q)
    d = tl.arange(0, HEAD_DIM)
    q_valid = q_abs < S
    # All inputs are contiguous [B,H,S,D].  Use int64 arithmetic for the
    # logical base and row offsets so large shape arithmetic is not truncated.
    bh = pid_bh.to(tl.int64)
    base = (bh * S) * HEAD_DIM
    q_offsets = base + q_abs[:, None] * HEAD_DIM + d[None, :].to(tl.int64)
    q = tl.load(q_ptr + q_offsets, mask=q_valid[:, None], other=0.0)

    m = tl.full((BLOCK_Q,), float("-inf"), tl.float32)
    l = tl.zeros((BLOCK_Q,), tl.float32)
    u = tl.zeros((BLOCK_Q, HEAD_DIM), tl.float32)
    scale = SCALE

    for kv0 in range(0, S, BLOCK_KV):
        k_abs = kv0 + tl.arange(0, BLOCK_KV)
        key_valid = k_abs < S
        k_offsets = base + k_abs[:, None].to(tl.int64) * HEAD_DIM + d[None, :].to(tl.int64)
        k = tl.load(k_ptr + k_offsets, mask=key_valid[:, None], other=0.0)
        v = tl.load(v_ptr + k_offsets, mask=key_valid[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        valid_score = q_valid[:, None] & key_valid[None, :]
        if CAUSAL:
            valid_score = valid_score & (k_abs[None, :] <= q_abs[:, None])
        # Mask the score, not just K/V.  Otherwise padded keys enter softmax.
        scores = tl.where(valid_score, scores, float("-inf"))
        block_m = tl.max(scores, axis=1)
        m_new = tl.maximum(m, block_m)
        safe_m = tl.where(m_new != float("-inf"), m_new, 0.0)
        alpha = tl.where(m != float("-inf"), tl.exp(m - safe_m), 0.0)
        p = tl.where(valid_score, tl.exp(scores - safe_m[:, None]), 0.0)
        u = alpha[:, None] * u + tl.dot(p, v, input_precision="ieee")
        l = alpha * l + tl.sum(p, axis=1)
        m = m_new

    denom = tl.where(l > 0.0, l, 1.0)
    out = u / denom[:, None]
    o_offsets = base + q_abs[:, None] * HEAD_DIM + d[None, :].to(tl.int64)
    tl.store(o_ptr + o_offsets, out, mask=q_valid[:, None])
~~~

host 侧先检查四维、同 shape、FP32、同 CUDA device、contiguous、正长度和 `D` 集合，再计算 `grid=(ceil(S/BLOCK_Q),B*H)`。`prefill_attention` 的 `out` 可以由调用方预分配；wrapper 只在没有传入时分配，benchmark 传入独立 output。finite scan 是显式 preflight 选项，不进入默认 timed call。输出检查用连续 byte interval 判断与 Q/K/V 的区间是否重叠，Q/K/V 之间只读共享不被拒绝。launch 在 `torch.cuda.device(q.device)` 上固定 `num_warps=4`，host 计算 `1/sqrt(D)` 后以 `SCALE` constexpr 传入。

尾块最容易写出“看起来合理”的错误。设 noncausal `Q=K=0`、真实 `S=3`、`V` 全1，KV tile 补到4。如果只把第四个 K/V load 成0，却没有把第四个 score 设为 `-inf`，四个相同 score 会进入分母，输出就是 `3/4` 而不是1。S=33、`BLOCK_KV=32` 时，错误实现会把最后一个 tile 的 64 个槽位当成有效，固定输入会直接暴露这个问题。正确顺序是：key load 可以 `other=0`，但 `valid_score` 必须同时包含 `key_valid` 和 causal 条件，然后 `scores=where(valid_score,scores,-inf)`；V 的零 padding 不能代替 score mask。

本地 CPU 检查用 `S=1,3,5,33,65` 和 `block_kv=1,2,3,4` 对 dense reference 做逐元素比较；GPU harness 还用不同 batch/head 常数填充 V，检验 BH base 是否串位。这个测试比随机误差更容易把尾块、batch、head 三类错误分开。

## 5. Causal 对角线、GQA 与 varlen 的形状边界

当前 baseline 的 causal 条件是同一个 sequence 起点上的 `k_abs≤q_abs`。它不处理 prefix cache、不同 query/key 起点或 packed varlen。例：prefix 长度 `p=3`，第一个 query 的 `query_local=0`，所以 `q_abs=3`；如果 KV buffer 含完整 prefix 并从绝对位置0开始，`k_abs` 依次是 `0,1,2,3,...`，不能把 K 也机械地加3。一般写成独立的 `q_abs=q_start+q_local`、`k_abs=k_start+k_local`，其中 `q_start` 与 `k_start` 由 cache API 合同决定；若 KV 只保留 suffix，才在 API 合同中把两者映射到同一局部起点。对 packed varlen，`cu_seqlens` 是物理存储起点，RoPE/causal 的 `pos_start` 是位置起点，两者不是同一个概念。典型 `[T,H,D]` 地址是 `(cu[b]+r)*H*D+h*D+d`，而本 kernel 是 BHSD 的 `((bH+h)S+s)D+d`。每个 batch 长度不同还需要长度表、cu-seqlens 和位置 offset，不能用一个方形 grid 替代。

GQA 也不能靠重复 Q head 的指针来假装完成。对于 `Hq` 个 query head 和 `Hkv` 个 KV head，要求 `Hq%Hkv=0`，映射是

$$
kv\_head(q)=\left\lfloor\frac{q}{H_q/H_{kv}}\right\rfloor.
$$

这正是 [#80 Grouped Query Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/80_grouped_query_attention) 的无 batch 题面边界；本章 `[B,H,S,D]` square wrapper 没有实现它。GQA 可以作为 CPU 映射模型加入，但不能把它写进当前 Triton kernel 后仍称“同一 baseline”。同理，varlen 需要每个 batch 的绝对 offset 和长度，causal diagonal 不能继续用 `k_abs≤q_abs` 的单一方形坐标替代。

## 6. FA1、FA2 与 warp 组织：不要压扁成一句“分块”

FlashAttention 初版的并行组织首先沿 batch/head 分工；论文 FlashAttention-2 说明 FA2 forward 又把 Q 的 row tiles 分给 CTA，循环结构可以读成 outer-Q、inner-KV。这里的“Q tile”是算法/程序组织，不等于旧 CUDA 教学例子中一个 `blockIdx.x` 就完整代表 FA1 原始实现。

§3.3 的差异更具体：初版在 warp 层把 K/V 工作切开时，需要归约不同 warp 产生的 partial output；FA2 采用 split-Q，让多个 warp 共享同一份 K/V 工作，减少 output fragments 的跨 warp 合并。它不意味着 FA2 不需要任何 warp 或 block 同步；shared-memory 加载、数据可见性和流水线阶段仍需遵守 CUDA 同步语义。FA2 的收益是减少非矩阵运算和不必要的 warp 间归约，不是“每一行永远只由一个 warp 做完”。

旧例子 tail Q 线程退出后，主要问题是它们原本应参与 cooperative KV tile load 的 lane 不再写入，其他线程随后读取未初始化的 shared 数据；不要把这件事泛化成“任何提前 exit 都必定死锁”，CUDA 同步语义对 exited thread 有专门规则，具体风险取决于剩余控制流和协作数据是否完整。

### 6.1 维护 O 还是维护 U

也可以维护已经归一化的输出 `O_old=U_old/l_old`。新 tile 的 `pV` 向量记为 `W=Σ_t p_tV_t`，则先把旧概率质量重标定，再归一化：

$$
l_{new}=\alpha l_{old}+\sum_t p_t,
$$

$$
O_{new}=\frac{\alpha l_{old}O_{old}+\sum_t\left(p_tV_t\right)}{l_{new}}.
$$

它和维护

$$
U_{new}=\alpha U_{old}+\sum_t\left(p_tV_t\right),
\qquad O=\frac{U}{\operatorname{where}(l>0,l,1)}
$$

是同一个数学结果：两者都必须对旧状态做 `alpha` rescale。区别是 O-state 每个 tile 都要对完整输出向量重新归一化；U-state 只维护一次向量累加，最后除一次，省掉重复的向量除法和归一化依赖，也更容易把 `l` 作为一个标量状态放在寄存器中。全 masked 行仍使用 `where(l>0,l,1)`，不把它改写成 `max(l,1)`。

### 6.2 一个并行规模算例

令 `B=1,H=8,S=4096,BQ=64`。只按 `(b,h)` 分工时有 `BH=8` 个任务；按 Q tile 展开后是

$$
BH\times\left\lceil S/B_Q\right\rceil=8\times64=512
$$

个 Q-tile CTA 任务。512 是 grid 中的任务总数，不等于同时驻留在 GPU 上的 CTA 数；同时驻留还受 SM 数、寄存器、shared memory、warp 数和调度波次限制。`BQ` 翻倍到128会把 Q tile 数从64减到32，减少每个 KV tile 被不同 Q program 重读的次数，但会增加 Q、U 和 score 的活跃元素，可能提高寄存器压力。`BKV` 翻倍到64会减少 KV loop 次数，但每次 K/V/score tile 变大；有效 K/V 总元素仍然是 `S*D` 每次 Q tile 扫描，不会因为 BKV 翻倍而自动减半。尾块、cache 和边界浪费要另算。

### 6.3 代码旋钮与观察量

| 代码旋钮 | 预期改变 | 应观察的资源/计数器 | 不能直接推出 |
|---|---|---|---|
| `BQ` | Q tile 数、Q/U 活跃量、K/V 重读次数 | registers/thread、occupancy、L2 read、kernel duration | 更大一定更快 |
| `BKV` | loop 次数、K/V/score tile 大小 | registers、shared、eligible warps、迭代指令数 | 有效 KV bytes 减半 |
| `num_warps=4` | program 内线程组织与并行归约 | registers、active warps、stall、launch metadata | 等于四个独立 CTA |
| score `-inf` mask | softmax 分母与边界正确性 | correctness、predicate/branch、尾块 workload | K/V load 为0就足够 |
| `SCALE`/输入 dtype | score 数值与乘法路径 | FP32/TC 指令、误差、bytes | FP32 accumulator 等于 FP32 multiply |

表中的“观察量”需要目标 GPU profiler 或编译报告才能填写；当前教材只固定代码旋钮和数学预期，不制造性能数字。

### 6.4 K、V 和概率 tile 的最后一次使用并不同步

一次 KV 迭代包含两个矩阵乘和一次行归约：先计算 QK 的 score，再更新在线 Softmax 状态，最后计算 PV。K 主要由 QK 使用，V 由 PV 使用；同一个 KV tile 的两个输入，并不是同时结束使用的。

| 缓冲区或状态 | 本轮读者 | 何时才能覆盖或更新 |
|---|---|---|
| K shared tile | QK 矩阵操作 | 相关 QK 操作完成读取之后 |
| score tile | mask、行最大值、exp | 这些消费者不再需要旧 score 之后 |
| P tile | PV 矩阵操作 | 相关 PV 操作完成读取之后 |
| V shared tile | PV 矩阵操作 | 相关 PV 操作完成读取之后 |
| U accumulator | 旧结果重标定、PV 累加 | 先建立相关矩阵操作完成和寄存器访问顺序 |
| Q tile | 多轮 QK | 最后一个需要它的 QK 操作结束之后 |

如果实现把 K 和 V 作为同一槽位的一对缓冲区管理，最简单的释放时刻是两者最后使用都结束之后，通常要等本轮 PV。若分别管理 K 和 V 的释放，可以更早复用 K 的存储，却需要多维护一套槽位状态和通知。不能只看数学上 QK 已经用过 K，就覆盖一个异步指令可能仍在读的物理地址。

这里说的是共享内存中的实际读者。若某种实现已将一份输入完整搬入寄存器 fragment，原 shared 的最后使用时刻可能更早；应沿真实加载和矩阵指令判断，而不是把表中的粗粒度阶段名当成所有实现的固定协议。

### 6.5 下一块 KV 可以预取，但下一次重标定不能乱序

对一个 query 行，在第 t 个 KV 块开始前，状态是 m_old、l_old、U_old。当前块算出最大值后，以新的 m_new 为统一指数基准：

$$
\alpha_t=\exp(m_{\mathrm{old}}-m_{\mathrm{new}}),\qquad
P_{t,j}=\exp(S_{t,j}-m_{\mathrm{new}}),
$$

$$
l_{\mathrm{new}}=\alpha_tl_{\mathrm{old}}+\sum_jP_{t,j},\qquad
U_{\mathrm{new}}=\alpha_tU_{\mathrm{old}}+
\sum_j\left[P_{t,j}V_{t,j}\right].
$$

上式针对已有有限最大值的非空状态；首次迭代和全 mask 行仍使用本章前面的保护分支。P 不是整行最终归一化后的概率，而是相对于当前最大值的指数权重；最终还要除以 l。

拿本章的具体例子看依赖。处理第一块后，m_old=ln2、l_old=1.5、U_old=[3.5,25]。第二块最大值为 ln4，因此 alpha=0.5。如果第二轮直接把 V 的贡献加到旧 U，却忘了先把旧 U 乘 0.5，就混合了两个不同的指数基准。这不是普通浮点误差，而是公式已经改变。

下一块 K/V 的地址通常不依赖本轮的 m、l、U，因此可以提前发起搬运；某些后续 QK 计算也可在独立的 score 存储中提前进行。但对同一个在线状态的更新仍有依赖。若要把 Softmax 与矩阵计算交错，需要明确哪一轮的 score、P、alpha 和 U 正在被谁使用，不能给原循环加一个异步调用就认为各轮独立了。

下面的协议伪代码保持在线状态顺序，只让加载走在计算前面。真实 CUDA/Triton 实现还需要将这些步骤映射成对应的完成机制：

~~~text
搬运角色：
    获得空闲槽位 → 发起第 t 块 K/V 搬运 → 由完成机制通知本代可读
    只要下一槽位已释放，就继续准备第 t+1 块

计算角色：
    等待第 t 块 K/V 就绪
    发起 QK，并在读取 score 前等待相关计算完成
    mask → 行最大值 → 得到 alpha 和 P
    确认旧 U 已可访问，再执行 U = alpha * U
    发起 PV，得到本轮对 U 的增量
    在下一次依赖 U 的更新前等待相关矩阵操作完成
    当前槽位的全部读者结束后，发布槽位可复用
~~~

这个安排表达的是安全的先后关系，不是 FlashAttention-3 完整 kernel，也不是性能最优的调度。更深入的交错方案会保留多个独立中间结果，并显式管理它们的完成顺序。FA1/FA2 的分块与在线归约提供算法基础；利用 TMA、异步矩阵操作和 warp specialization 是另一个实现层面的优化问题，不应回退成“FA1 一行、FA2 一块”的说法。

### 6.6 Attention 的资源账本比两份 KV 更大

以 BQ=64、BKV=64、D=128、FP16 Q/K/V 为例：一个 Q tile 为 16 KiB，一对 K/V tile 为 32 KiB；KV 双缓冲就是 64 KiB。一个 64×64 的 score tile 若以 FP32 保存，逻辑容量为 16 KiB；同形状 P 若转成 FP16 则为 8 KiB。U 有 64×128 个 FP32 值，逻辑容量为 32 KiB，m 和 l 还各有 64 个 FP32。

这些量不能直接相加后全部叫作 shared memory。Q、score、P、U 可能分别位于寄存器、shared 或其他架构专用存储，也可能复用生命周期不重叠的区域。正确的分析是逐个确定它们的存储位置和存活区间，再计算物理资源峰值。

这解释了为什么把 BQ 或 BKV 调大可能退化：不仅 K/V 变大，score、P、U 以及寄存器活跃区间也会变化。增加一个 KV stage 可能减少等待，也可能因 shared 占用而减少驻留 CTA；让矩阵组更深地在途，则可能增加 accumulator 或中间结果的存储需求。

实验应先固定精度、mask、shape 和基线，对一个变化提出可检查的判断。例如“增加 KV stage 后，等待数据的时间应下降，但 shared 用量会上升”；再查看目标 kernel 耗时、资源和时间线。若耗时没有改善，应继续判断是计算依赖、Softmax 非矩阵工作、资源限制还是不足的并行任务量，而不是只看 Tensor Core 峰值。

### 6.7 把一块 K/V 前视加载写进 Attention 循环

前面的 `_prefill_kernel` 每轮在 QK 前加载本轮 K/V，完成 QK、mask、行最大值、指数权重和 PV 后才进入下一轮。下面的独立 `examples/prefill_pipeline.py` 保留相同的 FP32 数学与分块维度，但显式保活下一块 K/V：先用当前 K 计算 QK，再发起下一块 K/V 的普通 `tl.load`，然后处理当前 tile 的 softmax 状态和 P@V。初始 K/V 在循环外 bootstrap；每轮结束后把下一块提升成当前块。

```text
当前 K ── QK ──┬── 当前 score mask → block max → alpha/P ── 当前 P@V → 更新 U/l/m
               └── 发起 next K/V load ───────────────────→ 下轮 QK/P@V
当前 V ─────────────────────────────────────── 当前 P@V ─┘
```

图中的两条分支表示数据依赖可并行，不表示硬件已经实现了重叠。代码中 next K/V 的地址只依赖下一块索引，和当前 tile 的 `m/l/U` 无关；这给编译器一个机会，把普通 global load 提前发射，并尝试与当前 tile 的 softmax 归约、指数计算、旧 `U` 重标定和 P@V 指令区间重叠。`tl.load` 不是这里显式控制的异步 copy；源码的文本顺序不能证明访存已提前发出或与运算重叠。K 的逻辑最后使用点是本轮 QK，V 的逻辑最后使用点是本轮 P@V，P 的逻辑最后使用点也是本轮 P@V；如果矩阵操作异步执行，物理缓冲区还必须等该操作完成后才可覆盖。`k_next/v_next` 则必须存活到下一轮相应的 QK/P@V。这里用 Triton 张量值表达活跃 tile，不手动覆盖同一块 shared-memory 槽位，也没有自行实现 copy commit/wait 或 MMA barrier。

在线状态依赖不能被前视加载打乱。对当前 tile，必须先由当前 score 得到 `m_new`，再确定 `alpha` 和 `p`；旧 `U` 以 `alpha` 转到新指数基准后，才能加上当前 `P@V`。`l` 采用同一 `alpha` 和 `p`，`m` 最后前进。把下一块的 `P@V` 提前累加到当前 U，或省略旧 U 的重标定，都改变了结果，而不只是引入舍入误差。

以下是 Triton kernel 中实际执行的循环主体，包含 next-tile loads 和完整的 `m/l/U` 更新：

<!-- source-check: examples/prefill_pipeline.py -->
~~~python
        for kv0 in range(0, N, BLOCK_KV):
            k_abs = kv0 + kv_rel
            key_valid = k_abs < N
            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE

            # Look one tile ahead. These K/V loads are independent of the
            # current tile's softmax/P@V arithmetic and remain live for the
            # next loop iteration. They are ordinary loads, not async copies.
            next_abs = kv0.to(tl.int64) + BLOCK_KV + kv_rel
            next_offsets = kv_base + next_abs[:, None] * D + d[None, :]
            next_mask = (next_abs < N)[:, None] & d_valid[None, :]
            k_next = tl.load(k_ptr + next_offsets, mask=next_mask, other=0.0)
            v_next = tl.load(v_ptr + next_offsets, mask=next_mask, other=0.0)

            valid_score = q_valid[:, None] & key_valid[None, :]
            if CAUSAL:
                valid_score = valid_score & (k_abs[None, :] <= q_abs[:, None])
            scores = tl.where(valid_score, scores, float("-inf"))
            block_m = tl.max(scores, axis=1)
            m_new = tl.maximum(m, block_m)
            safe_m = tl.where(m_new != float("-inf"), m_new, 0.0)
            alpha = tl.where(m != float("-inf"), tl.exp(m - safe_m), 0.0)
            p = tl.where(valid_score, tl.exp(scores - safe_m[:, None]), 0.0)

            # Recurrence order is mandatory: rebase old U, then add this tile's
            # P@V; l follows the same alpha and exponent base; m advances last.
            u = alpha[:, None] * u + tl.dot(p, v, input_precision="ieee")
            l = alpha * l + tl.sum(p, axis=1)
            m = m_new
            k, v = k_next, v_next
~~~

这段是一次 tile 的 lookahead 调度，不等于 Hopper/Blackwell 的异步 shared-memory 双缓冲。`tl.load` 返回的值何时真正发出、等待在哪里插入、是否和当前计算重叠，由 Triton IR 到后端代码生成决定；寄存器活跃区间也可能延长，造成寄存器压力或 spill。源码中的先后顺序只证明存在独立数据依赖，不证明生成的 SASS 保留该时序，更不证明延迟被隐藏或耗时下降。复核时要固定 GPU、shape、FP32、causal 语义和 baseline，在 Triton IR/TTGIR 与 PTX/SASS 中追踪 next K/V load、当前 dot/softmax/PV 的指令顺序与等待，再用 Nsight Compute Source/SASS 视图观察 global load、寄存器、spill、occupancy 与相关 stall；最终以同条件计时为准。[Triton `tl.range` 的 loop `num_stages` 说明](https://triton-lang.org/main/python-api/generated/triton.language.range.html)区分 loop 属性与 launch 的 `num_stages`：前者请求编译器把循环 pipeline 到多次迭代在途，launch 参数主要 pipeline 喂给 dot 的 loads；本实现两者都没有设置，而是在循环体内显式表达 lookahead。Triton Gluon 的 [NVIDIA async-copy 教程](https://triton-lang.org/main/getting-started/tutorials/gluon/async-copy.html)展示了真正的 global-to-shared 异步拷贝和多缓冲机制，本例未使用它，也不是 FlashAttention-3 复现。`warp_specialize` 亦未启用；当前文档把它限定在 Blackwell 且只适用于简单 matmul loop，不能据此声称任意 Attention loop 已有 warp specialization。[Nsight Compute Source 页面](https://docs.nvidia.com/nsight-compute/NsightCompute/)可关联 source、PTX 与 SASS，实际可见项依报告与编译信息而定。

同一独立文件还实现题面形状 `solve(Q,K,V,out,M,N,d)`：单 batch、单 head、矩形 M×N、noncausal、FP32；示例 `d=4` 会在 kernel 内补零到至少 16 个 dot 维度，但 softmax scale 仍使用原始 `d`。接口与核心 shape 检查如下：

<!-- source-check: examples/prefill_pipeline.py -->
~~~python
def solve(Q, K, V, out, M: int, N: int, d: int):
    """LeetGPU #6 contract: FP32 noncausal Q[M,d], K/V[N,d] -> out[M,d]."""
    _require_runtime()
    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError("Q/K/V must be rank-2")
    if tuple(Q.shape) != (M, d) or tuple(K.shape) != (N, d) or tuple(V.shape) != (N, d):
        raise ValueError("tensor shapes do not match M, N, d")
    if tuple(out.shape) != (M, d):
        raise ValueError("out must have shape [M,d]")
    if M <= 0 or N <= 0 or d <= 0 or d > 128:
        raise ValueError("M, N, d must be positive and d <= 128")
    if any(x.dtype is not torch.float32 for x in (Q, K, V, out)):
        raise TypeError("solve is FP32-only")
    if not (Q.is_cuda and K.is_cuda and V.is_cuda and out.is_cuda):
        raise ValueError("Q/K/V/out must be CUDA tensors")
    if not (Q.device == K.device == V.device == out.device):
        raise ValueError("Q/K/V/out must use the same CUDA device")
    if not (Q.is_contiguous() and K.is_contiguous() and V.is_contiguous() and out.is_contiguous()):
        raise ValueError("Q/K/V/out must be contiguous")
    if any(_ranges_overlap(out, x) for x in (Q, K, V)):
        raise ValueError("out must not overlap Q/K/V")
    pad_d = max(16, triton.next_power_of_2(d))
    with torch.cuda.device(Q.device):
        _attention_lookahead_kernel[(triton.cdiv(M, BLOCK_Q), 1)](
            Q, K, V, out, 1, 1, M, N, d,
            BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, PAD_D=pad_d,
            SCALE=1.0 / math.sqrt(d), CAUSAL=False, num_warps=4,
        )
    return out
~~~

Prefill 入口只接受相同的连续 `[B,H,S,D]` Q/K/V，可选同长度 causal；`solve` 是单 head 矩形 noncausal 接口。两者均为 IEEE FP32 路径，没有用 FP16/BF16 或 TF32 切换换取表面性能，也都不支持 GQA（`Hq != Hkv`）、varlen、dropout、backward 或自定义 mask。CPU 检查独立验证分块递推、`m/l/U` 数值、causal 有效区、部分尾块和全 mask 零状态；GPU 程序另测两个 kernel 对 PyTorch FP32 reference 的正确性，并在 benchmark 中并列现有 FP32 Triton baseline、lookahead 版本和 dense PyTorch FP32 reference。

### 6.8 作者实现的 work tile、warp 所有权与 stride

固定复核两个作者实现：FlashAttention-2 使用 Dao-AILab `v2.7.4` commit `979702c87a8713a8e0a5e9fee122b90d2ef13be5`；FlashAttention-3 固定在同一官方仓库 commit `8d3a3b80d4758ebde5a867c50d24d4351443cf2b`。这里描述的是这些版本中的 CUDA/CuTe 实现，不把较新的 main 分支行为倒推到旧代码，也不把本文 FP32 Triton 教学 kernel 等同于 FA2/FA3。

对一个 head，令

$$
Q\in\mathbb{R}^{S_q\times D},\qquad K\in\mathbb{R}^{S_k\times D},\qquad V\in\mathbb{R}^{S_k\times D_v},\qquad O\in\mathbb{R}^{S_q\times D_v}.
$$

一个 work tile 固定一个 batch/head 和 `B_Q` 个 query 行；它扫描若干 `B_K` 个 key/value 行。CTA 的 query 维 block id 决定 `m_block`，因此有效 query 行 `r` 的拥有者可写成 `(b,h,floor(r/B_Q))`。在常规 FA2 前向路径中，不同 Q row-block 进入不同 CTA；一个 CTA 内各 warp 协作完成该 Q tile 的 QK、行 softmax 与 PV，不需要先把多个 CTA 的 partial output 合并。FA2 CUDA 代码把 query 行交给 mask 的一个坐标表达式

```cpp
m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4
```

这里的 warp id 与 warp 内 4-lane 小组共同构成行坐标，说明它不是“一 warp 永远对应一行”。具体每个 lane 持有哪几个 score/output 元素还要结合 `TiledMma` 的 accumulator layout 与拷贝分区；不能仅凭上面一项坐标式推导寄存器映射。

两份作者 kernel 都把 logical shape 与 element stride 分开传递。通用地址写成

$$
offset(b,h,s,d)=b\,\sigma_b+h\,\sigma_h+s\,\sigma_s+d\,\sigma_d.
$$

例如原张量是连续 `[B,S,H,D]`，之后转成 `[B,H,S,D]` view，其 view strides 为 `(S H D, D, H D, 1)`；这不是连续 BHSD 的 `(H S D, S D, D, 1)`，但在 stride 正确时仍可寻址同一组元素。FA2 的 kernel 为 Q/K/V 分别使用 batch、row 和 head stride 来建立 tensor，head dimension 的相邻 feature 仍按连续元素处理；FA3 也把 `shape_Q/K` 与 `stride_Q/K` 分开交给 CuTe/TMA descriptor。于是“支持 stride view”不等于任意负 stride、重叠 view 或任意 feature stride 都合法：必须符合对应 descriptor 的 alignment、major mode 与内层连续性约束。当前教学 wrapper 仍明确只接受 contiguous `[B,H,S,D]`；不能把作者库的通用 stride 能力偷偷归入本 wrapper。

### 6.9 FA2：mask → 在线重标定 → P@V 的代码次序

FA2 的 K-block 区间由 Q block、实际 Q/K 长度、causal/local 窗口和 split 状态共同确定。边缘迭代先应用 key padding 与 causal mask，再更新行最大值；稳定权重可以写成 `p=exp(score-m_new)`，旧状态用 `alpha=exp(m_old-m_new)` 转到同一指数基准。作者的实现调用顺序中，`softmax_rescale_o` 先更新 softmax 行状态并重标定输出累加器，随后把当前 score fragment 转为 P 并交给 PV：

```cpp
? softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
```

其数学状态为

$$
m_t=\max(m_{t-1},\max_j s_{t,j}),\qquad
\alpha_t=\exp(m_{t-1}-m_t),\qquad
p_{t,j}=\exp(s_{t,j}-m_t),
$$

$$
l_t=\alpha_tl_{t-1}+\sum_jp_{t,j},\qquad
U_t=\alpha_tU_{t-1}+\sum_jp_{t,j}V_{t,j},\qquad O=U_T/l_T.
$$

操作次序很重要：`QK → mask → 更新 m/l 并重标定旧 U → 形成 P → P@V`。若在第二个 tile 中省略 `alpha*U_old`，就是把两个指数基准下的量直接相加。FA2 的源码按分开的 masking iterations 与 unmasked iterations 组织循环：causal 对角附近的多个 KV block 要逐块 mask，序列尾 block 也要对不完整 key tile 做 predicate；进入完全位于有效区的 block 后才可走免 mask 路径。只将尾部 K/V load 补零而不把 score 设为 `-inf`，仍会把虚构概率质量加入分母。

KV block 的物理生命周期也不相同。K 的矩阵读取者是 QK；V 的读取者是 PV；P 由 score/softmax 阶段产生，最后交给 PV。若 K/V 共用一个槽，安全复用点取两者最后读者的较晚者，也就是本轮 PV 完成；若分开管理，可在 QK 最后一次真实读取完成后回收 K，但必须确保异步矩阵操作已停止读取其地址。FA2 中 P 由 score fragment 转换为输入类型后送往 `gemm_rs`；这一路径不会因为变量名 P 就代表全局的 `[S_q,S_k]` 概率矩阵，它只是在寄存器/片上 tile 间短暂存活。

### 6.10 FA3：producer/consumer warpgroup、TMA 与 P 的位置

固定 commit 的 FA3 Hopper 前向 kernel 显式按 warpgroup 区分 producer 与 MMA consumer。作者 kernel 以 `warp_group_idx==0` 进入 producer 分支，其他参与组进入 consumer；其中 `NumMmaWarpGroups` 由 PV tiled MMA 的线程数除以 128 得出，可取 1、2 或 3。producer 的实际线程数还会随 `Transpose_V`、是否对 Q 使用 TMA 等编译期条件在一个 warp 与一个 warpgroup 之间变化。因此应分别说“多少个 producer/consumer warpgroup”和“producer branch 实际有多少线程”，不能把两者都简化成一个固定 warp 数。

```cpp
if (warp_group_idx == 0) { // Producer
```

producer 与 consumer 分别初始化 pipeline：K、V 各有自己的 stage 状态；每次 load 先 acquire 对应的可写代，再发 TMA 到其 shared destination，或在 paged/non-TMA 分支用 thread copy 并 commit。consumer 只在对应 pipeline ready 后才能用 QK/PV。FA3 在 SM90 路径中按 compile-time tile helper 选 `B_Q/B_K`，并设两级 K/V stage；这属于该固定实现路径的选择，不代表任意其他 FA3 commit 或其他硬件自动也是相同参数。

V 可能有额外转换阶段。`Transpose_V` 时 TMA 先将 V 放入 `sVt`；producer warpgroup 再经 shared-to-register、byte permutation、register-to-shared 写入 MMA 所需 `sV`，发出 shared-memory fence、提交 V pipeline，并同步 warpgroup 后才能释放 `sVt`。如果把 `sVt` 看成可在 TMA 返回时立即复用，就会与转换线程仍在读 shared 的事实冲突。非转置且启用 intra-warpgroup overlap 的分支会让 V load 与 K 的相邻 block 调度错开；这不是普通 attention 数学递推的重排许可，softmax 状态仍需按 score tile 的因果顺序更新。

FA3 的 softmax 将 `softmax_scale*log2(e)` 作为指数尺度，因而用 `exp2` 表达与自然指数相同的稳定重标定。源码在更新 row maximum 后先缩放旧归一化因子：

```cpp
scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
row_sum(mi) *= scores_scale(mi);
```

随后当前 tile 的 score 以新的 max 为基准转换成指数权重并进入 row sum；同一个 `scores_scale` 也用于重标定寄存器中的输出累加器，然后才能加入当前 P@V。最终的 `finalize` 对 row sum 做 warpgroup 内归约，以倒数归一化输出，并生成每行 log-sum-exp。FA3 的两项注意事项是：1) 这些是 block-online 状态，不是先保存完整 softmax；2) FP8 分支可能改变缩放和指数偏移，不能把它的低精度数值路径与本章 IEEE FP32 baseline 混成一张性能/误差表。

P 是否驻留 shared 由 PV MMA 的 RS/SS specialization 决定。`MmaPV_is_RS` 为真时，P 留在寄存器 fragment，shared-storage 类型不分配 `smem_p`；为假时建立 `smem_p`，P 的 shared 写入和 PV shared 读取构成必须维护的 producer/consumer 生命周期。输出 `tOrO` 则是寄存器 accumulator。大 `D_v`、FP8、V transpose 与 `MmaPV_is_RS` 的组合会受静态约束，例：FA3 代码对 `D_v>256` 要求相应 head-dim 对齐、`B_Q≤64` 且 PV 走 SS；不是任意把 `B_Q` 或 `D_v` 调大都仍可实例化。

### 6.11 causal offset、空块、小 shape 与退化判断

同长度 self-attention 的 causal 条件是 `k≤q`。作者 FA2/FA3 接口对不同 Q/K 长度采用右下角对齐；局部索引的条件是

$$
0\le q<S_q,\qquad 0\le k<S_k,\qquad k\le q+(S_k-S_q).
$$

例如 `S_q=2,S_k=5` 时，两行分别可见 keys `[0,1,2,3]` 与 `[0,1,2,3,4]`；这和“Q/K 都从局部位置 0 开始、直接用 `k≤q`”不同。前缀缓存、变长 `cu_seqlens`、左 padding 和 paged KV 还要把物理地址偏移与 rotary/causal 的位置偏移分开。对 causal tile，CTA 先算出 `n_block_min/n_block_max`；如果没有与该 Q tile 相交的有效 KV block，作者实现有专门的 invalid/zero-output 路径，不应发出越界 TMA 或把未初始化 shared 当零。

小 shape 的主要退化原因常是任务不足，而非算术复杂度式改变。方形 dense prefill 的 Q-tile 任务数约为

$$
N_{CTA}=B\,H_q\left\lceil S_q/B_Q\right\rceil.
$$

当 B、heads、`S_q` 都小，`N_CTA` 可能远小于可并行处理的 SM 数；每个 CTA 仍承担 TMA、barrier、online softmax 与 epilogue 管理成本，且最后一个 Q/KV block 的有效比例可能很低。增大 `B_Q` 会减少 CTA 数和 K/V 重读次数，却增加 Q、P、U 的活跃量；增大 `B_K` 会减少循环次数，却增大 score/P tile 和 V 生命周期；在 P 使用 shared 的分支里，还会额外提高 shared 峰值。给定 `B_Q,B_K,D,D_v`，可先估计：

$$
S_K\approx n_{stage}B_KD\,bytes_{elem},\qquad
S_V\approx n_{stage}B_KD_v(1+I_{transpose})\,bytes_{elem}.
$$

$$
S_P\approx\begin{cases}0,&MmaPV\_is\_RS\\ B_QB_K\,bytes_{elem},&\text{PV 用 shared P}\end{cases},\qquad
R_O\propto B_QD_v\cdot4\text{ bytes}.
$$

其中 `I_transpose=1` 表示该分支同时保留 TMA 目标 `Vt` 与供 PV 使用的 `V` tile；非转置分支取 0。`S_K`、`S_V`、`S_P` 是各逻辑缓冲的容量估算，不代表三者一定同时占用或在同一段 shared storage 相加；真实 shared layout 有对齐、swizzle 与临时区，FA3 kernel 还可能把 epilogue output 与 V 的 shared 区域在不重叠的生命周期内复用。Q shared、barrier 和编译器临时空间也须另核。寄存器由 QK/PV warpgroup 拆分与 ptxas 分配决定；producer/consumer 分配策略会变，不能从元素字节数直接推出 occupancy。应先检查编译报告的 registers/thread、shared/CTA、spill 和所选 `NumMmaWarpGroups`，再用相同输入精度、mask、shape、stride 和计时范围测性能。本文没有这些 GPU 编译/计时数据，不把资源估算写成实测退化比例。

CPU-only 检查把 stride 地址、Q row owner、right-aligned causal mask、online recurrence 与 K/P/V 的最晚读者作为可执行合同；它不模拟真实 warp、TMA、WGMMA 或硬件 stage 时序：

```bash
python roadmap/curriculum/operators/05-prefill-attention/examples/cpu_author_attention_checks.py
```

### 6.12 作者实现的现场验证命令

FA2 pinned README 标注 CUDA 12.0+ 与受支持的 NVIDIA GPU；FA3 在该 pinned commit 的说明中要求 H100/H800 与 CUDA 12.3+。两者均需匹配的 PyTorch/CUDA 环境。以下命令各自固定源码 commit，并运行作者测试；FA2 与 FA3 的 Python 包/API 可能冲突，建议分开虚拟环境。测试通过只说明该测试矩阵正确性通过，不提供本章自写 kernel 的性能数字。

```bash
# FA2 v2.7.4 author tests
git clone https://github.com/Dao-AILab/flash-attention.git flash-attention-fa2
git -C flash-attention-fa2 checkout 979702c87a8713a8e0a5e9fee122b90d2ef13be5
(
  cd flash-attention-fa2
  python setup.py install
  pytest -q -s tests/test_flash_attn.py
)

# FA3 pinned Hopper implementation tests
git clone https://github.com/Dao-AILab/flash-attention.git flash-attention-fa3
git -C flash-attention-fa3 checkout 8d3a3b80d4758ebde5a867c50d24d4351443cf2b
(
  cd flash-attention-fa3/hopper
  python setup.py install
  export PYTHONPATH="$PWD"
  pytest -q -s test_flash_attn.py
)
```

在作者代码支持的 shape/mask/dtype 组合上，再与同语义 PyTorch math/reference 比较输出与误差；逐项改变 `B_Q/B_K`、GQA/varlen、causal 与 head dimensions 时，先看实际 dispatch 选择，再检查 registers/thread、shared memory、spill 和 kernel 时间。真实构建、设备 correctness、计时与 profiler 需在目标设备现场记录，本页的 CPU 通过不替代其中任何一项。

## 7. 精度路线和官方源诊断

本章 baseline 固定 FP32 输入、FP32 accumulator、`tl.dot(..., input_precision="ieee")`。验证参考用 `torch.matmul`、TF32 关闭、full FP32 score/softmax/PV，容差 `rtol=atol=1e-4`。FP16/BF16 Tensor Core 路线应另列输入 bytes、乘法精度、累加 dtype、转换位置和容差；不能拿 FP32 baseline 的 FLOPs/bytes 表冒充半精度性能，也不能把“accumulator 是 FP32”写成所有中间乘法都是 IEEE FP32。

官方 Triton 当前 [06-fused-attention.html](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) 的实现会在 PV 前把概率转换到与 V 兼容的 dtype；如果选择 `exp2`，score 还要先乘 `log2(e)`。本章 baseline 用自然 `exp`，不把两条路径混写。旧 reference 的问题保留为诊断，不改旧源：

<!-- source-check: reference/triton/flash_attention/flash_attn.py -->
~~~python
    pid = tl.program_id(0)
    q_start = pid * BLOCK_Q
~~~

旧 wrapper 的注释是 `[B,S,D]`，但实际调用片段如下：

<!-- source-check: reference/triton/flash_attention/flash_attn.py -->
~~~python
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Forward: O = softmax(QK^T / sqrt(d)) × V"""
    assert q.dim() == 3  # (batch, seq, dim)
    B, seq_len, head_dim = q.shape

    o = torch.empty_like(q)
    BLOCK_Q = 32
    BLOCK_KV = 64
    grid = (triton.cdiv(seq_len, BLOCK_Q),)

    flash_attn_kernel[grid](
        q, k, v, o,
        seq_len, head_dim,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV,
    )
    return o
~~~

因此它把 `[B,S,D]` 的 `stride(0)` 传作 sequence stride、`stride(1)` 传作 feature stride，却没有把 batch stride 作为 kernel 的 base，也没有在 grid 中展开 batch。kernel 中 `head_dim` 不是 constexpr 却用于 `tl.arange`；K tail load 为0却没有把对应 score mask 成 `-inf`；`p` 是 FP32 而 `v` 是 FP16 时，`tl.dot` 输入 dtype 合同也没有处理。这些是可定位的 shape、边界和 dtype 问题，不是把两行 `pid` 代码换成四维就自动修好的。

旧 CUDA 教学 reference 的 tail-Q 诊断同样不能称完整 FA2：

<!-- source-check: reference/cuda/flash_attention/flash_attn.cu -->
~~~cpp
  if (qr >= (q_end - q_start)) return;
~~~

它让无效 query 线程不再参与后续 KV tile cooperative load；tail tile 中可能缺失本应由这些 tid 搬运的元素，剩余线程读取未初始化 shared 数据。该问题是协作加载和数据完整性问题，不是对所有 CUDA `return` 的笼统结论。

## 8. 当前题面与本章实现的关系

当前 [AlphaGPU leetgpu-challenges 总入口](https://github.com/AlphaGPU/leetgpu-challenges) 中，准确题名为 **#6 Softmax Attention**，题面接口是 `Q[M,d], K[N,d], V[N,d], out[M,d], solve(Q,K,V,out,M,N,d)`，FP32、noncausal、`rtol=atol=1e-4`，示例是 `M=2,N=3,d=4`。它是矩形单 batch/单 head 题，不是本章的 `[B,H,S,D]` 方形 causal wrapper；本章代码不能直接提交并声称适配 #6，也不沿用旧 reference 的错误 `[B,S,D]`/`N,d` 索引合同。独立 `reference_leetgpu_6` 只测试矩形数学：Q/K 全0、V 为三行不同已知常数时，输出应为三行均值；它不把 square kernel 伪装成 #6 solve。

[#80 Grouped Query Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/80_grouped_query_attention) 是 `Q[Hq,S,D]、K/V[Hkv,S,D]`、无 batch、noncausal，使用 `kvhead=qhead//(Hq/Hkv)`；[#61 RoPE Embedding](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/61_rope_embedding) 是 `Q/cos/sin/output[M,D]`，split-half，cos/sin 由输入提供。三者都是 free challenge，但题面入口、题名、shape、容差各自独立；平台实现必须从题面空白编辑器开始并原样归档当次 `solve`/kernel。

## 9. LeetGPU：正确性与代码归档

本章不把方形教学 kernel 当作 #6 的提交物。平台路线从 [LeetGPU 官方题目入口](https://leetgpu.com/challenges) 搜索准确题名 **Softmax Attention**，再核对公开题面的精确目录 [6_softmax_attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/6_softmax_attention) 和 `solve(Q,K,V,out,M,N,d)` 合同；之后再单独完成 #80 GQA 与 #61 RoPE（如果进入这些题）。GitHub 是公开题库源，不是平台入口。每个题面都要保存当次平台原始代码，记录 shape、FP32、容差和免费题面链接；平台通过只由对应题面结果证明。

## 10. 服务器：真实性能

`examples/prefill_pipeline.py` 是独立完整程序：默认校验既有 FP32 Triton baseline、lookahead kernel、PyTorch dense FP32 reference 及强库对照 PyTorch SDPA math backend；另测矩形 `solve(Q,K,V,out,M,N,d)`、Q/K tails、batch/head offsets 和 causal future-key isolation。支持的 square prefill 形状为连续同形 `[B,H,S,D]`、`D∈{16,32,64,128}`，FP32 IEEE dot；causal 只支持相同 Q/K 长度。它不支持 GQA、varlen、dropout 或 backward。benchmark 预热后以同步 host wall time 比较两个 Triton wrapper，二者都传入预分配 output，因而是直接同边界比较；SDPA 与 dense reference 都返回新 tensor，单独报告其分配开销，不能拿该差值宣称 kernel 加速。无 PyTorch/Triton 或无 CUDA device 时程序返回 77（SKIP）；`--help` 不要求 GPU 运行。

## 运行入口

从仓库根目录运行：

~~~bash
python roadmap/curriculum/operators/03-gemm/examples/cpu_gemm_checks.py
python roadmap/curriculum/operators/04-activation-and-fusion/examples/rope_qknorm_reference.py
python roadmap/curriculum/operators/05-prefill-attention/examples/cpu_attention_checks.py
python roadmap/curriculum/operators/05-prefill-attention/examples/cpu_pipeline_checks.py
python roadmap/curriculum/operators/05-prefill-attention/examples/cpu_author_attention_checks.py

# 需要 CUDA、PyTorch 和 Triton；默认同时检查 causal/noncausal correctness。
python roadmap/curriculum/operators/05-prefill-attention/examples/validate_prefill.py

# lookahead 实现的完整接口、正确性测试与命令行帮助；无 GPU 返回 77（SKIP）。
python roadmap/curriculum/operators/05-prefill-attention/examples/prefill_pipeline.py --help
python roadmap/curriculum/operators/05-prefill-attention/examples/prefill_pipeline.py

# 预热并排除 JIT/allocation，再测预分配 output 的 wrapper 墙钟平均。
python roadmap/curriculum/operators/05-prefill-attention/examples/validate_prefill.py --benchmark
python roadmap/curriculum/operators/05-prefill-attention/examples/validate_prefill.py --benchmark --causal

# 同精度对比现有 Triton baseline 与 lookahead kernel；另列 SDPA math 和 dense reference。
python roadmap/curriculum/operators/05-prefill-attention/examples/prefill_pipeline.py --benchmark
python roadmap/curriculum/operators/05-prefill-attention/examples/prefill_pipeline.py --benchmark --causal --batch 2 --heads 8 --seq-len 1024 --head-dim 64

# 可复现的长序列 wrapper 墙钟形状；只在 --benchmark 时读取这些参数。
python roadmap/curriculum/operators/05-prefill-attention/examples/validate_prefill.py --benchmark --batch 2 --heads 8 --seq-len 256 --head-dim 64
python roadmap/curriculum/operators/05-prefill-attention/examples/validate_prefill.py --benchmark --batch 2 --heads 8 --seq-len 1024 --head-dim 64
~~~

`validate_prefill.py` 的长序列示例仍用于观察原教学 baseline；`prefill_pipeline.py --benchmark` 则固定 shape、mask、FP32 IEEE 算术和预分配输出，直接比较 baseline 与 lookahead。PyTorch SDPA 被限定到 math backend 并关闭 TF32，保持 FP32 数值路线；它返回新输出，报告单列且不与 Triton 的预分配输出时间混算。dense reference 会显式构造 `S×S` score/prob 逻辑，显存随 `S^2` 增长，仅作为独立参考路径，不把它写成 Flash kernel 的显存占用或优化收益证据。

## 参考阅读

[FlashAttention-2 论文](https://tridao.me/publications/flash2/flash2.pdf) · [FA2 作者实现目录（固定 commit）](https://github.com/Dao-AILab/flash-attention/tree/979702c87a8713a8e0a5e9fee122b90d2ef13be5/csrc/flash_attn/src) · [FlashAttention-3 论文](https://arxiv.org/abs/2407.08608) · [FA3 Hopper 实现目录（固定 commit）](https://github.com/Dao-AILab/flash-attention/tree/8d3a3b80d4758ebde5a867c50d24d4351443cf2b/hopper) · [CUDA Programming Guide 13.3](../../../../downloads/cuda-programming-guide.pdf)。

## 章节导航

- [上一章：Activation 与 Fusion](../04-activation-and-fusion/README.md)
- [算子总览](../README.md)
- 下一章：[Decode / PagedAttention](../06-decode-paged-attention/README.md)
