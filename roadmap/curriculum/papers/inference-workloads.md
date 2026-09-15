# Scaling Law、强化学习与长任务推理

同一块 GPU 可以执行预训练前向、生成训练样本，也可以处理在线请求。三种场景都运行矩阵乘和 Attention，却不一定应该优化同一个指标。要判断某个 kernel 的价值，先弄清它服务的是训练预算、数据生成速度，还是用户等待时间。

本节把模型规模、训练信号和推理系统连接起来。数学例子使用缩小的数据和明确假设，不把某个框架在某次实验中的速度表当成通用排名。

## 1. Prefill/Decode 与训练/推理不在同一个分类轴上

Prefill 和 Decode 描述自回归生成时如何处理 token。训练和推理描述是否利用损失更新参数。RL（Reinforcement Learning，强化学习）训练中有生成样本的 rollout，因此一个训练系统内部也会反复执行 Prefill 和 Decode。

| 阶段 | 输入和计算 | 损失或反馈 | 是否更新参数 |
|---|---|---|---|
| 预训练 | 已有序列，通常用 teacher forcing 同时计算多个位置 | 下一个 token 的负对数似然等 | 是 |
| SFT | 指令与示范回答，在指定 token 上算损失 | 监督信号 | 是 |
| RL rollout | 用行为策略逐步生成回答或工具调用 | 收集轨迹，之后评估奖励 | 生成阶段本身通常不做梯度更新 |
| RL update | 回放轨迹，计算概率和目标函数 | 奖励、优势、正则项等 | 是 |
| 在线推理 | 已部署权重执行请求 | 对外返回结果 | 普通生成步骤不更新权重 |

SFT 是 Supervised Fine-Tuning（监督微调）。teacher forcing 用已有 token 作为下一位置输入，所以大量位置可以并行处理；生成未知 token 时，下一步需要先知道上一步采到了什么。前者看起来像大矩阵 Prefill，不代表“训练就是 Prefill”。训练还要保留或重计算中间激活，执行反向、梯度通信与优化器更新。

SFT 不是只能记忆，RL 也不天然保证更强推理。两者的泛化、稳定性与收益依赖数据、奖励、优化过程和评估；不能把“会答/会想”当成算法边界。SFT 后接 RL 是常见安排，不是所有后训练路线必须遵循的唯一顺序。

## 2. Scaling Law：固定预算下为什么存在中间最优点

先用一个简化的 dense 模型损失拟合式：

$$
\mathcal L(N,D)=\mathcal L_\infty+
A N^{-\alpha}+B D^{-\beta},
\qquad C\approx\kappa ND.
$$

N 是模型参数数，D 是训练 token 数，C 是训练计算预算。A、B、α、β 是在指定数据、模型族和训练方法上拟合的系数；κ 把参数与 token 的乘积换成 FLOPs。常见 dense Transformer 粗算可用 κ≈6，但长序列 Attention、embedding、重计算等开销会使它不够精确。

固定 C 时，不能同时任意增大 N 和 D。令 X=C/κ，则 D=X/N，损失中与选择有关的部分为：

$$
f(N)=AN^{-\alpha}+B(N/X)^\beta.
$$

N 太小时，第一项大，模型容量不足；N 太大时，可用于训练它的 D 变少，第二项变大。求导：

$$
f'(N)=-\alpha A N^{-\alpha-1}
+\beta B X^{-\beta}N^{\beta-1}=0.
$$

移项并乘 N：

$$
\alpha A N^{-\alpha}=
\beta B X^{-\beta}N^\beta,
\qquad
N^{\alpha+\beta}
=\frac{\alpha A}{\beta B}X^\beta.
$$

所以：

$$
N^*=
\left[\frac{\alpha A}{\beta B}(C/\kappa)^\beta\right]^{1/(\alpha+\beta)},
\qquad
D^*=\frac{C}{\kappa N^*}.
$$

只有在 α≈β 时，两个量随 C 的幂次才都接近 1/2。某篇工作的拟合系数和 token/parameter 比例有实验语境，不能把“每参数固定配 20 token”当作所有数据、模型和生命周期成本下的定律。

<!-- source-check: examples/workload_checks.py -->
~~~python
def compute_optimum(compute, kappa, a, b, alpha, beta):
    if min(compute, kappa, a, b, alpha, beta) <= 0:
        raise ValueError("positive model constants required")
    product = compute / kappa
    n = ((alpha * a / (beta * b)) * product ** beta) ** (1 / (alpha + beta))
    return n, product / n
~~~

这段代码逐项对应闭式解。示例 C=240000、κ=6、A=4、B=1、α=β=1，得到 N=400、D=100。它使用人为系数检验代数，不是建议训练一个这样的模型。随附检查把结果和一维网格搜索比较，验证公式最小点没有写反。

### 2.1 为什么不能直接把这个公式套到 MoE

MoE 的总权重、每 token 激活矩阵、批内实际触达的专家集合，以及专家并行通信，是不同对象。不能把 dense 拟合式中的 N 一会儿换成总参数、一会儿换成激活参数，然后继续沿用同一组系数。

固定总参数下改变路由数，会同时改变每 token 计算、专家被训练到的频次和负载分布。要建立 MoE 的预算模型，至少记录共享层、激活专家、路由与通信；能力拟合也需要相应实验。稀疏激活不会免除未激活权重的保存与放置问题。

## 3. 训练最优与部署最优为什么不同

训练只花一次，服务可能产生大量 token。把两种候选模型的生命周期成本写成：

$$
C_{\mathrm{life}}=C_{\mathrm{train}}+
Q\,c_{\mathrm{serve}},
$$

Q 为预期服务工作量，c_serve 是相同质量和服务约束下的单位成本。单位可以是货币或 GPU 时间，但不能一项用 FLOPs、一项用秒而直接相加。

若较小模型多训练一次的成本增量为 ΔC_train，而每单位服务节省 Δc_serve>0，那么仅按此模型估算的盈亏平衡点为：

$$
Q_{\mathrm{break-even}}=
\frac{\Delta C_{\mathrm{train}}}{\Delta c_{\mathrm{serve}}}.
$$

例如额外训练成本是 100 个预算单位，每百万输出 token 节省 2 个单位，则 50 百万 token 是算术平衡点。真实决策还要纳入质量是否达标、输入长度、响应延迟、容量利用率、模型更新频率和工程维护成本。参数少不保证在所有小 shape 上 kernel 都更高效。

增加数据量也可能遇到重复、质量退化或分布不匹配。因此“多训小模型”是一种可比较方案，不是保证同时省钱和变强的规则。

## 4. RL 的生成循环里，各个 tensor 在做什么

设一个 prompt 为 x，回答为 y=(y_1,...,y_T)。行为策略 μ 在生成时采样：

$$
y_t\sim\mu(\cdot\mid x,y_{<t}).
$$

生成后，环境、规则验证器或奖励模型给出 R(x,y)。更新阶段希望增加高奖励轨迹的概率，最简单的目标是：

$$
J(\theta)=E_{y\sim\pi_\theta}[R(x,y)].
$$

实际算法还可能包含 KL 约束、价值估计、裁剪和组内比较。不能把整段回答奖励直接当成每个 token 的“正确标签”；它提供的是优化信号，梯度通过模型对实际生成动作的 log probability 计算。

对于同一 prompt 的 G 条回答，可以先构造简单的组内中心化优势：

$$
A_i=R_i-\frac1G\sum_{j=1}^{G}R_j.
$$

有的实现再除以标准差，有的只中心化，分母、ε 和归一化层次要以具体算法为准。不同 prompt 难度不同，不能为了凑 batch 就把它们的奖励随意混组。若一组奖励都相同，这种中心化优势为零，说明这一组没有提供相对区分信号。

### 4.1 为什么需要保存旧 log probability

更新时参数可能已经变成 θ_new，而动作来自 μ。对采到的动作计算比率：

$$
r_t=
\frac{\pi_{\theta_{\mathrm{new}}}(y_t\mid h_t)}
{\mu(y_t\mid h_t)}
=
\exp\bigl(\log\pi_{\theta_{\mathrm{new}}}(y_t\mid h_t)
-\log\mu(y_t\mid h_t)\bigr).
$$

h_t 包含该动作之前的完整上下文。只保存 token ID 而不保存行为策略的概率和身份，之后不能从最新模型可靠恢复分母。

一个 token-wise clipped surrogate 可写作：

$$
J_{\mathrm{clip}}=
\frac1{|\mathcal M|}\sum_{t\in\mathcal M}
\min\left(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right).
$$

这里 M 是有效模型动作 token 集合。下面省略 KL、critic、分布式归一化和其他算法细节，只演示比率、裁剪和 mask：

<!-- source-check: examples/workload_checks.py -->
~~~python
def clipped_objective(new_logp, old_logp, advantage, valid, clip=0.2):
    new, old, adv, mask = [np.asarray(x) for x in (new_logp, old_logp, advantage, valid)]
    if not (new.shape == old.shape == adv.shape == mask.shape) or clip < 0:
        raise ValueError("aligned token arrays and nonnegative clip required")
    mask = mask.astype(bool)
    if not mask.any():
        raise ValueError("no valid action tokens")
    ratio = np.exp(new[mask] - old[mask])
    unclipped = ratio * adv[mask]
    clipped = np.clip(ratio, 1 - clip, 1 + clip) * adv[mask]
    return float(np.minimum(unclipped, clipped).mean())
~~~

A>0、r 很大时，裁剪限制收益上界；A<0 时，min 的另一边可能更重要，不能只把 ratio 截断后乘 advantage 代替整个式子。

工具返回的文本可以成为下次模型输入，却不是模型自主采样的动作。它通常不应该直接作为 policy-gradient 的动作 token。padding、被排除的过旧 token 和不同 loss 项各自的 mask 也应分清。训练 logits 可具有 [batch,sequence,vocab] 形状，但实际动作 log probability 是按 token ID gather 后的 [batch,sequence] 数组。

### 4.2 裁剪不是任意陈旧样本都安全的证明

若新旧策略差异很大，重要性比率可能极端、有效样本数变小。逐 token 裁剪是有偏的稳定化手段，不等于对任意旧策略数据恢复了最新策略的精确目标。

需要同时观察策略版本差、log-ratio 分布、KL、被 mask 比例、样本年龄和质量曲线。固定“允许旧多少步”并无通用答案：一次 update 的大小、学习率、任务分布和生成温度都影响策略变化。

### 4.3 一条 rollout 从请求到更新的状态流

把一次强化学习样本当作有版本的记录，而不是“prompt + 最终字符串”。一条可复核的路径如下：

| 阶段 | 随状态一起保存 | 进入下一阶段的条件 | 不能提前做的事 |
|---|---|---|---|
| 派发 | `group_id`、`sample_id`、数据集、prompt、行为策略版本 | worker 接到明确版本 | 不能因任务完成顺序重建 prompt group |
| 生成 | 动作 token ID、逐 token `old_logp`、loss mask、KV owner/version | EOS、长度上限或工具动作结束 | 不能用更新后的策略概率覆盖 `old_logp` |
| 环境等待 | 工具请求 ID、输入/输出、超时/取消状态、尚存 KV 引用 | 所有依赖的工具结果已回填，或明确标失败 | 主机发出取消不代表设备和环境资源已释放 |
| 验证 | reward、验证器版本、终止原因、是否可训练 | 样本可接受且属于原 group | 工具输出不能自动变成 policy action token |
| 组装 | 同一 prompt 的 G 个样本及其版本年龄 | 算法要求的组内统计/归一化条件满足 | 不得把不同 prompt 的奖励混为一组 |
| 更新 | 新策略 logp、ratio/KL、有效 token mask、陈旧样本策略 | loss 计算与版本门槛通过 | 不能声称任意旧策略数据经 clipping 就无偏 |
| 回收 | cache/page 引用、未完成异步通信与最后 consumer | 设备事件和工具依赖都结束 | 不能在提交异步 send/cancel 后立即复用 buffer |

异步派发允许 `sample_id=s2` 先于 `s0` 返回，但记录必须仍归入同一 `group_id`；若某个样本太旧而丢弃，组装器必须按算法规定补采或判组无效，不能悄悄用别组样本补数。每个样本保存生成版本下的 `old_logp`，训练版本重新计算 `new_logp`，二者才组成重要性比率。输入 token、环境返回文本、模型动作与 padding 分别有 mask；它们不是一个“序列长度”字段能概括的。

想象同一 prompt 的三个 rollout 以 s2、s0、s1 的顺序完成，reward 分别为 1、0、0.5；组均值为 0.5，于是优势是 `{s0:-0.5,s1:0,s2:0.5}`，无论返回次序如何都不变。若先到两个结果就立刻归一化，它们的相对奖励与最终三样本组不同，产生的是另一种训练批次。

可运行的 CPU 状态机同时检查乱序归组、old logp 身份、候选回滚和工具等待/KV 版本门槛：

~~~bash
python roadmap/curriculum/papers/examples/workload_lifecycle.py
~~~

它只检验状态转换和整数/tuple 账本，不启动 RL 框架，不执行真实 rollout，也不模拟网络或设备轨迹。

## 5. Rollout 的长尾为什么浪费计算资源

假设两条生成槽位上依次处理六条任务，时长为 [1,10,1,1,10,1]。若每两条组成一个同步批，下一批必须等待两条都完成，总时间为：

$$
T_{\mathrm{sync}}=10+1+10=21.
$$

若任一槽位空闲就取下一条，长任务执行时另一槽位可以继续消化短任务。在没有共享资源干扰和调度成本的玩具模型中，总时间为 13。

<!-- source-check: examples/workload_checks.py -->
~~~python
def worker_makespan(durations, workers, synchronous=False):
    durations = [float(t) for t in durations]
    if workers < 1 or any(not np.isfinite(t) or t < 0 for t in durations):
        raise ValueError("positive workers and finite nonnegative durations required")
    if synchronous:
        return sum(max(durations[i:i + workers]) for i in range(0, len(durations), workers))
    available = [0.] * workers
    heapq.heapify(available)
    for duration in durations:
        start = heapq.heappop(available)
        heapq.heappush(available, start + duration)
    return max(available)
~~~

heap 保存每个槽位最早空闲时间，不代表真实 GPU 内部调度。实际 continuous batching 会让新请求改变 batch 形状、每步计算量和 KV 压力，任务时长并非固定。这个模型只隔离“等待整批”造成的损失。

可以先用下面的下界检查任何调度结果：

$$
T\geq \max\left(\max_i t_i,\frac{\sum_i t_i}{P}\right),
$$

P 为独立槽位数。少于最长单任务时长或少于总工作量除以槽位数，说明计时口径或模拟写错了。

### 5.1 调度变快后，训练样本也可能变了

若训练一收够若干完成任务就开始更新，短回答在早期批次中更容易出现。工具慢的任务、长推理任务和某些数据域会被系统性延迟。这是选择偏差，不是仅靠更高 tokens/s 能解决的。

需要保留任务、prompt group、数据集、生成版本和完整结束原因。按数据域限制并发、等待组内必要样本、调整早期样本接纳策略、限制陈旧度，都是不同旋钮。不能在补充新请求时顺便重定义奖励比较组。

指标也应从“生成了多少 token”扩展到有效完成轨迹/s、可用于更新的动作 token/s、奖励合格率、长度分布、被丢弃或 mask 的比例，以及达到同一质量需要的总成本。

## 6. 同一个 prompt 的多次采样，能共享什么

G 条回答可以共享相同权重和前缀下的 KV，但生成后缀不同，不能一直共用同一组可写物理页。

设公共前缀 P 个 token，各回答后缀长度为 L_i，每 token 缓存字节为 b：

$$
M_{\mathrm{dup}}=b\left(GP+\sum_iL_i\right),\qquad
M_{\mathrm{shared}}=b\left(P+\sum_iL_i\right).
$$

理想节省为 b(G-1)P。这个比值只描述缓存数据，不等于吞吐提升；还没计入页尾、引用计数、copy-on-write 和元数据。

<!-- source-check: examples/workload_checks.py -->
~~~python
def prefix_storage(prefix_tokens, suffix_lengths, bytes_per_token):
    lengths = [int(x) for x in suffix_lengths]
    if prefix_tokens < 0 or bytes_per_token < 0 or any(x < 0 for x in lengths):
        raise ValueError("nonnegative sizes required")
    duplicated = (len(lengths) * prefix_tokens + sum(lengths)) * bytes_per_token
    shared = (prefix_tokens + sum(lengths)) * bytes_per_token if lengths else 0
    return duplicated, shared
~~~

G=4、P=1000、后缀为 [10,20,30,40]、b=16 时，重复存储为 65600 bytes，共享前缀后为 17600 bytes。随着生成后缀变长，共享前缀占比下降，节省比例也会变化，不能固定声称“采样四次就省四倍显存”。

需要核对的缓存身份至少包括 token 与位置、模型参数版本、adapter、影响结果的预处理与 attention 配置。不是看字符串前缀相同就一定可以复用。

## 7. 权重同步与 KV 生命周期为什么相互制约

一层最简单的投影为 K=HW_K。权重改成 W_K+ΔW 时：

$$
K'=H(W_K+\Delta W)=K+H\Delta W.
$$

只要 HΔW 非零，旧 K 就不再等于新 K；更早的层更新还可能改变 H。即使差异很小，也只能说近似误差可能可接受，不能说早期层缓存天然不失效。

这给出一个明确的工程选择：

| 策略 | 数学含义 | 成本与要求 |
|---|---|---|
| 更新后失效并重算 | 按新模型重新建立状态 | 重算和传输有成本 |
| 同一轨迹固定生成版本 | 轨迹内部继续用同一策略 | 需要管理多版本权重或等待切换 |
| 跨版本复用近似状态 | 生成条件含旧状态 | 必须明确训练处理方式和误差验证 |

一些后训练系统选择第三条，并记录历史路由或处理陈旧样本；这不能自动变成普通在线推理缓存的正确性规则。

若同步传输的数据总量为 B，有效链路吞吐为 BW，则只看通信也有 t≥B/BW。完整同步还包含训练分片到推理分片的转换、打包、barrier 和引擎重建。shared memory 不等于 GPU 权重免费更新，LoRA 热切换也不能代表任意全量更新。

不要用缺少网络拓扑、分片方式和计时边界的框架排名表决定架构。需要对相同模型、batch 和设备测实际权重发布路径。

## 8. 共驻、分离与一致性是三个问题

共驻是训练与 rollout 使用同一组设备，可能分时切换；分离是二者使用不同资源池，并通过传输发布权重。数值一致性则是在给定版本与输入下，训练和推理路径的结果应满足什么误差合同。

共驻可以减少闲置和某些搬运，也可能被优化器、激活与 KV 同时挤占容量。分离允许两种负载独立扩缩容，但要承担网络、策略延迟与资源配比成本。没有一种形态对所有模型、任务和集群始终更优。

参数 shard、rollout KV、训练 optimizer state、通信 buffer、图捕获内存可能同时存活。峰值容量要按时间线的活跃对象求和，不能把各阶段单独测出的峰值最大值当成共驻总峰值。

## 9. 智能体长任务：不能把所有等待都算成 Decode

一次任务可能包含多轮模型调用、工具执行和结果回填。对有依赖的执行图，每个节点 v 的完成时间是：

$$
E(v)=t(v)+\max_{u\in\operatorname{pred}(v)}E(u).
$$

没有前驱时 max 取 0。任务端到端时间由终点的最长依赖路径决定，而总资源消耗与所有节点工作量有关。并行工具可以降低关键路径，但不会让它们的总 CPU、网络或 GPU 消耗消失。

例如模型花 1 秒生成工具调用，两个独立工具分别用 2 秒与 5 秒，随后模型用 1 秒汇总。工具完全并行时端到端约为 7 秒，串行时约为 9 秒。若长工具依赖短工具结果，则不能为了追求并行把它们同时启动。

每一轮至少区分 queue、prefill、decode、tool、cache restore 和结果提交时间。任务级 P95 不能通过把这些组件各自的 P95 相加获得；最慢组件来自的请求不一定相同。

### 9.1 模型等工具时，KV 怎么办

让 KV 驻留会占用其他请求可使用的显存；卸载到 host/SSD 会引入传输与恢复延迟；直接驱逐会增加下轮重算。是否卸载要看预期等待、再次使用概率、上下文大小、传输带宽和其他请求的容量压力。

粗略比较可写成：

$$
C_{\mathrm{keep}}\approx
\lambda_{\mathrm{mem}} B_{\mathrm{KV}}t_{\mathrm{idle}},
\qquad
C_{\mathrm{offload}}\approx
C_{\mathrm{transfer}}+C_{\mathrm{restore}}+C_{\mathrm{metadata}}.
$$

λ_mem 表示本系统对显存占用的机会成本权重，不是硬件固定常数。若只优化单请求延迟，可能倾向全驻留；若服务整体吞吐受容量限制，部分卸载可能更合算。必须说明优化目标，才能比较两个方案。

工具失败、超时和取消还会留下缓存与环境资源。释放要跟实际最后消费者完成绑定，不能在主机提交取消请求时就假设设备访问已经结束。

## 10. 自适应投机：少生成几步，可能多占一次大 batch

固定草稿长度忽略了任务难度、草稿匹配程度和系统负载。设第 i 个草稿位置在前缀存活条件下接受概率为 p_i，则期望接受长度是：

$$
E[L_{\mathrm{acc}}]=\sum_{i=1}^{k}\prod_{j=1}^{i}p_j.
$$

如果一个周期还有额外的纠正或 bonus token，且没有结束截断，可比较：

$$
\frac{1+E[L_{\mathrm{acc}}]}
{t_{\mathrm{draft}}(k)+t_{\mathrm{verify}}(k)+t_{\mathrm{commit}}(k)}.
$$

增大 k 会同时增加分子与分母。验证可能让 batch 从一个高效区域跨到另一个区域，也可能挤占其他请求。设备还有空闲计算单元，不代表搬运、临时 KV、launch 和提交成本都为零。

置信度可以用于选择是否投机和验证长度，但不能替代正确的接受/拒绝规则。分布保持要求使用实际草稿分布与目标分布进行校正，不能因为候选“很可信”就无条件接收。

实际一次验证要把“暂存 cache”与“已提交上下文”分开。设当前目标 KV 已提交到 T，草稿器暂存了 k 个 token `d[0:k]`。目标模型以因果 mask 对整段候选做一次块前向，得到每个候选位置的 next-token logits `[B,k,V]`。Greedy 路径在首个不匹配位置 a 停止：只保留目标 KV 到 `T+a`，输出已接受的 `d[:a]` 和目标模型在 a 处给出的纠正 token，丢弃 `d[a:]` 与其草稿状态。纠正 token 已成为输出上下文，但它的 KV 通常要在后续前向消费该 token 时才写入。

| 验证结果 | 可见输出 | 目标 KV 提交长度 | 草稿 KV 处理 |
|---|---|---:|---|
| 第 a 位首次失败 | `d[:a] + correction` | `T+a` | 回滚到已接受前缀，删除失败位及之后的暂存页 |
| k 位全通过 | `d[:k] + bonus`（若仍有预算） | `T+k` | 保留已接受段，继续从 bonus 后同步 |
| 请求 EOS/长度上限提前结束 | 只提交截止边界之前的 token | 按有效前缀截断 | 不得提交 padding 或未验证候选 |

若候选 token 来自随机分布 q，目标分布是 p，逐 token 的 exact speculative sampling 要按概率比接受，并在拒绝时从目标分布的残差部分采样；仅比较 token ID 是否相同只对应 greedy 验证。无论哪种算法，提交边界必须由 verifier 决定，而不能按草稿长度直接前移 cache block table。CPU 检查 `workload_lifecycle.py` 只执行 greedy prefix commit/rollback 的状态算术，不是某个引擎的调度记录。

### 10.1 为什么候选长度分档有工程价值

动态 k 会影响输入形状、图捕获、workspace 和调度预算。可以只支持一组档位，例如 {0,1,2,4}，对每个档位测当前服务条件下的 cost，并给调度器有限候选。

这减少了图和布局组合，却可能错过某个中间最优长度。padding 后的无效草稿必须 mask，不能计作实际验证或有效吞吐。图复用还需要满足地址与更新条件，不能只因为 shape 命中档位就认定可直接 replay。

在线更新草稿模型是另一条适应路径，它引入训练、权重发布和版本管理成本。草稿变好、调度变好和目标模型变了，要分开对照实验，否则看不出收益来自哪里。

## 11. 把优化价值放回完整循环

若 rollout 占一次完整训练循环的比例为 f，只把 rollout 加速 s 倍，其他部分不变，则总加速比为：

$$
S_{\mathrm{total}}=\frac1{(1-f)+f/s}.
$$

f=0.6、s=1.5 时，总加速为 1.25 倍，耗时减少 20%，不是总时间直接减少 50%。如果权重同步、评分或环境已经成为瓶颈，继续加速生成会把等待移动到别处。

但这仍假设训练需要的有效数据和更新次数不变。若吞吐优化改变了样本质量、长度偏差或策略陈旧度，必须比较达到同一能力的总成本，不能只比较每轮秒数。

一个有说服力的分析至少同时报告：生成有效 token、完成轨迹、训练更新、同步传输、工具等待、峰值显存以及质量指标。单个 kernel 的速度是其中一项证据，不是最终结论。

## 12. 读懂以后应该能回答什么

- 为什么 RL 训练里会出现 Prefill 和 Decode，但训练又不能等同 Prefill？
- 固定训练计算预算下，N 和 D 为什么不能同时无限增长？闭式解用了哪些假设？
- 为什么 dense 的预算拟合不能直接把 N 替换为 MoE 激活参数？
- 异步调度减少长尾后，训练数据分布可能怎样改变？
- 同 prompt 多采样共享了什么，又在哪里开始需要独立可写 KV？
- 为什么最新策略概率不能充当旧行为策略的概率？
- 工具调用的结果 token 和模型动作 token 在 loss mask 上有何区别？
- 为什么权重更新后保留缓存是一项需要证据的策略，而不是默认安全操作？
- 为什么投机接受率更高，却可能让整个服务更慢？

可以用随附的 CPU 例子核对代数与反例：

~~~bash
python roadmap/curriculum/papers/examples/workload_checks.py
~~~

它不启动训练框架、推理服务或 GPU 实验；调度时间是人为设定的示例时长，不是任何产品的性能测量。

## 参考阅读

[大模型推理实践](../../../downloads/大模型推理实践.pdf)提供问题背景；[DeepSeek-V4.1-Flash 技术报告](../../../downloads/DeepSeek_V41_Tech_Report.pdf)提供异步后训练与长任务系统案例。框架实现、同步链路与实验数字应按具体源码版本和测量条件核查，不从讲义中的简表推断普遍结论。
