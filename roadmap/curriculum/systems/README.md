# 第三章 vLLM 应用：把算子能力放进真实请求路径

vLLM 将请求转成 token 批次，安排模型执行，管理 KV，并将结果提交给请求。调度决策会改变 kernel 形状、缓存占用和等待时间。

分析从一次请求的调用路径开始：确认执行的 backend、输入布局和计时范围，再比较替换算子前后的单步与端到端结果。

## 1. 从请求到设备执行

文本请求经过输入处理、tokenization、调度、模型执行、采样和输出处理。请求进入服务时，GPU 不一定立即执行，还要满足 token 预算、活动请求数与 KV 容量。

下面是教学伪代码，省略执行器通信、异步返回和错误处理，不是可直接替换框架的 API：

~~~python
request = prepare_input(raw_request)
scheduler.add_request(request)
while scheduler.has_work():
    plan = scheduler.schedule()
    outputs = model_runner.execute_model(plan)
    scheduler.update_from_output(plan, outputs)
    emit_ready_outputs(outputs)
~~~

plan 要说明计算哪些位置、每个请求已有多少 KV、结果写到哪里。模型返回后再推进计算与生成状态。计算若干输入 token，不代表返回相同数量的输出 token；未完成的 Prefill chunk 可能还没有可采样输出。

具体安装版本的引擎、scheduler、worker/model runner、模型层与 Attention backend 可能有不同调用方式。类名相同也不保证同步/异步行为相同，需要结合实际配置和时间线确认。

## 2. token 预算不是请求数量

一个 Decode 请求可能只需计算一个新位置，Prefill 请求则可能还剩数百个位置。限制最多几个请求，与限制本轮计算几个 token，是不同约束。

<!-- source-check: examples/scheduler_reference.py -->
~~~python
def allocate_tokens(demands, budget):
    if budget < 0 or any(n<0 for _,n in demands):
        raise ValueError("nonnegative demand and budget required")
    if len({name for name,_ in demands}) != len(demands):
        raise ValueError("request identities must be unique")
    assignments=[]
    for name,need in demands:
        take=min(need,budget)
        if take:
            assignments.append((name,take))
        budget-=take
    return assignments
~~~

预算为 4，先处理一个需求为 1 的 Decode，再处理需求为 8 的 Prefill，分配为 1+3；反过来则 Prefill 占满 4。这个模型演示顺序与预算的影响，不代表 vLLM 固定采用某一种优先级。

Chunked Prefill 把长输入拆小，为其他请求留下调度机会。块太小可能增加调度与 kernel 开销，块太大则延长其他请求的等待；要同时观察 TTFT、TPOT、吞吐与长尾。

Continuous batching 在步与步之间加入或移除请求。batch 行号可以变化，请求身份不能丢：length、位置、KV 映射、采样状态和输出归属需要一起更新。

## 3. KV 容量与前缀复用

在无共享和 COW 的简单情形下，已有长度 L、每页 T 个位置，追加 n 个位置需要的新页数为

$$
\Delta P=\left\lceil\frac{L+n}{T}\right\rceil-
\left\lceil\frac{L}{T}\right\rceil.
$$

T=16，L=16 再追加一个位置需要新页，L=17 再追加一个可能仍使用尾页。共享前缀、投机预留与不同层的 cache 结构还会改变预算，不能只统计当前有效 token。

### 相同 token 片段不一定对应相同 KV

KV 依赖前面的上下文、模型权重、位置规则和 adapter。两个末尾 block 的 token 相同，之前的 prefix 不同，hidden state 也可能不同；按 block 缓存时，需要父前缀身份，而不只 hash 当前 block。

下面用完整 prefix 演示身份组成，不是 vLLM 内部 hash 协议：

<!-- source-check: examples/scheduler_reference.py -->
~~~python
def prefix_identity(tokens, model_revision, adapter, position_base, tenant):
    # tokens is the complete prefix here; block-based caches also need parent identity.
    payload=[list(tokens),model_revision,adapter,position_base,tenant]
    return hashlib.sha256(json.dumps(payload,separators=(",",":")).encode()).hexdigest()
~~~

model_revision 应覆盖影响计算的权重与配置，adapter 和位置也须一致；需要租户隔离时加入相应隔离域。多模态还要考虑图像、音频等状态。本例只处理文本，hash 不替代访问控制。

前缀命中主要减少重复 Prefill。普通 Decode 仍要访问 Attention 所需的历史 KV，不能把“省去重算”理解为“后面不用读”。

### Offload 与 PD 分离

KV offload 改变容量与传输路径，恢复时可能增加等待、预取与设备临时空间。H2D、网络和 HBM 带宽不是同一个指标。

Prefill/Decode 分离让两阶段独立配置资源，但必须交接 KV 和请求状态。收益可能来自减少干扰或独立扩缩容，代价包括传输、排队与调度；短请求、低并发下未必划算，应和同条件共置基线比较。

## 4. 确认真实 backend，再替换 kernel

Attention 路径可能随 dtype、head dimension、cache layout、设备架构和 Prefill/Decode 阶段变化。先从启动配置与时间线确认实际 kernel，再检查调用处的 shape、stride、mask、输出与 workspace。

捕获一组相同输入，先对比局部结果，再替换实现并检查模型输出与请求指标。连续 FP32 Attention 不能直接替代需要分页、低精度和动态长度的 backend；适配增加的 transpose、cast 和 copy 必须计入成本。

CUDA Graph 能减少部分 host launch 开销，但地址和执行结构有约束。动态 shape 通常需要 buffer 复用、形状分档或回退；图回放不表示数学工作量消失。比较 eager 与图执行时，保持其他条件不变。

## 5. 实践：本地模型与同条件基线

以下命令在自己的 GPU 环境中执行。先准备能放入设备的本地模型，再核对当前版本参数；不自动下载安装，也不修改公共配置。

~~~bash
python -c "import importlib.metadata as m; print(m.version('vllm'))"
vllm serve --help
vllm bench serve --help
~~~

启动只监听本机的服务示例：

~~~bash
MODEL_DIR=/data/models/your-model
vllm serve "$MODEL_DIR" \
  --served-model-name teaching \
  --host 127.0.0.1 --port 8000 \
  --dtype auto --max-model-len 2048 \
  --gpu-memory-utilization 0.8
~~~

MODEL_DIR 是需要替换的本地路径。模型支持、dtype 和容量取决于环境，这不是任意模型都能启动的保证。服务保持运行，在另一个终端设置同一模型路径，先检查小请求再基准测试：

~~~bash
MODEL_DIR=/data/models/your-model
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"teaching","prompt":"Hello","max_tokens":8,"temperature":0}'
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8000 \
  --model teaching --tokenizer "$MODEL_DIR" \
  --dataset-name random \
  --random-input-len 128 --random-output-len 64 \
  --random-range-ratio 0 \
  --num-prompts 32 --request-rate 4 \
  --save-result --result-dir ./results/vllm-small
~~~

先检查单请求返回的文本、finish_reason 和 usage；/v1/models 能访问不代表模型前向已经成功。用 help 确认安装版本参数。随机输入用于控制长度和压力，不代表业务质量；需要记录实际生成长度、EOS、错误、warmup 和 cache 状态。request rate 是到达速率，不保证服务吞吐达到该值，并发限制也可能压低实际发送速率。

比较前缀缓存时，分别控制重复/不同前缀以及冷/热状态。不要同时换模型、dtype、输入长度和调度参数，再把结果归因于一个 kernel。

## 6. 判断优化是否真正生效

| 观察 | 需要排除的混淆 |
|---|---|
| TTFT 下降 | 前缀命中、排队或输入长度变化 |
| TPOT 下降 | batch、输出长度、采样或结束策略变化 |
| 吞吐增加 | 错误增加、SLO 超限、实际生成 token 减少 |
| 显存下降 | cache 变小，导致更多重算或换出 |
| kernel 更快 | 层外新增转换、通信或同步 |

SLO（Service Level Objective，服务等级目标）约束下的有效吞吐，比无约束峰值更有意义。保存输入配置、软件版本、设备身份和逐请求结果，按同一边界比较。

无 GPU 时可以先核对预算与缓存身份：

~~~bash
python roadmap/curriculum/systems/examples/scheduler_reference.py
~~~

系统组合没有对应的 LeetGPU 题。局部题面正确，还要在这里验证 backend 与请求路径；能返回文本不等于性能优化有效。

## 7. 整网与算子：从请求形状追到设备执行

一次推理不是一次 kernel launch，而是跨越请求状态、模型张量、框架调度和设备执行的多层路径：

| 层次 | 主要工作 | 需要观察的对象 |
|---|---|---|
| 请求与调度 | 接收 token、选择本轮请求与 token 预算 | 请求状态、Prefill/Decode 阶段、KV 位置 |
| ModelRunner | 整理输入、位置与 Attention 元数据 | 输入 shape、slot/block 映射、采样位置 |
| 模型层 | 执行投影、归一化、Attention、MLP 与采样 | 张量维度、dtype、mask、残差数据流 |
| 框架执行 | 按配置走 eager、编译或图回放 | 捕获/编译范围、shape 分桶、CPU 提交间隙 |
| 设备执行 | 运行 kernel，并按需传输或集合通信 | kernel、copy、通信、同步与空闲时间 |

GPU 与昇腾的分层可以用来互相理解，但接口和实现不能直接互换：

| 观察层 | 昇腾常见组成 | NVIDIA GPU 上的对应问题 | 可迁移的判断方法 |
|---|---|---|---|
| 请求与批处理 | 推理服务、调度器、ModelRunner | 服务入口、Scheduler、ModelRunner | 本轮处理多少 token、请求状态和 KV 映射是否一致 |
| 图与执行 | PyTorch、CANN Runtime、GE/ACLGraph 等路径 | PyTorch、CUDA Runtime、编译或 CUDA Graph 等路径 | 图是否捕获目标工作、动态形状如何处理、CPU 是否仍逐算子调度 |
| 计算算子 | CANN 算子、Ascend C、自定义 NPU 算子 | CUDA/Triton kernel、CUDA 库或融合算子 | 输入输出 shape、stride、dtype、mask、workspace 和数值合同 |
| 设备通信 | HCCL 与实际 NPU 互连 | NCCL 与实际 GPU 互连 | 通信量、拓扑、同步点及其与计算的重叠 |
| 设备内存 | NPU HBM、片上存储与 KV 分配 | GPU HBM、SMEM/寄存器与 KV 分配 | 读写字节、驻留时间、临时 buffer、碎片和并发容量 |

这张表是分析视角的对应，不是 API 或硬件单元的一对一映射。比如，知道算子“可以入图”还不等于当前请求真的走了图执行；需核对启动配置、shape 分桶、编译/捕获日志和设备时间线。eager 下小 Decode kernel 可能被 CPU launch 间隙隔开；图执行可能压低重复提交开销，但不会减少 Attention 必须读取的历史信息，也不能证明某个 kernel 更快。

### 用一层模型把 shape 追到底

本地 openPangu-Embedded-7B-V1.1 配置给出隐藏维度 `H=4096`、34 层、32 个 Query Head、8 个 KV Head、词表 153,376。每个 head 的宽度为

$$
d=H/32=128.
$$

对 `T` 个 token，投影的逻辑形状为：

| 张量 | 形状 | 含义 |
|---|---:|---|
| 输入 `X` | `[T, 4096]` | 每个 token 一个 4096 维隐藏向量 |
| `Q` | `[T, 32, 128]` | 32 个 Query Head |
| `K`、`V` | 各 `[T, 8, 128]` | 每种张量 8 个 KV Head |
| Attention 输出 | `[T, 4096]` | 拼接 32 个 head 后的隐藏维度 |
| LM Head logits | `[T, 153376]` | 每个位置对应词表分数 |

Prefill 的 `T` 是本轮实际送进模型的 prompt token 数，chunked prefill 时不一定等于完整 prompt 长度。Decode 常见情形是每条活跃请求只添一个新位置，输入可写成 `[B,4096]`；它为本轮 token 生成 Q/K/V，同时从请求自己的 KV 映射读取历史上下文。每 4 个 Query Head 共享一组 KV Head，因此 GQA 减少 KV 的存储和读取量；它没有把每个 Query Head 的计算自动变成一个更短的历史序列。

下面是 `vllm-ascend-pangu-v023` 中盘古适配类的源码原样短摘录，展示模型层如何把投影张量交给通用 `Attention` 模块。它不是 CUDA/NPU kernel，也没有固定某一个 Attention backend：

```python
qkv, _ = self.qkv_proj(hidden_states)
query, key, value = qkv.split(
    [self.q_size, self.kv_size, self.kv_size], dim=-1
)
query, key = self.rotary_emb(positions, query, key)
attn_output = self.attn(query, key, value)
output, _ = self.o_proj(attn_output)
return output
```

**教学改写（伪代码，不是框架 API）：**

```python
q, k, v = project_and_split(x, query_heads=32, kv_heads=8, head_dim=128)
q, k = apply_position_encoding(q, k, positions)
context = attention_backend(q, k, v, request_kv_map, mask)
y = output_projection(context)
```

第一段保留实际模型调用结构；第二段只把数据依赖摊开，明确 backend 还需要 KV 映射和 mask 等信息。源码调用 `self.attn(...)`，不等于模型类直接调用某个 `flash` 或 `paged` 函数。backend 会受软件版本、设备、dtype、shape、KV 布局、调度阶段和执行模式影响。对 GPU 也一样：读到模型层的 `Attention` 调用，只能说明模块边界，不能据此声称运行了 FlashAttention、PagedAttention 或某个特定 kernel。

## 8. 同一品牌，不同模型：Embedded-7B 与 openPangu 2.0

“盘古”是系列名，不代表模型结构相同。以下三列分开描述，不能把 2.0 的设计当成 Embedded-7B 的实现细节：

| 属性 | openPangu-Embedded-7B-V1.1 | openPangu-2.0-Flash | openPangu-2.0-Pro |
|---|---|---|---|
| 规模标称 | 约 7B，配置可逐张量复算 | 约 92B 总 / 6B 激活 | 约 505B 总 / 18B 激活 |
| 主体结构 | 34 层 Dense decoder | 46 层 MoE | 50 层 MoE |
| 隐藏宽度 | 4096 | 2560 | 5120 |
| Attention | GQA：32 Q heads、8 KV heads、head dim 128 | MLA + DSA/SWA 混合，SWA:DSA=2:1 | MLA + DSA/SWA 混合，SWA:DSA=2:1 |
| MoE 路由 | 无 | 256 routed experts，每 token 选 8；另有 1 shared expert | 384 routed experts，每 token 选 8；另有 1 shared expert |
| 上下文配置 | 32,768 | 524,288 | 524,288 |

教材中的 Flash/Pro 描述有混合之处；应以官方模型卡和配置按版本区分：Flash 是约 92B/6B、256 选 8，Pro 是约 505B/18B、384 选 8。两者的专家数不同，不能互相套用，也都不是本地 Embedded-7B Dense 模型。

2.0 的 Attention 也不能用 Embedded-7B 的 GQA 公式代替。官方模型卡按 DSA:SWA=1:2 给出混合层配比，即 SWA:DSA=2:1。Flash 官方配置包含 `q_lora_rank=1024`、`kv_lora_rank=512`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、48 个 attention heads；Pro 对应值为 Q rank 1536、KV rank 512、NoPE/RoPE 维度 128/64、64 heads。MLA 以低秩 latent 表达压缩 K/V 路径，DSA 再按索引选择稀疏上下文，SWA 只保留局部窗口。要把这些数字换算成真实 KV 分配，仍需跟随匹配版本的推理实现，检查每种层的 cache tensor、索引数据、dtype、block 组织和请求映射；不能只把 `kv_lora_rank` 乘层数就宣称是进程显存。

### 2.0 方法如何映射到系统成本

官方模型资料确认 2.0 使用 MLA、DSA/SWA 混合层、4-stream mHC、3 头 MTP；模型配置还给出 512K 上下文与 MoE 路由参数。把这些结构转换为系统问题时：

- **SWA 与 DSA**：SWA 把可见上下文限制到窗口，数据访问较局部；DSA 先用索引机制选候选位置，随后对稀疏位置做 Attention，带来索引和不规则访问。两类层的 shape、读写和并行策略不同，不能用同一条 dense Attention kernel 时间代表整网。
- **MoE 384 选 8（Pro）/256 选 8（Flash）**：一个 token 只路由到部分 routed expert，减少其 MLP 计算；但“激活参数少”不表示全部未激活专家权重都无需驻留。运行成本还包括 router、token 重排、专家分配、设备间 dispatch/combine、负载不均和同步。低并发时通信启动/同步占比可能重要，高并发时传输量与负载均衡也必须测量。
- **长上下文**：512K 是模型支持的最大位置规模，不代表任意 batch 都能在单设备上保留全部状态。需要分别核算 KV/索引状态、临时 workspace、权重分片与并发请求，并以实际模型版本和部署配置的容量检查为准。
- **MTP**：额外预测头改变候选 token 的产生与验证路径；端到端收益取决于接受率、额外计算和调度，不是把单个 decode kernel 的时间简单除以 3。
- **PD 分离与并行策略**：官方 2.0 推理仓库把 Flash（92B）和 Pro（505B）分别配置；其公开 README 给出 A3 上 Flash 的 1P1D 示例与 Pro 的独立部署配置。这说明部署方案与版本绑定，不能拿旧 vLLM-Ascend 的 TP4 静态图当作 2.0 运行配置或性能证据。

这些是从公开结构推导出的待验证系统变量；模型配置本身不能证明对应 backend 或 kernel 的运行方式和性能。切换到 GPU 时，分析仍要追踪张量布局、算术强度、内存流量、调度粒度、通信域与同步点；DSA/MoE backend、通信策略和 fused kernel 必须由目标 GPU 软件栈的实现与执行证据确认，不能从昇腾算子名推导 CUDA 路径。

读 2.0 源码时要把模型定义与推理部署分开配对。HF 官方 Flash/Pro 各自的配置都声明 `OpenPanguV2ForCausalLM` / `openpangu_v2`，但 hidden width、层数、dense 起始层、DSA 层集合、专家总数和每 token top-k 必须从同一个模型目录读取；不能把 Flash 配置套在 Pro checkpoint 上。`architectures` 字符串是模型类声明，配置文件描述模型张量与结构；二者都不足以确定容器内实际载入的 adapter 或 kernel backend。

官方 openPangu-2.0-Infer 仓库再按 92B Flash 与 505B Pro 分别提供部署目录：Flash 的示例拓扑是 1P1D（两台 A3），Pro 的示例拓扑是 2P1D（八台 A3），启动模板、inventory、镜像版本和 served model name 按模型分别匹配。README_EN 列出镜像内依赖 `omni-npu 0.2.0` 与 `vllm 0.14.0+empty`，并指定镜像 `omniinfer-a3-arm:release_1.2.1.post1-202607241407-vllm`；它建议用 `pip list` 并进入容器定位 omni-npu 安装路径以修改推理代码。可见 GitCode main commit `5ddd125e` 下的 `components/omni-npu` 目录为空或不存在；官方 Gitee 组织索引则列出 `omniai/omni-npu`，说明它是 vLLM out-of-tree NPU plugin。该 Gitee 仓库代码页面标注外部成员不可访问，公开 GitCode 树与 README_EN 未提供匹配镜像版本的 adapter 源码或 commit。由此，公开可访问的模型配置与部署 YAML 不能确定 backend；这不代表实现未公开或镜像内只有二进制。模型卡与部署 YAML 说明配置意图，不构成单请求执行轨迹；服务日志和 profiler 才能证明对应配置下的模型载入、请求成功、backend/kernel 分派与性能。

在获准进入且与服务相同版本的容器后，可只查 Python distribution metadata 来定位安装文件；这不会 import `omni_npu`、初始化 NPU 或执行 adapter：

~~~python
from importlib import metadata

dist = metadata.distribution("omni-npu")
print("distribution version:", dist.version)
print("distribution root:", dist.locate_file(""))

files = dist.files or ()
print("recorded file count:", len(files))
for entry in files:
    if str(entry).endswith((".py", ".so", "entry_points.txt")):
        print(dist.locate_file(entry))
~~~

此检查只能找到当前容器里安装的发行文件和版本；若要核对实际 model adapter，还需在该版本源码允许访问的前提下静态查看注册、load_weights、forward 与 vLLM 调用点。它本身不是执行轨迹，不能证明服务载入了哪个 adapter 或调用了哪个 backend。

部署前需选定并记录官方仓库 commit，复用对应模型目录的 inventory 与服务模板，填入真实节点地址、持久日志目录、权重路径和 served model name；再按 README 创建容器并按 Flash/Pro 对应的 `run_server,run_proxy` 标签启动。应通过服务日志确认模型 ID、Prefill/Decode 分组、容器内框架版本和请求结束状态；启动命令返回 0 只证明编排步骤成功，不能代替设备端生成或性能验收。

共享集群不得直接照抄下面的命令。执行前先审核 inventory 中的 P/D/C 节点、host/IP/device 列表、模板引用的 `DOCKER_NAME_*`、image tag、日志目录和 P/D 命令；确认同名容器的归属并得到节点/容器 owner 同意。官方 README 明确提示：相同镜像时重复 `run_docker` 会覆盖同名容器；`run_server,run_proxy` 除容器内 nginx/proxy 外，还会在 master host 启动 nginx。因此先在隔离且获准的执行环境确认宿主机影响与端口占用，不能在共享 host 上未经审核启动。源码修改也须先确认路径是在容器文件系统、挂载卷还是宿主机；仅改任务获准的容器工作副本，不改共享宿主机 `/etc`、启动服务或公共配置。以下命令体现相应部署入口和作用域风险，须经上述预检后才能在获准环境执行。

在 A3 执行节点上，应按官方 README 中与模型规格匹配的 inventory 和模板执行：

~~~bash
# Flash: use the 92B / 1P1D inventory and service template documented by that checkout
(cd tools/ansible/92B && ansible-playbook -i omni_infer_inventory_used_for_1P1D.yml omni_infer_server_template_performance1P1D_92B_bf16_open.yml --tags run_docker)
(cd tools/ansible/92B && ansible-playbook -i omni_infer_inventory_used_for_1P1D.yml omni_infer_server_template_performance1P1D_92B_bf16_open.yml --tags run_server,run_proxy)

# Pro: use the 505B / 2P1D inventory and its separate service template
(cd tools/ansible/505B && ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template_performance2P1D_505B_bf16_open.yml --tags run_docker)
(cd tools/ansible/505B && ansible-playbook -i omni_infer_inventory_used_for_2P1D.yml omni_infer_server_template_performance2P1D_505B_bf16_open.yml --tags run_server,run_proxy)
~~~

运行前必须在同一官方 checkout 中填好 inventory 的实际节点地址及模板环境值，不能把 Flash inventory 和 Pro template 混用。服务启动后先保存 commit、inventory、template、image tag 与日志目录，再用 README 的示例 API 请求做功能 smoke；只有目标设备日志和 profiler 证据才允许写设备通过或吞吐结果。

## 9. 手算容量：权重与 KV 分开

对 Embedded-7B，权重参数可按实际结构逐项计数。令 `H=4096`、`I=12800`、`Hkv=8×128=1024`：

$$
P_{\mathrm{attn}}=2H^2+2H H_{kv},\qquad P_{\mathrm{attn\ bias}}=2H+2H_{kv}.
$$

前式包含 Q、O 两个 `H×H` 权重和 K、V 两个 `H×Hkv` 权重；后式是四个投影的 bias 数量。门控 MLP 与 RMSNorm 的参数量为

$$
P_{\mathrm{MLP}}=3HI,\qquad P_{\mathrm{norm}}=2H.
$$

| 参数组 | 每层/每矩阵计算 | 参数量 |
|---|---:|---:|
| Attention 投影权重 | `2H²+2HHkv` | 41,943,040 / 层 |
| Attention bias | `2H+2Hkv` | 10,240 / 层 |
| MLP 权重 | `3HI` | 157,286,400 / 层 |
| 两个 RMSNorm | `2H` | 8,192 / 层 |
| Decoder Layer | 上述四项之和 | 199,247,872 / 层 |
| 34 个 Decoder Layer | `34×199,247,872` | 6,774,427,648 |
| Embedding 与 LM Head | `2×V×H`，因 `tie_word_embeddings=false` | 1,256,456,192 |
| Final RMSNorm | `H` | 4,096 |
| **总计** | `34×P_layer+2VH+H` | **8,030,887,936** |

按 BF16 每参数 2 bytes，理论权重 payload 为 `16,061,775,872 bytes = 14.958694 GiB`。这是参数张量字节数，不包含运行时 workspace、allocator、图、通信 buffer、临时权重副本或对齐开销。

GQA 的逻辑 KV 容量为：

$$
M_{KV}=B\times S\times L\times 2\times H_{kv}\times d\times b,
$$

其中 `B` 是请求数，`S` 是每条序列实际保留的 token 数，`L=34`，因子 2 表示 K 和 V，`b=2 bytes` 表示 BF16。故每个请求、每个 token、跨所有层需要 `34×2×8×128×2=139,264 bytes=136 KiB`；32,768 token 的未分页逻辑值是 4.25 GiB。若按 128-token block 分配，129 个有效 token 要占两个 block，即 256 个槽位；实际 allocator 还需考虑共享、预留、量化、元数据与其它工作区。

2.0 的总参数量与激活参数量也不能混作容量。以 Pro 标称 505B 参数为例，若所有参数都以 BF16 常驻，单看参数 payload 的粗略下限就是 `505×10⁹×2 / 2³⁰ ≈ 940.64 GiB`；实际部署还需权重切分/量化、buffer 和 KV 空间。`18B active`描述每 token 参与计算的参数规模，不代表只需加载 18B 权重。Flash 的 92B/6B 同理，不能由 6B 激活直接推导出权重可驻留大小。

核心账本只用整数算术即可复核形状、权重与 KV 的数量级：

~~~python
head_dim = HIDDEN // QUERY_HEADS
kv_width = KV_HEADS * head_dim
attention_weights = 2 * HIDDEN**2 + 2 * HIDDEN * kv_width
mlp_weights = 3 * HIDDEN * INTERMEDIATE
layer_params = attention_weights + (2 * HIDDEN + 2 * kv_width) + mlp_weights + 2 * HIDDEN
total_params = LAYERS * layer_params + 2 * VOCAB * HIDDEN + HIDDEN
kv_bytes_per_token = LAYERS * 2 * KV_HEADS * head_dim * BYTES_PER_BF16

def require_integer(name, value, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")

def paged_kv_bytes(tokens, batch, block_size):
    require_integer("tokens", tokens, 0)
    require_integer("batch", batch, 1)
    require_integer("block_size", block_size, 1)
    blocks = (tokens + block_size - 1) // block_size
    return blocks * block_size * batch * kv_bytes_per_token
~~~

完整程序只复算 Embedded-7B 的 shape、参数与分页账本，不加载权重、不依赖 PyTorch，也不访问加速器：

<details>
<summary>展开完整 CPU 账本程序</summary>

<!-- source-check: examples/pangu_embedded7b_ledger.py -->
~~~python
"""CPU-only shape, BF16 weight, and KV-cache ledger for Embedded-7B.

The constants below come from the checked-in openPangu-Embedded-7B-V1.1
config.json. This is arithmetic verification, not a model load or NPU/GPU run.
"""

HIDDEN = 4096
INTERMEDIATE = 12800
LAYERS = 34
QUERY_HEADS = 32
KV_HEADS = 8
VOCAB = 153376
BYTES_PER_BF16 = 2
BLOCK_SIZE = 128


def require_integer(name, value, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def calculate():
    assert HIDDEN % QUERY_HEADS == 0
    assert QUERY_HEADS % KV_HEADS == 0
    head_dim = HIDDEN // QUERY_HEADS
    kv_width = KV_HEADS * head_dim

    # Q, K, V, O projection weights and their configured biases.
    attention_weights = 2 * HIDDEN * HIDDEN + 2 * HIDDEN * kv_width
    attention_bias = 2 * HIDDEN + 2 * kv_width
    mlp_weights = 3 * HIDDEN * INTERMEDIATE  # gate + up + down, no bias
    norm_weights = 2 * HIDDEN
    layer_params = attention_weights + attention_bias + mlp_weights + norm_weights

    embedding_params = VOCAB * HIDDEN
    lm_head_params = VOCAB * HIDDEN  # tie_word_embeddings=false
    final_norm_params = HIDDEN
    total_params = LAYERS * layer_params + embedding_params + lm_head_params + final_norm_params

    # One K and one V vector per KV head, per layer, per token.
    kv_bytes_per_token = LAYERS * 2 * KV_HEADS * head_dim * BYTES_PER_BF16

    def kv_bytes(tokens, batch=1):
        require_integer("tokens", tokens, 0)
        require_integer("batch", batch, 1)
        return tokens * batch * kv_bytes_per_token

    def paged_kv_bytes(tokens, batch=1, block_size=BLOCK_SIZE):
        require_integer("tokens", tokens, 0)
        require_integer("batch", batch, 1)
        require_integer("block_size", block_size, 1)
        blocks = (tokens + block_size - 1) // block_size
        return blocks * block_size * batch * kv_bytes_per_token

    return {
        "head_dim": head_dim,
        "kv_width": kv_width,
        "attention_weights": attention_weights,
        "attention_bias": attention_bias,
        "mlp_weights": mlp_weights,
        "layer_params": layer_params,
        "embedding_params": embedding_params,
        "lm_head_params": lm_head_params,
        "final_norm_params": final_norm_params,
        "total_params": total_params,
        "weight_bytes": total_params * BYTES_PER_BF16,
        "kv_bytes_per_token": kv_bytes_per_token,
        "kv_bytes": kv_bytes,
        "paged_kv_bytes": paged_kv_bytes,
    }


def gibibytes(byte_count):
    return byte_count / (1024**3)


def main():
    ledger = calculate()

    def assert_rejected(call):
        try:
            call()
        except ValueError:
            return
        raise AssertionError("invalid ledger input was accepted")

    assert ledger["head_dim"] == 128
    assert ledger["kv_width"] == 1024
    assert ledger["layer_params"] == 199_247_872
    assert ledger["total_params"] == 8_030_887_936
    assert ledger["kv_bytes_per_token"] == 139_264
    assert ledger["kv_bytes"](32768) == 4.25 * (1024**3)
    assert ledger["kv_bytes"](0) == 0
    assert ledger["paged_kv_bytes"](0) == 0
    assert ledger["paged_kv_bytes"](128) == 128 * ledger["kv_bytes_per_token"]
    assert ledger["paged_kv_bytes"](129) == 256 * ledger["kv_bytes_per_token"]
    assert ledger["paged_kv_bytes"](256) == 256 * ledger["kv_bytes_per_token"]
    huge_tokens = 10**100
    huge_slots = ((huge_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    assert ledger["paged_kv_bytes"](huge_tokens) == huge_slots * ledger["kv_bytes_per_token"]

    for invalid_call in (
        lambda: ledger["kv_bytes"](-1),
        lambda: ledger["kv_bytes"](1.5),
        lambda: ledger["kv_bytes"](True),
        lambda: ledger["kv_bytes"](1, batch=0),
        lambda: ledger["kv_bytes"](1, batch=1.0),
        lambda: ledger["kv_bytes"](1, batch=True),
        lambda: ledger["paged_kv_bytes"](-1),
        lambda: ledger["paged_kv_bytes"](True),
        lambda: ledger["paged_kv_bytes"](1.5),
        lambda: ledger["paged_kv_bytes"](1, batch=0),
        lambda: ledger["paged_kv_bytes"](1, batch=-1),
        lambda: ledger["paged_kv_bytes"](1, batch=1.5),
        lambda: ledger["paged_kv_bytes"](1, batch=False),
        lambda: ledger["paged_kv_bytes"](1, block_size=0),
        lambda: ledger["paged_kv_bytes"](1, block_size=-1),
        lambda: ledger["paged_kv_bytes"](1, block_size=1.5),
        lambda: ledger["paged_kv_bytes"](1, block_size=True),
    ):
        assert_rejected(invalid_call)

    print("openPangu-Embedded-7B-V1.1 | config-derived CPU ledger")
    print(f"head_dim={ledger['head_dim']}; K/V width per projection={ledger['kv_width']}")
    print(f"attention projection weights/layer={ledger['attention_weights']:,} params")
    print(f"attention projection biases/layer={ledger['attention_bias']:,} params")
    print(f"MLP weights/layer={ledger['mlp_weights']:,} params")
    print(f"one decoder layer={ledger['layer_params']:,} params")
    print(f"all model weights={ledger['total_params']:,} params; BF16 payload={gibibytes(ledger['weight_bytes']):.6f} GiB")
    print(f"KV/token across all layers={ledger['kv_bytes_per_token']:,} bytes")
    print(f"KV at 32,768 tokens, batch 1={gibibytes(ledger['kv_bytes'](32768)):.2f} GiB")
    print(f"paged KV at 129 tokens, block {BLOCK_SIZE}={ledger['paged_kv_bytes'](129):,} bytes")
    print("PASS: arithmetic, page-boundary, large-integer, and invalid-input assertions")
    print("No model weights, framework, accelerator, or performance claim used")


if __name__ == "__main__":
    main()
~~~
</details>

运行以下命令可复算并执行程序中的断言：

```bash
python roadmap/curriculum/systems/examples/pangu_embedded7b_ledger.py
```

运行结果应包括 128 head dimension、1024 K/V 投影宽度、单层 199,247,872 参数、BF16 权重 14.958694 GiB、每 token 139,264 bytes 和 32K 上下文 4.25 GiB。这个检查验证算式与整数关系，不验证框架数值正确性、实际显存分配、性能或任意 2.0 部署。

## 10. 证据边界：模型源码、backend 与运行记录

对本地 vLLM-Ascend 案例，检查架构名如何注册、类如何构造、Attention 如何分派，比仅看命令行或分析笔记更可靠。已核对的 `vllm-ascend-pangu-v023` checkout 中，`PanguEmbeddedForCausalLM` 显式映射到 `vllm_ascend.models.open_pangu:PanguEmbeddedForCausalLM`；模型类创建 `PanguEmbeddedAttention`，后者调用 vLLM `Attention` 抽象。NPU backend 的源码包含多种分支和条件，模型类本身并不硬编码“Prefill 必为 Flash、Decode 必为 paged”。需要有同版本、同配置的日志或 profiler 轨迹，才能把具体设备算子归属到一次真实执行。

运行记录须按证据类型解读：

- **TP1 smoke 功能案例：**保存的成功日志尾段包含一条 Embedded-7B 请求的生成文本；运行命令记录的配置为 TP1/PP1、BF16、`enforce_eager=True`、单请求。这说明该次功能 smoke 返回过文本，不提供当前 checkout 的 commit、完整算子轨迹或性能基线。
- **加载失败案例：**一次较早的初始化日志显示 `make_layers` 用关键字 `prefix` 调用层构造函数时产生 `TypeError`，原因是该次构造 `lambda` 的参数签名不匹配。此记录说明该次加载未完成，不是成功执行证据；也不能用失败快照描述当前 checkout。
- 另一份 TP4 材料是旧静态调用链分析；在没有相同 HEAD、启动参数、完整日志和硬件轨迹前，它不是 TP4 实测，更不能替代 TP1 smoke 或 2.0 Pro/Flash 的部署事实。
- 参数与 KV 容量是 CPU 算术结果，不代表实际显存分配或性能；GPU/NPU 执行需在对应硬件、软件版本和配置下另行验证。

将算子优化放进整网时，至少同时保存：模型配置与源码版本、设备和并行配置、eager/compile/graph 选择、输入/输出长度与并发、局部算子 shape 和数值误差、整网 TTFT/TPOT/吞吐/显存，以及 profiler 中的 kernel、copy、通信和 idle 区间。只有这些记录能把“硬件机制 → 代码旋钮 → 预期轨迹变化 → 实测结果”闭合起来。

## 参考阅读

[vLLM 服务参数](https://docs.vllm.ai/en/latest/cli/serve/) · [vLLM 在线基准](https://docs.vllm.ai/en/latest/cli/bench/serve/) · [自动前缀缓存](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) · [大模型推理实践](../../../downloads/大模型推理实践.pdf)（第 4、13 章） · [openPangu-2.0-Flash 官方配置与模型卡（固定版本）](https://huggingface.co/openpangu/openPangu-2.0-Flash/tree/510b2395075d6107985d74311d777e99e7564f13) · [openPangu-2.0-Pro 官方配置与模型卡（固定版本）](https://huggingface.co/openpangu/openPangu-2.0-Pro/tree/6619c11f420ea5e2dfacf2438a81026302d873bb) · [openPangu-2.0-Infer 官方部署说明](https://gitcode.com/ascend-tribe/openPangu-2.0-Infer/blob/main/README_EN.md) · [omniai/omni-npu 上游仓库入口（内容访问受限）](https://gitee.com/omniai/omni-npu)

## 章节导航

- [上一章：Mini Transformer](../model-analysis/mini-transformer/README.md)
- [下一章：多 GPU 与通信](multi-gpu/README.md)
