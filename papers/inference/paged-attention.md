# PagedAttention: Efficient Memory Management for LLM Serving

**Authors**: Kwon, Li, Zhuang et al. (UC Berkeley)  
**Venue**: SOSP 2023 | **arxiv**: [2309.06180](https://arxiv.org/abs/2309.06180)  
**实现**: vLLM | **优先级**: P0 | **状态**: ✅ 精读 | **日期**: 2026-05-28

> **一句话**: PagedAttention 把每条序列的逻辑 KV 位置映射到固定大小、可非连续分配的物理 block，并在请求共享时维护引用关系；它改善 KV 管理，不改变注意力数学。

---

## 为什么这篇论文重要

论文把一个具体瓶颈拆成两部分：KV cache 随请求增长而动态变化，连续预留会产生内部/外部碎片；共享前缀或 beam 又会重复保存相同历史。PagedAttention 的页表式映射和引用管理分别处理这两个问题，系统吞吐仍取决于调度、kernel、模型和负载。

---

## 解决了什么问题

### LLM Serving 的显存瓶颈

推理时，每个请求都要存 **KV Cache**（历史 key/value，避免重算已经处理的上下文）：
```
每层每个 token 的 KV: 2 × num_kv_heads × head_dim × element_bytes
若是 MHA 且 hidden_dim = num_kv_heads × head_dim，才可写成 2 × hidden_dim × element_bytes
总容量还要乘层数、实际 token 数、分页对齐和元数据
```

可用并发由权重、KV、激活、临时 buffer、allocator 保留和服务目标共同决定；不能从显存容量减去权重后直接推出并发数。

### 问题 1: 内存碎片化（Memory Fragmentation）

**传统做法**（FasterTransformer / TGI 早期）：
```python
# 为每个请求预分配连续显存
kv_cache = allocate(max_seq_len * hidden_dim)  # 预分配 2048 token 的空间
```

**碎片来源**：
1. **内部碎片**：请求实际只用 500 token，预分配了 2048，浪费 75%
2. **外部碎片**：请求结束后留下奇怪大小的空洞，新请求放不进去（虽然总空间够）

论文 Figure 1 展示了其对比系统在特定请求设置下的浪费；该比例是实验观察，不是所有模型或框架的固定常数。

### 问题 2: 不支持高级采样（Beam Search / Parallel Decoding）

Beam search 需要 **共享 prefix 的 KV Cache**：
```
Prompt: "Translate to French: Hello"
Beam 1: "Translate to French: Hello" → "Bonjour"
Beam 2: "Translate to French: Hello" → "Salut"
        ^^^^^^^^^^^^^^^^^^^^^^^^^ 这段 KV 应该共享
```

传统方案：复制整个 KV Cache → 显存浪费 × beam_width。

---

## 核心思想：虚拟内存 for GPU

### OS 虚拟内存的类比

| OS 虚拟内存 | PagedAttention | 目的 |
|---|---|---|
| 虚拟地址空间 | 逻辑 block ID | 请求看到的"连续"地址 |
| 物理页 (4KB) | Physical KV block (固定大小) | 实际分配的显存块 |
| 页表 | Block table | 逻辑→物理映射 |
| 页面调度 | Block Manager | 分配/释放/共享 block |
| Copy-on-write | Copy-on-write | Beam search 时延迟复制 |

### PagedAttention 的三大组件

#### 1. Block Table（映射表）

每个请求维护一张表：
```python
# 请求 A: "Hello world, how are you?"（5 token）
# block_size = 4 token/block

block_table_A = [
    0 -> 7,   # 逻辑 block 0 → 物理 block 7
    1 -> 3,   # 逻辑 block 1 → 物理 block 3
]

# 物理显存布局（非连续）：
Physical Block 3: [KV of "you"]
Physical Block 7: [KV of "Hello", "world", "how", "are"]
```

请求自己看到的是"连续"的逻辑 block 0, 1, 2...，实际存在任意位置的物理 block。

#### 2. Block Manager（分配器）

全局管理所有物理 block：
```python
class BlockManager:
    def __init__(self, num_blocks):
        self.free_blocks = set(range(num_blocks))  # 空闲块池
        self.ref_count = [0] * num_blocks          # 引用计数（共享用）
    
    def allocate(self):
        if not self.free_blocks:
            raise OutOfMemory
        block_id = self.free_blocks.pop()
        self.ref_count[block_id] = 1
        return block_id
    
    def free(self, block_id):
        self.ref_count[block_id] -= 1
        if self.ref_count[block_id] == 0:
            self.free_blocks.add(block_id)
    
    def share(self, block_id):  # Copy-on-write
        self.ref_count[block_id] += 1
```

#### 3. Attention Kernel 改造

标准 attention：
```cuda
// K, V 是连续的 [num_tokens, num_kv_heads, head_size]
// 这里只展示按 token 的寻址；dot 是该 token 内 head_size 维归约
for (int i = 0; i < num_tokens; i++) {
    scores[i] = dot(Q[q_idx], K[i]);  // 连续访问 K[i]
}
```

PagedAttention：
```cuda
// K, V 是分散的 blocks
for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
    int physical_block = block_table[block_idx];  // 查表
    
    for (int offset = 0; offset < BLOCK_SIZE; offset++) {
        int token_idx = block_idx * BLOCK_SIZE + offset;
        if (token_idx >= num_tokens) break;
        
        // K 在 (physical_block, offset) 位置
        score += Q[q_idx] * K[physical_block][offset];
    }
}
```

**关键**：逻辑 token 先按 `logical_block = token_idx // BLOCK_SIZE` 定位，再由 `block_table[logical_block]` 找到物理 block，最后使用块内 offset 读取 K/V。页表查找发生在访问路径中，但布局、并行映射和缓存行为由具体 kernel 决定。

---

## 算法详解

### Prefill 阶段（处理 prompt）

```python
def prefill(prompt_tokens, model):
    # 1. 分配 blocks
    num_tokens = len(prompt_tokens)
    num_blocks_needed = ceil(num_tokens / BLOCK_SIZE)
    block_table = [block_manager.allocate() for _ in range(num_blocks_needed)]
    
    # 2. 计算当前 prompt 的 K/V（标准 Transformer forward）
    K, V = model.forward(prompt_tokens)  # 当前层的 [num_tokens, num_kv_heads, head_dim]
    
    # 3. 写入 physical blocks
    for i, token_kv in enumerate(zip(K, V)):
        logical_block = i // BLOCK_SIZE
        offset = i % BLOCK_SIZE
        physical_block = block_table[logical_block]
        
        kv_cache[physical_block][offset] = token_kv
    
    return block_table
```

### Decode 阶段（生成 token）

```python
def decode_layer_step(hidden_t, position, request, layer):
    # 单层、单 Query/KV head 的教学示意；q/k/v 均为一维向量
    # request 仅表示该请求在当前层的缓存状态，不是全模型共享的长度计数
    # hidden_t 是当前层当前 token 的 hidden；外部模型在所有层完成后再采样
    q, k_new, v_new = layer.project_qkv(hidden_t)
    q, k_new = apply_rope(q, k_new, position)

    # 当前 token 的 KV 先写入可写位置；因果 Attention 可以看到自己
    append_pos = request.num_tokens
    if append_pos % BLOCK_SIZE == 0:
        request.block_table.append(block_manager.allocate())
    physical_block = request.block_table[append_pos // BLOCK_SIZE]
    offset = append_pos % BLOCK_SIZE
    kv_cache[physical_block][offset] = (k_new, v_new)
    visible_tokens = append_pos + 1

    # 查表读取历史与当前 KV；这是寻址示意，真实 kernel 会并行计算 dot
    keys, values, scores = [], [], []
    for logical_block, physical_block in enumerate(request.block_table):
        for offset in range(BLOCK_SIZE):
            token_idx = logical_block * BLOCK_SIZE + offset
            if token_idx >= visible_tokens:
                break
            k, v = kv_cache[physical_block][offset]
            keys.append(k)
            values.append(v)
            scores.append(dot(q, k) / sqrt(q.shape[-1]))

    attn_weights = softmax(scores)
    layer_output = sum(attn_weights[i] * values[i] for i in range(visible_tokens))
    request.num_tokens = visible_tokens  # 只推进本层缓存长度
    return layer_output
```

这里的 `decode_layer_step` 只表示一层、单头的 Q/K/V 投影、KV 写入和 Attention 输出；真实模型还要完成多头映射、后续层、残差与输出头，采样发生在完整前向结束后。各层以相同位置处理当前 token，全模型的已提交长度不能在每层重复加一。示例假设尾页已私有且可写；共享尾页必须先执行后文的 Copy-on-Write。`keys/values/scores` 是便于说明逻辑的临时列表，生产 kernel 使用分块归约而不物化这些完整列表。

---

## Copy-on-Write for Beam Search

**场景**：Beam search 要从同一个 prefix 分叉出多个候选。

**朴素做法**：复制整个 KV Cache × beam_width → **显存炸裂**。

**Copy-on-Write**：
```python
# 初始：所有 beam 共享 prefix blocks
request_beam1.block_table = [7, 3]  # 指向同一批物理 block
request_beam2.block_table = [7, 3]  # 共享
block_manager.ref_count[7] = 2
block_manager.ref_count[3] = 2

# Beam 1 生成新 token，需要修改 block 3
if block_manager.ref_count[3] > 1:  # 被共享，不能直接写
    new_block = block_manager.allocate()
    copy(kv_cache[3] -> kv_cache[new_block])  # 复制这一个 block
    request_beam1.block_table[-1] = new_block
    block_manager.free(3)  # Beam 1 不再用旧 block 3

# 现在 Beam 1 和 Beam 2 各自独立
request_beam1.block_table = [7, new_block]  # 独立
request_beam2.block_table = [7, 3]          # 还在共享 prefix
```

**显存节省**：
- 朴素：prefix 1024 token × 4 beams = 4096 token KV
- CoW：prefix 1024 token × 1 份（共享）+ 新生成的独立部分

---

## Block Size 选择

**Trade-off**：
- **大 block**（如 128 token/block）：内部碎片多（请求长度不是 128 的倍数）
- **小 block**（如 4 token/block）：block table 长（查表开销大）

block size 没有跨实现的固定最优值：块越大，页表和分配管理开销可能较低，但尾块内部浪费更大；块越小，尾块浪费可能下降，却会增加页表项、地址计算和 kernel 元数据。论文中的曲线只适用于其模型、设备和请求分布；部署时要按目标版本的配置、KV dtype、head 布局和负载复测。

---

## 性能数据（论文 Table 1-3）

论文在指定模型、设备和请求负载上比较了 vLLM 与当时的系统，并报告吞吐/延迟和 KV 利用率变化。复现实验时应保留模型版本、输入/输出长度分布、并发、batch 调度、block size、KV dtype 和计时边界；PagedAttention 主要改变容量与共享，端到端收益还要看调度和 kernel 是否成为新瓶颈。

## 求职追问：从页表到请求指标

以下是教学示例，不代表个人实测。回答“PagedAttention 为什么有用”时，先说明请求的逻辑 token 序列如何按 block 切分，再说明 block table 如何把它映射到非连续物理块；如果涉及 beam 或前缀共享，还要说明引用计数和 copy-on-write 在第一次写入时如何断开共享。最后把 block 分配、页表查找、KV 读取、调度和请求级 TTFT/TPOT 放到同一条时间线，不能只用显存节省推断吞吐。

追问“怎么验证”时，先用小程序检查逻辑位置、物理 block、尾块和引用计数，再以相同请求分布比较连续 cache 与分页 cache 的峰值显存、有效 batch、kernel 时间和端到端分位数；任何 block size 或利用率结论都限定在记录过的版本和设备上。

---

## 实现细节（vLLM）

### 数据结构

```python
# vllm/core/block_manager.py
class BlockSpaceManager:
    def __init__(self, block_size, num_gpu_blocks):
        self.block_size = block_size  # 由目标版本/配置决定
        self.gpu_blocks = [PhysicalBlock(i) for i in range(num_gpu_blocks)]
        self.free_blocks = list(self.gpu_blocks)
    
    def allocate(self, seq_group):
        num_blocks = ceil(seq_group.num_tokens / self.block_size)
        blocks = [self.free_blocks.pop() for _ in range(num_blocks)]
        seq_group.block_table = blocks

# vllm/worker/model_runner.py
class ModelRunner:
    def execute_model(self, seq_group_metadata_list):
        # 1. 收集所有 block_table
        block_tables = [seq.block_table for seq in seq_group_metadata_list]
        
        # 2. 调用 PagedAttention kernel
        attn_output = paged_attention(
            query, key_cache, value_cache,
            block_tables, block_size
        )
```

### CUDA Kernel（简化）

```cuda
// csrc/attention/attention_kernels.cu
__global__ void paged_attention_kernel(
    float* out,              // [num_seqs, num_heads, head_size]
    const float* q,          // [num_seqs, num_heads, head_size]
    const float* k_cache,    // [num_blocks, block_size, num_heads, head_size]
    const float* v_cache,
    const int* block_tables, // [num_seqs, max_num_blocks]
    int block_size
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    
    // 读这个请求的 block table
    const int* block_table = block_tables + seq_idx * max_num_blocks;
    
    // 每个 token 产生一个 score；dot 在 head_size 维度内归约
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        int physical_block = block_table[block_idx];  // 查表
        
        for (int offset = 0; offset < block_size; offset++) {
            int token_idx = block_idx * block_size + offset;
            if (token_idx >= num_tokens[seq_idx]) break;
            
            // K 在 (physical_block, offset) 位置
            float* k = k_cache + physical_block * block_size * num_heads * head_size
                                + offset * num_heads * head_size
                                + head_idx * head_size;
            
            scores[token_idx] = dot(q, k, head_size);
        }
    }
    // Softmax + 乘 V（同理）
    ...
}
```

**关键**：`block_table[block_idx]` 查表找物理 block，然后按 offset 读 KV。

---

## 与 Flash Attention 的关系

**Flash Attention 解决的是"算 attention 时的 IO 优化"**（tiling + online softmax）。  
**PagedAttention 解决的是"存 KV Cache 时的内存管理"**（分页 + CoW）。

**组合使用**（vLLM 实际实现）：
```
PagedAttention kernel 内部用 Flash Attention 的 tiling 策略
↓
既省显存（PagedAttention），又快（Flash Attention）
```

某个 vLLM 版本的 paged kernel 是否采用何种 tiling、在线归约或版本化实现，应以对应 checkout 的 kernel 和 profiler 为准；PagedAttention 的页表语义与 FlashAttention 的片上计算策略是两个正交层次。

---

## 在 Ascend 的对应

| GPU serving 实现 | 其他设备的候选实现 | 需要核对 |
|---|---|---|
| block table 与物理 KV block | 目标运行时的页表/块管理 | 分配粒度、地址空间与引用语义 |
| block manager | 目标框架内存管理 | allocate/free/share 与异步生命周期 |
| paged attention kernel | 目标设备 kernel | 布局、同步、数值和实际 trace |

**可移植性**：PagedAttention 的思想跨平台通用（OS 虚拟内存是通用概念），只是 kernel 要重写。

---

## 与我何干（学习路径）

### C1 — Prefill vs Decode（推理系统基础）
理解为什么推理要分两阶段：
- **Prefill**：prompt 一次性过 Transformer，生成各位置的 KV；工作量与 shape、batch 和实现有关
- **Decode**：每次生成一个 token，复用历史 KV；每步瓶颈与权重/KV/launch、batch 和 cache 有关

PagedAttention 主要优化的是 **Decode 阶段的 KV Cache 管理**。

### C2 — PagedAttention / KV Cache（本篇）
- 读这篇论文
- 理解 block table 怎么查
- 知道 vLLM 为什么比 FT 快

### C3 — 调度 continuous batching
vLLM 的另一半：怎么把不同长度的请求打包进一个 batch（dynamic batching）。

### 求职追问

**Q1: vLLM 何时可能更快？**
A: PagedAttention 减少 KV 分配浪费并支持共享，continuous batching 改善请求混合；实际收益还取决于 block size、kernel、调度和同条件请求负载。

**Q2: PagedAttention 的核心思想？**  
A: 借鉴 OS 虚拟内存，KV Cache 切成固定大小 block，用 block table 做逻辑→物理映射，消除碎片 + 支持 CoW 共享。

**Q3: Block size 怎么选？**  
A: 在尾块内部浪费、页表/地址计算、kernel 元数据和请求分布之间权衡；用目标版本和负载测峰值显存、有效 batch、kernel 时间与请求分位数。

**Q4: 和 Flash Attention 什么关系？**  
A: Flash Attn 优化"算 attention"（IO-aware tiling），PagedAttn 优化"存 KV"（分页管理）。vLLM 两个都用。

**Q5: Beam search 怎么共享 KV？**  
A: Copy-on-Write。所有 beam 初始指向同一批 block（引用计数 >1），谁要修改谁复制那一个 block，其他 beam 继续共享 prefix。

---

## 代码对照

### vLLM 源码（关键文件）
- `vllm/core/block_manager.py` — BlockSpaceManager（分配器）
- `vllm/worker/model_runner.py` — 调用 PagedAttention kernel
- `csrc/attention/attention_kernels.cu` — CUDA kernel（查表 + Flash-style tiling）

### 读代码路径
1. 先看 `block_manager.py` 怎么 allocate/free/share block
2. 再看 `model_runner.py` 怎么把 block_table 传给 kernel
3. 最后看 `attention_kernels.cu` 怎么查表读 KV

### 本仓库参考
可用不依赖 vLLM 的简化 block manager 与伪 kernel 检查逻辑映射，再将结果与目标版本的实现和 trace 对照。

---

## 扩展：后续优化

### 版本化实现的扩展
- **Prefix caching**：多个请求在身份、模型版本和位置条件一致时共享 prompt 的 KV blocks；具体命中规则依版本实现。
- **Chunked prefill**：长 prompt 切块处理并与 decode 调度交错；块大小与调度开销需要实测。

### 其他 serving 实现
不同 serving 框架可能提供 paged KV 或类似块管理，但 API、块布局、共享语义和性能不能仅凭名称判断，应按各自官方文档和版本源码核对。

### 学术后续
- **DistServe** (OSDI'24): 跨节点的 KV Cache 分页（多 GPU serving）
- **Infinite-LLM** (arxiv'24): KV Cache 换出到 CPU/SSD（超长上下文）

---

## 参考资料

- **论文**: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- **vLLM 官方 repo**: [vllm-project/vllm](https://github.com/vllm-project/vllm)
- **作者博客**: [vLLM: Easy, Fast, and Cheap LLM Serving](https://blog.vllm.ai/2023/06/20/vllm.html)
- **配套课程**: C2 推理系统（算子线 C），[roadmap/vllm.md](../../roadmap/vllm.md)
- **Flash Attention 论文**: [papers/attention/flash-attention.md](../attention/flash-attention.md)

---

*本章的复述重点是 block table、KV 生命周期、copy-on-write 条件，以及它们如何进入请求级测量。*
