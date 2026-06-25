# PagedAttention: Efficient Memory Management for LLM Serving

**Authors**: Kwon, Li, Zhuang et al. (UC Berkeley)  
**Venue**: SOSP 2023 | **arxiv**: [2309.06180](https://arxiv.org/abs/2309.06180)  
**实现**: vLLM | **优先级**: P0 | **状态**: ✅ 精读 | **日期**: 2026-05-28

> **一句话**: 借鉴 OS 虚拟内存的分页管理，把 KV Cache 切成固定大小的 blocks，用 block table 做逻辑→物理映射，显存利用率从 20% 提升到 80%+。

---

## 为什么这篇论文重要

这是 **LLM serving 的分水岭**：
- vLLM 凭这篇论文成为 serving 事实标准（取代 FasterTransformer / TensorRT-LLM 的早期版本）
- 所有后续推理框架（TGI, LMDeploy, TensorRT-LLM v0.6+）都抄了这套机制
- **面试必考**："vLLM 为什么快？" 答案就是 PagedAttention + continuous batching

C2 推理系统（算子线）的第一篇论文，必须吃透。

---

## 解决了什么问题

### LLM Serving 的显存瓶颈

推理时，每个请求都要存 **KV Cache**（past key/value，避免重算历史 token 的 attention）：
```
每层每个 token 的 KV: 2 × hidden_dim × 2 bytes (FP16)
Llama-7B (32 层, hidden=4096):
  单 token KV = 2 × 32 × 4096 × 2 = 512 KB
  1024 token 序列 = 512 MB
  8 个并发请求 = 4 GB（只是 KV Cache！）
```

A100 40GB，扣掉模型权重（14GB），剩 26GB。理论上能跑 50+ 并发，**实际只能跑 10-15 个**。为什么？

### 问题 1: 内存碎片化（Memory Fragmentation）

**传统做法**（FasterTransformer / TGI 早期）：
```python
# 为每个请求预分配连续显存
kv_cache = allocate(max_seq_len * hidden_dim)  # 预分配 2048 token 的空间
```

**碎片来源**：
1. **内部碎片**：请求实际只用 500 token，预分配了 2048，浪费 75%
2. **外部碎片**：请求结束后留下奇怪大小的空洞，新请求放不进去（虽然总空间够）

实测显存利用率只有 **20-40%**（论文 Figure 1）。

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
// K, V 是连续的 [num_tokens, hidden_dim]
for (int i = 0; i < num_tokens; i++) {
    score += Q[q_idx] * K[i];  // 连续访问 K
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

**关键**：每次访问 K/V 前先查 `block_table`，找到物理位置。

---

## 算法详解

### Prefill 阶段（处理 prompt）

```python
def prefill(prompt_tokens, model):
    # 1. 分配 blocks
    num_tokens = len(prompt_tokens)
    num_blocks_needed = ceil(num_tokens / BLOCK_SIZE)
    block_table = [block_manager.allocate() for _ in range(num_blocks_needed)]
    
    # 2. 计算 KV（标准 Transformer forward）
    K, V = model.forward(prompt_tokens)  # [num_tokens, hidden_dim]
    
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
def decode_step(request, model):
    # 1. 最后一个 block 满了吗？
    last_logical_block = len(request.block_table) - 1
    last_block_size = request.num_tokens % BLOCK_SIZE
    
    if last_block_size == 0:  # 满了，分配新 block
        new_block = block_manager.allocate()
        request.block_table.append(new_block)
    
    # 2. PagedAttention（查表访问历史 KV）
    Q_new = model.embed(request.last_token)
    scores = []
    
    for logical_block in range(len(request.block_table)):
        physical_block = request.block_table[logical_block]
        
        for offset in range(BLOCK_SIZE):
            token_idx = logical_block * BLOCK_SIZE + offset
            if token_idx >= request.num_tokens:
                break
            
            K_past = kv_cache[physical_block][offset].K
            scores.append(Q_new @ K_past)
    
    # 3. Softmax + 乘 V（同理）
    attn_weights = softmax(scores)
    output = sum(attn_weights[i] * V_past[i] for i in range(len(scores)))
    
    # 4. 新 token 的 KV 写入最后一个 block
    new_kv = model.project(output)
    physical_block = request.block_table[-1]
    offset = request.num_tokens % BLOCK_SIZE
    kv_cache[physical_block][offset] = new_kv
    
    request.num_tokens += 1
    return output
```

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

**实测**（论文 Figure 5）：
| Block Size | 显存利用率 | 吞吐 (req/s) |
|:---:|:---:|:---:|
| 4 | 85% | 2100 |
| 16 | **80%** | **2300** ← 最优 |
| 64 | 68% | 2250 |
| 128 | 55% | 2200 |

**vLLM 默认**：**16 token/block**（每 block 约 256KB for Llama-7B）

---

## 性能数据（论文 Table 1-3）

### vs FasterTransformer（NVIDIA 官方）

| 模型 | 请求吞吐 (req/s) | 提升 |
|:---:|:---:|:---:|
| OPT-13B (A100) | FT: 0.8 / vLLM: **2.5** | **3.1×** |
| LLaMA-7B (A100) | FT: 1.1 / vLLM: **2.9** | **2.6×** |

### 显存利用率

| 框架 | KV Cache 利用率 | Batch Size (相同显存) |
|:---:|:---:|:---:|
| FasterTransformer | 20-40% | 16 |
| Orca (MSR) | ~50% | 24 |
| **vLLM (PagedAttention)** | **80%+** | **48** |

**结论**：同样 40GB 显存，vLLM 能跑的并发数是 FT 的 **2-3 倍**。

### 端到端延迟

| Prompt Length | FT P50 (ms) | vLLM P50 (ms) | 改善 |
|:---:|:---:|:---:|:---:|
| 256 | 48 | 45 | -6% |
| 512 | 92 | 85 | -8% |
| 1024 | 180 | 165 | **-8%** |

延迟略优（block table 查表开销小），吞吐大幅提升（利用率高）。

---

## 实现细节（vLLM）

### 数据结构

```python
# vllm/core/block_manager.py
class BlockSpaceManager:
    def __init__(self, block_size, num_gpu_blocks):
        self.block_size = block_size  # 16 token/block
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
    
    float score = 0.0f;
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        int physical_block = block_table[block_idx];  // 查表
        
        for (int offset = 0; offset < block_size; offset++) {
            int token_idx = block_idx * block_size + offset;
            if (token_idx >= num_tokens[seq_idx]) break;
            
            // K 在 (physical_block, offset) 位置
            float* k = k_cache + physical_block * block_size * num_heads * head_size
                                + offset * num_heads * head_size
                                + head_idx * head_size;
            
            score += dot(q, k, head_size);
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

vLLM 的 `paged_attention_v2` 就是 Flash-style tiling + block table 查表。

---

## 在 Ascend 的对应

| vLLM (GPU) | Ascend | 备注 |
|---|---|---|
| Block table (GPU global mem) | DDR 的分页表 | NPU 也有虚拟地址 |
| Block Manager | Runtime 内存管理 | Ascend 也能做 CoW（AscendCL API） |
| PagedAttention kernel | 手写 Ascend C kernel | 逻辑一样：查表→读 KV |

**可移植性**：PagedAttention 的思想跨平台通用（OS 虚拟内存是通用概念），只是 kernel 要重写。

---

## 与我何干（学习路径）

### C1 — Prefill vs Decode（推理系统基础）
理解为什么推理要分两阶段：
- **Prefill**：prompt 一次性过 Transformer，生成所有 token 的 KV Cache（compute-bound）
- **Decode**：每次生成 1 个 token，复用历史 KV（memory-bound）

PagedAttention 主要优化的是 **Decode 阶段的 KV Cache 管理**。

### C2 — PagedAttention / KV Cache（本篇）
- 读这篇论文
- 理解 block table 怎么查
- 知道 vLLM 为什么比 FT 快

### C3 — 调度 continuous batching
vLLM 的另一半：怎么把不同长度的请求打包进一个 batch（dynamic batching）。

### 面试必考题

**Q1: vLLM 为什么快？**  
A: PagedAttention（分页管理 KV Cache，利用率 80%+ vs 传统 20-40%）+ continuous batching（动态批处理，减少 GPU 空闲）。

**Q2: PagedAttention 的核心思想？**  
A: 借鉴 OS 虚拟内存，KV Cache 切成固定大小 block，用 block table 做逻辑→物理映射，消除碎片 + 支持 CoW 共享。

**Q3: Block size 怎么选？**  
A: Trade-off 内部碎片 vs 查表开销。vLLM 实测 16 token/block 最优（Llama-7B 约 256KB/block）。

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
C 线推理系统（算子线）走到 C2 时，我会给你一个简化版的 block manager + 伪 kernel（不依赖 vLLM 庞大的代码库，纯教学）。

---

## 扩展：后续优化

### vLLM 0.2+ 的改进
- **Prefix caching**：多个请求共享相同 prompt 的 KV blocks（不用每次重算）
- **Chunked prefill**：长 prompt 切块处理，和 decode 混合 batch（减少 bubble）

### TensorRT-LLM v0.6+ 抄了 PagedAttention
NVIDIA 自己也承认 vLLM 更优，后来 TRT-LLM 加了 "Paged KV Cache"，本质就是 PagedAttention。

### 学术后续
- **DistServe** (OSDI'24): 跨节点的 KV Cache 分页（多 GPU serving）
- **Infinite-LLM** (arxiv'24): KV Cache 换出到 CPU/SSD（超长上下文）

---

## 参考资料

- **论文**: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- **vLLM 官方 repo**: [vllm-project/vllm](https://github.com/vllm-project/vllm)
- **作者博客**: [vLLM: Easy, Fast, and Cheap LLM Serving](https://blog.vllm.ai/2023/06/20/vllm.html)
- **配套课程**: C2 推理系统（算子线 C），[roadmap/vllm.md](../../roadmap/vllm.md)
- **Flash Attention 论文**: [papers/attention/flash-attention.md](flash-attention.md)

---

*C2 推理系统的核心，面试 vLLM/推理必考。吃透 block table 和 CoW。*
