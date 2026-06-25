# Flash Attention 2: Faster Attention with Better Parallelism and Work Partitioning

**Authors**: Tri Dao (Princeton / Together AI)  
**Venue**: ICLR 2024 | **arxiv**: [2307.08691](https://arxiv.org/abs/2307.08691)  
**优先级**: P0 | **状态**: ✅ 精读

> Flash Attention 2 通过重新设计工作分区策略、减少非矩阵乘法运算量、优化线程块并行度，在 A100 上实现了高达 225 TFLOPS（72% MFU），比 Flash-1 快 2x，比 PyTorch 标准 attention 快 9x。

---

## 为什么需要 Flash-2（Flash-1 的问题）

Flash-1 已经解决了 IO 复杂度问题（O(N²) → O(N) HBM reads），但在 GPU 实际利用率上仍留有大量空间：

**问题 1: 线程块间工作不均衡**
- Flash-1 的并行维度仅为 batch size × head 数量
- 对于长序列推理（batch=1，1 个 head），只有 1 个线程块在工作，GPU 严重闲置
- A100 有 108 个 SM，batch=1, heads=8 时只用了 8/108

**问题 2: forward pass 中 non-matmul FLOPs 占比过高**
- Flash-1 对每个 block 的 rescaling（更新 O accumulator）是逐元素运算
- A100 上 CUDA Core 吞吐只有 Tensor Core 的 1/16（FP32）或 1/2（FP16）
- 这些 rescale 操作时间占比 ~40%，严重拖慢整体

**问题 3: backward pass 分区与 forward 不对称**
- Flash-1 backward 对 K/V 的写入存在竞争（多个线程块更新同一 dK/dV）
- 需要 atomic 操作，大幅降低并行度

**问题 4: 序列并行度不足**
- 在超长序列（128K tokens）时，batch/head 维度已不够填满 SM

---

## 核心改进 1: Work Partitioning（工作分区重设计）

### Flash-1 的分区方式

```
线程块分配：一个 (batch, head) pair → 一个线程块
并行度 = batch_size × num_heads
```

batch=1, heads=8 时，只有 8 个线程块，108 个 SM 中 100 个空闲。

### Flash-2: Q 参与并行（Forward）

```
并行维度 = batch_size × num_heads × ⌈seq_len / Br⌉

seq_len=4096, Br=64 → 64 个 Q-blocks
batch=1, heads=8 → 8 × 64 = 512 个线程块
A100 108 SM → 每 SM 约 4-5 个线程块，充分利用
```

每个线程块负责 Q 的一个行块（Br 行），串行遍历所有 K/V 块。

### Flash-2: K/V 外层并行（Backward）

Flash-1 backward 中多个线程块竞争写 dK/dV，Flash-2 换方向：

```
外层并行：K/V 的列块（每个 K/V 块只有一个线程块负责 → 无竞争写）
内层循环：Q 的行块
```

代价：forward/backward 分区方向相反，但换来无锁写入。

### Causal Masking 优化

Flash-2 识别出上三角（j > i）的 K/V 块整块被 mask，**直接跳过**，只对对角线块做 masking：

```
Br=Bc=64, N=4096：
  跳过的块：64×63/2 = 2016 个（上三角）
  有效计算量减少约 50%
```

---

## 核心改进 2: 减少 non-matmul FLOPs

### 硬件背景

A100 算力分布（BF16）：
- Tensor Core: 312 TFLOPS
- CUDA Core (element-wise): ~19 TFLOPS（差距 ~16×）

**non-matmul 运算是严重瓶颈**。

### Flash-1 的问题

```python
# 每次新 K/V block 后，必须 rescale 已有的 O accumulator
O = exp(m_old - m_new) * O + P_j @ V_j
#   ^^^^^^^^^^^^^^^^^^^
#   对整个 O 矩阵（Br × d）做逐元素乘法，CUDA Core 运算，N/Bc 次
```

对 O 的 rescaling 每次内层循环都要做，共 N/Bc 次。

### Flash-2 的优化

**关键洞察**：不需要在每步都保持 O 的归一化状态，推迟到最后处理一次。

Flash-2 将 rescaling 与矩阵乘法通过循环结构调整融合，非矩阵运算占比从 ~13% 降到 ~3-5%。

量化收益：
```
Flash-1: non-matmul FLOPs ≈ 总 FLOPs 的 10-15%，但占 ~40% 时间
Flash-2: non-matmul FLOPs 降至 ~5%，时间占比 <10%
净效果：同样 wall-clock time，做了更多有效 matmul
```

---

## 算法伪代码（Forward，单头）

```python
def flash_attention_2_forward(Q, K, V, causal=False):
    O = zeros(N, d)
    L = zeros(N)          # logsumexp，供 backward 使用

    for i in range(ceil(N / Br)):   # 可并行
        Q_i = Q[i*Br : (i+1)*Br]
        O_i = zeros(Br, d)
        m_i = full(Br, -inf)
        l_i = zeros(Br)

        j_end = (i+1) if causal else ceil(N/Bc)  # causal 跳上三角
        for j in range(j_end):
            K_j = K[j*Bc : (j+1)*Bc]
            V_j = V[j*Bc : (j+1)*Bc]

            S_ij = Q_i @ K_j.T / sqrt(d)         # Tensor Core
            if causal and j == i:
                S_ij = apply_causal_mask(S_ij)

            m_new  = maximum(m_i, rowmax(S_ij))
            P_ij   = exp(S_ij - m_new[:, None])   # unnormalized
            l_new  = exp(m_i - m_new) * l_i + rowsum(P_ij)

            O_i = diag(exp(m_i - m_new)) @ O_i + P_ij @ V_j
            m_i, l_i = m_new, l_new

        O[i*Br:(i+1)*Br] = O_i / l_i[:, None]     # 最后一次 normalize
        L[i*Br:(i+1)*Br] = m_i + log(l_i)         # logsumexp 存 HBM

    return O, L
```

**与 Flash-1 的关键区别**：
1. 外层 `for i` 循环可并行（多线程块各自负责一个 `i`）
2. 最终 normalize 只做一次（不是每步）
3. 存 `L = m + log(l)` 供 backward，不存 N×N attention 矩阵

---

## 性能数据

### Forward Pass 吞吐（A100 SXM4 80GB，BF16）

| 方法 | seq=512 | seq=1K | seq=2K | seq=4K | seq=8K |
|------|:-------:|:------:|:------:|:------:|:------:|
| PyTorch standard | 155 | 160 | 145 | 105 | OOM |
| Flash-1 | 166 | 175 | 170 | 162 | 149 |
| **Flash-2** | **185** | **200** | **210** | **215** | **218** |

单位：TFLOPS（A100 理论峰值 312 TFLOPS）

**Flash-2 最高达 225 TFLOPS ≈ 72% MFU**

### 端到端训练加速（GPT-3 style，175B，A100）

| 框架 | MFU |
|------|-----|
| Megatron-LM 原始 | 30-35% |
| + Flash-1 | 46% |
| **+ Flash-2** | **72%** |

### 内存占用

| 方法 | seq=8K, batch=1 的 activation memory |
|------|--------------------------------------|
| Standard Attention | 8 GB（存完整 N×N attention） |
| Flash-1/2 | **20 MB**（只存 L, m） |

---

## Flash-3 (H100) 简述

Flash-3 针对 H100 两个新硬件特性：

**TMA（Tensor Memory Accelerator）**：异步 SMEM↔HBM 数据搬运，完全解耦计算与 IO，消除 memory stall。

**WGMMA（Warpgroup MMA）**：替代 A100 的 HMMA，支持更大 tile（64×256），减少指令 dispatch overhead。

额外支持 FP8 attention（实验性），理论达 1979 TFLOPS。

**性能**：Flash-3 on H100 达 ~740 TFLOPS（75% MFU），约 Flash-2 on A100 的 3.3×。

关键区别：
- Flash-2: 软件层面 work partitioning
- Flash-3: H100 硬件特性挖掘（TMA + WGMMA + FP8）

---

## 在 Ascend 的对应

| GPU | Ascend |
|-----|--------|
| SRAM/Shared Memory | L1 Buffer / UB |
| CUDA Core element-wise | Vector Unit |
| TMA (H100) | AIPP / DMA Engine |
| WGMMA | Cube Unit + AscendC API |

Ascend Vector Unit 与 Cube Unit 物理分离，可用 `pipe` 流水并行——这正是 Flash-2 减少 non-matmul 开销的同等思路。

---

## 与我何干

**A5 Flash Attn 读代码**：理解 Flash-2 的 work partition 后，读 [reference/cuda/flash_attention/flash_attn.cu](../../reference/cuda/flash_attention/flash_attn.cu) 里的双层循环和 online softmax，一下就能对上。

**B3 Triton Flash Attention**：Triton 教程 `06-fused-attention.py` 实现的就是 Flash-2（带 causal + GQA 支持），是 B3 的核心任务。

**[面试] 必考数字**：
- Flash-2: 225 TFLOPS, 72% MFU on A100
- Flash-1 → Flash-2: ~2× speedup
- Flash-2 → Flash-3 (H100): ~3.3×
- Memory: O(N²) → O(N)

## 参考

- **论文**: [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- **官方 repo**: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- **前置**: [flash-attention.md](flash-attention.md) · [notes/algorithms/online-softmax.md](../../notes/algorithms/online-softmax.md)
- **课程**: [Lesson 05](../../lessons/05-flash-attn-reading.md)
