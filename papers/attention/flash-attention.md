# Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Authors**: Dao, Fu, Ermon, Rudra, Ré (Stanford)  
**Venue**: NeurIPS 2022 | **arxiv**: [2205.14135](https://arxiv.org/abs/2205.14135)  
**优先级**: P0 | **状态**: ✅ 精读 | **日期**: 2026-05-28

> **一句话**: 通过 IO-aware tiling + online softmax + recomputation，把标准 attention 从 O(N²) 显存降到 O(N)，提速 2-4×。

---

## 为什么这篇论文重要

这是 **第一篇把 attention 变成实际可用**的工作（长序列不 OOM、速度还更快），直接催生了：
- GPT-3/4 等长上下文模型的训练可行性
- vLLM / TensorRT-LLM 的 PagedAttention（Flash Attn 的推理版）
- 所有后续的 attention 优化（Flash-2/3, Ring Attention, Grouped-Query Attention）都基于这套 tiling 框架

面试必考，A5 读代码必看，C2 推理系统的基础。

---

## 解决了什么问题

### 标准 Attention 的瓶颈

Self-attention 公式：
```
S = Q @ K^T        # [N, N]  attention scores
P = softmax(S)     # [N, N]  attention weights
O = P @ V          # [N, d]  output
```

**问题 1: 显存 O(N²)**  
N×N 的 S 和 P 矩阵必须存在 HBM 里（GPU 显存）。N=4096, FP16 → 32MB（单头）；N=16K → 512MB。Multi-head 再乘 32 头就炸了。

**问题 2: Bandwidth-bound（带宽瓶颈）**  
A100 的 HBM 带宽 1.5TB/s，SRAM (L2/shared memory) 带宽 19TB/s，**相差 10×+**。标准实现要：
```
HBM → SRAM: 读 Q, K, V
SRAM: 算 Q @ K^T
SRAM → HBM: 写 S           ← 第一次回 HBM
HBM → SRAM: 读 S
SRAM: 算 softmax(S)
SRAM → HBM: 写 P           ← 第二次回 HBM
HBM → SRAM: 读 P, V
SRAM: 算 P @ V
```
每个 N×N 矩阵都要过一遍慢速 HBM，成为性能瓶颈。

### 数据对比（问题严重性）

| Seq Length | Attention 矩阵显存 (FP16, 单头) | A100 40GB 能放几头? |
|:---:|:---:|:---:|
| 1K | 2 MB | 20000 头 (够用) |
| 4K | 32 MB | 1250 头 (32 头 OK) |
| 16K | 512 MB | 78 头 (**32 头就吃紧**) |
| 64K | 8 GB | 5 头 (**OOM**) |

Llama-7B (32 头 × 128 维 × 4096 seq) 训练一个 batch，仅 attention 矩阵就要 **1GB**。

---

## 核心思想：IO-Aware Tiling

### 关键洞察

> **不要把中间矩阵写回 HBM。** 把 Q/K/V 切成小块，每次只在 SRAM 里算一个 tile 的 attention，算完立即用它更新输出，tile 本身丢弃。

**类比 Ascend**：就像你用 L1 Buffer tiling 做 GEMM——不可能把整个 C 矩阵放进 L1，每次只算一个 TILE×TILE 的小块，算完写回 Global Memory，L1 里的 tile 复用。

### Tiling 策略

```
Q: [N, d] 按行切成 Tr 块，每块 Br 行    (Br ≈ 32~128)
K: [N, d] 按行切成 Tc 块，每块 Bc 行    (Bc ≈ 32~128)
V: [N, d] 同 K

O: [N, d] 输出，初始化为 0
```

双层循环：
```python
for i in range(Tr):         # 遍历 Q 的每一块
    Qi = Q[i*Br:(i+1)*Br]  # [Br, d]
    
    for j in range(Tc):     # 遍历 K 的每一块
        Kj = K[j*Bc:(j+1)*Bc]  # [Bc, d]
        Vj = V[j*Bc:(j+1)*Bc]
        
        Sij = Qi @ Kj.T     # [Br, Bc] ← 小矩阵，放得进 SRAM
        Pij = softmax(Sij)  # 但 softmax 需要全局 max/sum...
        
        Oi += Pij @ Vj      # 增量更新输出
```

**问题**：Softmax 需要**全局 max**（数值稳定）和**全局 sum**（归一化），但我们只看到一个 tile，怎么算？

---

## 核心技术 1: Online Softmax

**标准 softmax（需要全局信息）**:
```python
m = max(S[i, :])              # 全局 max
num = exp(S[i, :] - m)
denom = sum(num)              # 全局 sum
output = num / denom
```

**Online softmax（增量维护）**:  
详细推导见 [notes/algorithms/online-softmax.md](../../notes/algorithms/online-softmax.md)，这里给结论：

维护 **running max (m)** 和 **running sum (l)**，每来一个新 tile 就更新：

```python
# 初始
m_old = -inf
l_old = 0

# 读到 Sij (第 j 个 K tile)
m_new = max(m_old, max(Sij))          # 更新全局 max
correction = exp(m_old - m_new)        # 旧块的 exp 要修正
l_new = l_old * correction + sum(exp(Sij - m_new))

# 输出也要修正（因为之前用的是旧 max）
O_new = O_old * correction + exp(Sij - m_new) @ Vj

m_old = m_new
l_old = l_new
O_old = O_new
```

最后归一化：`O_final = O / l_final`

**关键**：每个 tile 算完后，**不存 Sij 和 Pij**，直接更新 O、m、l，然后 tile 就丢了。显存只需要存 O (N×d) + m (N×1) + l (N×1) = O(N)。

---

## 核心技术 2: Recomputation（反向传播）

### 前向不存中间矩阵

标准实现的反向传播需要：
- 存 S (N×N) 用于算梯度
- 存 P (N×N)

Flash Attention **不存**，只存：
- Q, K, V 本身（本来就要存）
- 每行的 m, l（2N 个 FP32，总共 8N 字节）

### 反向时重新计算

```python
# Backward pass
for i in range(Tr):
    Qi = Q[i*Br:(i+1)*Br]
    mi, li = m[i*Br:(i+1)*Br], l[i*Br:(i+1)*Br]  # 前向存的
    
    for j in range(Tc):
        Kj, Vj = K[j*Bc:(j+1)*Bc], V[j*Bc:(j+1)*Bc]
        
        # 重新算 Sij 和 Pij（用存储的 mi, li 保证数值稳定）
        Sij = Qi @ Kj.T
        Pij = exp(Sij - mi[:, None]) / li[:, None]
        
        # 算梯度 dQ, dK, dV（标准 attention 的反向公式）
        ...
```

**为什么这样更快？**  
- 存 N×N 矩阵：需要 2N² × 2字节(FP16) = 4N² 字节 HBM
- 重算：从 SRAM 读 Q/K，算 Sij，全程在 SRAM（19TB/s），比 HBM 读写（1.5TB/s）快 **10×+**

IO 分析（论文核心定理）：
- 标准实现：HBM 读写量 = Θ(N²·d + N²)
- Flash Attn：HBM 读写量 = Θ(N²·d² / M)，其中 M 是 SRAM 大小

当 d ≪ N 时（典型：d=64~128, N=4K~16K），Flash Attn 的 HBM 访问量是标准的 **1/10 ~ 1/50**。

---

## 算法伪代码（Forward, 单头）

```python
def flash_attention_forward(Q, K, V, Br, Bc):
    """
    Q, K, V: [N, d]
    Br, Bc: block size
    Returns: O [N, d], m [N], l [N]
    """
    N, d = Q.shape
    Tr = ceil(N / Br)
    Tc = ceil(N / Bc)
    
    O = zeros(N, d)
    m = full(N, -inf)
    l = zeros(N)
    
    for i in range(Tr):
        # Load Q block to SRAM
        Qi = Q[i*Br : (i+1)*Br, :]  # [Br, d]
        Oi = zeros(Br, d)
        mi = full(Br, -inf)
        li = zeros(Br)
        
        for j in range(Tc):
            # Load K, V blocks to SRAM
            Kj = K[j*Bc : (j+1)*Bc, :]  # [Bc, d]
            Vj = V[j*Bc : (j+1)*Bc, :]
            
            # Compute attention scores (in SRAM)
            Sij = Qi @ Kj.T  # [Br, Bc]
            
            # Online softmax update
            mij_new = maximum(mi, rowmax(Sij))  # [Br]
            Pij = exp(Sij - mij_new[:, None])   # [Br, Bc]
            lij_new = exp(mi - mij_new) * li + rowsum(Pij)  # [Br]
            
            # Update output (with correction for old max)
            Oi = diag(exp(mi - mij_new)) @ Oi + Pij @ Vj
            
            mi = mij_new
            li = lij_new
        
        # Normalize and write back to HBM
        Oi = Oi / li[:, None]
        O[i*Br : (i+1)*Br, :] = Oi
        m[i*Br : (i+1)*Br] = mi
        l[i*Br : (i+1)*Br] = li
    
    return O, m, l
```

**数据流动**（关键）：
1. Q 块从 HBM → SRAM（一次）
2. K/V 块从 HBM → SRAM（Tc 次，内层循环）
3. Sij, Pij 在 SRAM 里算，**不写回 HBM**
4. Oi 累加完后 HBM ← SRAM（一次）

中间的 N×N 矩阵从未出现在 HBM 里。

---

## 核心技术 3: Causal Masking 优化

Decoder 的 causal attention（因果注意力）：每个 token 只能看到自己和之前的 token，attention 矩阵是下三角：

```
     K0  K1  K2  K3
Q0  [✓] [-∞] [-∞] [-∞]
Q1  [✓] [✓] [-∞] [-∞]
Q2  [✓] [✓] [✓] [-∞]
Q3  [✓] [✓] [✓] [✓]
```

Flash Attention 的优化：第 i 个 Q 块只需要处理 **前 i+1 个 K 块**，后面的直接跳过（连算都不算）。

```python
for i in range(Tr):
    Qi = Q[i*Br : (i+1)*Br]
    
    for j in range(min(i+1, Tc)):  # ← causal: 只到第 i 块
        Kj, Vj = K[j*Bc:(j+1)*Bc], V[j*Bc:(j+1)*Bc]
        ...
```

**加速**：计算量和访存都减半（上三角不碰）。

---

## Block Size 选择（实战关键）

**原则**：让 Qi, Kj, Vj, Sij, Pij 同时放进 SRAM。

A100 的 shared memory per block = 164 KB，假设 FP16：
```
SRAM 占用 = Br·d·2 + Bc·d·2 + Br·Bc·2  (Q块 + K块 + S块)
```

典型配置（d=64）：
- Br=Bc=128 → 128×64×2 + 128×64×2 + 128×128×2 = 32KB + 32KB + 32KB = 96KB ✅
- Br=Bc=256 → 256KB ❌ 放不下

**实战**（论文 + 开源实现）：
- d=64, Br=Bc=128
- d=128, Br=Bc=64

Br=Bc 时（方块 tile）效果最好，因为 Q 和 K 的复用率平衡。

---

## 性能数据（论文 Table 1-3）

### 训练加速（GPT-2 Medium, A100）

| Seq Length | 标准 Attn (ms/iter) | Flash Attn (ms/iter) | 加速 |
|:---:|:---:|:---:|:---:|
| 512 | 85 | 45 | 1.9× |
| 1024 | 170 | 55 | **3.1×** |
| 2048 | 680 | 180 | **3.8×** |
| 4096 | OOM | 710 | ∞ |

### 显存占用（BERT-Large, batch=8）

| Seq Length | 标准 Attn | Flash Attn | 节省 |
|:---:|:---:|:---:|:---:|
| 512 | 8.2 GB | 6.1 GB | 26% |
| 1024 | 18.3 GB | 8.7 GB | **52%** |
| 2048 | OOM | 14.9 GB | ∞ |

### 端到端训练（GPT-2, 125M）

| 配置 | 标准实现 | Flash Attn | 提速 |
|:---:|:---:|:---:|:---:|
| Seq=1K, batch=64 | 1.0× | **2.4×** | - |
| Seq=2K, batch=16 | 1.0× | **3.0×** | - |

---

## 数值稳定性验证

**问题**：Recomputation 会不会导致数值误差累积？

论文实验（Section 4.3）：
- 对比标准实现 vs Flash Attn 的输出
- L2 误差 < 10⁻⁴（FP16）
- **原因**：前向存了 m 和 l，反向重算 softmax 时用同样的 max 做数值稳定，数值路径完全一致

---

## 在 Ascend 的对应

| CUDA (Flash Attn) | Ascend Da Vinci | 备注 |
|---|---|---|
| SRAM (shared memory) | L1 Buffer / UB | Ascend 更大（1MB vs 164KB） |
| Tiling (Br×Bc) | Cube tiling | 思想一致 |
| Online update | Pipe 流式处理 | 增量更新，不存完整矩阵 |
| Recomputation | 前向省 UB，反向重算 | 同样的 trade-off |
| `__syncthreads()` | `pipe_barrier()` | 同步机制 |

**你的优势**：写过 Ascend Cube 算子的 tiling，Flash Attn 的思路一看就懂。区别只是 CUDA 要手写线程循环，Ascend 是指令调。

---

## 与我何干（学习路径）

### A5 — Flash Attn 读代码 (Lesson 05)
你会读 [reference/cuda/flash_attention/flash_attn.cu](../../reference/cuda/flash_attention/flash_attn.cu)，对照上面的伪代码找：
- 双层 for 循环（Tr, Tc）在哪
- `m_new`, `l_new`, `correction` 怎么算
- `__syncthreads()` 在哪些关键位置
- Causal mask 怎么实现（`if (j > i) break`）

### B3 — Triton Flash Attn
用 Triton 重写，会发现：
- Tiling 逻辑更简洁（`tl.load` 自动切块）
- Online softmax 还是那套公式（逃不掉）
- 但不用管 `__syncthreads__`（Triton 编译器插）

### C2 — PagedAttention (vLLM)
Flash Attn 是 PagedAttention 的基础：
- PagedAttention 把 KV Cache 切成 physical blocks
- 每个 block 的 attention 计算 = Flash Attn 的一个 tile
- 论文：[papers/inference/paged-attention.md](../inference/paged-attention.md)

### 面试必考题

**Q1: Flash Attention 为什么快？**  
A: IO-aware tiling。不把 N×N attention 矩阵写回 HBM，全程在 SRAM 里算小 tile，HBM 访问量降 10-50×。

**Q2: 显存复杂度怎么从 O(N²) 降到 O(N)？**  
A: 不存 S 和 P 矩阵，只存 Q/K/V (O(Nd)) + m/l (O(N))。

**Q3: Recomputation 不是会更慢吗？**  
A: 不会。重算在 SRAM (19TB/s)，比从 HBM 读写 (1.5TB/s) 快 10×+。Compute 便宜，memory access 贵。

**Q4: 能处理任意 mask 吗？**  
A: 可以，但非结构化 mask（稀疑 attention）加速效果打折扣。Causal (下三角) 是最优情况。

**Q5: 和 Triton 什么关系？**  
A: Flash Attn 1 是手写 CUDA。后续 Flash-2 用 Triton 重写了部分（生产力更高），但核心算法一样。

---

## 代码对照

### 官方实现
- [HazyResearch/flash-attention](https://github.com/Dao-AILab/flash-attention)
- CUDA kernel 在 `csrc/flash_attn/src/flash_fwd_kernel.h`（~1000 行，heavily optimized）

### 本仓库参考实现
- [reference/cuda/flash_attention/flash_attn.cu](../../reference/cuda/flash_attention/flash_attn.cu)
- 简化版（单头 causal, Br=Bc=32），~200 行，适合学习

### A5 读代码时重点看
1. `flashAttnKernel` 的双层循环（line ~80-150）
2. Online softmax 的 `m_new`, `l_new`, `correction` 计算（line ~110-120）
3. `__syncthreads()` 的两个位置（防什么 race）
4. Shared memory 分配（`__shared__ float s_Q[...]`）

---

## 扩展：Flash Attention 2 / 3

### Flash-2 (2023)
- **改进**：Work partitioning（每个 block 处理更多 Q，减少跨 block 通信）
- **加速**：~2× vs Flash-1
- **论文**: [2307.08691](https://arxiv.org/abs/2307.08691)

### Flash-3 (2024, H100)
- **改进**：利用 H100 的异步 WGMMA (Warp Group Matrix Multiply Accumulate) + TMA (Tensor Memory Accelerator)
- **加速**：~1.5× vs Flash-2 on H100
- **核心算法没变**，只是压榨新硬件

---

## 参考资料

- **论文**: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- **作者博客**: [Tri Dao's blog](https://tridao.me/publications/flash2/flash2.pdf)
- **配套理论笔记**: [notes/algorithms/flash-attention-mechanism.md](../../notes/algorithms/flash-attention-mechanism.md)
- **Online softmax 推导**: [notes/algorithms/online-softmax.md](../../notes/algorithms/online-softmax.md)
- **课程**: [Lesson 05 — Flash Attn 读代码](../../lessons/05-flash-attn-reading.md)

---

*这是 A5/B3/C2 的基础，必须吃透。面试 99% 会问。*
