# MoE（混合专家模型）推理挑战

> 模型架构类 · 理解 MoE 的 serving 痛点与解法，非架构概述

---

## 解决了什么问题

**为什么大模型要用 MoE？**  
Dense 模型每 token 激活所有参数（如 LLaMA 70B 的 70B 参数全部参与计算），算力线性增长。MoE 将 FFN 层替换为多个"专家"（Expert），每个 token 只路由到 K 个（通常 K=2）专家，**参数量大但激活参数量小**。

| 模型 | 总参数 | 激活参数 | KV Cache 大小 |
|------|:------:|:--------:|:------------:|
| LLaMA 3 70B (dense) | 70B | 70B | 大 |
| Mixtral 8×7B | 47B | ~13B | 小（激活参数少） |
| DeepSeek-V3 | 671B | ~37B | 中 |

**MoE 的推理挑战**：虽然激活参数少（快），但总参数大（需要 Expert 权重常驻显存或频繁换入），serving 时每个 token 可能路由到不同专家，引发大量**随机访存**。

---

## 核心思路

### 1. Expert Routing 机制

每个 token 经过一个 Router（小型线性层 + softmax），选出 Top-K 专家：

```python
# Router: [d_model] → [num_experts]
router_logits = x @ W_router                    # [seq_len, num_experts]
top_k_scores, top_k_indices = topk(router_logits, k=2)  # Top-2

# Gating（每个专家的贡献权重）
gates = softmax(top_k_scores)   # [seq_len, 2]

# 路由：将每个 token 发到对应专家
output = sum(gates[t][i] * Expert[top_k_indices[t][i]](x[t]) for i in range(k))
```

**问题 1：专家利用率不均（load imbalance）**  
如果大部分 token 都路由到少数几个"热门专家"，那些专家串行处理，效率低。

**解决方案（DeepSeek-V3 等）**：
- **Auxiliary loss**：训练时加负载均衡惩罚项，鼓励均匀分布
- **Expert capacity**：给每个专家设置 buffer（capacity = tokens × K / num_experts × capacity_factor），超载则 token 被丢弃（dropped）或用残差跳过

### 2. Expert Parallelism（EP）

MoE 天然支持跨 GPU 分布：每个 GPU 负责一部分专家。

```
4 GPU × 8 experts = 32 experts total
GPU 0: Expert 0-7
GPU 1: Expert 8-15
...

问题：token 要到它被路由的那个 GPU 上计算
解法：AllToAll 通信（每个 GPU 把 token 发给对应的 GPU，收回计算结果）
```

**AllToAll 的带宽开销**：每个 MoE 层需要两次 AllToAll（发送 + 收回），通信量 = `seq_len × d_model × dtype_bytes`。

以 DeepSeek-V3（61 层 MoE，d_model=7168，FP8）为例：  
每层 AllToAll = 61 × 2 × seq_len × 7168 × 1 ≈ 875 MB per 1K tokens，**通信成为瓶颈**。

### 3. Expert 权重的显存管理

Mixtral 8×7B：每个专家权重约 3.5B × 2B = 7GB，8 个专家共 56GB。  
实际 serving 时并非所有专家同时活跃 → **offloading**：将不常用的专家 offload 到 CPU/NVMe，需要时 prefetch。

DeepSeek 的解法：**expert speculation**（预测下一个 token 最可能用哪些专家，提前预取）。

---

## 关键数据/取舍

| 方案 | 吞吐 | 延迟 | 显存 | 适用 |
|------|:----:|:----:|:----:|------|
| 全 GPU（所有专家常驻） | 高 | 低 | 极高 | H100 × 8 集群 |
| Expert offloading | 中 | 高 | 中 | 单机推理 |
| Expert parallelism | 高 | 中（AllToAll） | 低/GPU | 多机推理 |
| Shared Expert（DeepSeek-V3）| 高 | 低 | 中 | 质量-效率均衡 |

**DeepSeek-V3 的共享专家**：2 个专家永远激活（所有 token 必过），256 个路由专家中选 Top-8。共享专家缓解了 load imbalance，提高了知识共享。

---

## 在 Ascend 的对应

MoE 的 AllToAll 通信在 Ascend 用 **HCCL AllToAll** 实现，语义相同。  
Expert 并行在 Ascend 集群（Atlas 900 等）已有实现（MindSpore / CANN 框架层）。  
核心 kernel 挑战一致：**稀疏矩阵乘**（scatter/gather + GEMM）。

---

## 与我何干

**C 线推理系统**：vLLM 支持 MoE（Mixtral 等），底层就是 Expert Parallelism + AllToAll。读 vLLM 的 `mixtral.py` 模型实现时会遇到。

**理论线后续**：MoE kernel 优化（grouped GEMM、排序 token、减少 AllToAll）是热门研究方向。

**[面试]**：
- "Mixtral 为什么比 LLaMA 70B 推理快？" → 激活参数少（13B vs 70B），虽然总参数大但每 token 计算量小
- "MoE 推理的主要挑战？" → load imbalance + expert 权重显存 + AllToAll 通信开销
- "Expert parallelism 怎么做通信？" → AllToAll：每 GPU 把要处理的 token 发给对应 Expert 所在 GPU，两次 AllToAll 一个 MoE 层

## 参考

- Mixtral 8×7B: [arxiv 2401.04088](https://arxiv.org/abs/2401.04088)
- DeepSeek-V3: [arxiv 2412.19437](https://arxiv.org/abs/2412.19437)
- Expert offloading: [Pre-gated MoE (2023)](https://arxiv.org/abs/2308.12066)
