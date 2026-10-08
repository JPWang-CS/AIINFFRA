# MoE（混合专家模型）推理挑战

> 模型架构类 · 从路由、专家计算和通信理解 MoE 推理路径

---

## 解决了什么问题

**为什么大模型要用 MoE？**  
稠密（dense）模型在每个 token 上调用同一层的全部前馈网络参数；混合专家（MoE，Mixture of Experts）把这层拆成多个专家（expert），由路由器（router）为每个 token 选择少数专家。于是总参数容量可以增长，而单个 token 参与的专家矩阵乘数量由路由选择决定；这只描述计算路径，不等于未选专家不需要存储或不产生调度成本。

| 模型 | 总参数 | 激活参数 | KV Cache 大小 |
|------|:------:|:--------:|:------------:|
| LLaMA 3 70B (dense) | 70B | 70B | 由层数和 KV head 决定 |
| Mixtral 8×7B | 47B | 约 13B | 由层数和 KV head 决定 |
| DeepSeek-V3 | 671B | 约 37B | 由 MLA/层配置决定 |

**MoE 的推理挑战**：总专家参数可能超过单卡容量，且一个 batch 内的 token 会被分散到不同专家。执行器必须先按专家编号重排 token，再运行不同大小的专家 GEMM，最后按原 token 顺序合并；专家放置跨设备时还要加入 dispatch/combine 通信。是否更快取决于这些步骤与稠密基线的关键路径比较，而不是只看激活参数量。

---

## 核心思路

### 1. Expert Routing 机制

每个 token 经过一个 Router（小型线性层，通常产生路由分数），再按目标模型定义选择 Top-K 专家。下面的 Top-2 仅是示例，实际 K、归一化和共享专家规则必须从模型配置/论文读取：

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
如果大部分 token 都路由到少数几个热门专家，分桶后的 token 数就会偏斜：热门专家的 GEMM 排队，其他设备空闲，尾部完成时间由最慢的桶决定。均匀路由只改善一种负载形状，不能保证专家 GEMM 的矩阵尺寸已经适合硬件。

**可能的训练期缓解手段**：
- **Auxiliary loss**：训练时加入负载均衡目标，改变路由分布；它会影响训练目标，不能在推理端单独补救既有偏斜。
- **Expert capacity**：给每个专家预留容量。超出容量时，具体实现可能丢弃 token、走残差或采用其他溢出策略，必须和模型训练时的语义保持一致。

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

**AllToAll 的带宽开销**：每个 MoE 层通常有一次 dispatch 和一次 combine。若本轮有 T 个 token、每个 token 选择 K 个专家、隐藏宽度为 D、元素占 b bytes，所有选择都跨 rank 时，发送与回收的 payload 上界约为 `2 × T × K × D × b`；同一 rank 内的选择、打包格式、重复 token 合并和量化会改变实际链路字节数。先按专家所在 rank 统计本地与远端桶，再用通信 trace 计算有效流量和等待时间。

因此，不能仅凭层数和隐藏宽度断言通信成为瓶颈。需要把 router、排序/打包、专家 GEMM、两次 collective、反排序和残差合并放到同一时间线，比较哪一段位于关键路径。

### 3. Expert 权重的显存管理

专家权重是否常驻取决于模型总量、精度、设备容量和请求覆盖的专家集合。若需要 **offloading**，不常用专家可放到 CPU 或 NVMe，并在路由后预取；传输和预取会占用关键路径，必须与全 GPU 常驻基线比较。

一种待验证的系统策略是 **expert speculation**：根据历史路由预测下一步可能访问的专家并提前预取；它会引入预测错误、额外显存和调度成本，不能当作某个模型已采用的事实。

---

## 关键数据/取舍

| 方案 | 吞吐 | 延迟 | 显存 | 适用 |
|------|:----:|:----:|:----:|------|
| 全 GPU（所有专家常驻） | 避免换入 | 少一条传输路径 | 高 | 权重可驻留且通信不是主瓶颈 |
| Expert offloading | 受预取命中影响 | 可能增加恢复等待 | 较低 | 容量受限、专家访问有局部性 |
| Expert parallelism | 取决于桶大小 | 增加 collective 等待 | 分摊到各 GPU | 通信拓扑和负载可控 |
| Shared Expert（DeepSeek-V3）| 仍需测路由与共享层 | 取决于实现 | 由完整配置决定 | 需要共享能力且保留稀疏路由 |

DeepSeek-V3 报告了共享专家与路由专家并存的设计；讨论其收益时，应分别统计共享路径、路由路径和通信，而不是把共享专家直接当作负载均衡保证。

---

## 在 Ascend 的对应

在 Ascend 迁移时，AllToAll 的数学语义可对应 HCCL 等通信原语；具体 MoE expert parallel、打包布局和 kernel 是否由目标 MindSpore/CANN 版本提供，需要查对应版本文档与 trace。可迁移的核心问题仍是 token 分桶、scatter/gather、专家 GEMM 和通信重叠。

---

## 与我何干

**C 线推理系统**：vLLM 支持 MoE（Mixtral 等），底层就是 Expert Parallelism + AllToAll。读 vLLM 的 `mixtral.py` 模型实现时会遇到。

**理论线后续**：MoE kernel 优化（grouped GEMM、排序 token、减少 AllToAll）是热门研究方向。

**求职追问段落见下方独立小节。**

## 求职追问与示范回答

以下是教学示例，不代表个人实测：
- “为什么不能只用激活参数量判断 MoE 更快？”——激活参数量只估算专家矩阵乘；还要把 router、按专家分桶、不同桶的 GEMM 效率、dispatch/combine、权重驻留或预取放入同一条关键路径。与 dense 基线比较时，要固定模型、请求形状、精度和 batch。
- “Expert Parallelism 的通信量怎么估？”——先记录每个 token 的 top-k 专家和专家所在 rank，再按本地/远端桶统计元素数；全跨 rank 的粗略上界是两次 `T×K×D×b`，实际值还受打包、重复目标和 collective 实现影响。用通信 trace 与每 rank 的等待时间核对。
- “如何定位负载不均？”——同时看每个专家 token 数、每个 rank 的 GEMM 时间、collective 时间和空闲区间；若只看平均吞吐，会掩盖最慢桶决定尾延迟的事实。

## 参考

- Mixtral 8×7B: [arxiv 2401.04088](https://arxiv.org/abs/2401.04088)
- DeepSeek-V3: [arxiv 2412.19437](https://arxiv.org/abs/2412.19437)
- Expert offloading: [Pre-gated MoE (2023)](https://arxiv.org/abs/2308.12066)
