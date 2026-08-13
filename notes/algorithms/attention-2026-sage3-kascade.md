# 2026 注意力新变体：SageAttention3（FP4）与 Kascade（anchor 稀疏）

> 注意力演进类 · 两条新路线：量化注意力 + 训练免稀疏注意力 · 论文整理
> 挂靠：主线 A 注意力实现侧 · 量化字典 · 面试全景

---

## 解决了什么问题

FlashAttention 已经把"别读写 N×N 中间矩阵"做到极致。2026 年的新优化从两个方向继续压长序列推理的成本：

1. **带宽再降一个量级**：把 Q/K 直接量化到 FP4（SageAttention3），attention 的矩阵乘走 Blackwell 的 FP4 tensor core
2. **只给重要 token 花算力**：跨层复用 top-k 索引，只算最相关的历史（Kascade）

两者都不改整体架构，近似/无损各有取舍，目标是让长上下文更便宜。

---

## SageAttention3：FP4 量化注意力

### 核心思路

- 把 Q/K 量化到 **FP4（microscaling 格式：每块带一个小 scale）**，V 保持更高精度
- 用 Blackwell 的 FP4 tensor core 做 QK^T 和 PV 两个矩阵乘
- Q/K 的 HBM 带宽直接降 **4x**（相对 BF16）；长序列时 attention 是带宽瓶颈，收益最大
- 论文还探索了 8-bit 训练（SageBwd），训练侧仍是开放问题

### 关键数据与取舍

- 长序列推理相对 FlashAttention 快 **2-5x**（论文口径，长序列优势最明显）
- 依赖 Blackwell FP4 硬件；4090（Ada）没有 FP4 tensor core，只能学原理
- 属于"插拔式"：不需要重新训练模型，推理时直接换 kernel（HF 上有现成模型权重直接跑）

取舍：

- FP4 只有 2-3 bit 尾数，精度靠 microscaling 的 per-block scale 兜底
- 量化误差对长上下文/检索类任务更敏感，上线前必须评测
- 和 FP8 时代一样的问题：量化注意力最终是"默认选项"还是"可选优化"，取决于精度-速度权衡

---

## Kascade：训练免的跨层稀疏注意力

### 核心思路

两个观察：

1. **post-softmax 注意力天然稀疏**：多数历史 token 的权重接近 0
2. **高权重 key 的身份在相邻层之间很稳定**：这一层重要的 token，下一层通常也重要

做法：

```text
在少数 anchor layer 上：算 exact top-k（head-aware，每个 head 独立选）
在中间 reuse layer 上：直接复用 anchor 的索引，只对这 k 个 key 做 attention
```

- anchor 层不是随便选的：用动态规划在开发集上挑"跨层相似度最大"的层组合
- 训练免：任何现成模型都能套，不需要重新训练
- kernel 做 tile 级操作（tile_size=32），目前主要支持 fp16，vLLM 集成在 experimental 分支

### 关键数据与取舍

- top-k = 10% 时：H100 上 decode 最快 **4.1x**、prefill **2.2x**
- anchor 层越多越准但越贵；DP 选层是部署前要做一次的工作
- 稀疏率越高精度掉得越快；10% 附近是论文推荐的工作点

取舍：

- **和 DSA 对比**：DSA 用可学习 indexer（要训练、每层每头动态选），Kascade 用结构复用（免训练、跨层共享同一批索引）——两条路正好是"学习 vs 结构"的对照
- 跨层复用假设"相邻层高权重 key 稳定"，如果模型不满足这个性质（深层语义变化大），需要更多 anchor 层兜底

---

## 与主线的关系：2026 注意力优化全景

```text
可编程：FlexAttention + FA4      → 任意 mask/块稀疏，编译器生成 kernel
量化：  SageAttention3           → Q/K 压到 FP4，带宽 4x 下降
稀疏：  DSA（可学习 indexer）     → top-k 动态选择，生产已验证
稀疏：  Kascade（跨层复用）       → 免训练，anchor layer 静态复用
```

面试口径："FlashAttention 之后 attention 还能怎么优化？" 标准答法就是上面四条线，能各说一句核心机制 + 一个数字。

## 与我何干

- **理论线**：SageAttention3 挂"量化"子类，Kascade 挂"注意力演进"子类——正好把你已有的 INT8/FP8 知识和 FA 知识接上
- **硬件限制**：两个方案都主要在 H100/Blackwell 上验证，本地 4090 只做概念、不做实机
- **C 阶段**：读 vLLM 源码时可以看 Kascade 的 paged kernel 集成分支，和 PagedAttention 的 block 结构直接相关

---

*配套：[DSA 稀疏注意力](dsa-sparse-attention.md) · [FA4/FlexAttention](fa4-flexattention.md) · [量化 INT8/FP8](quantization-int8-fp8.md)*