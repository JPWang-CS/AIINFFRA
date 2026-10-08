# FlashAttention-4 与 FlexAttention：可编程的注意力

> 注意力演进类 · 2026-03 PyTorch 公布 FA4 backend 集成 · FlexAttention 编译路径 · 官方博客 + 仓库整理
> 挂靠：主线 A/B 注意力实现侧 · B3 之后 · torch.compile 方向

---

## 解决了什么问题

注意力变体会改变 score、mask 和可见的 tile：causal、ALiBi、soft-capping、窗口、文档 mask 和块稀疏都可能需要不同的数据流。若每种变体都手写 CUDA kernel，研究验证和维护成本都会上升。FlexAttention 的目标是让用户描述 score/mask 规则，再由编译器和后端把规则嵌入 fused attention；这解决的是编程接口与实现复用问题，不保证每个规则都得到特化 kernel 的速度。

## 核心思路

### 1. FlexAttention：把 mask 变成可编程的

PyTorch 的 FlexAttention 提供两个"钩子"，用户不写 kernel，只写规则：

```text
score_mod：score = score_mod(score, b, h, q_idx, kv_idx)
           逐元素修改 score 的函数
           causal / ALiBi / softcap / 文档 mask 都只是这个函数的不同实现

block_mask：声明哪些块稀疏（整块跳过）
            用块级可见性元数据支持跳过整块计算
```

FlexAttention 会把这些规则编译进 kernel。用户可以快速改数学规则，但仍要确认编译支持、动态 shape、mask 元数据和目标后端；“改一行函数”不等于所有配置都能无代价复用编译结果。

### 2. FlashAttention-4：高性能后端

- **2026-03-05 PyTorch 官方博客公布** FlexAttention 的 FlashAttention-4 backend（Hopper / Blackwell）；这条证据支持“已公开集成”，不等于所有 CUDA wheel 或所有设备都默认可用
- 把 score/mask 修改内联进 FA 的实现，而不是先物化完整 scores；PyTorch 博客明确说明后端和编译器集成覆盖 forward/backward 的开发路径，但具体算子、动态标量和梯度能力仍随版本变化
- 支持 FlexAttention 的块稀疏元数据；官方博客说明 Blackwell FA4 路径的最小调度单元受双 Q-tile pipeline 约束，可为 256×128，而 Triton 路径的粒度不同。这是该后端的实现条件，不是所有 block mask 的通用默认值
- 官方博客报告在 compute-bound workload 上相对 Triton backend 的 1.2–3.2×，这是该文给定 workload 和实现版本的结果，不是所有 mask 或模型的端到端加速保证

## 关键数据与取舍

| 维度 | 传统特化 FA kernel | FlexAttention + FA4 |
|---|---|---|
| 新 mask/稀疏变体 | 可能需要调整特化实现及调度 | 用 score_mod 和 BlockMask 表达，并验证支持范围 |
| 性能 | 针对特定模式优化，仍需测量 | 受规则、稀疏结构和编译实现影响 |
| 稀疏粒度 | 由实现的分块与掩码路径决定 | 受后端调度单元限制，例如博客中的 Blackwell 256×128 |
| 硬件 | FA2 覆盖较广；FA3 的作者实现面向 Hopper | FA4 backend 面向 Hopper / Blackwell；Blackwell 使用新的异步管线与 TCGEN05/TMEM 路径 |

取舍：

- **灵活性换性能**：元素级 score 修改仍可能保留完整 tile 的读写；只有 block mask 让后端跳过整块 KV tile 时，才有机会减少对应的 global/HBM 访问。即便是块稀疏，也要看块密度、元数据、尾块和调度开销，不能保证一定节省端到端 HBM 流量
- **FA4 目标硬件是 Hopper / Blackwell**；4090（Ada）不能据此宣称拥有同一 FA4 实机路径，FlexAttention 的 Triton backend 仍覆盖更广硬件
- 和 FA1→FA2→FA3 的关系：FA2 是较广硬件上的并行/访存改进，FA3 的重点是 Hopper，FA4 则针对 Hopper/Blackwell 的新管线并加入 Flex 的扩展点；不能把 FA3 的 Hopper 能力或布局直接外推到 FA4/Blackwell

## 与我何干

- **B3 Triton Flash Attention**：你写 causal mask 时，本质就是在写一个 score_mod（`scores = tl.where(offs >= offs_kv, scores, -inf)`）。理解"mask 是 score 修改的特例"，以后加窗口/文档 mask 不用重写结构。
- **C 阶段**：FlexAttention 是 PyTorch 编译栈的方向，和 torch.compile 是同一套思路（写规则、编译器生成 kernel）；面试聊"2026 注意力编程模型趋势"提这个比只背 FA 论文更显深度。
- **4090 限制**：Ada 上不能把 FA4 backend 的硬件性能当作可运行事实。可以用 PyTorch 的数学/reference 或可用的 Triton backend 检查 `score_mod`/`mask_mod` 的语义；这与“目标后端存在、能编译、在 Hopper/Blackwell 上达到 FA4 性能”是三件事。

## 求职核对题

- `score_mod` 修改的是 softmax 前的 score，`block_mask` 影响的是哪些 tile 进入循环；为什么元素级稀疏不能自动带来与块跳过相同的 HBM 收益？
- FlexAttention 生成的 kernel 与手写特化 kernel 比较时，哪些 shape、mask、dtype、编译缓存和 backward 条件必须固定？
- 4090 上可以验证 API 语义，但为什么不能把 CPU/Triton fallback 的结果写成 FA4 的硬件性能？

---

*配套：FA2 机制 [flash-attention-2.md](flash-attention-2.md) · [PyTorch FA4 官方博客](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/) · [FA4 官方实现](https://github.com/Dao-AILab/flash-attention) · B3 Triton Flash Attention（Lesson 按需生成）*
