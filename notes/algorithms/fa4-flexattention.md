# FlashAttention-4 与 FlexAttention：可编程的注意力

> 注意力演进类 · 2026-03 FA4 发布 · PyTorch FlexAttention 集成 · 官方博客 + PR 整理
> 挂靠：主线 A/B 注意力实现侧 · B3 之后 · torch.compile 方向

---

## 解决了什么问题

模型侧的注意力变体越来越多：causal、ALiBi、soft-capping、窗口注意力、文档 mask、多模态 grid、块稀疏……每个变体都需要对应的高性能 kernel。

传统做法是"一个变体写一个 kernel"，成本高、维护难。需要的是：**用户用高层 API 描述"怎么改 scores、哪些块跳过"，编译器和后端生成高性能 kernel**。

## 核心思路

### 1. FlexAttention：把 mask 变成可编程的

PyTorch 的 FlexAttention 提供两个"钩子"，用户不写 kernel，只写规则：

```text
score_mod：score = score_mod(score, b, h, q_idx, kv_idx)
           逐元素修改 score 的函数
           causal / ALiBi / softcap / 文档 mask 都只是这个函数的不同实现

block_mask：声明哪些块稀疏（整块跳过）
            结构化稀疏才有收益，粒度是块不是元素
```

FlexAttention 会把这些规则编译进 kernel：用户改一行函数，就得到一个新变体，不用碰底层实现。

### 2. FlashAttention-4：高性能后端

- **2026-03 正式发布**，PyTorch FlexAttention 新增 FA4 backend（Hopper / Blackwell）
- 把 score 修改**内联进 FA 的实现**（forward 和 backward 都支持），而不是"先算完整 scores 再改"——否则就回到 HBM 写中间矩阵的老路
- 支持 FlexAttention 的块稀疏元数据；Blackwell 上默认稀疏块大小 (256, 128)
- PyTorch 2026-07 的 PR 把 FA4 设为**数据中心 Blackwell 上 FlexAttention 的默认后端**（CUDA wheel 外接 FA4 b21，配 QuACK / CuTeDSL 栈）
- 官方宣传口径是"把 attention 推到矩阵乘法级别的速度"——即把 non-matmul 开销继续压到接近零

## 关键数据与取舍

| 维度 | 传统特化 FA kernel | FlexAttention + FA4 |
|---|---|---|
| 新 mask/稀疏变体 | 重写 kernel | 改一个 score_mod 函数 |
| 性能 | 特化场景最优 | 结构化稀疏场景接近特化 |
| 稀疏粒度 | 整块跳过（causal 等） | block 稀疏，Blackwell 默认 (256,128) |
| 硬件 | Ampere+（FA2/3） | FA4 主要 Hopper / Blackwell |

取舍：

- **灵活性换性能**：非结构化 mask（任意元素级稀疏）收益有限，只有块级稀疏才能真正省 HBM 读写
- **FA4 深度绑定 Blackwell**（稀疏硬件 + 新指令栈），4090（Ada）上没有 FA4 的实机路径
- 和 FA1→FA2→FA3 的关系：FA 家族把"已知模式"做到极致，FlexAttention/FA4 把"未知模式"也纳入同一个框架

## 与我何干

- **B3 Triton Flash Attention**：你写 causal mask 时，本质就是在写一个 score_mod（`scores = tl.where(offs >= offs_kv, scores, -inf)`）。理解"mask 是 score 修改的特例"，以后加窗口/文档 mask 不用重写结构。
- **C 阶段**：FlexAttention 是 PyTorch 编译栈的方向，和 torch.compile 是同一套思路（写规则、编译器生成 kernel）；面试聊"2026 注意力编程模型趋势"提这个比只背 FA 论文更显深度。
- **4090 限制**：FA4 跑不了，但 FlexAttention API 在 CPU/任何 torch 上都能写——可以先在 torch 里体验 score_mod 和 block_mask，再回 Triton 自己实现。

---

*配套：FA2 机制 [flash-attention-2.md](flash-attention-2.md) · B3 Triton Flash Attention（Lesson 按需生成）*