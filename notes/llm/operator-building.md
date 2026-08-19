# 最新模型与算子构建能力

> 定位：从“能读懂最新模型”升级到“能构建/复现最新模型关键组件与算子”。
> 约束：不另起学习线。构建任务挂回 PATH 执行参考的 Triton 主线、模型结构理论和推理/训练阶段。
> 状态：🚧 草稿，需要按任务逐个完成并留下正确性与性能数字。

---

## 1. 为什么需要“构建能力”

面试和工程里，区别“看过”和“掌握”的关键是：

- 能不能不看参考实现，写出简化版？
- 能不能把最新结构拆成可运行的 kernel？
- 能不能用数字说明构建版本与 reference 的差距？

本路线把最新模型拆成“可构建组件”，并映射到 Triton/CUDA/Python 任务。

---

## 2. 构建能力总路线

### 2.0 统一落地流程

新组件一律按以下顺序推进：

```text
自己从空文件写 -> LeetGPU 验证 -> 真实 GPU 验证 -> benchmark / profiler -> 记录面试口径
```

纯 CUDA kernel 只作为后置深钻，不替代 Triton 主线；除非当前任务本身就是 CUDA 底层机制，否则不提前展开 warp shuffle、手写 FlashAttention 等内容。

```text
Dense Transformer 基础件
  -> GQA / MLA / MoE
  -> FlashAttention 1/2
  -> PagedAttention / Serving
  -> 量化 / 投机解码
  -> 长序列 / SSM
```

每个组件都按同一标准完成：

```text
1. 读懂结构
2. 写出简化实现
3. 对比 reference
4. 记录正确性 + 性能/显存
5. 准备 1 分钟面试口径
```

---

## 3. 组件构建清单

### 3.1 Dense Transformer 基础件

| 组件 | 构建方式 | 验收 |
|------|---------|------|
| RoPE | PyTorch/Triton 旋转位置编码 | 和 HF 输出对齐 |
| RMSNorm | CUDA/Triton row reduce | 和 `torch.nn.RMSNorm` 对齐 |
| GQA | PyTorch/Triton attention | KV cache 尺寸符合手算 |
| SwiGLU | Triton fused MLP | 和 `nn.Linear` 对齐 |

### 3.2 Attention 变体

| 变体 | 构建重点 |
|------|---------|
| MHA | 标准 scaled dot-product attention |
| MQA | 所有 Q head 共享一组 K/V |
| GQA | 按组共享 K/V |
| MLA | 低秩 latent KV，再投影展开 |

构建验收：

- [ ] 能用手算 KV cache 解释省多少显存
- [ ] 能用 Triton 写出至少 GQA 版本
- [ ] 能和 PyTorch reference 对齐

### 3.3 MoE

构建内容：

```text
router
top-k selection
expert forward
load balance loss（可选）
AllToAll 模拟（多卡或虚拟 rank）
```

构建验收：

- [ ] 能写出简化 router + top-k
- [ ] 能画出 token 到 expert 的映射
- [ ] 能说明负载不均衡如何影响延迟

### 3.4 FlashAttention 1/2

构建内容：

```text
tiling
online softmax
register accumulator
causal mask
FA2 work partitioning（warp/tile 分工）
```

构建验收：

- [ ] Triton 版本和 PyTorch ref 对齐
- [ ] 记录显存和速度
- [ ] 能讲 FA1 解决 IO，FA2 解决并行度

### 3.5 PagedAttention / Serving

构建内容：

```text
block table
逻辑 block -> 物理 block
prefix sharing
copy-on-write（可选）
continuous batching 调度模拟
```

构建验收：

- [ ] 能画 block table 映射
- [ ] 能写一个简单 block manager
- [ ] 能说明减少多少碎片

### 3.6 量化

构建内容：

```text
per-tensor / per-channel scale
symmetric / asymmetric quant
dequant
GPTQ/AWQ 简化版（可选）
KV cache 量化
```

构建验收：

- [ ] 能手算 scale / zero_point
- [ ] 能量化/反量化一个 tensor
- [ ] 能对比 FP16 / INT8 / FP8 误差

### 3.7 投机解码

构建内容：

```text
draft model
verify loop
accept / reject
```

构建验收：

- [ ] 能写一个简化 draft-verify 循环
- [ ] 能记录接受率和加速比

### 3.8 SSM / Mamba（可选强化）

构建内容：

```text
状态方程 h_t = A h_{t-1} + B x_t
selective scan 简化版
```

构建验收：

- [ ] 能写出线性 scan
- [ ] 能和 Attention 对比复杂度

---

## 4. 映射到现有学习计划

| 构建任务 | 对应 PATH 阶段 | 执行参考 |
|---------|----------------|---------|
| RoPE / RMSNorm / SwiGLU | 理论线模型结构 + M2 Triton | M2 B2/B5 |
| GQA / MLA | 理论线注意力演进 + M2 Triton | M2 B5 |
| MoE | 理论线模型架构 | M1.5 + M2 B5 |
| FlashAttention 1/2 | 算子线 A5 + M2 Triton | M2 B4 |
| PagedAttention | 算子线 C 推理系统 | M3 |
| 量化 | 理论线量化 | M3 |
| 投机解码 | 理论线推理技术 | M3 |
| SSM / Mamba | 理论线模型架构 | M1.5 |

---

## 5. 完成标准

一个“构建能力”任务算完成，必须满足：

- [ ] 有代码：`solutions/` 或 `scripts/`
- [ ] 有正确性：和 PyTorch / reference 对齐
- [ ] 有数字：GFLOPS / GB/s / 显存 / 误差 / 接受率
- [ ] 有讲解：能画图、能推公式、能说取舍
- [ ] 不是只读笔记：至少有一个简化实现

---

## 6. 关联材料

- [模型结构](architectures.md)
- [推理系统](inference-systems.md)
- [训练系统](training-systems.md)
- [最新模型与结构](../algorithms/latest-model-architectures.md)
- [PATH 执行参考](../../roadmap/ai-infra-curriculum.md)
