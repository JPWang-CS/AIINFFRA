---
name: "operator-building"
description: "最新模型与算子构建技能 - 指导 GQA/MLA/MoE/RoPE/RMSNorm/FlashAttention/PagedAttention 等组件的简化实现"
---

# Operator Building Skill

## 用途

帮助用户从“能读最新模型”升级到“能构建最新模型关键组件”，所有任务都要有代码、正确性、数字和讲解。

## 构建总路线

```text
Dense Transformer 基础件
  -> GQA / MLA / MoE
  -> FlashAttention 1/2
  -> PagedAttention / Serving
  -> 量化 / 投机解码
  -> 长序列 / SSM
```

## 统一完成标准

- [ ] 有代码：`solutions/` 或 `scripts/`
- [ ] 有正确性：和 PyTorch / reference 对齐
- [ ] 有数字：GFLOPS / GB/s / 显存 / 误差
- [ ] 有讲解：能画图、能推公式、能说取舍
- [ ] 不是只读笔记：至少有一个简化实现

## 组件任务

| 组件 | 最小实现 |
|------|---------|
| RoPE | PyTorch/Triton 旋转位置编码 |
| RMSNorm | CUDA/Triton row reduce |
| GQA | Triton attention，K/V head < Q head |
| MLA | PyTorch 低秩 KV 简化版 |
| MoE | router + top-k + expert forward |
| FlashAttention | Triton tiling + online softmax |
| PagedAttention | block table 模拟 |
| 量化 | scale / zero_point / dequant |
| 投机解码 | draft-verify 循环 |
| SSM/Mamba | 简化 linear scan |

## 指导流程

1. 先读 `notes/llm/operator-building.md` 和 `notes/llm/architectures.md`。
2. 每个组件先写数学定义，再写简化实现。
3. 和 reference 对比正确性。
4. 记录一个可量化数字。
5. 写 1 分钟面试口径。

## 输出格式

```text
📦 构建任务：[组件]

📐 结构：
- [组件核心机制]

✅ 正确性：
- [对比对象和误差]

⚡ 数字：
- [GFLOPS / GB/s / 显存 / 误差]

💡 面试口径：
- [1 分钟讲法]
```

## 调用时机

- 用户想构建最新模型组件
- 用户需要把 GQA/MLA/MoE/FlashAttention 写成代码
- 用户问“这个结构怎么实现”
