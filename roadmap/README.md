# Roadmap — 一年总路线 + 未来阶段

> 这是**长期视野**：一年要去哪、各阶段大致顺序。日常执行看 [PATH.md](../PATH.md)，当前焦点看 [NOW.md](../NOW.md)。
> 这里的未来阶段（vLLM/分布式/Agent/面试）现在只有计划，走到了再展开。

## 全景

```
两条平行路径（地位一样，每周并行）：

算子线（动手）  A CUDA打底 ─→ B Triton ─→ C 推理系统 ─→ D 分布式 ─→ E Agent
                ← 现在在 A（A2✅ → A3）
理论线（理解）  GPU优化算法 · 量化 · 注意力演进 · 模型架构 · 推理系统技术 · 训练/并行
```

权重和阶段出口的**权威定义在 [PATH.md](../PATH.md)**，这里不重复。

## 未来阶段详细计划

走到算子线对应阶段时再激活，现在只是占位计划：

| 文件 | 对应阶段 | 内容 |
|------|---------|------|
| [vllm.md](vllm.md) | C 推理系统 | vLLM 源码深挖：Scheduler / PagedAttention / Worker / 量化通路 |
| [distributed.md](distributed.md) | D 分布式 | DDP / FSDP / TP / PP demo + 通信模式 |
| [agents.md](agents.md) | E Agent | MCP demo / Tool Use / RAG project |
| [interviews.md](interviews.md) | 最后 2-3 月 | CUDA / 推理 / 分布式 / 行为面试题库 + 叙事 |
| [leetgpu-ladder.md](leetgpu-ladder.md) | ⭐ 可选深钻 | 各算子超出 B 级的进阶优化层（vec4 / double buffer / tensor core） |

## 关于 leetgpu-ladder（旧的 CUDA 深钻方案）

`leetgpu-ladder.md` 是早期定的"5 算子 × 多层优化到 tensor core"的深钻方案。**现在的方向是 CUDA 只到 B 级（读得懂）**，所以它降级为**可选**：CUDA 打底够用即可，有余力或面试需要再回头啃 tensor core / double buffering。

## 每周节奏

- **工作日**：~30min 读 + ~30min 动手
- **周末**：深度工作（写算子 / 读源码）+ 一条理论线 + 论文扫读
- 不并行多主题，一次一个焦点（[NOW.md](../NOW.md) 给出）

## 参考资源

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA MODE Lectures](https://github.com/cuda-mode/lectures)
- [Triton Language Docs](https://triton-lang.org/)
- [vLLM Source](https://github.com/vllm-project/vllm)
- [Anthropic MCP Docs](https://modelcontextprotocol.io/)
