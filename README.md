# AI Infra Journey

从昇腾 NPU 到 GPU 生态的系统性学习路线，覆盖 CUDA、Triton、vLLM、分布式训练、Agent 系统。

> **入口文件 → [MAIN.md](./MAIN.md)** — 当前进度、本周任务、下一步

## 路线图

```
Phase 1 (8-10w)          Phase 2 (6-8w)         Phase 3 (10-12w)       Phase 4 (8-10w)
CUDA 基础 + 算子          Triton + Flash Attn     vLLM 源码深挖          分布式训练
████████████             ████████                ████████████           ██████████

                    Phase 5: Agent 系统（贯穿全程，每周 2-3h）
                    ████████████████████████████████████████████
```

## 目录

| 目录 | 内容 | 状态 |
|------|------|------|
| [cuda-kernels/](./cuda-kernels/) | CUDA 算子实现（GEMM/Softmax/LayerNorm/FlashAttn） | Phase 1 |
| [triton-kernels/](./triton-kernels/) | Triton 算子实现（Matmul/FusedMLP/FlashAttn） | Phase 2 |
| [vllm-notes/](./vllm-notes/) | vLLM 源码分析（PagedAttention/Scheduler/Worker） | Phase 3 |
| [distributed-demos/](./distributed-demos/) | 分布式训练 Demo（DDP/FSDP/TP/PP） | Phase 4 |
| [agent-lab/](./agent-lab/) | Agent 实验（MCP/ToolUse/RAG） | Phase 5 |
| [papers/](./papers/) | 论文笔记和索引（[流程](./papers/process.md)） | 持续 |
| [interviews/](./interviews/) | 面试准备 | 最后 2-3 月 |
| [scripts/](./scripts/) | 工具脚本（arxiv 抓取等） | 辅助 |
| [leetgpu-roadmap.md](./leetgpu-roadmap.md) | LeetGPU 42 题刷题计划 | 持续 |
| [weekly-log-template.md](./weekly-log-template.md) | 每周学习日志模板 | 持续 |

## 里程碑

- [ ] 第 10 周：完成 CUDA 基础，用 Nsight 分析 kernel 瓶颈
- [ ] 第 18 周：完成 Triton + Flash Attention 实现
- [ ] 第 30 周：完成 vLLM 核心模块分析
- [ ] 第 40 周：完成分布式训练，能画 FSDP/TP/PP 通信图
- [ ] 持续：精读论文 ≥ 30 篇
- [ ] 持续：Agent 方向有 1 个可展示项目

## 每周节奏

- **工作日**：30min 阅读 + 30min 动手
- **周六**：4-6h 深度工作（代码/源码分析）
- **周日**：3-4h 笔记整理 + 论文扫读 + Agent 学习

## 参考资源

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA MODE Lectures](https://github.com/cuda-mode/lectures)
- [Triton Language Docs](https://triton-lang.org/)
- [vLLM Source](https://github.com/vllm-project/vllm)
- [Anthropic MCP Docs](https://modelcontextprotocol.io/)
