# AI Infra Journey

从昇腾 NPU 到 GPU 生态的系统性学习。方向：ML 系统工程师 · Triton 为主力 · CUDA 为底层 · 推理系统并行。

> ## 👉 现在学什么 → [NOW.md](./NOW.md)
> 完整路径和进度 → [PATH.md](./PATH.md)　|　长期路线 → [roadmap/](./roadmap/)

---

## 这个仓库怎么用

**拉模式**：不预排课表。你问"学什么"，我读 [PATH.md](./PATH.md) 进度，生成 [NOW.md](./NOW.md)（指向具体课 + 代码任务 + 验收 + 一条理论）。学完告诉我，我更新进度 + 按 git 活动自动写回顾。

两条平行路径（地位一样，每周并行）：
- **算子线** — 动手写代码，学 GPU/算子（CUDA → Triton → 推理…）
- **理论线** — 学算法/理论（量化、新架构、GPU 优化算法…）

三个控制文档：
- **[NOW.md](./NOW.md)** — 进来先看：现在做什么 + 接下来（两条线并列）
- **[PATH.md](./PATH.md)** — 知识地图：找任何知识一跳到位 + 权威进度
- **本文件** — 知识库索引（你在这）

---

## 知识库（按domain查）

### 学习内容
| 区 | 内容 |
|----|------|
| [lessons/](./lessons/) | 主题课：CUDA 基础 → GEMM → Softmax → Flash Attn → Triton |
| [notes/cuda/](./notes/cuda/) | CUDA 速查 + 深入：内存模型、warp、架构对比、Triton 底层对照、LeetGPU 题库 |
| [notes/triton/](./notes/triton/) | Triton 速查 + Triton vs CUDA |
| [notes/algorithms/](./notes/algorithms/) | 理论线：量化、新算法、GPU 优化算法、推理技术（6 子类） |
| [papers/](./papers/) | 论文笔记 + 索引（[流程](./papers/process.md)） |

### 代码
| 区 | 内容 |
|----|------|
| [solutions/](./solutions/) | **我自己写的**算子（跑通才进） |
| [reference/cuda/](./reference/cuda/) | CUDA 参考实现（看不抄）+ CMake + 工具库 |
| [reference/triton/](./reference/triton/) | Triton 参考实现 |

### 计划 / 未来
| 区 | 内容 |
|----|------|
| [roadmap/](./roadmap/) | 一年总路线 + 未来阶段（vLLM / 分布式 / Agent / 面试 / 可选 CUDA 深钻） |
| [weekly/](./weekly/) | 回顾周报（完成一轮后写） |
| [scripts/](./scripts/) | 工具脚本（arxiv 抓取） |

---

## 当前位置

进度只记在一处 → **[NOW.md](./NOW.md)**（现在 + 接下来）和 [PATH.md](./PATH.md)（全貌）。
环境：4090（CUDA 12.4 / PyTorch 2.5.1），日常用 LeetGPU。

---

## 规则

1. **自己写** —— [reference/](./reference/) 是参考，从空文件开始，跑通后存进 [solutions/](./solutions/)
2. **LeetGPU 为主** —— 跑通才算完成
3. **一次一个焦点** —— 不并行多主题，听 [NOW.md](./NOW.md)
4. **两条线并行** —— 算子线写代码 + 理论线补一条（[notes/algorithms/](./notes/algorithms/)）
5. **完成一步 commit** —— `<主题>: <实际做了什么>`，回顾我来写
