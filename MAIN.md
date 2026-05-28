# MAIN — 学习主线

> 每次打开这个文件就知道：**现在到哪了、这周做什么、下一步是什么**
> 更新频率：每周日回顾时更新进度

---

## 当前状态

| 项目 | 状态 |
|------|------|
| **当前 Phase** | Phase 1 — CUDA 基础 (W1-10) |
| **当前周** | Week 1 (2026-05-29 ~ 2026-06-04) |
| **本周主题** | GPU 架构 + CUDA 编程模型 |
| **总进度** | ░░░░░░░░░░░░░░░░░░░░ 0/40 周 |

---

## Phase 进度

```
Phase 1 CUDA     ░░░░░░░░░░ 0/10 周
Phase 2 Triton   ░░░░░░░░░░ 0/8 周
Phase 3 vLLM     ░░░░░░░░░░ 0/12 周
Phase 4 分布式    ░░░░░░░░░░ 0/10 周
Phase 5 Agent    持续进行中
```

---

## 本周任务 (Week 1)

### 阅读
- [ ] CUDA C++ Programming Guide — Ch1: Introduction
- [ ] CUDA C++ Programming Guide — Ch2: Programming Model (grid/block/thread)
- [ ] CUDA C++ Programming Guide — Ch3: Memory Hierarchy
- [ ] 整理笔记到 `cuda-kernels/notes/gpu-architecture.md`

### 动手 (LeetGPU)
- [ ] [1_vector_add](https://leetgpu.com) — 第一个 kernel，理解 grid/block/thread
- [ ] [31_matrix_copy](https://leetgpu.com) — 练习 global memory 读写
- [ ] [19_reverse_array](https://leetgpu.com) — stride access pattern

### 论文（本周不强制）
- [ ] 扫 arxiv 标题，有感兴趣的存档到 `papers/README.md`

### 输出
- [ ] 填 `weekly-log-template.md` → 另存为 `logs/week-01.md`

---

## 下一步 (Week 2 预览)

```
主题: GPU 架构深入 + 内存层级
LeetGPU: 3_matrix_transpose, 8_matrix_addition, 9_1d_convolution
阅读: CUDA Guide Ch4-5, shared memory + coalescing
```

---

## 里程碑检查

- [ ] 第 10 周 — CUDA 基础完成，Nsight profiling
- [ ] 第 18 周 — Triton + Flash Attention 实现
- [ ] 第 30 周 — vLLM 核心模块分析
- [ ] 第 40 周 — 分布式训练，通信图
- [ ] 持续 — 论文精读 ≥ 30 篇
- [ ] 持续 — Agent 可展示项目

---

## 快捷链接

| 需求 | 文件 |
|------|------|
| 刷题计划 | [leetgpu-roadmap.md](./leetgpu-roadmap.md) |
| 刷题平台 | [leetgpu.com](https://leetgpu.com) |
| 论文索引 | [papers/README.md](./papers/README.md) |
| 论文流程 | [papers/process.md](./papers/process.md) |
| 抓论文 | `python scripts/fetch_papers.py --days 3` |
| 本周日志 | [logs/week-01.md](./logs/week-01.md) |
| GPU 架构对比 | [cuda-kernels/notes/gpu-architecture.md](./cuda-kernels/notes/gpu-architecture.md) |
| GEMM 代码 | [cuda-kernels/gemm/gemm.cu](./cuda-kernels/gemm/gemm.cu) |
| 面试大纲 | [interviews/README.md](./interviews/README.md) |

---

## 规则

1. **每周日更新此文件**：推进 week 数字 + 写下周任务
2. **每天打开先看此文件**：5 秒知道今天做什么
3. **不要跳周**：每道 LeetGPU 题做完在 roadmap 里打 ✅
4. **遇到值得记录的知识点**：直接写到对应 notes 文件，不要新建碎片文件
5. **每周 commit 一次**：`git add -A && git commit -m "week-XX: <本周主题>"`

---

*最后更新: 2026-05-29 · Week 1*
