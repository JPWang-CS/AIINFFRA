# AIINFFRA 统一学习执行系统

> 这是“怎么学、怎么验、怎么留下证据”的唯一流程说明。
> 学什么由 [NOW.md](../NOW.md) 决定，完成状态由 [PATH.md](../PATH.md) 维护；本文件不维护进度。

---

## 1. 仓库只保留五种对象

| 对象 | 回答的问题 | 位置 | 是否维护进度 |
|------|------------|------|:--:|
| 路线 | 先后顺序和验收是什么？ | `roadmap/` | 否 |
| 课程 | 当前单元需要理解什么？ | `lessons/` | 否 |
| 知识 | 某个概念/系统到底怎么工作？ | `notes/` | 否 |
| 论文 | 某篇工作解决了什么、证据是什么？ | `papers/` | 否 |
| 产物 | 我亲手写并验证了什么？ | `solutions/`、`weekly/` | 否，状态回写 `PATH.md` |

`reference/` 只用于完成自写版本后的对照，不属于个人产物。

---

## 2. 单个学习单元的七段闭环

```text
S0 定义问题
 -> S1 最小知识
 -> S2 从空文件实现
 -> S3 平台正确性门
 -> S4 真实服务器 benchmark
 -> S5 profiler + 单变量优化
 -> S6 归档 + 面试口径
```

### S0：定义问题

开始前只写清四件事：

- 输入、输出、shape、dtype；
- 正确性 reference 和误差标准；
- FLOPs / bytes / communication volume；
- 预计是 compute、memory、launch 还是 communication bound。

### S1：最小知识

只读当前实现必需的内容。输出不是长笔记，而是一张最小知识卡：

```text
概念：
为什么需要：
关键公式/数据流：
Ascend -> NVIDIA 映射：
实现时最容易错的点：
```

### S2：从空文件实现

- 新算子从空文件开始。
- 未跑通前留在临时工作区或题目编辑器，不放入 `solutions/`。
- 可以读 lesson 的结构/TODO；完成第一版前不复制 `reference/` 或公开答案。

### S3：平台正确性门

实验类型不同，门槛不同：

| 类型 | 正确性门 | 通过证据 |
|------|----------|----------|
| CUDA/Triton 算子且 LeetGPU 有对应题 | LeetGPU | 题号、语言、通过日期、关键边界 case |
| CUDA/Triton 算子但平台无对应题 | 自建 harness + PyTorch/CPU reference | seed、shape、dtype、误差、命令 |
| 通信原语 | `nccl-tests` / PyTorch distributed reference | ranks、bytes、算法带宽、bus bandwidth |
| 训练/推理系统 | 单卡 baseline → 单机多卡 → 多机 | loss/输出一致、吞吐、显存、扩展效率 |
| 纯阅读/论文 | 手算、画图或口述验收 | 一页笔记和反例/边界 |

LeetGPU 适用于算子正确性和受控性能对照；不能把不支持的分布式实验硬塞进 LeetGPU。算子有对应题时，LeetGPU 未通过不进入服务器 benchmark。

### S4：真实服务器 benchmark

每次先固定环境，再记录结果：

```text
date / git commit:
GPU / Compute Capability / GPU count:
driver / CUDA / PyTorch / Triton / NCCL:
topology / interconnect:
shape / dtype / seed:
warmup / repeats / synchronization:
baseline:
time / GB/s / GFLOPS / tokens/s / communication bandwidth:
correctness:
```

服务器阶段的最低要求：

1. 记录实际 GPU 型号，不能写“服务器 GPU”。
2. 排除首次编译/JIT，明确 warmup 与重复次数。
3. CUDA event 或框架 benchmark 前后正确同步。
4. 与 PyTorch、cuBLAS、NCCL baseline 或上一版本对比。
5. 不用不同 GPU、dtype、shape 的数字做直接排名。

### S5：Profiler + 单变量优化

一次只改变一个变量，例如 tile、`num_warps`、stages、fusion、bucket size、parallel degree。

| 场景 | 工具 | 至少回答 |
|------|------|----------|
| 单 kernel | Nsight Compute | throughput、occupancy、stall、memory transaction |
| 多 kernel/系统 | Nsight Systems | launch、copy、stream、compute-communication overlap |
| 通信 | `nccl-tests`、NCCL debug/topology dump | algbw、busbw、路径、slow rank |
| 训练 | PyTorch profiler、memory stats | step time、通信占比、peak memory、MFU/吞吐 |
| 推理 | vLLM/SGLang benchmark | TTFT、TPOT、tokens/s、KV cache、batch |

### S6：归档与口径

只有 S3–S5 的证据齐全，代码才进入 `solutions/`。每个完成单元至少留下：

- `README` 或 benchmark 表；
- 正确性结果；
- 性能数字和环境；
- 一个失败版本及原因；
- 1 分钟面试口径：是什么、瓶颈、优化、证据、取舍。

最后才更新 `PATH.md`；若当前焦点改变，再更新 `NOW.md`。

---

## 3. LeetGPU 与服务器的职责边界

```text
本机/解释器：接口、索引、边界、最小逻辑排错
LeetGPU：受控题面下的正确性门和平台性能反馈
真实服务器：目标 GPU 上的可复现 benchmark
Profiler：解释为什么，不只记录快慢
```

常见错误：

- LeetGPU 通过就宣称“性能优化完成”；
- 服务器跑通但没有平台/自建正确性门；
- 只记录 leaderboard percentile，不记录题面、语言与提交版本；
- 服务器只贴耗时，不记录 GPU、shape、dtype 和 baseline；
- 多次同时改 tile、dtype、融合策略，无法建立因果。

---

## 4. 三种标准实验卡

### 4.1 算子卡

```text
problem -> formula -> baseline -> own kernel
-> LeetGPU/reference gate -> real GPU -> profiler -> optimization log
```

指标：误差、µs/ms、GB/s、GFLOPS、相对 baseline、关键 profiler counter。

### 4.2 系统卡

```text
single process -> single GPU -> single-node multi-GPU
-> multi-node -> topology/failure injection -> scaling analysis
```

指标：正确性、step latency、throughput、peak memory、communication ratio、strong/weak scaling efficiency。

### 4.3 论文卡

```text
inbox -> relevance triage -> attach to current PATH node
-> reproduce one claim/figure -> one-page note -> interview statement
```

论文不因“进入 inbox”算已学；只有产生可验证输出才改变状态。

---

## 5. 建议目录约束

```text
roadmap/       顺序与验收
lessons/       当前单元教学
notes/         跨任务复用的知识
papers/inbox/  自动抓取，未筛选
papers/        已筛选/精读论文
reference/     外部参考，不直接改写成自己的成果
solutions/     通过正确性门和服务器验证的产物
weekly/        阶段复盘
scripts/       抓取、验证、benchmark 辅助工具
```

不要为同一主题同时维护多份状态；所有状态只回写 `PATH.md`。

---

## 6. 当前主线如何使用本流程

当前仍是 Triton MatMul：

1. S0：写清 FP32 row-major 题面、FLOPs 与误差。
2. S1：只学 program mapping、K tiling、`tl.dot`、mask。
3. S2：从空文件做单 tile，再加 K loop。
4. S3：LeetGPU #02 通过。
5. S4：RTX 3090 记录至少 3 组 tile/`num_warps` 与 GFLOPS。
6. S5：解释最快/最慢配置的资源和数据复用差异。
7. S6：代码、表格、失败案例、1 分钟口径入库。

GPU 架构补强按 [gpu-foundations.md](gpu-foundations.md) 挂载；不会另开一条并行大计划。
