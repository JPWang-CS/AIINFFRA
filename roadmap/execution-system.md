# AIINFFRA 统一学习执行系统

> 这是“怎么学、怎么验、怎么留下证据”的唯一流程说明。
> 学什么由 [NOW.md](../NOW.md) 决定，完成状态由 [PATH.md](../PATH.md) 维护；本文件不维护进度。

---

## 1. 仓库只保留五种对象

| 对象 | 回答的问题 | 位置 | 是否维护进度 |
|------|------------|------|:--:|
| 路线 | 先后顺序和验收是什么？ | `roadmap/` | 否 |
| 课程 | 当前单元需要理解什么？ | `lessons/` | 只维护当前单元卡，权威状态仍在 `PATH.md` |
| 知识 | 某个概念/系统到底怎么工作？ | `notes/` | 否 |
| 论文 | 某篇工作解决了什么、证据是什么？ | `papers/` | 否 |
| 产物 | 我亲手写并验证了什么？ | `solutions/`、`weekly/` | 否，状态回写 `PATH.md`；可含明确标记的 WIP |

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

其中对用户可见的章节只保留两段：**S3 = LeetGPU 正确性与代码归档**，**S4–S6 = 服务器真实性能与最终归档**。S0–S2 是写作前的内部准备，不再在 lesson 里重复拆成多个流程表。

### 统一状态模型

| 状态 | 必须具备 | 能否进入下一段 |
|------|----------|----------------|
| `WIP` | 当前代码或 lesson 快照，明确来源和未完成问题 | 不能进入服务器 |
| `LEETGPU_PASS` | 题号、语言、通过日期、原始 `solve` 已归档 | 可以进入服务器 |
| `GPU_VALIDATED` | 实际 GPU 型号、正确性复核、性能数字和 baseline | 可以整理完成材料 |
| `COMPLETE` | 上述证据、失败案例、性能解释、1 分钟口径 | 完成 |

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
- 未跑通前可以留在题目编辑器，也可以保存到 `solutions/` 作为 `WIP` 快照；文件和 lesson 必须明确标记 `WIP`，不能进入完成清单。
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

### LeetGPU 代码归档与学习计划索引

算子通过 LeetGPU 后，必须保留当次平台提交的原始 `solve`/kernel，而不是只留下本地 wrapper 或 `reference/` 参考实现。对应 lesson、`PATH.md` 和算子 README 至少要能一眼查到：

```text
LeetGPU 题目/题号 → 原始 solve 代码 → 本地 solutions 文件 → 正确性/性能证据
```

如果题目已通过但原始代码尚未同步，状态写成“已通过、代码待归档”；如果只有 wrapper/reference，不能写成“LeetGPU 版本已保存”。当前 lesson 必须把代码快照或路径直接放在 LeetGPU 章节，不让用户再跨多个目录寻找。

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

普通算子完成一次有证据的瓶颈定位即可；MatMul、Softmax/Norm、FlashAttention、Fused MLP/GQA 四类性能锚点继续执行 [P0–P8 极致性能阶梯](gpu-foundations.md#32-核心算子的极致性能阶梯)。每次改动必须记录“硬件假设 → 代码旋钮 → 预期 counter → 实测”，不能只保存 autotune 最优参数。

| 场景 | 工具 | 至少回答 |
|------|------|----------|
| 单 kernel | Nsight Compute | throughput、occupancy、stall、memory transaction |
| 多 kernel/系统 | Nsight Systems | launch、copy、stream、compute-communication overlap |
| 通信 | `nccl-tests`、NCCL debug/topology dump | algbw、busbw、路径、slow rank |
| 训练 | PyTorch profiler、memory stats | step time、通信占比、peak memory、MFU/吞吐 |
| 推理 | vLLM/SGLang benchmark | TTFT、TPOT、tokens/s、KV cache、batch |

### S6：归档与口径

`solutions/` 可以提前保存明确标记的 WIP；只有完成门槛齐全，才把它从 WIP 提升为完成产物。每个完成单元至少留下：

- `README` 或 benchmark 表；
- 正确性结果；
- 性能数字和环境；
- LeetGPU 原始 `solve`/kernel、题号、语言和通过日期（有对应题时）；
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

不要为同一主题同时维护多份权威状态；`PATH.md` 是唯一状态源，lesson 只展示与它同步的当前单元卡，README 只展示代码和证据入口。

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
