# 学习计划补密度 + A5 进度校准 — 2026-07-22 ~ 2026-08-03

> 复盘型周报 · 这次没有新的算子代码提交，主要完成两件事：校准真实进度，把计划从“地图”升级为“课表+验收”。

## 实际状态

- 07-25 最后一次提交后，仓库没有新的 CUDA/Triton 代码提交。
- A5 Flash Attn CUDA 读码只完成准备（lesson/参考/机制笔记），实际逐段读码未完成。
- A4 Softmax：仓库里只有 3-pass `softmax_naive.cu` 和 2-pass fused `softmax_online.cu`；NOW/lesson 里提到的 1-pass `maxSumkernel` 是 LeetGPU 上的实践，未落盘到 `solutions/`。
- warp shuffle 优化版 `softmax_opt.cu` 仍是 3-pass 占位，没有 `__shfl_down_sync` 实现。

## 为什么改计划

用户反馈“干货太少”。对照 [AIInfraGuide](https://github.com/caomaolufei/AIInfraGuide) 后，问题不是知识分类缺失，而是每个阶段的产出和验收太薄：

- 原计划很多条目是“学 X / 读 X”，没有强制要求代码、数字、面试口径三样产出。
- 理论笔记存在“Agent 生成了但用户没消化”的混淆，容易被当成进度。
- 推理系统/分布式只有主题占位，缺少可执行的 benchmark 和源码阅读路径。

## 本轮改动

- 新增 [roadmap/ai-infra-curriculum.md](../roadmap/ai-infra-curriculum.md)：AI Infra 密集课表，按 M0-M5 分模块，每个任务写清最小产出和验收。
- PATH/NOW 修正 A4 状态：3-pass ✅、2-pass fused ✅、1-pass true online 待落盘、warp shuffle/benchmark 待做。
- README / lessons / roadmap 索引接入新课表与外部参考。
- 新增并扩展 [最新模型与结构](../notes/algorithms/latest-model-architectures.md)、[剩余理论主题速览](../notes/algorithms/remaining-theory-primer.md) 和 [模型追踪表](../notes/algorithms/model-tracker.md)，PATH 理论线接入 M1.5 模型结构追踪；状态标记为草稿，未计入已学。
- agent memory 同步到 2026-08-03。

## 下一步（按新课表 M0）

1. A5 读 Flash Attn CUDA，输出逐段注释笔记。
2. 把 LeetGPU 上实践过的 1-pass true online 提交为 `softmax_1pass.cu`。
3. 给 3-pass / 2-pass fused / warp shuffle 三版跑 `KERNEL=... ./run.sh`，留下 GB/s 对比。
