# Triton Matrix Transpose V2 性能验证记录

日期：2026-10-15

## 结论与范围

Matrix Transpose V2 的 LeetGPU 通过状态保留，服务器小尺寸正确性和 4096×4096 FP32 profiler workload 正确性通过，阶段状态为 `GPU_VALIDATED`。本记录归档 NCU 诊断和用户提供的 AutoDL 计时；NCU 的 `32.224 μs` 是 profiler 采集时的 kernel duration，不是正式 benchmark。本阶段结束后不单独继续转置调参，极致优化留待后续模型实现及多算子组合中结合真实工作负载开展。

## 服务器正确性与 NCU 诊断

**来源：** 本地核验的任务收据 `.codex/ncu-kernel-replay-2026-10-15/artifacts/kernel-replay-2026-10-15-b95c7c7cb3c0/verified/`。完整原始 NCU 终端输出归档于[NCU 原始终端日志](./logs/2026-10-15-transpose-l20x-ncu.txt)；原始报告 `transpose.ncu-rep` SHA256 为 `bd911ec679f6efb1f4f37717355c849aabc37517a2dbb2f84d5c9fbdc9f177b8`。

- 设备与软件：NVIDIA L20X，NCU/CUDA runtime 报告 CC 9.0、132 SM；Torch `2.13.0+cu126`、Triton `3.7.1`、CUDA runtime `12.6`、driver `570.172.08`、Nsight Compute `2025.1.1`。这里按报告中的 CC 9.0 记录，不据此套用消费级 GPU 型号参数。
- 正确性：独立环境五个 FP32 小尺寸 `1×1`、`3×5`、`32×32`、`37×65`、`65×37` 检查通过；本次 NCU 入口的 `4096×4096` FP32 workload 也报告 `correctness=PASS`。task `kernel-replay-2026-10-15-b95c7c7cb3c0` 状态为 `capture_pass=true`、`effective_exit=0`、`timeout=false`、`source_unchanged=true`，报告非空；NCU 十个内部 passes 完成。
- 采集条件：连续 `4096×4096` FP32 输入和输出；32×32 tile，grid `128×128`、共 16,384 CTA，128 threads/block，25 registers/thread，动态 shared memory 4096 B。warmup 20、repeats 100；匹配 launch 后跳过 50 次并采集 1 次；`--replay-mode kernel`、`cache-control none`、`clock-control none`，启用 Speed Of Light 与 Launch Statistics。任务的 90 秒超时未触发。
- 指标：Duration `32.224000 μs`；L2 throughput `91.071394%`；DRAM throughput `66.773761%`；SM throughput `27.449247%`；ALU active `27.449247%`；LSU instruction throughput `21.752234%`；Tensor pipe `0%`；achieved occupancy `89.170925%`。
- 释义：各 throughput/activity 百分比按各自指标口径归一化，不能相加。L2 throughput 不是 cache hit rate；SM throughput 不是 occupancy。L2 可作为后续检查方向，但这些计数器不足以单独确定瓶颈。逻辑读写量为 64 MiB + 64 MiB = 128 MiB；这里不据此派生实际 DRAM 带宽，也不将 NCU replay duration 当作 CUDA-event benchmark。
- 收据还记录了七轮、每两秒一次的状态快照，任务匹配残留为 0。profile 前后的元数据确认 PyTorch、Triton、CUDA、driver 和编译工具版本一致，三个受保护的 kernel/bench/profile Git blob 与预期值一致。

## AutoDL RTX 4090 用户提供数据

以下数据来自用户此前贴出的 AutoDL 原始终端结果，与上面的 L20X/NCU 诊断分开记录，不能合并计算跨设备速度倍数。

- 环境：RTX 4090，CC 8.9；Torch `2.12.1+cu130`、Triton `3.7.1`、CUDA runtime `13.0`。
- 边界正确性：`(1,1)`、`(3,5)`、`(32,32)`、`(37,65)`、`(65,37)` 均通过。
- 比较：`4096×4096` FP32，warmup 20、repeats 100、相同 buffer 的 CUDA-event loop。Triton 为 `0.152741 ms`、有效带宽 `878.727 GB/s`；预分配输出并用 `out.copy_(input.T)` 物化转置为 `0.298428 ms`、`449.748 GB/s`。
- 用户提供的 Nsight Systems kernel 统计包含 100 次调用：Triton mean `152008.1 ns`（`152.0081 μs`）、median `152528 ns`；PyTorch mean `317077 ns`。此组是 RTX 4090 的独立采样，不与 L20X NCU replay duration 对比。
- AutoDL 未开放 NCU hardware counters；本次没有改驱动或配置。

## 失败路径与归因边界

此前 application replay 在输出 correctness PASS 后，后续启动/握手阶段挂住。经身份核对后仅对本任务 PID `3551791` 发送 SIGTERM；本次使用新的短私有临时目录执行单次 kernel replay 并成功。由于运行条件同时发生变化，不能把成功唯一归因于某项 NCU 行为或单一目录差异。失败路径与退出 1 的证据仍保留在本地任务收据中。

## 后续安排

Matrix Transpose V2 本阶段收口。后续进入模型实现和多算子组合时，再按真实工作负载选择优化对象、比较同语义 baseline，并配合 profiler 与端到端测量；当前不开展独立转置参数扫描。
