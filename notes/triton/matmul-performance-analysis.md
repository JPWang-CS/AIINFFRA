# Triton MatMul 性能分析记录

> 日期：2026-08-26
> 对象：`solutions/triton/matmul.py` 服务器验证版
> 状态：服务器 `GPU_VALIDATED`；LeetGPU 原始代码仍为 `WIP`

## 1. 实验口径

```text
GPU: NVIDIA GeForce RTX 3090
shape: M=8192, N=6144, K=4096
dtype: FP32
tl.dot: input_precision="ieee"
PyTorch 对照: torch.backends.cuda.matmul.allow_tf32 = False
正确性: (1,1,1), (64,32,64), (65,33,67), (257,513,129) 全部通过
```

FLOPs 统一按：

```text
2 × M × N × K
```

本次 `torch.mm` 为 17.138 ms / 24,058.3 GFLOPS。不同运行的 cuBLAS 时间有轻微波动，比较时以同一次 sweep 中的 PyTorch 对照为准。

## 2. 配置 sweep

| 配置 | 耗时 | GFLOPS | torch.mm 比例 | 结论 |
|---|---:|---:|---:|---|
| `64×32×64, w4, s3` | 24.924 ms | 16,542.7 | 68.8% | baseline |
| `128×32×64, w4, s3` | 22.298 ms | 18,491.3 | 76.9% | 当前测试中明显更好 |
| `128×32×128, w4, s3` | 28.354 ms | 14,541.8 | 60.4% | `BLOCK_K` 增大后反而变慢 |
| `128×64×128, w4, s3` | 编译失败 | — | — | shared memory 超限 |
| `128×32×256, w8, s3` | **22.033 ms** | **18,713.5** | **77.8%** | 当前最佳 |

最佳配置：

```text
BLOCK_M=128, BLOCK_N=32, BLOCK_K=256
num_warps=8, num_stages=3
```

相对 baseline：

```text
耗时：24.924 → 22.033 ms，下降约 11.6%
GFLOPS：16,542.7 → 18,713.5，提升约 13.1%
```

## 3. 结果解释

### 3.1 为什么 `BLOCK_M=128` 有收益

`BLOCK_M` 增大后，一个 program 负责更多输出行：

```text
C tile: [BLOCK_M, BLOCK_K]
```

在相同归约循环下，A/B tile 的加载和地址计算更容易被更多 FMA 摊薄，所以 `128×32×64` 比 `64×32×64` 快。

但这不是“tile 越大越好”。输出 accumulator 也随 `BLOCK_M × BLOCK_K` 增长，会增加寄存器压力，可能降低 resident blocks 或触发 spill。

### 3.2 为什么 `BLOCK_K=128` 没有继续变快

当前命名中：

- `BLOCK_N` 是归约维度 tile；
- `BLOCK_K` 是 C 的输出列 tile。

`BLOCK_K` 从 64 增到 128，会让 accumulator 从 `128×64` 变成 `128×128`。这可能带来：

- 更多寄存器占用；
- 更低 occupancy；
- 更大的 tile 传输和布局压力；
- 编译器选择不同的 MMA/加载路径。

因此 `128×32×128` 的 14,541.8 GFLOPS 说明当前配置已经受到资源或执行路径影响，不能只按“数据复用更多”推断性能。

### 3.3 为什么 `BLOCK_N=64` 编译失败

`128×64×128, w4, s3` 报告：

```text
Required shared memory: 131,072 B
Hardware limit: 101,376 B
```

`num_stages=3` 会增加 A/B pipeline staging；`BLOCK_N` 增大又扩大了每轮归约 tile。两者叠加后超过 RTX 3090 的单个 SM shared-memory 上限。

这个失败是资源约束，不是 kernel 逻辑错误。降低 `BLOCK_N`、降低 `num_stages` 或缩小 tile 才能继续编译。

### 3.4 为什么 `BLOCK_K=256, w8` 当前最好

`128×32×256, w8, s3` 在本次固定 shape 上达到 18,713.5 GFLOPS，可能同时获得了：

- 更大的输出 tile，摊薄 program 启动和地址计算；
- 8 warps 对更大的 accumulator 提供更高并行度；
- 归约循环次数相对减少。

但现在只能说它是**当前 shape 和当前候选集的最佳**，不能直接断言是全局最优。必须用 profiler 确认是否有 register spill、occupancy 下降或 memory stall。

## 4. 后续优化顺序

### P0：先 profile 当前最佳配置

先不要继续盲扫 tile，针对当前最佳配置单独运行 Nsight Compute：

```bash
ncu --set roofline -o matmul_k256 \
  python solutions/triton/matmul.py \
  --config k256-128x32x256-w8-s3
```

重点记录：

| 观察项 | 要回答的问题 |
|---|---|
| registers/thread | `BLOCK_K=256` 是否造成寄存器压力 |
| local memory / spill | 是否把 accumulator 溢出到 local memory |
| shared memory/block | 哪些 stage/tile 组合接近上限 |
| achieved occupancy | 是否因为资源占用无法驻留足够多 block |
| Tensor Core/MMA 指令 | IEEE FP32 是否走 CUDA Core，而不是 TF32 Tensor Core |
| DRAM/L2 throughput | 瓶颈是 HBM/L2 还是计算吞吐 |
| warp stall 原因 | 是等待内存、依赖、barrier 还是执行单元不足 |

### P1：围绕最佳点做小范围搜索

只扩展邻域，避免再次盲扫：

```text
BLOCK_M: 128
BLOCK_N: 16 / 32
BLOCK_K: 128 / 256
num_warps: 4 / 8
num_stages: 2 / 3
```

优先测试：

```text
128×32×256, w8, s2
128×16×256, w8, s3
128×32×128, w8, s3
128×32×256, w4, s3
```

每个候选必须先跑 4 组 correctness，再记录耗时；不能只用性能 shape 测出一个数字就标成完成。

### P2：检查 program 排布和 L2 复用

当前 `pid_m / pid_k` 是简单二维映射。下一步可以测试 grouped ordering，让相邻 program 更长时间复用同一批 A 行块或 B 列块，观察：

- L2 hit rate 是否上升；
- global load throughput 是否下降；
- 总耗时是否真的下降。

如果 L2 已经命中率很高，继续改排布可能没有收益，应停止这条分支。

### P3：单独做 TF32 对照

IEEE FP32 和 TF32 必须分成两张成绩单：

```text
IEEE：回答严格 FP32 kernel 有多快
TF32：回答允许精度损失后 Tensor Core 能换来多少吞吐
```

TF32 对照要额外记录 `max_abs_error`、`max_rel_error`，不能把 TF32 的速度提升归因于 tile 优化。

### P4：再考虑 autotune

当前已经有手动 sweep，下一步可以把经过 profiler 筛选的少量候选放进 `triton.autotune`。autotune 不是替代分析：

- 先删除会编译失败或明显 spill 的配置；
- 保留不同资源/性能路径的代表配置；
- 用 `key=["M", "N", "K"]` 覆盖多个 shape；
- 记录 autotune 选择结果，而不是只记录最终最快数字。

## 5. 当前停止条件

当前先停在：

```text
正确性：通过
IEEE FP32：已有 baseline + sweep
最佳结果：18,713.5 GFLOPS，torch.mm 的 77.8%
资源失败：已记录 shared memory 上限
下一证据：Nsight Compute 当前最佳配置
```

在 LeetGPU 仍无法运行的情况下，这些结果属于服务器适配版的 `GPU_VALIDATED` 证据，不把它提前升级成 `LEETGPU_PASS` 或 `COMPLETE`。

## 相关入口

- [Lesson 06 Triton MatMul](../../lessons/06-triton-intro.md#56-服务器真实性能)
- [Lesson 07 Triton Debugging](../../lessons/07-triton-debugging.md)
- [服务器验证代码](../../solutions/triton/matmul.py)
- [Triton 代码 README](../../solutions/triton/README.md)
- [GPU 底层架构与优化路线](../../roadmap/gpu-foundations.md)
