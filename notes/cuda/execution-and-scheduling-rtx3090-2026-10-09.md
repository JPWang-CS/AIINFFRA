# CUDA 执行模型与指令调度：RTX 3090 实验记录（2026-10-09）

本次实验是第一篇第二章第 6 节的官方/reference 实验，不是 LeetGPU 题目。用户在 RTX 3090 上执行了课程原有的 `execution_and_scheduling.cu`，本笔记把代码、输入、线程组织、输出检查和测量边界串起来。默认目标的原始输出保存在[sm52原始日志](logs/2026-10-09-execution-and-scheduling-rtx3090.txt)，本轮 native 目标的三组输出保存在[sm86原始日志](logs/2026-10-09-execution-and-scheduling-rtx3090-sm86.txt)；课程正文仍保留内联源码，日志不是源码的替代品。

## 1. 环境、命令与证据边界

用户提供的设备是 NVIDIA GeForce RTX 3090，`nvcc` 为 CUDA 12.4（`V12.4.131`，build `cuda_12.4.r12.4/compiler.34097967_0`）。驱动版本、时钟、功耗和服务器 git revision 未记录。编译命令是：

```bash
nvcc -O3 -std=c++17 -lineinfo --resource-usage \
  examples/execution_and_scheduling.cu -o /tmp/cuda-execution
```

用户日志中的 `ptxas` 资源报告首先是 **sm_52** 默认目标的编译报告：`predicated_select` 8 registers、`divergent_branch` 7、`independent_accumulators` 10、`dependent_chain` 7；另有 `cmem[0]` 340/344 bytes、独立与依赖版本各 `cmem[2]` 4 bytes，四者均为 0 stack、0 spill stores、0 spill loads、0 gmem。CUDA 12.4.1 的默认目标为 `sm_52`，这份表描述的是 sm52 代码生成。

本轮按 `-arch=sm_86` 重新编译，得到另一份整理自用户日志的报告：`predicated_select` 12 registers、`divergent_branch` 10、`independent_accumulators` 14、`dependent_chain` 9；对应 `cmem[0]` 为 372/376 bytes，四个 kernel 均为 0 stack、0 spill stores、0 spill loads，编译单元报告 0 gmem。该报告没有列出 `cmem[2]`。本轮资源分析以 sm86 表为准；最终 SASS 仍待读取。

相关编译器语义可查 [CUDA 12.4 NVCC Compiler Driver](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-compiler-driver-nvcc/index.html)；架构兼容和 PTX/JIT 边界见 [Ampere Compatibility Guide](https://docs.nvidia.com/cuda/ampere-compatibility-guide/index.html)。内存检查器的官方说明见 [Compute Sanitizer 文档](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。

按 [Ampere Tuning Guide 的 occupancy 说明](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#occupancy)，CC 8.6 每 SM 有 48 warp、64K 个 32-bit registers、最多 16 个 block。256 threads/block 是 8 warp，按 warp 上限可放 6 block/SM；9 和 14 registers/thread 乘 256 得每 block `R*T` 下界 2304 和 3584，6 block 分别为 13824 和 21504，均低于 64K。实际寄存器分配按硬件粒度取整；下一步用 occupancy API 读取可驻留 block 数，再解释它能否隐藏延迟。

## 2. 从代码到一次 kernel launch

程序先生成 `input[i] = 0.001f * (i % 1000)`，在设备上分配 `device_input` 和 `device_output`，并把输入复制到 `device_input`；主机侧的 `output` 用于把结果拷回后验证，`reference` 保存当前 kernel 对应的 CPU 结果。固定 `threads = 256`，网格为

```cpp
const int blocks = (n + threads - 1) / threads;
```

每个 kernel 中的全局元素编号都是 `i = blockIdx.x * blockDim.x + threadIdx.x`，并先执行 `if (i >= n) return`。所以小形状 `n=257` 使用 `grid=2`：第一个 block 覆盖 0–255，第二个 block 只有 `i=256` 有效，其余 255 个线程在边界返回。这是本实验的尾块正确性检查。大形状 `n=1,048,576` 正好是 `4096` 个 256-thread block，没有尾部不足问题。

每个 `run` 先生成对应的 CPU 参考实现；两个累加版本使用 `std::fma`，两个分支版本使用普通乘加表达式。`measure` 先预热一次，排除首次启动的准备开销；然后重复启动 `repeats` 次，用 CUDA event 测总耗时并除以 repeats 得到平均单次 `ms`。每次启动都从同一份 `device_input` 重新计算，`steps` 只影响两个 FMA kernel，两个 branch kernel 不循环 `steps`。`max_abs_error` 是所有元素的最大绝对差，有限且不超过 `1e-4` 才是 PASS；`source_FMA_GFLOP_s` 是每秒十亿次源码浮点操作，每个 FMA 计 2 次操作。device-to-host copy 在 event 外，launch 之间的 host 提交空隙可能包含在 event 内。

## 3. 四个 kernel 各在做什么

`dependent_chain_kernel` 每个线程只有一个 `x`，每轮执行

```cpp
x = fmaf(x, 1.000001f, 0.00001f);
```

下一轮必须等上一轮的 `x`，因此这是单条数据依赖链。`independent_accumulators_kernel` 使用 `x0`、`x1`、`x2`、`x3` 四个独立累加器，每轮做四次 FMA；最后先将四条链相加、减去 `0.6f`，再乘 `0.25f`。它不是把 dependent 改写成“点积”而保持完全相同的数学函数：四条链的初值分别是 `input[i] + 0`、`+0.1`、`+0.2`、`+0.3`，迭代函数为 `f(x)=a*x+b`。用实数写，独立版本输出为

$$
z = F_s(x) + 0.15(a^s-1),
$$

其中 `F_s(x)` 是从 `x` 出发的 `s` 次迭代；末尾减去 `0.6` 只是减掉四个初始偏移之和，并不会让每条链在迭代后重新回到同一条 dependent 数学路径。因此两个 kernel 各自对各自的 CPU reference 检查，`max_abs_error=0` 表示各自对应的 float reference 通过，不表示两个 kernel 输出相同，也不表示真实数值运算没有舍入。

`fmaf` 将乘法和加法融合，并在最终结果处做一次 float 舍入；CPU reference 使用 `std::fma` 对应这条链。分支版本 `divergent_branch_kernel` 根据 `i` 奇偶只计算一条路径：偶数用 `even_path`，奇数用 `odd_path`。选择版本 `predicated_select_kernel` 先计算两条路径，再按奇偶选择结果。函数名并不能证明最终一定编译成 predicate；真实分支/谓词形态必须看 PTX/SASS。CPU 的 `reference_branch` 使用普通的 `input*scale + bias`，而 GPU 路径使用 FMA，所以出现 `5.9604645e-08` 或 `1.1920929e-07` 级误差是合理的；阈值是 `1e-4`，因此仍 PASS。

## 4. 运行结果怎么读

本轮命令使用 `n / steps / repeats` 参数：`257 / 20 / 5` 检查 2 个 256-thread block 的尾部，`1,048,576 / 200 / 50` 比较吞吐。正式性能只看普通运行：

| Kernel | n | steps | repeats | threads/block | correctness / max abs error | 平均调用 ms | 源码归一化 GFLOP/s |
|---|---:|---:|---:|---:|---|---:|---:|
| dependent | 257 | 20 | 5 | 256 | PASS / 0 | 0.0035 | 2.953 |
| independent | 257 | 20 | 5 | 256 | PASS / 0 | 0.0029 | 14.342 |
| divergent | 257 | 20 | 5 | 256 | PASS / 5.9604645e-08 | 0.0027 | 不适用 |
| predicated | 257 | 20 | 5 | 256 | PASS / 5.9604645e-08 | 0.0027 | 不适用 |
| dependent | 1,048,576 | 200 | 50 | 256 | PASS / 0 | 0.0201 | 20,855.398 |
| independent | 1,048,576 | 200 | 50 | 256 | PASS / 0 | 0.0576 | 29,132.291 |
| divergent | 1,048,576 | 200 | 50 | 256 | PASS / 1.1920929e-07 | 0.0118 | 不适用 |
| predicated | 1,048,576 | 200 | 50 | 256 | PASS / 1.1920929e-07 | 0.0113 | 不适用 |

每个版本与自己的参考函数比较；两个累加版本的初值和计算路径不同。dependent 每元素做 200 次 FMA，共 `209,715,200` 个 FMA，即 `419,430,400` FLOPs；independent 每元素做 800 次 FMA，共 `838,860,800` 个 FMA，即 `1,677,721,600` FLOPs。四倍工作用了约 2.87 倍时间，因此按源码 FMA 数计单位时间工作量约高 40%，对应约 20.9 TFLOP/s 和 29.1 TFLOP/s。四个累加器属于同一线程，彼此不依赖，增加可连续发射的独立计算指令。

sm86 报告中 dependent 使用 9 registers/thread，independent 使用 14；每个 kernel 为 0 stack、0 spill stores、0 spill loads，编译单元为 0 gmem。按 [Ampere Tuning Guide 的 occupancy 说明](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#occupancy)，CC 8.6 最多同时驻留 48 个 warp、16 个 block，并有 64K 个 32-bit registers；256 threads/block 是 8 warp，按 warp 上限可放 6 block/SM。未计分配取整时，每块分别是 `9×256=2304` 和 `14×256=3584` 个寄存器，6 block 分别为 13824 和 21504；实际分配按硬件粒度取整，下一步用 occupancy API 读取可驻留 block 数。

偶数位置走 `even_path(x)=fmaf(x,1.25,0.5)`，奇数位置走 `odd_path(x)=fmaf(x,0.75,-0.25)`。divergent 版本按条件只算对应路径，predicated 版本先算两条路径再选择。CPU reference 使用普通乘加，GPU 使用 FMA，所以出现 `1.1920929e-07` 的舍入差；`1e-4` 容差下仍 PASS。大输入两者为 11.8 us 和 11.3 us，差 0.5 us。

首次默认构建的对照见[实验记录](logs/2026-10-09-execution-and-scheduling-rtx3090.txt)，本轮逐字输出见[sm86 原始日志](logs/2026-10-09-execution-and-scheduling-rtx3090-sm86.txt)。`n=257` 的 `grid=2` 中第二个 block 只有一个有效线程，四个 kernel 小输入均 PASS；小输入 memcheck 为 0 errors。接着看 SASS 中的条件跳转、谓词与 FMA，确认这两种源码分别生成了哪些指令。

## 5. 当前结论与下一检查

截至本轮用户日志，RTX 3090 上的设备正确性已达到 `GPU_VALIDATED`：sm52 默认构建与 sm86 native 构建的小尾块、大输入均通过，小输入的两次 memcheck 均为 0 errors；本轮没有 n=1,048,576 的 sanitizer 结果。课程整体仍是 `WIP`。本次没有 LeetGPU 门槛，也不推进第 7 节，更不改变 MLA 或历史算子成绩。

sm86 native 复测已经完成；下一步是检查最终 SASS，并做多轮同 shape 计时。

为下一次同机复测，源码新增 `--branch-probe`：同一个 kernel 接收 `input`、`unsigned char flags`、输出和 `steps`，分别使用 lane-split 与 warp-uniform 两组 flags；脚本扫描 `steps=1/32/256`，交替测量两种 flags，并保存 SASS。probe 的 GPU 结果尚未执行；本机已用 `--branch-probe --self-test` 做 CPU 边界、参考值、guard/NaN 和非法参数检查。
