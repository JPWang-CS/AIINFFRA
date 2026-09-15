# 第五章 资源模型、Occupancy与Roofline

> 状态：`WIP`（Work in Progress，进行中）。正文已建立，尚未完成用户阅读和真实GPU验收。

本章把thread、warp、CTA、register和shared memory写成可计算的资源账本，再用Occupancy和Roofline判断瓶颈。实验身份与软件栈归入第八章，本章只讨论kernel资源和性能上界。

术语：Occupancy（占用率）是每SM活跃warp数与硬件最大warp数之比；Arithmetic Intensity（AI，算术强度）是运算量与数据流量之比；Roofline（屋顶线模型）用计算峰值和带宽上界共同约束可达吞吐。

## 1. Occupancy是资源结果，不是性能目标

Occupancy通常定义为：

$$
Occupancy=\frac{Active\ Warps\ per\ SM}{Maximum\ Warps\ per\ SM}
$$

一个CTA能否驻留由多个上限共同决定。设每CTA线程数为$T_b$、warp数为$W_b$、register需求为$R_b$、shared需求为$S_b$，则理论resident CTA近似受以下最小值限制：

$$
B_{resident}=\min\left(
B_{hw},
\left\lfloor\frac{T_{SM}}{T_b}\right\rfloor,
\left\lfloor\frac{W_{SM}}{W_b}\right\rfloor,
\left\lfloor\frac{R_{SM}}{R_b}\right\rfloor,
\left\lfloor\frac{S_{SM}}{S_b}\right\rfloor
\right)
$$

真实分配还受到register/shared粒度和架构规则影响，应使用Occupancy API、编译器报告或Nsight Compute确认。

高occupancy通常有助于隐藏延迟，但不是越高越快。减少每线程register可能提高occupancy，却导致spill；缩小tile可能增加驻留CTA，却降低数据复用。判断必须回到瓶颈：当前warp是在等内存、等依赖、等barrier，还是执行pipe已经饱和？

## 2. 静态资源卡要为每个设备单独建立

后续tile、occupancy和理论上界依赖的字段包括：

- Compute Capability和SM数量；
- warp size、每block/SM线程与warp上限；
- registers/SM、每thread或block限制与分配粒度；
- shared memory/SM、每block默认和opt-in上限；
- L2容量；
- 显存容量、介质、时钟与bus width；
- 支持的数值格式、矩阵与异步搬运能力；
- 官方理论带宽及计算口径。

运行时可查询值、官方产品规格和架构/CC上限要分列。它们可能名称相似，但回答的问题不同。

```bash
python - <<'PY'
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"device[{i}]")
    print(p)
PY
```

再使用CUDA Samples中的`deviceQuery`或Runtime API补齐属性，并与官方Compute Capability资料交叉确认。

## 3. 资源表怎样变成kernel判断

资源卡不是资产登记表。每个字段都要能落回kernel：

| 资源 | 影响 | 不能这样推断 |
|---|---|---|
| SM数 | grid并行度、wave数量 | SM翻倍则任意kernel加速两倍 |
| Warp/线程上限 | 理论驻留上界 | block越大occupancy越高 |
| Register | 驻留CTA和spill | 源码变量少就一定register少 |
| Shared memory | tile、stage与CTA驻留 | 能编译就代表资源利用合理 |
| L2 | 跨CTA和重复访问缓存机会 | 数据集小于L2就一定全部命中 |
| 显存带宽 | memory-bound roof | 理论带宽等于实际有效带宽 |
| CC/目标 | 指令和dtype可用性 | 目标更新就自动使用专用路径 |

### 用tile算一次shared memory

设GEMM每个stage缓存：

$$
A_{tile}\in\mathbb{R}^{B_M\times B_K},\qquad
B_{tile}\in\mathbb{R}^{B_K\times B_N}
$$

元素宽度为$b$字节，stage数为$s$，忽略padding与额外buffer时：

$$
Shared\ Bytes\approx s\times(B_M\cdot B_K+B_K\cdot B_N)\times b
$$

例如把$B_M$或$B_N$翻倍，不仅增加单CTA shared，还可能扩大accumulator、提高register压力。最终驻留CTA数由线程、warp、register、shared和硬件block上限共同取最小值。

实际编译结果还可能包含padding和编译器生成的额外空间，所以手算用于预判，`ptxas -v`、Triton编译元数据和Profiler用于确认。

## 4. Arithmetic Intensity与Roofline

Arithmetic Intensity（AI，算术强度）是计算量与数据流量之比：

$$
AI=\frac{Operations}{Bytes}
$$

Roofline把峰值计算吞吐$P_{peak}$和带宽$BW$合并为上界：

$$
P_{attainable}\le\min(P_{peak},\ AI\times BW)
$$

GEMM通过tile复用提高AI；Vector Add即使实现完美也需要为少量计算搬运多份数据，通常更靠近memory roof。实际bytes应同时给出算法最小流量和Profiler观测流量，避免cache与重复访问造成误判。

多层memory有不同roof：DRAM、L2、shared和register数据路径可能分别限制kernel。点落在DRAM roof下方并不自动证明DRAM饱和，还要检查请求效率、latency和并行度。

## 5. 资源限制的算例

假设一个CTA有256 threads，每thread使用96 registers，每CTA使用48 KiB shared。先分别根据目标设备的thread/warp、register、shared和block上限计算resident CTA，再取最小值。若把stage从2增到3导致shared变成72 KiB，resident CTA可能发生阶梯下降；若同时stall下降，总性能仍可能提高。

实验必须记录：配置、理论resident CTA、编译register/shared、achieved occupancy、eligible warp、memory/compute throughput和耗时。只给occupancy百分比不能解释性能。



## 章节导航

[上一章](../04-compute-pipelines-numerics-and-tensor-cores/README.md) · [返回第一篇目录](../README.md) · [下一章](../06-synchronization-concurrency-async-copy-and-pipelines/README.md)

