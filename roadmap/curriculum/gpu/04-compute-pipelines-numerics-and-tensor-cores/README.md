# 第四章 GPU计算管线、数值格式与Tensor Core

> 状态：`WIP`（Work in Progress，进行中）。正文已建立，尚未完成用户阅读和真实GPU验收。

本章解释普通浮点/整数管线、特殊函数单元、Load/Store单元和Tensor Core怎样共同完成Transformer算子，并区分存储格式、乘法精度和累加精度。

术语：Special Function Unit（SFU，特殊函数单元）、Load/Store Unit（LSU，加载/存储单元）、Matrix Multiply-Accumulate（MMA，矩阵乘加）和Fused Multiply-Add（FMA，融合乘加）。Tensor Core是SM内部矩阵计算管线，不是一颗独立处理器。

## 1. Tensor Core怎样嵌入SM，而不是替代SM

Tensor Core是SM中的矩阵执行管线。线程仍负责地址计算、数据搬运、控制流、同步和epilogue。典型GEMM采用分层tiling：

```text
Grid：覆盖输出矩阵
  └── CTA tile：决定一组CTA负责的C区域
       └── Warp tile：分配给CTA内不同warp
            └── MMA instruction tile：硬件矩阵指令
```

数据也沿层次移动：

```text
Global A/B
  → CTA共享的shared tile
  → warp使用的register/fragment
  → Tensor Core MMA
  → register accumulator
  → epilogue
  → Global C
```

CTA tile越大，global数据复用机会通常越多，但shared和accumulator也更大；warp tile越大，单warp ILP与复用可能更好，但warp数量和并行度会变化。这是CUTLASS等高性能GEMM实现采用分层tile的原因。

`tl.dot`或cuBLAS调用隐藏了部分指令选择，但不会取消这些资源约束。是否走Tensor Core仍取决于dtype、shape、layout、对齐、目标架构和编译器生成代码。

## SM里不只有一种“核心”

Warp Scheduler发射的是指令，具体指令进入不同pipe：普通Floating Point（FP，浮点）或Integer（INT，整数）管线处理标量算术、逻辑与地址相关操作；Load/Store Unit（LSU，加载/存储单元）执行访存指令；Special Function Unit（SFU，特殊函数单元）处理部分`exp`、`rcp`、`rsqrt`等；Tensor Core处理受支持的矩阵乘加。

一个Transformer kernel往往混合使用多条pipe。以Attention为例：QK和PV可进入Tensor Core，在线Softmax的max/sum由归约指令完成，`exp`使用特殊函数路径，指针与mask使用普通整数/谓词指令，Q/K/V仍需Load/Store供数。

## 存储dtype、乘法dtype和累加dtype

三者必须分开：

```text
内存中的A/B格式
→ 载入与可能的转换
→ 乘法采用的有效精度
→ accumulator采用的格式
→ epilogue转换成输出格式
```

FP16输入配FP32累加，表示乘法操作数来自16位格式，但部分和保存在更大范围/精度的accumulator中。TF32则常接收FP32存储输入，在Tensor Core乘法路径中使用TF32有效精度并进行FP32累加。它们的误差与性能不能放进IEEE FP32同一栏。

## 为什么矩阵指令有shape和layout要求

Tensor Core不是接收任意二维数组的黑盒。Matrix Multiply-Accumulate（MMA，矩阵乘加）指令规定操作数dtype、fragment shape、layout和协作线程范围。库或编译器必须把shared/register中的数据排成指令需要的形式。

```text
CTA tile
→ warp tile
→ instruction tile
→ 每个lane持有fragment的一部分
```

对齐不足、K维过小、边界tile过多或layout转换昂贵，都会让Tensor Core路径收益下降。看到输入是FP16并不能证明使用了MMA；要检查PTX/SASS中的矩阵指令和Profiler Tensor pipe指标。

## Tensor Core也需要被“喂饱”

设一个CTA对A/B tile复用$r$次。增大tile通常提高Arithmetic Intensity（AI，算术强度），但会同时增加shared和register。Pipeline需要在MMA消费当前fragment时预取下一fragment，否则Tensor pipe会等待数据。

优化顺序应是：确认数学与dtype→确认生成矩阵指令→检查grid是否足够→检查global/shared供数→检查Tensor pipe利用→再调tile与stage。

## 算子实验

- GEMM分别运行IEEE FP32、TF32、FP16/BF16输入+FP32累加，设置各自正确性容差；
- 用相同shape检查PTX/SASS，确认普通FMA与MMA路径；
- 记录Tensor pipe、memory throughput、register与shared；
- 对很小M/N/K做size sweep，观察Tensor Core启动和padding成本何时盖过收益；
- 在Softmax上验证“没有Tensor Core热点”不能推出GPU没有计算瓶颈，仍需看SFU和归约。

## 浮点格式为什么影响范围和精度

二进制浮点通常拆成sign、exponent和fraction：

$$
x=(-1)^s\times 2^{e-bias}\times(1.f)
$$

指数位决定可表示范围，尾数位决定相邻可表示数的间隔。BF16保留接近FP32的指数范围但尾数较短，FP16尾数更多但指数范围较小；二者都占16 bit，却不能只按“同位宽”推断数值行为。

Accumulator需要承受K维大量乘积求和。低精度输入使用FP32累加，主要是降低部分和舍入和溢出的风险；最后写回低精度输出仍会发生一次转换。正确性测试要包含大K、不同幅值、正负抵消和非规则shape。

## SFU与近似数学

Softmax中的`exp`、Norm中的`rsqrt`可能使用硬件特殊函数或编译器近似序列。Fast math可以减少指令或使用低精度近似，但会改变误差。评测要同时记录最大绝对/相对误差，不能只比较速度。

Triton的`tl.exp`、`tl.rsqrt`等最终怎样实现取决于目标和编译器。验证路径需要查看生成PTX/SASS和Profiler，而不是由API名称断言“使用了一个SFU周期”。

## 普通FMA与矩阵MMA的工作粒度

普通Fused Multiply-Add（FMA，融合乘加）可由单个thread对标量执行；MMA由一个warp或更大的warp group协作处理矩阵fragment。矩阵指令的结果分散在参与线程的register中，因此fragment不能按普通连续数组的线程私有布局理解。

```text
SIMT FMA：thread i持有a_i、b_i、c_i并执行一个标量融合乘加
MMA：参与warp共同持有A/B/C fragments并提交矩阵乘加
```

高性能库负责建立这种分布式fragment布局。自写PTX或CuTe/CUTLASS代码则必须显式满足layout和指令契约。

## Epilogue为什么也属于计算路径

GEMM主循环结束后通常还要做bias、activation、scale、residual或量化写回。若单独启动kernel，会再次读写C；融合epilogue可直接消费register accumulator，减少global traffic。

但融合会延长accumulator和附加参数的活跃区间，增加register与指令压力。是否融合应比较减少的bytes和增加的资源，而不是默认“少一个kernel一定更快”。

## 本章应保留的证据

```text
输入/输出/accumulator dtype
误差标准与失败case
目标Compute Capability
PTX/SASS中的FMA或MMA证据
Tensor/SFU/FP/INT/LSU相关吞吐
register、shared与occupancy
同语义库baseline
```

## 章节导航

[上一章](../03-registers-shared-memory-and-memory-hierarchy/README.md) · [返回第一篇目录](../README.md) · [下一章](../05-resource-model-occupancy-and-roofline/README.md)

