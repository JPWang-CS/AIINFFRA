---
name: "interview-prep"
description: "面试准备技能 - 构建 Ascend→GPU 跨平台优化者叙事，准备高频面试题"
---

# Interview Prep Skill

## 用途
帮助用户准备 ML 系统工程师面试，构建"跨平台优化者"叙事。

## 核心叙事框架

### 主题：跨平台优化者
```
"我理解的不是某个 vendor 的 API，而是异构计算的本质——
计算与访存的权衡、数据搬运的开销、并行度的挖掘。
从 Cube Unit 到 Tensor Core，从 L1 Buffer 到 Shared Memory，
底层原理完全同构。"
```

### 自我介绍模板（1-2分钟）
```
"我是[姓名]，有[X]年昇腾 NPU 算子开发经验。
过去主要负责[具体工作]，优化过[具体算子]，
性能提升[X]倍。

我意识到 GPU 生态是行业标准，而我的核心竞争力
不是绑定某个平台，而是理解算子优化的本质方法论。
从 Ascend 到 GPU 的迁移，1-2个月就上手了优化。

这是我的[GEMM/Flash Attention]实现，达到[XX]%峰值。
我能快速适配新平台，因为我懂的是底层原理。"
```

## 高频面试题库

### CUDA 基础
1. **"讲讲 warp 和 divergence"**
   - [面试] 必考题
   - 要点：32 threads 同步执行、if-else 导致 divergence
   - Ascend 类比：Vector Unit SIMD width

2. **"什么是 shared memory bank conflict"**
   - [面试] 必考题
   - 要点：32 banks、4B stride、连续访问避免
   - Ascend 类比：相同的 bank conflict 机制

3. **"什么时候用 shared memory"**
   - 要点：tiling、reduce、数据复用
   - 判断：带宽瓶颈时考虑

### 性能优化
4. **"怎么判断算子是 memory-bound 还是 compute-bound"**
   - 要点：Roofline 模型、算术强度、Nsight 指标
   - 必提：A100/4090 的带宽和算力数值

5. **"优化 GEMM 的思路"**
   - [面试] 高频
   - 要点：naive → tiled → vector → tensor core
   - 必提：每个优化点带来什么

6. **"occupancy 是什么？越高越好吗？"**
   - 要点：并行度指标，不是越高越好
   - 关键：register pressure、shared memory trade-off

### 推理系统
7. **"讲讲 PagedAttention"**
   - 要点：KV Cache 管理、虚→实映射、内存碎片解决
   - 必提：vLLM 论文核心

8. **"Prefill 和 Decode 瓶颈有什么不同"**
   - 要点：Prefill memory-bound、Decode compute-bound
   - 优化方向不同

### 量化
9. **"AWQ 和 GPTQ 的区别"**
   - 要点：AWQ 训练激活、GPTQ 后量化
   - 各自适用场景

## 模拟面试流程

### 第一步：叙事构建
```
- 帮用户提炼"跨平台"主题
- 准备 1-2-3 分钟版本自我介绍
- 准备 3-5 个代表性项目案例
```

### 第二步：题库准备
```
- 按主题分类（CUDA/推理/量化）
- 每题准备 2-3 分钟回答框架
- 标记 [必须] [加分] [了解] 优先级
```

### 第三步：模拟练习
```
- 用户随机选题
- 计时回答
- 反馈改进点
```

## 调用时机
- 用户说"准备面试"时
- 用户问"这个问题面试会考吗"时
- 用户请求"帮我打磨自我介绍"时
