---
name: "concept-explain"
description: "概念解释技能 - 用 Ascend→CUDA 映射解释 GPU/ML 系统概念"
---

# Concept Explain Skill

## 用途
用用户熟悉的 Ascend C 概念来解释 CUDA/ML 系统新概念。

## 核心映射表

| CUDA 概念 | Ascend C 对应 | 解释要点 |
|----------|--------------|---------|
| Thread/Block/Grid | 无直接对等 | 用 tiling 语义类比，强调 SIMT 范式差异 |
| Warp (32 threads) | Vector Unit SIMD | 解释 warp divergence 为 CUDA 独有问题 |
| Shared Memory | L1 Buffer/Unified Buffer | 都是程序员管理的片上内存 |
| `__syncthreads()` | `pipe_barrier`/`block_sync` | 同步语义完全相同 |
| Tensor Core (`mma.sync`) | Cube Unit (`mmad`) | 都是矩阵乘加速器 |
| Global Memory Coalescing | Merge access | 32B/128B 对齐要求相同 |
| Double Buffering | Pipe `InitBuffer<PIPE_BUF>` | Ascend 更显式，CUDA 需手动管理 |
| L2 Cache | L2 Cache | 完全相同 |
| HBM | HBM | 完全相同 |
| `__shfl_down_sync` | Vector reduce instruction | CUDA 用 warp shuffle 模拟 |
| Occupancy | 无直接对等 | 必须深入教：register pressure + shared memory trade-off |
| Nsight Compute | Ascend Profiling Tool | 都是性能分析工具 |
| Bank Conflict | Bank Conflict | 通用概念，bank 数可能不同 |

## 教学模式

### 模式一：新概念引入
1. 先问用户在 Ascend 上对应的是什么
2. 然后给出 CUDA 版本
3. 强调相同点和不同点
4. 给代码示例对比

### 模式二：类比解释
当用户问"这就像 Ascend 的 XX 吗？"：
1. 先验证类比是否正确
2. 如果正确，强化理解
3. 如果不正确，温和纠正并解释差异

## 输出格式
```
📚 概念：[CUDA 概念]

🔄 Ascend 对应：[Ascend C 对应概念]

✅ 相同点：[列表]
❌ 不同点：[列表]

💡 记忆口诀：[一句话助记]

[面试] 标记高频考点
```
