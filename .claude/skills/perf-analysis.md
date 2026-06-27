---
name: "perf-analysis"
description: "性能分析技能 - 分析 CUDA/Triton 算子性能瓶颈，提供优化建议"
---

# Performance Analysis Skill

## 用途
分析 CUDA/Triton 算子性能，识别瓶颈并提供优化方向。

## 分析维度

### 1. Roofline 模型分析
```
算术强度 = FLOPs / Bytes
判断位置：memory-bound vs compute-bound
理论峰值对比（A100/4090 带宽 + 算力）
```

### 2. Nsight Compute 指标
```
Occupancy: sm__warps_active.avg.pct_of_peak
Memory: l1tex__data_pipe_lsu_wave_total
Compute: smsp__sass_thread_inst_executed_op_*.sum
带宽利用率：drambytes / theoretical bandwidth
```

### 3. 瓶颈诊断
```
Memory-bound 症状：
- 计算吞吐远低于带宽
- 内存读写时间长

Compute-bound 症状：
- 带宽已接近理论值
- 计算单元利用率低

Occupancy 问题：
- Register pressure 过高
- Shared memory 用量过大
- Block 配置不当
```

### 4. 优化建议排序
```
1. 算法级：减少访存（tiling、融合）
2. 内存级：coalescing、减少 bank conflict
3. 计算级：tensor core、向量化
4. 配置级：调整 block size、grid size
```

## 输出格式
```
⚡ 性能分析报告

📊 当前数据：
- GFLOPS: [数值]
- 带宽利用率: [百分比]
- Occupancy: [百分比]

🔍 瓶颈诊断：
- 主要瓶颈：[memory/compute/occupancy]
- 证据：[具体指标]

💡 优化建议（按优先级）：
1. [建议1] - 预期提升：[X]x
2. [建议2] - 预期提升：[Y]x

🎯 下一步：
- [ ] [具体行动]
```

## 调用时机
- 用户报告性能不达标时
- 跑完 benchmark 需要解读结果时
- 请求"怎么优化这个算子"时
