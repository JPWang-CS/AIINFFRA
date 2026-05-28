# 面试准备 (最后 2-3 个月激活)

系统性的面试准备，覆盖 CUDA/GPU、推理系统、分布式训练、系统设计。

## 文件规划

| 文件 | 内容 | 状态 |
|------|------|------|
| `cuda-questions.md` | CUDA 高频面试题（内存模型、warp、bank conflict、occupancy） | ⏳ |
| `system-design.md` | 系统设计：设计一个 LLM serving 系统 / 分布式训练平台 | ⏳ |
| `behavioral.md` | 行为面试：项目经历包装、Ascend→GPU 叙事线 | ⏳ |
| `coding-questions.md` | 算法题 + 系统编程题（CUDA/多线程/内存管理） | ⏳ |

## CUDA 高频考点

- [ ] GPU 内存层级（global/shared/register/L1/L2）
- [ ] Warp divergence 的代价和避免方法
- [ ] Shared memory bank conflict 的原理和解决
- [ ] Global memory coalescing 的条件
- [ ] Tensor Core 的 tile size 和精度支持
- [ ] CUDA Stream 的异步执行和同步
- [ ] Occupancy 的计算和影响因素
- [ ] Nsight 工具链的使用场景

## 推理系统高频考点

- [ ] Prefill vs Decode 的计算/访存差异
- [ ] PagedAttention 的设计和实现细节
- [ ] Continuous batching 的调度策略
- [ ] KV Cache 的量化（FP8/INT8）
- [ ] Speculative decoding 的原理
- [ ] 模型量化的方法对比（AWQ/GPTQ/SmoothQuant）

## 分布式训练高频考点

- [ ] AllReduce 的 ring 算法和通信量
- [ ] FSDP/ZeRO Stage 1-3 的区别
- [ ] TP vs PP 的通信模式
- [ ] MoE 的 expert parallel 和负载均衡
- [ ] 梯度累积和 micro-batching

## 行为面试叙事线

**核心故事**："从 Ascend NPU 到 GPU 的跨平台优化经验，证明我理解异构计算的本质"

关键话术：
- "昇腾 NPU 和 GPU 在内核架构上高度同构（HBM 层级、tiling、stream 异步）"
- "我的经验不是绑定某个 vendor，而是理解计算和访存的本质权衡"
- "1-2 年内系统性地补全了 CUDA 生态知识，并在 vLLM/分布式训练上做了深度实践"
