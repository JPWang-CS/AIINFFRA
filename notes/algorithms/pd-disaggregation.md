# Prefill-Decode 分离（PD Disaggregation）

> 推理系统技术类 · Splitwise / DistServe 方案，解决 P 和 D 的硬件冲突

---

## 解决了什么问题

### Prefill 和 Decode 的本质差异

| 阶段 | 做什么 | 计算特点 | GPU 最优配置 |
|------|--------|---------|------------|
| **Prefill** | 处理输入 prompt，生成所有 token 的 KV | Compute-bound（大矩阵乘）| 高 TFLOPS（A100/H100） |
| **Decode** | 逐 token 自回归生成 | Memory-bandwidth bound（加载权重）| 高 HBM 带宽，或多卡摊薄 |

**两者在同一个 GPU 上运行时相互干扰**：
1. 长 prompt 的 prefill（数秒）阻塞同 GPU 上的 decode 请求（延迟毛刺）
2. Prefill 需要大计算算力，decode 需要大内存带宽——理想 GPU 不同
3. KV cache 争抢显存导致两者都不能满负荷运行

### 结果

```
混合 serving 时的问题:
- TTFT (Time to First Token):  prefill 长 → decode 被阻塞 → TTFT 毛刺大
- TBT (Time Between Tokens):   KV cache 碎片 → decode 吞吐不稳定
- GPU 利用率:                  prefill 期间 memory-bound 算子浪费算力
                                decode 期间 compute-bound 算子浪费带宽
```

---

## 核心思路：物理分离 P 和 D 节点

**Splitwise (Patel et al., 2023) / DistServe (Zhong et al., 2024)**：

```
           ┌─────────────────────┐
请求  →    │   Prefill Cluster   │  compute-bound → 用 A100/H100 高算力 GPU
           │  (P1, P2, P3, P4)   │
           └──────────┬──────────┘
                      │ KV Cache 传输 (RDMA/NVLink)
           ┌──────────▼──────────┐
           │   Decode Cluster    │  memory-bound → 可用 A100 或更便宜的 GPU（如 L40S）
           │  (D1, D2, D3, D4)   │
           └──────────┬──────────┘
                      │ output tokens
                      ▼
```

**工作流程**：
1. 请求到达 → 路由到空闲的 P 节点
2. P 节点完成 prefill，生成 KV Cache
3. **KV Cache 通过高速互联（RDMA、NVLink over Ethernet 或 InfiniBand）迁移到 D 节点**
4. D 节点接管 decode，逐 token 生成
5. P 节点立即腾出处理下一个请求的 prefill

---

## 关键数据/取舍

### KV Cache 传输开销

```
LLaMA-2-70B (GQA, G=8, d_k=128, 80 layers, FP16):
  每 token KV = 320 KB
  prompt = 1024 tokens: KV = 320 MB
  传输时间（InfiniBand HDR 200Gb/s = 25 GB/s）: 320 MB / 25 GB/s = 12.8 ms
  vs. prefill 时间（1024 tokens）: ~1-2 s
```

KV 传输开销 (~13ms) 远小于 prefill 时间 (~1-2s)，**传输不是瓶颈**。

### 性能收益（DistServe 论文数据）

| 指标 | 混合 Serving | PD 分离 | 提升 |
|------|:-----------:|:-------:|:----:|
| P99 TTFT | 5.8s | **1.2s** | 4.8× |
| P99 TBT | 0.21s | **0.08s** | 2.6× |
| Overall Throughput | 1.0× | **2.1×** | 2.1× |

测试：LLaMA-13B，ShareGPT 真实 workload，A100 × 8。

### 硬件选型建议

| 集群 | 推荐 GPU | 原因 |
|------|---------|------|
| Prefill | A100/H100 | 高 TFLOPS，prefill 是 compute-bound |
| Decode | A100 或 L40S | L40S 有更大 VRAM density（48GB @ 300W vs 80GB @ 400W），成本效益更好 |
| 异构组合 | P: H100, D: A100 | H100 prefill 速度是 A100 的 3× |

**成本分析**：相同 SLA 下，PD 分离可降低 ~40% 硬件成本（因为 decode 集群可用更便宜 GPU）。

---

## 进一步优化：Chunked Prefill

即使不做 PD 分离，**chunked prefill** 也能缓解问题：

```python
# 不分离，但将长 prompt 切成 chunk
for chunk in prompt.split(chunk_size=512):
    prefill_chunk(chunk)   # 短 prefill，不长时间阻塞
    # 中间执行若干 decode step（interleaved）
```

效果：消除 TTFT 毛刺，但 prefill 总时间略增（每个 chunk 的 KV 要分批处理）。

vLLM 0.4+ 支持 chunked prefill（`--enable-chunked-prefill`），是 PD 分离的轻量替代方案。

---

## 在 Ascend 的对应

PD 分离是 serving 架构层面的优化，与硬件无关。  
华为 CloudMatrix 和 MindIE 框架已有 PD 分离的实现（用于 Atlas 300 集群）。  
关键通信：KV cache 迁移用 **HCCL** 或 **RoCE**（基于以太网的 RDMA）。

---

## 与我何干

**C3 调度：continuous batching**：理解 PD 分离后，continuous batching 的调度逻辑（何时 preempt P 让 D 先跑）更容易理解。

**系统设计面试**：设计高吞吐 LLM serving 系统时，PD 分离是加分点。

**[面试]**：
- "Prefill 和 Decode 为什么要分离？" → 两者计算特性不同（P: compute-bound, D: bandwidth-bound），在同一 GPU 上互相干扰
- "KV Cache 如何在 P 和 D 节点间传输？" → RDMA（InfiniBand 或 RoCE），传输延迟 <10ms，远小于 prefill 时间
- "Chunked prefill 解决什么问题？" → 不分离情况下，长 prompt 阻塞 decode → 切成小块交替执行，降低 P99 TTFT

## 参考

- Splitwise: [arxiv 2311.18677](https://arxiv.org/abs/2311.18677)
- DistServe: [arxiv 2401.09670](https://arxiv.org/abs/2401.09670)
- Sarathi (chunked prefill): [arxiv 2308.16369](https://arxiv.org/abs/2308.16369)
- Mooncake (字节跳动 PD 分离实现): [arxiv 2407.00079](https://arxiv.org/abs/2407.00079)
