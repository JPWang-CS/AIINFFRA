# Prefill-Decode 分离（PD Disaggregation）

> 推理系统技术类 · 用阶段隔离和资源配比管理 Prefill/Decode 干扰

---

## 解决了什么问题

### Prefill 和 Decode 的本质差异

| 阶段 | 做什么 | 计算特点 | GPU 最优配置 |
|------|--------|---------|------------|
| **Prefill** | 处理输入 prompt，生成各位置的 KV | 常有较大矩阵工作量，受 shape/batch 影响 | 以矩阵效率、KV 容量和 TTFT 约束选择 |
| **Decode** | 逐 token 自回归生成 | 每步工作量小，可能受权重/KV/launch 影响 | 以每步时延、KV 容量和并发选择 |

**两者在同一个 GPU 上运行时相互干扰**：
1. 长 prompt 的 Prefill 占据一段连续计算时间，可能推迟 Decode 请求的下一步，形成 TTFT/TPOT 尾部等待；
2. 两阶段的矩阵形状、并行度和权重/KV 访问模式不同，资源配置与 batching 目标也不同；
3. KV cache、临时 buffer 和请求并发共同占用显存，调度器必须在容量与服务目标之间取舍。

“Prefill 一定 compute-bound、Decode 一定 memory-bound”只是某些形状下的工作负载倾向。短 prompt、大 batch、融合和缓存驻留都会改变结论，应以设备时间线和请求指标确认。

### 结果

```
混合 serving 时的问题:
- TTFT (Time to First Token):  新请求排在长 prefill 后，排队与提示词处理时间上升
- TPOT/TBT (每输出 token/相邻 token 时间): 已在生成的请求被 prefill 插入延迟，尾部变长
- 资源利用率:                  两阶段的 shape、缓存和调度需求不同，可能互相挤占
```

---

## 核心思路：物理分离 P 和 D 节点

**Splitwise (Patel et al., 2023) / DistServe (Zhong et al., 2024)**：

```
           ┌─────────────────────┐
请求  →    │   Prefill Cluster   │  按 Prefill shape 与 TTFT 约束配置
           │  (P1, P2, P3, P4)   │
           └──────────┬──────────┘
                      │ KV Cache 传输 (RDMA/NVLink)
           ┌──────────▼──────────┐
           │   Decode Cluster    │  按 KV 容量、TPOT 与并发约束配置
           │  (D1, D2, D3, D4)   │
           └──────────┬──────────┘
                      │ output tokens
                      ▼
```

**工作流程**：
1. 请求到达 → 路由到空闲的 P 节点
2. P 节点完成 prefill，生成 KV Cache
3. **KV Cache 与请求元数据通过选定互联迁移到 D 节点**
4. D 节点接管 decode，逐 token 生成
5. P 节点立即腾出处理下一个请求的 prefill

---

## 关键数据/取舍

### KV Cache 传输开销

KV 传输时间的下界是 `payload_bytes / effective_bandwidth`，实际还包含打包、协议、拥塞、同步和目标端恢复。payload 由层数、KV head、head dimension、token 数、dtype、分页和压缩方式共同决定；因此必须在目标拓扑与请求长度下测量，不能把一次链路算术外推成“传输不是瓶颈”。

### 性能收益如何比较

论文报告的是特定模型、请求分布、硬件和 SLO 下的结果，不能直接作为当前部署的倍数。复测时固定模型版本、输入/输出长度分布、并发、P/D 资源配比和网络拓扑，分别记录 TTFT、TPOT、尾延迟、有效吞吐、KV 传输字节与迁移耗时。

### 硬件选型建议

| 集群 | 推荐 GPU | 原因 |
|------|---------|------|
| Prefill | 适合目标矩阵形状与并发的设备 | 以 Prefill 关键路径和 TTFT 约束选择 |
| Decode | 能容纳权重/KV 并满足 TPOT 的设备 | 以每步生成、容量和尾延迟选择 |
| 异构组合 | P/D 分别测资源配比 | 传输、利用率和 SLO 共同决定成本 |

成本分析应以满足同一 TTFT/TPOT/SLO 的设备数量、利用率、网络和运维成本核算；PD 分离有可能提高资源匹配，也可能因空闲容量与 KV 迁移增加成本。

---

## 进一步优化：Chunked Prefill

即使不做 PD 分离，**chunked prefill** 也能缓解问题：

```python
# 不分离，但将长 prompt 切成 chunk
for chunk in prompt.split(chunk_size=512):
    prefill_chunk(chunk)   # 短 prefill，不长时间阻塞
    # 中间执行若干 decode step（interleaved）
```

效果：给 Decode 插入调度机会，可能降低长请求对尾延迟的干扰；但 chunk 边界会增加调度与 kernel 组织成本，Prefill 总时间和 TTFT 需要实测。

某些引擎提供 chunked-prefill 选项；参数名和默认行为随版本变化，应先用目标版本的 `--help`、源码和日志确认，再与 PD 分离基线比较。

---

## 在 Ascend 的对应

PD 分离是 serving 架构层面的优化；迁移到其他设备时，需确认目标框架是否提供 P/D 调度、KV 传输和一致性管理。关键通信可能使用设备集体通信或 RDMA，但具体 API、拓扑与可用能力必须以目标版本文档和运行记录为准。

---

## 与我何干

**C3 调度：continuous batching**：理解 PD 分离后，continuous batching 的调度逻辑（何时 preempt P 让 D 先跑）更容易理解。

**系统设计面试**：设计高吞吐 LLM serving 系统时，PD 分离是加分点。

**求职追问段落见下方独立小节。**

## 求职追问与示范回答

以下是教学示例，不代表个人实测：
- “为什么考虑 Prefill/Decode 分离？”——两阶段的形状、资源需求和时延目标不同；分离可以独立配置并减少互相干扰，但新增 KV 迁移和排队。先用共置基线测 TTFT、TPOT、尾延迟和有效吞吐，再看分离后的关键路径。
- “KV 如何交接？”——保存与请求版本匹配的 KV 布局、页/块映射和元数据，通过目标互联传输；接收端确认数据完整并建立本地映射后，Decode 才能读取。有效带宽和同步决定迁移成本，不能预设固定毫秒数。
- “Chunked Prefill 解决什么问题？”——把长 prompt 拆块，让调度器在块边界插入 Decode；块大小同时决定提交开销、TTFT 和 TPOT，应在长短请求混合负载下比较分位数。

## 参考

- Splitwise: [arxiv 2311.18677](https://arxiv.org/abs/2311.18677)
- DistServe: [arxiv 2401.09670](https://arxiv.org/abs/2401.09670)
- Sarathi (chunked prefill): [arxiv 2308.16369](https://arxiv.org/abs/2308.16369)
- Mooncake (字节跳动 PD 分离实现): [arxiv 2407.00079](https://arxiv.org/abs/2407.00079)
