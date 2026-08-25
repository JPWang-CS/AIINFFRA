# 2026 AI Infra 论文与项目观察池

> 核对日期：2026-08-25。这里维护“值得观察的增量”，不维护学习进度。
> P0/P1 仍按当前 PATH 挂载；最新不等于当前必须读。

## 当前主线直接相关

| 条目 | 日期 | 类型 | 优先级 | 为什么值得看 | 何时读 |
|------|------|------|:--:|----------------|--------|
| [FlashAttention-4](https://arxiv.org/abs/2603.05451) | 2026-03 | paper | P1 | Blackwell 非对称流水、异步 MMA、2-CTA/TMEM、CuTe DSL | Triton FA 完成后 |
| [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) | 2025-12 | paper | P0 | DSA indexer、稀疏 attention 与 serving 影响 | FA2→MLA 后 |
| [SageAttention3](https://arxiv.org/abs/2505.11594) | 2025-05 | paper | P1 | FP4/低比特 attention 与精度取舍 | 量化枝干 |
| [Kascade](https://arxiv.org/abs/2512.16391) | 2025-12 | paper | P1 | 跨层 top-k 复用与稀疏 attention | DSA 后 |

## 多机多卡与网络

| 条目 | 日期 | 类型 | 优先级 | 学习重点 | 挂载 |
|------|------|------|:--:|----------|------|
| [Collective Communication for Distributed LLM Systems](https://arxiv.org/abs/2608.15118) | 2026-08-15 | survey/tutorial | P1 | planning、runtime adaptation、compute-communication coordination 三层 taxonomy | D1–D5 总览 |
| [CommBench](https://arxiv.org/abs/2608.04450) | 2026-08-05 | benchmark | P2 | P2P、collective、EP、融合通信任务与正确性/性能联合评价 | D1 后选题库 |
| [NCCL EP](https://arxiv.org/abs/2603.13606) | 2026-03 | paper/system | P1 | LL decode、HT prefill/training、hierarchical EP dispatch/combine | D5 |
| [HetCCL](https://arxiv.org/abs/2601.22585) | 2026-01 | paper/system | P2 | NVIDIA/AMD 异构 collective 与 RDMA | 了解生态边界 |
| [FlexLink](https://arxiv.org/abs/2510.15882) | 2025-10 | paper/system | P2 | 聚合 NVLink、PCIe、RDMA 异构链路 | D4 进阶 |
| [ByteScale](https://arxiv.org/abs/2502.21231) | 2025-02 | paper/system | P1 | 12k GPU 长上下文、动态 mesh、DP/CP 统一与负载均衡 | D4/CP |
| [MegaScale](https://arxiv.org/abs/2402.15627) | 2024-02 | NSDI paper | P0 | 10k+ GPU 的通信重叠、网络调优、可观测性、故障与 straggler | D3–D4 |

## 官方项目观察

| 项目 | 当前关注点 | 学习挂载 |
|------|------------|----------|
| [NCCL](https://github.com/NVIDIA/nccl) / [nccl-tests](https://github.com/NVIDIA/nccl-tests) | topology、multi-NIC、IB/RoCE、MNNVL、collective baseline | D1–D3 |
| [PyTorch distributed](https://docs.pytorch.org/tutorials/beginner/dist_overview.html) | DeviceMesh、DTensor、FSDP2、TP | D2–D4 |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | TP/PP/CP/EP、distributed optimizer、overlap | D4 |
| [DeepEP](https://github.com/deepseek-ai/DeepEP) | EP V2、NCCL Gin、low-SM/zero-SM、NVLink+RDMA | D5 |
| [DeepSeek profile-data](https://github.com/deepseek-ai/profile-data) | V3/R1 prefill/decode communication-compute overlap | D5 |
| [vLLM EP deployment](https://github.com/vllm-project/vllm/blob/main/docs/serving/expert_parallel_deployment.md) | 多节点 EP backend 与部署参数 | M3 serving |

## 筛选规则

- P0：基础系统论文，必须产出完整笔记或复现实验。
- P1：与近期 PATH 节点直接相关，到挂载点再读。
- P2：只读 abstract/conclusion，除非实验遇到相同问题。
- 项目 README/博客的性能数字不能替代论文、官方文档或目标集群实测。
- 新条目先进入 [inbox](inbox/README.md)，核验后才进入本页。
