# 多机多卡实验记录模板

## 拓扑

- nodes × GPUs/node：
- GPU / NIC：
- intra-node：PCIe / NVLink / NVSwitch
- inter-node：InfiniBand / RoCE / Ethernet
- `nvidia-smi topo -m` 摘要：
- rank mapping：

## 软件

- Driver / CUDA / NCCL / PyTorch：
- container / image：
- launch command：
- 关键环境变量：

## 正确性

- world size：
- reference：
- loss/output checksum：
- timeout/error handling：

## 通信基线

| primitive | bytes | ranks | algbw | busbw | latency |
|-----------|-------|-------|-------|-------|---------|
| AllReduce | | | | | |
| AllGather | | | | | |
| ReduceScatter | | | | | |
| AllToAll | | | | | |

## 端到端

| 配置 | step time | throughput | peak memory | comm ratio | scaling efficiency |
|------|-----------|------------|-------------|------------|--------------------|
| | | | | | |

## 故障定位

- NCCL debug/topology：
- slow rank：
- IB/RoCE link 状态：
- hang/timeout 根因：

## 结论

- 瓶颈层：GPU / PCIe / NVLink / NIC / fabric / synchronization
- 下一次单变量实验：
- 一分钟面试口径：
