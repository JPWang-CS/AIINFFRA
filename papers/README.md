# 论文索引

> 工作流：[process.md](./process.md) | 笔记模板：[template.md](./template.md)
> 抓取脚本：`python ../scripts/fetch_papers.py --days 3`

## 阅读规则

- 每篇笔记 ≤ 一页：核心思想 + 关键设计 + 一句话启发
- 每周扫 arxiv cs.DC/cs.LG，仅存档不深入
- 每月一个主题，主题相关论文精读，其余存档
- 优先级：P0 = 必须精读，P1 = 主题相关时深读，P2 = 存档备用

## 📌 和 notes/algorithms/ 的边界（硬规矩）

> **单位是"一篇论文" → 这里；单位是"一个技术/概念" → [notes/algorithms/](../notes/algorithms/)。**

有标志性论文的技术（AWQ、GQA、ZeRO）两边各一份：这里写"论文讲了什么"（精读，带 arxiv），algorithms 那边写"机制 + 怎么实现"并互链。判断：**"读论文" ≠ "会实现"**。非论文技术（online softmax 等）只进 algorithms。

---

## Attention & Kernels

| 论文 | 年份 | 优先级 | 状态 | 笔记 |
|------|------|--------|------|------|
| Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness | 2022 | P0 | ✅ | [笔记](./attention/flash-attention.md) |
| Flash Attention 2: Faster Attention with Better Parallelism and Work Partitioning | 2023 | P0 | ✅ | [笔记](./attention/flash-attention-2.md) |
| GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | 2023 | P0 | ✅ | [笔记](./attention/gqa.md) |
| Ring Attention with Blockwise Transformers for Near-Infinite Context | 2023 | P1 | ⏳ | |

## Inference

| 论文 | 年份 | 优先级 | 状态 | 笔记 |
|------|------|--------|------|------|
| Efficient Memory Management for Large Language Model Serving with PagedAttention | 2023 | P0 | ✅ | [笔记](./inference/paged-attention.md) |
| SGLang: Efficient Execution of Structured Language Model Programs | 2024 | P1 | ⏳ | |
| AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration | 2023 | P1 | ⏳ | |
| GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | 2023 | P1 | ⏳ | |
| FP8 Formats for Deep Learning | 2023 | P1 | ⏳ | |
| Speculative Decoding (Leviathan et al.) | 2023 | P1 | ⏳ | |

## Training

| 论文 | 年份 | 优先级 | 状态 | 笔记 |
|------|------|--------|------|------|
| ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | 2020 | P0 | ✅ | [笔记](./training/zero-paper.md) |
| Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism | 2019 | P0 | ⏳ | |
| Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM | 2021 | P1 | ⏳ | |
| PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel | 2023 | P1 | ⏳ | |
| GSPMD: General and Scalable Parallelization for ML Computation Graphs | 2021 | P1 | ⏳ | |
| MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs | 2024 | P2 | ⏳ | |

## Compiler

| 论文 | 年份 | 优先级 | 状态 | 笔记 |
|------|------|--------|------|------|
| Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations | 2019 | P0 | ✅ | [笔记](./compiler/triton-paper.md) |
| MLIR: Scaling Compiler Infrastructure for Domain Specific Computation | 2021 | P1 | ⏳ | |
| TorchDynamo / TorchInductor (PyTorch 2.0 blog) | 2023 | P1 | ⏳ | |
| TVM: An Automated End-to-End Optimizing Compiler for Deep Learning | 2018 | P2 | ⏳ | |

## Agents

| 论文 | 年份 | 优先级 | 状态 | 笔记 |
|------|------|--------|------|------|
| ReAct: Synergizing Reasoning and Acting in Language Models | 2022 | P0 | ⏳ | |
| SWE-Agent: Agent-Computer Interfaces Enable Automated Software Engineering | 2024 | P0 | ⏳ | |
| Toolformer: Language Models Can Teach Themselves to Use Tools | 2023 | P1 | ⏳ | |
| MemGPT: Towards LLMs as Operating Systems | 2023 | P1 | ⏳ | |
| Generative Agents: Interactive Simulacra of Human Behavior | 2023 | P2 | ⏳ | |

---

状态说明：⏳ 待读 / 📖 在读 / ✅ 已精读 / 📦 存档
