# 论文管理工作流

## 节奏

- **每天 5min**：扫 arxiv RSS / X / 公众号标题，有意思的存链接
- **每周日 30min**：把本周存的链接过一遍 → 补充到索引 → 标记优先级
- **每月底 30min**：回顾本月精读、归档过时的 P2 条目

## 发现管道

### 主要来源

| 来源 | 频率 | 方法 |
|------|------|------|
| arxiv cs.DC | 每日 | `scripts/fetch_papers.py` 自动抓 |
| arxiv cs.LG | 每日 | 同上 |
| arxiv cs.CL | 每日 | 同上 |
| MLSys / OSDI / SOSP / ASPLOS | 每年 | 关注 proceedings |
| 知乎/公众号 | 日常 | 看到就存 |
| X/Twitter (AI Infra 圈) | 日常 | 关注 @cHHillee, @tri_dao, @vllm_project 等 |

### 自动抓取

```bash
# 抓最近 3 天 AI Infra 相关论文（cs.DC + cs.LG + cs.CL）
python scripts/fetch_papers.py --days 3

# 抓指定关键词
python scripts/fetch_papers.py --query "attention AND (GPU OR CUDA OR inference)"

# 只看新论文标题，快速筛选
python scripts/fetch_papers.py --days 7 --titles-only
```

## 分类标准

| 优先级 | 含义 | 动作 |
|--------|------|------|
| P0 | 必须精读，领域基础 | 写完整笔记，搞懂每个设计决策 |
| P1 | 与当前学习主题相关 | 主题月内精读 |
| P2 | 知道即可 | 只读 abstract + conclusion，存档 |

## 笔记模板

每篇论文笔记 ≤ 一页，固定四个部分：

```
# 标题
**作者/年份/会议** | **状态** | **优先级**

## 解决了什么问题
（1-2 句话）

## 怎么解决的
（核心方法，3-5 个要点）

## 关键数据
（性能数字、消融实验结论）

## 与我何干
（一句话：对我的工作/学习有什么启发）
```

## 每月主题轮换

| 月份 | 主题 | 精读重点 |
|------|------|---------|
| 2026.06 | CUDA + GPU 架构 | GPU 体系结构论文、CUDA 最佳实践 |
| 2026.07 | Attention 机制 | Flash Attn 1/2、GQA/MQA、Ring Attn |
| 2026.08 | Triton + 编译器 | Triton、MLIR、TorchDynamo |
| 2026.09 | LLM 推理 | PagedAttention、量化、调度 |
| 2026.10 | LLM 推理（续） | SGLang、SpecDecode、MoE 推理 |
| 2026.11 | 分布式训练 | ZeRO、FSDP、Megatron、MoE |
| 2026.12 | Agent 系统 | ReAct、Tool Use、MCP、RAG |
| 2027.01+ | 待定 | 根据业界动态调整 |

## 归档规则

- P2 论文如果 3 个月没碰 → 移到 `archive/`，只保留索引条目
- P0/P1 精读过的论文永久保留笔记
- 每年年底清理一次索引
