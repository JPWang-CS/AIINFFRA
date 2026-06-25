# Agent 实验室（算子线 E：熟悉 + 1 个 demo）

> 进度看 [PATH.md](../PATH.md)。

理解 Agent 架构和 MCP 协议，跟踪前沿论文，搭建可展示的 Agent 应用。

## 子项目

### mcp-demo

MCP (Model Context Protocol) 的 server + client 实现：
- 实现一个简单的 MCP server（如 SQLite 查询、文件系统操作）
- 对接 Claude / GPT 作为 client
- 理解 tool/resource/prompt 三种 MCP 原语

```
mcp-demo/
├── server.py        # MCP server 实现
├── client.py        # MCP client (调用 LLM + tools)
└── README.md
```

### tool-use-lab

Function calling / Tool use 实验：
- ReAct loop 实现
- Tool schema 设计（JSON Schema → function call）
- 多轮 tool calling 的状态管理
- 错误恢复（tool 调用失败时的重试策略）

```
tool-use-lab/
├── react_loop.py    # ReAct agent 主循环
├── tools.py         # 工具定义
└── README.md
```

### rag-project

RAG pipeline 实践：
- 文档分块 + embedding + 向量检索
- 多轮对话中的上下文管理
- Rerank + 混合检索（BM25 + 语义）

```
rag-project/
├── ingest.py        # 文档摄入
├── retrieve.py      # 检索 pipeline
├── generate.py      # 生成 + 引用
└── README.md
```

## 论文跟进

详见 `papers/agents/` 目录和 `papers/README.md` 中 Agent 部分。

## Agent 核心概念速查

| 概念 | 说明 | 关键论文/资源 |
|------|------|-------------|
| ReAct | Reasoning + Acting 交替 | ReAct (Yao et al., 2022) |
| Tool Calling | LLM 调用外部工具 | Toolformer (2023), GPT-4 Function Calling |
| MCP | 标准化的 tool/resource 协议 | Anthropic MCP 规范 |
| RAG | 检索增强生成 | 多篇（持续更新中） |
| Multi-Agent | 多个 Agent 协作/竞争 | AutoGen, CrewAI |
| Planning | Agent 自主分解任务 | Plan-and-Execute, Tree of Thoughts |
| Memory | Agent 的长期记忆管理 | MemGPT (2023) |
