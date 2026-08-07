# Agent 实验室计划（算子线 E：熟悉 + 1 个 demo）

> 对应 PATH 算子线 E，以及 [notes/llm](../notes/llm/README.md)。
> 目标：理解 Agent 架构、Tool Use、MCP、RAG，完成 1 个可展示 demo。

---

## 1. 前置

- [ ] 理解 LLM 基本生成流程
- [ ] 能调用 OpenAI-compatible API
- [ ] 理解函数调用 / tool calling
- [ ] 有 Python 环境

## 2. 核心概念

| 概念 | 说明 | 要能回答 |
|------|------|---------|
| Agent | LLM 驱动工具选择和执行的循环 | 和普通 chatbot 区别 |
| Tool Use | LLM 输出结构化工具调用 | schema 怎么设计 |
| ReAct | 推理 + 行动交替 | 为什么能减少幻觉 |
| MCP | 标准化的 tool/resource 协议 | server/client 怎么通信 |
| RAG | 检索增强生成 | 分块/检索/生成怎么串 |
| Memory | 多轮状态和长期记忆 | 和 KV cache 的区别 |

## 3. 学习阶段

### Phase 1：Tool Calling

目标：先理解一个 LLM 如何调用工具。

步骤：
1. 定义工具：名称、描述、JSON Schema。
2. 让 LLM 生成 tool call。
3. 执行工具。
4. 把结果回填给 LLM。
5. 循环直到回答完成。

完成定义：
- [ ] 跑通“计算器/天气/文件查询”工具
- [ ] 能画 tool call 时序图
- [ ] 能处理工具返回错误

### Phase 2：ReAct

目标：理解 Agent 主循环。

```text
观察问题 -> 思考 -> 行动 -> 观察结果 -> 再思考 -> 最终回答
```

完成定义：
- [ ] 实现 `react_loop.py`
- [ ] 能解释为什么需要限步数
- [ ] 能处理 max_iterations 和错误重试

### Phase 3：RAG

目标：跑通最小 RAG pipeline。

步骤：
1. 文档分块。
2. embedding。
3. 向量检索。
4. 把检索结果放入 prompt。
5. 生成带引用回答。

可选：
- BM25 + 语义混合检索
- reranker

完成定义：
- [ ] `ingest.py`、`retrieve.py`、`generate.py` 跑通
- [ ] 能测量检索 top-k 对回答的影响

### Phase 4：MCP

目标：理解 MCP server/client。

步骤：
1. 写一个简单 MCP server，暴露一个 tool。
2. 写 client 调用。
3. 理解 tool / resource / prompt 三种原语。

完成定义：
- [ ] `server.py` + `client.py` 跑通
- [ ] 能画 MCP 通信流程
- [ ] 能说 MCP 和直接 function calling 的区别

### Phase 5：Agent 系统设计

目标：把单个 Agent 扩成可维护系统。

要回答：
- 多个工具怎么注册？
- 权限和错误边界怎么设计？
- 长任务怎么规划？
- 多 Agent 怎么协作？
- 如何加可观测性？

完成定义：
- [ ] 设计一个“客服 / 代码助手 / 数据查询”Agent
- [ ] 画架构图
- [ ] 列出风险和控制手段

## 4. 输出物

- [ ] `solutions/agents/mcp-demo/`
- [ ] `solutions/agents/tool-use-lab/`
- [ ] `solutions/agents/rag-project/`
- [ ] `notes/llm/agents.md`

## 5. 常见坑

| 坑 | 解法 |
|----|------|
| 工具调用格式不稳定 | 使用结构化 JSON Schema |
| 无限循环 | 限制 max_iterations |
| 工具报错导致崩溃 | 错误捕获 + 重试 |
| RAG 分块太随意 | 按语义/标题/长度实验 |
| 只调 API 不理解协议 | 至少读一次 MCP 文档 |