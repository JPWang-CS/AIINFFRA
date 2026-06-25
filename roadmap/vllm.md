# vLLM 源码深挖 (Phase 3: Week 19-30)

理解 vLLM 的完整推理链路，从请求入队到 token 响应。

## 分析路线

### 模块拆解

```
请求 → Scheduler → Worker → ModelRunner → Attention → KV Cache → 返回
  │        │          │          │             │           │
  ▼        ▼          ▼          ▼             ▼           ▼
 排队    抢占策略   权重管理   forward    PagedAttention  block管理
```

### 分析计划

| 周 | 模块 | 文件 | 核心问题 |
|----|------|------|---------|
| W19-20 | 基础概念 | - | Prefill vs Decode 的计算/访存差异 |
| W21-22 | PagedAttention | `vllm/attention/` | block table 如何实现虚拟→物理映射？Copy-on-write 机制？ |
| W23-24 | Scheduler | `vllm/core/scheduler.py` | 如何决定先跑哪个请求？抢占策略？Chunked prefill？ |
| W25-26 | Worker + Runner | `vllm/worker/`, `vllm/worker/model_runner.py` | 模型加载流程？权重分片？KV Cache 初始化？ |
| W27-28 | 量化支持 | AWQ/GPTQ/FP8 通路 | 量化权重如何加载？INT8/FP8 的 kernel 调用链？ |
| W29-30 | SGLang 对比 | - | RadixAttention vs PagedAttention 的取舍 |

### 输出

- [ ] `paged-attention.md` — PagedAttention 核心代码注解
- [ ] `scheduler.md` — 调度策略分析 + 流程图
- [ ] `inference-pipeline.md` — 端到端请求链路分析
- [ ] `quantization.md` — 量化方案对比 + 代码路径
- [ ] `vllm-vs-sglang.md` — 架构对比

## 关键源码入口

```
vllm/
├── entrypoints/          # API 入口
│   └── llm.py            # LLM class — 用户接口
├── engine/
│   ├── llm_engine.py     # 核心引擎 — 调度循环
│   └── async_llm_engine.py
├── core/
│   ├── scheduler.py      # 调度器 ★★★
│   └── block_manager.py  # KV Cache block 管理 ★★★
├── worker/
│   ├── worker.py         # GPU worker
│   └── model_runner.py   # 模型执行 ★★
├── attention/
│   └── ops/
│       └── paged_attn.py # PagedAttention kernel 入口 ★★★
└── model_executor/
    └── models/           # 各模型适配
```

**★ 数越多越重要**，建议按 ★★★ → ★★ → ★ 的顺序读。
