# vLLM 源码深挖计划（算子线 C：推理系统）

> 对应 PATH 算子线 C，以及大模型板块 [推理系统](../notes/llm/inference-systems.md)。
> 目标：从请求入队到 token 返回，完整理解 vLLM 的调度、KV cache、模型执行链路。

---

## 1. 先满足前置

- [ ] 理解 Prefill vs Decode 的瓶颈差异
- [ ] 理解 Flash Attention / online softmax
- [ ] 理解 GQA / KV cache
- [ ] 安装 vLLM，能跑起一个最小服务

```bash
pip install vllm
python -c "import vllm; print(vllm.__version__)"
```

## 2. 整体链路

```text
请求
 -> LLM / AsyncLLMEngine
 -> Scheduler
 -> Worker
 -> ModelRunner
 -> Attention（PagedAttention）
 -> KV Cache
 -> 采样
 -> 返回
```

## 3. 学习阶段

### Phase 1：基础概念

目标：先建立推理系统的语言，不急着读源码。

| 主题 | 要回答的问题 |
|------|-------------|
| Prefill vs Decode | 为什么一个 compute-bound、一个 memory-bound |
| TTFT / TPOT | 分别衡量什么 |
| KV cache | 占多少显存，怎么估算 |
| Continuous batching | iteration 调度是什么 |
| PagedAttention | block table 解决什么问题 |

完成定义：
- [ ] 能手算一个模型在固定 seq 下的 KV cache 大小
- [ ] 能画 prefill/decode 的时间线

### Phase 2：PagedAttention

目标：读懂 KV cache 的虚拟到物理映射。

核心概念：

```text
逻辑 block：请求连续编号
物理 block：显存中任意位置
block table：把逻辑编号映射到物理编号
```

示例：

```text
逻辑 block:  [0] [1] [2]
block table: [8] [3] [11]
物理 block:  8 -> 3 -> 11
```

源码入口：

- `vllm/attention/ops/paged_attn.py`
- `vllm/core/block_manager.py`
- `vllm/worker/cache_engine.py`

完成定义：
- [ ] 能画 block table 映射图
- [ ] 能解释 copy-on-write 为什么省显存
- [ ] 能解释前缀共享怎么复用 KV

### Phase 3：Scheduler

目标：理解请求如何被调度。

要回答的问题：
1. Prefill 和 Decode 请求怎么混排？
2. 显存不足时怎么抢占？
3. Chunked prefill 如何切块？
4. 新请求什么时候进入？

源码入口：

- `vllm/core/scheduler.py`
- `vllm/core/block_manager.py`
- `vllm/engine/llm_engine.py`

完成定义：
- [ ] 画出一个 iteration 的调度循环
- [ ] 能解释请求完成即退出的逻辑

### Phase 4：Worker / ModelRunner

目标：理解模型加载和 forward 链路。

要回答的问题：
1. 权重如何加载、分片？
2. KV cache 如何初始化？
3. 模型 forward 如何调用 attention 后端？
4. 多卡 TP 怎么切？

源码入口：

- `vllm/worker/worker.py`
- `vllm/worker/model_runner.py`
- `vllm/model_executor/`

完成定义：
- [ ] 能画模型从权重到 forward 的链路
- [ ] 能说出 TP 下权重如何分片

### Phase 5：量化通路

目标：知道量化权重如何加载和调用。

| 方案 | 量化对象 | 主要文件 |
|------|---------|---------|
| AWQ | 权重 | `vllm/model_executor/layers/quantization/awq.py` |
| GPTQ | 权重 | `vllm/model_executor/layers/quantization/gptq.py` |
| FP8 | 权重/激活 | `vllm/model_executor/layers/quantization/fp8.py` |
| KV cache 量化 | KV | `vllm/attention/ops/paged_attn.py` 相关配置 |

完成定义：
- [ ] 跑一个量化模型
- [ ] 对比 FP16 / INT8 / FP8 的显存和延迟

### Phase 6：端到端 benchmark

```bash
python -m vllm.entrypoints.openai.api_server --model <model> --max-model-len 4096
```

或离线：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="<model>")
out = llm.generate(["Explain KV cache."], SamplingParams(max_tokens=128))
```

记录：
- TTFT
- TPOT / TBT
- throughput
- peak memory
- KV cache 使用率

## 4. 输出物

- [ ] `notes/llm/paged-attention.md`
- [ ] `notes/llm/scheduler.md`
- [ ] `notes/llm/inference-pipeline.md`
- [ ] `notes/llm/quantization.md`
- [ ] `notes/llm/vllm-vs-sglang.md`

## 5. 常见坑

| 坑 | 解法 |
|----|------|
| 源码版本和文档不一致 | 以本地安装版本为准 |
| 只看代码不跑服务 | 先跑起来，再用代码验证 |
| 把 block 和 token 混淆 | block 是 KV 容器，token 是序列单元 |
| 只看一个模块 | 从 scheduler 到 attention 串起来 |