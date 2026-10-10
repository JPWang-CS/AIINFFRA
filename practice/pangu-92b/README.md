# 盘古92B部署实操

操作入口：[实操教程](./tutorial.md)。

## 目标与执行范围

按基础环境→7B 单机推理与服务→92B PD 混部→92B PD 分离推进。混部和分离阶段均分析 Prefill、Decode、KV、通信与完整请求 trace，持续记录部署问题和调试方法。

允许目标为 13、5、4、11 号机对应的 liteserver；本阶段使用 11 号机与 4 号机。远端执行进入自有容器 `w00939120`，物理工作根 `/data/docker/w00939120`，部署操作位于个人 `pangu-deploy` 目录。每次连接检查实际目录；shell 禁用 history 并清除本次会话历史。

## 当前状态

92B四卡混部在4号机完成50/50分片读取，四rank均加载26.76GiB；使用配置回退后已完成DP同步、KV初始化并达到HTTP health ready。首个请求在自定义算子加载阶段报缺少 aclnnAiInfraKvQuantSparseFlashAttentionV2 或对应 GetWorkspaceSize，尚未形成请求成功证据。详见 [92B启动记录](./records/2026-10-09-dp-sync.md)。

2026-10-09 已完成双节点 HCCL 与 7B TP2 最小推理。两侧各使用物理 5 号卡，10 组集合通信通过；11 号机 API 与 4 号机 headless worker 共同执行 TP2，四组请求通过，两侧服务退出 0，卡和端口释放。个人 CANN 库映射、BOS/SSE/stop、两侧 rank/权重与启动问题均已归档。

- [双节点 HCCL 记录](./records/2026-10-09-two-node-hccl.md)
- [7B 双机 TP2 记录](./records/2026-10-09-seven-b-tp2.md)

2026-10-08 已完成两台版本依赖、个人 CANN/Python 环境、基础 NPU 运算、完整 7B 权重校验和单机离线推理。两台自我介绍取得完整回答，17+25 返回 42；请求均 stop，退出码均为 0。运行使用个人 CANN 9.2.0、vLLM 0.14.0 与 Omni-NPU 后端。

2026-10-09 两台依次完成独立单机 HTTP 验证：健康/模型列表、非流式自我介绍与算术、流式算术和显式 stop。服务端 BOS、输入/输出 token、usage、SSE 和请求 ID 日志映射已核对。两台服务均 exit 0，本轮进程退出、端口关闭，物理 1 号卡空闲。

- [两节点部署与调试记录](./records/2026-10-08-seven-b-execution.md)
- [两台 HTTP 验证与请求日志](./records/2026-10-09-seven-b-http.md)
- [环境快照](./environment.md)
- [两节点阶段计划](./two-node-7b-plan.md)
- [运行环境](./runtime-stack-plan.md)
- [92B 权重对比](./model-artifact-comparison.md)
- [项目立项与证据](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/ANALYSIS_STATE.md)

当前先定位私有自定义算子缺失；P/D 分段、长输入、并发扫描和 profile 按后续阶段采集。课程网页入口为 [course-site/index.html](../../roadmap/curriculum/gpu/course-site/index.html)；本模块当前以 Markdown 和项目证据交付，专属网页在后续分析阶段整理。

## 每次进入模块先读

按顺序读取本入口、[既有部署经验参考](./prior-deployment-reference.md)、[通用整网分析方法](../llm-analysis/README.md)、当前本地工作区的 [fullnet-vllm-ascend skill](D:/Desktop/Code/CC4Ascend/.codex/skills/fullnet-vllm-ascend/SKILL.md)、[fullnet memory](D:/Desktop/Code/CC4Ascend/.codex/agent-memory/fullnet/MEMORY.md)，以及 AIINFFRA 的 [PATH](../../PATH.md)、[NOW](../../NOW.md)、[HISTORY](../../HISTORY.md)。本轮已核对这些本地路径可读；进入远端主机或容器后必须重新核查访问结果、引用和版本。先梳理既有经验和缺口，再连接环境。既有 7B 材料只能作为方法线索，不能作为 92B 参数或支持结论。

各阶段按实际结果整理以下记录：

```text
environment.md
model-and-startup.md
pd-colocated.md
pd-disaggregated.md
request-trace.md
comparison.md
troubleshooting.md
```

## 阶段顺序

小模型阶段作为 92B 基线前置；完成基础环境核对后，再讨论小模型最小生成/服务，随后进入 92B 混部和分离。

| 阶段 | 输入 | 计划记录 | 可核查退出条件 |
|---|---|---|---|
| 1. 连接与只读盘点 | 已验证目标/容器入口；后续核对运行时与资料可达性 | 连接身份、主机/容器只读信息、可读 skill/memory、目录与工具版本 | 目标环境和权限边界已确认；未写入配置；缺口列为待确认 |
| 2. 模型与资源确认 | 具体模型标识、权重、dtype、tokenizer、框架/容器/源码版本（待确认） | 模型配置、权重与激活估算、KV/临时/通信缓冲账本、总显存和设备拓扑 | 版本和资源预算可追溯；未知字段仍保持未知 |
| 3. 小模型最小生成/服务 | 基础环境快照、已确认的小模型、固定输入输出与停止条件 | 启动链、请求链、完成/失败/超时、最小日志 | 小模型最小请求有完整退出结果；不把它当作 92B 性能证明 |
| 4. PD 混部最小基线 | 最小请求、固定输入输出与停止条件 | 启动链、请求链、端到端指标、完成/失败/超时、基线日志 | 最小请求有完整退出结果和原始日志；未启 profile 的正式基线可复现 |
| 5. 混部 P/D 分析 | 基线 run、短/长输入输出、多并发负载 | P/D 分段、调度/排队、Prefill/Decode 时间线、CPU/设备 trace、慢 rank/通信 | 至少一次完整请求 trace；每个瓶颈有证据或保持未知 |
| 6. PD 分离部署 | 已确认的框架/连接器/网络和两侧资源（待确认） | P/D 启动、代理/路由、P/D 配额、KV 产生/传输/接收/释放与等待 | 两侧 ready；至少一次请求完成；KV 交接有证据；返回 token 与 finish reason 可核对。服务退出码仅在实际退出时记录 |
| 7. 分离分析与公平对比 | 混部和分离同口径负载、资源与缓存状态 | TTFT/TPOT/ITL、吞吐、尾延迟、显存、KV/代理开销、条件差异、完整请求 trace、Prefill/Decode 分段 | 只在资源和测量口径匹配时比较；否则明确不可直接称公平优胜 |
| 8. 问题与调试复盘 | 失败 run、日志、trace、源码版本 | 症状、最深已知层、预测、检查、失败尝试、最小修复、新 run | 修复用新 run 复验，原失败记录保留；经验注明适用范围 |

## 后续产物

已生成 [环境快照](./environment.md) 和 [7B 部署记录](./records/2026-10-08-seven-b-execution.md)。后续按阶段整理 `model-and-startup.md`、`pd-colocated.md`、`pd-disaggregated.md`、`request-trace.md`、`comparison.md`、`troubleshooting.md`。

每个阶段都先写“拟执行动作、预期观测、停止条件”，讨论通过后再执行。真实记录入口见 [records/README.md](./records/README.md)，通用方法见 [整网分析方法](../llm-analysis/README.md)。

## 2026-10-09 7B ACS 评测

两台分别完成并发 1/2、每档 32 条短输入 prof，四档全部成功；C-Eval 四科 val 前 8 条由 ACS 采集回答并使用项目精确判分，两台均 22/32。原始结果与指标见 [项目 ACS 记录](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/execution-record-2026-10-09-acs.md) 和 [本地证据目录](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/evidence/2026-10-09-seven-b-acs/README.md)。本轮未执行 OpenCompass eval。

- [92B 只读预检记录](./records/2026-10-09-92b-preflight.md)


2026-10-09 92B 组合 vendor 复验已完成四个 HTTP cases：health/models、intro、算术普通/流式和显式 stop 均通过；BOS/EOS、SSE、usage、token 序列和清理已核对。详见 [组合 vendor 记录](./records/2026-10-09-custom-ops-combo.md)。

## 2026-10-10 92B PD 分离阶段A

- [PD 分离方案](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/92b-disaggregated-plan.md)
- [阶段A记录](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/execution-record-2026-10-10-pd.md)

阶段A仅完成两节点运行时、连接器和 DataDist 的只读核对；尚无跨机 KV 传输、请求或性能结论。

## 2026-10-10 PD 阶段 C

- [阶段 C HIXL micro 记录](./execution-record-2026-10-10-pd.md)

已完成基础 HIXL uint8 register/link/pull 样例；92B P/D 模型部署尚未完成。
