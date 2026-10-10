# 92B PD 分离部署方案

本方案定义 92B 的 P/D 分离部署：11 号机作为 Prefill（P），4 号机作为 Decode（D），两侧各使用四张空闲且 TLS 状态一致的卡。目标流程是跨机 KV 小测试、双实例启动、四案例请求与多 block 请求、正常退出和清理。阶段 A 的现场读取、依赖差异和失败项记录在 `execution-record-2026-10-10-pd.md`，不把方案内容当作已经完成的运行证据。

## 边界与固定拓扑

两台容器均使用 `/data/docker/w00939120/pangu-deploy` 作为私有根，CANN 从个人目录 `/data/docker/w00939120/pangu-deploy/cann/9.2.0/cann-9.2.0` 显式加载；模型权重保持共享路径只读。自定义算子使用 `custom-ops/20261009T-layer14-vendor-combo-01`，不修改宿主、Docker、公共配置、设备 TLS 或共享模型。每次运行使用新的 run id、nonce、日志目录和临时目录。

| 角色 | 主机 | 物理卡 | TP/DP/EP | 运行方式 |
|---|---|---|---|---|
| P | 11 号机 | 4/5/6/7 | TP1/DP4/EP4 | eager、MXFP8 权重、BF16 激活、MTP 关闭 |
| D | 4 号机 | 4/5/6/7 | TP1/DP4/EP4 | eager、MXFP8 权重、BF16 激活、MTP 关闭 |

两侧使用 `max_model_len=2048`、`max_num_batched_tokens=2048`、`max_num_seqs=4`、`block_size=128`、`kv_cache_dtype=hif8_ds_mla`、`gpu_memory_utilization=0.6`。两机合计八张卡，资源口径不同于此前四卡单机混部方案，不能直接比较性能。功能验收完成后再另定 ACS 与昇腾 profiler 的采集方案；它们不作为本方案的功能验收结束条件。

## 连接器与运行时前提

P/D 两侧须加载私有 CANN、DataDist 和自定义算子，并通过同一套 Omni-NPU 路线注册 KV connector。承重源码为 `omni_npu/connector/llmdatadist_connector_v1.py`：`LLMDataDistConnector` 在第 77 行继承 `KVConnectorBase_V1, SupportsHMA`，第 85 行读取角色；`omni_npu/platform.py` 第 61 行读取 `omni.kv_connectors` entry point，第 91 行绑定 KV cache，第 152–162 行选择 attention backend。注册、动态库加载或 `SupportsHMA` 存在，只能作为前置条件，不能替代跨机 KV 证据。

P/D 的地址、端口、rank/world size、角色和 HMA 能力必须在运行前写入本次配置，并由两端日志互相确认。网络接口、TLS、DataDist 端点和自定义算子版本采用同一次现场快照；任一端版本或角色不一致，停止本次运行并保留日志。

## 实施阶段与验收门

### A. 依赖与拓扑门

复核两端私有 CANN、DataDist、Omni-NPU、模型和自定义算子版本，检查四张目标卡的实时占用、TLS 状态、端口和公共文件哈希。确认 P/D 地址、端口、rank/world size、角色、HMA 和 KV 生命周期配置。只读注册检查不能替代后续设备实测。

### B. 模型前 NPU cache 门

在加载模型前，用实际目标设备和最终角色配置完成 uint8 NPU cache 的 register、link、pull 最小校验。记录物理卡、逻辑设备、rank、P/D 角色、KV handle 摘要、返回码、设备同步和清理结果。必须由 NPU 真实调用完成；Python 注册、库映射或 CPU 合成矩阵均不足以关闭此门。任一 register/link/pull 失败，停止进入模型加载。

### C. 双实例就绪门

分别启动 P、D 实例，确认各自模型配置、权重加载、KV connector、端口和健康状态。两侧日志应给出相同的 run id、nonce、world size 和互补角色；服务未就绪时不发送请求。

### D. 跨机 KV 小测试

先发送一条短 prompt，使 P 产生可交给 D 的 KV；再由 D 拉取并继续 decode。记录 KV register/link/pull 的请求 id、字节或 handle 摘要、P/D 日志对应行、首个 decode 结果和 cleanup。该步骤先于完整请求矩阵执行，失败时保留最深的 DataDist 或设备层错误。

### E. 请求验收

依次验证中文自我介绍、`17+25` 算术、流式请求和显式 stop；再增加多 block 输入，检查 prompt token、输出 token、BOS/EOS、finish reason、SSE `[DONE]`、usage 和请求 id。每个案例保存原始请求、响应、P/D 日志关联和退出结果。通过四案例不等于多 block 或长上下文通过。

### F. 结束与清理

停止本次 P/D 进程，等待子进程回收，复核端口、选卡进程、KV/DataDist 会话和私有临时目录；只清理本次会话 history，保留运行记录。公共五个配置文件前后哈希应一致，任何残留或异常都进入执行记录。

## 实施约束

每阶段使用独立 run id 和证据目录，阶段门关闭前不进入下一阶段。实施前仍需核对 P/D 角色、地址、端口、rank/world size、HMA、TLS、KV 生命周期和真实目标设备；未核验字段保持待确认。模型加载、跨机 KV 和请求验收完成后，才讨论正式 ACS/profiler、长上下文、并发和 PD 性能比较。
