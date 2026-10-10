- [92B custom-ops 组合 vendor](./2026-10-09-custom-ops-combo.md)：inference/training vendor、Metadata symbol 与 combo-loader。
- [92B custom-ops 检查](./2026-10-09-custom-ops.md)：layer14 vendor、libcust_opapi symbol、个人目录 loader。
- [92B DP 同步与首请求记录](./2026-10-09-dp-sync.md)：最小同步验证、回退配置、health ready 与首请求自定义算子错误。
# 运行记录索引

- [7B 双机 TP2 推理](./2026-10-09-seven-b-tp2.md)：两侧 rank/权重与 CANN 库、四组请求、BOS/SSE/stop、启动修正与清理。

- [双节点 HCCL 验证](./2026-10-09-two-node-hccl.md)：FP32/BF16 SUM、4096 元素比较、AllGather、TLS 选卡与清理证据。

- [两台 7B HTTP 验证](./2026-10-09-seven-b-http.md)：BOS、非流式/流式、stop、请求 ID 映射、日志与清理。
- [7B ACS 评测](./2026-10-09-seven-b-acs.md)：两台并发 1/2 短输入 prof、C-Eval 子集质量试评、指标与清理证据。
- [两节点 7B 部署与推理](./2026-10-08-seven-b-execution.md)：版本、环境、权重、两台单机推理与调试过程。
- [CANN 9.2.0 统一](./2026-10-08-cann-920-unification.md)：个人目录安装与加载路径。
- [Registry 与参考检查](./2026-10-08-registry-and-reference-check.md)：镜像及参考脚本来源。
- [两节点预检](./2026-10-08-two-node-7b-preflight.md)：早期环境快照。
- [运行时准备](./2026-10-08-runtime-preparation.md)：早期下载与版本评估。
- [容器预检](./2026-10-08-container-preflight.md)
- [连接检查](./2026-10-08-connection-check.md)

每次运行保存配置、命令、输入输出、退出状态、日志路径和下一步。失败与修复后的运行分别保存；benchmark 与 profile 分开采集。

- [双机网络只读盘点](./2026-10-09-two-node-network.md)：网卡、命名空间、设备可见性与 HCCN 工具缺口。
- [双机 CPU TCP/Gloo 验证](./2026-10-09-two-node-cpu-comm.md)：TCP 互认、TCPStore、Gloo 三组 AllReduce、barrier 与清理。

- [2026-10-09 92B 只读预检](./2026-10-09-92b-preflight.md)

- [2026-10-09 92B 依赖与 smoke](./2026-10-09-92b-deps-and-smoke.md)

- [92B四卡启动与冷加载](./2026-10-09-92b-http.md)
- [92B DP 同步诊断与回退](./2026-10-09-dp-sync.md)：最小通信验证、配置回退与新启动状态。
