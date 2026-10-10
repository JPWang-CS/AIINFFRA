# openPangu Embedded 7B 两节点计划

## 目标

在 11 号机、4 号机各自的 `w00939120` 容器中完成 7B 单机推理，再进入 HTTP 服务与双机通信。部署根为 `/data/docker/w00939120/pangu-deploy`。

## 阶段与验收

| 阶段 | 内容 | 验收条件 |
|---|---|---|
| 1 | 版本依赖 | 核对实际 Python、torch、torch_npu、vLLM、Omni-NPU 版本与插件入口 |
| 2 | 个人环境与基础 NPU | 个人 CANN 9.2.0 显式加载；FP32/BF16 add、matmul 回读；exit 0 |
| 3 | 两台权重 | 两台完整模型文件的大小、SHA256 与可信清单一致 |
| 4 | 两台单机离线推理 | 固定问题取得最终答案；核对 token、stop、退出码和进程释放 |
| 5 | HTTP 服务 | 讨论服务参数后，逐台验证 API 请求、输出与服务日志 |
| 6 | 双机通信 | 讨论网卡、rank、端口和通信方式后，验证初始化与一次完整请求 |

## 运行约定

两台均使用个人 CANN 9.2.0 / C12B096、vLLM 0.14.0 与 Omni-NPU 后端。7B 使用 native Embedded GQA 实现，plugins 为 `omni-npu,omni_custom_models,omni_npu_patches`。

单机基线采用 BF16、TP1、eager、max_model_len 1024、chunked prefill True、async scheduling False；输入按模型官方流程添加 BOS，最小验收使用 `/no_think`。模型启动前重新核对目标卡占用。

当前 NPU 运算使用自有容器 root；安装、源码、cache、日志与模型均在个人部署根。单机 Gloo 使用 lo；双机阶段另行核对互通网卡。

92B 阶段按混部→分离推进，分别分析 P/D 时间、KV 传输、调度、内存和通信。7B 基线的参数与实测结果作为后续检查入口。

执行进度与结果见 [部署记录](./records/2026-10-08-seven-b-execution.md)。
