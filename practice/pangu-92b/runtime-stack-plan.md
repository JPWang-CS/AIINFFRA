# 盘古推理运行环境

## 当前软件组合

两台使用个人 CANN 9.2.0 / C12B096、Python 3.11.6、torch 2.9.0、torch_npu 2.9.0.post7.dev20260817、vLLM 0.14.0（metadata 0.14.0+empty）、omni-npu 0.2.0、Transformers 4.57.6。版本与实际导入、NPU 运算、7B 离线推理结果见 [部署记录](./records/2026-10-08-seven-b-execution.md)。

运行环境从指定参考镜像的已校验层构造，置于个人目录。镜像的 PyTorch-2.10.0 目录名保留为来源标签，实际版本以包 metadata 和 import 为准。早期 vLLM-Ascend 0.23.0 资料用于版本比较，当前运行后端为 Omni-NPU。

## 入口与依赖

部署根 `/data/docker/w00939120/pangu-deploy`；CANN 根 `cann/9.2.0/cann-9.2.0`；Python 环境 `envs/pangu7b-image`。

`with-pangu7b.py` 经 CANN wrapper 显式加载个人 set_env，再进入个人 Python 环境。框架源码、原生依赖、cache、HOME、临时目录、Ascend 日志与 OmniInfer dump 均配置在部署根。

7B 使用 native `PanguEmbeddedForCausalLM`，worker 为 `omni_npu.worker.npu_worker.NPUWorker`。选用 `omni-npu,omni_custom_models,omni_npu_patches`，实际主进程和 worker 的 ACL/HCCL/opapi 加载路径已保存。

## 后续核对

HTTP 阶段讨论监听地址、端口、请求模板与进程管理；双机阶段讨论互通网卡、rank、通信路径与日志。92B MXFP8 阶段独立核对模型分支、量化算子、A5 配套和资源预算；PD 分离阶段再核对连接器与 KV 传输。

现有 92B 权重结构与 HF 对比见 [模型权重对比](./model-artifact-comparison.md)；阶段安排见 [两节点计划](./two-node-7b-plan.md)。

## 来源与历史记录

- [CANN 9.2.0 统一记录](./records/2026-10-08-cann-920-unification.md)
- [Registry 与参考检查](./records/2026-10-08-registry-and-reference-check.md)
- [早期运行时准备](./records/2026-10-08-runtime-preparation.md)
- [vLLM-Ascend 0.23.0 安装资料](https://raw.githubusercontent.com/vllm-project/vllm-ascend/v0.23.0/docs/source/installation.md)
- [OmniInfer](https://github.com/omni-ai-npu/omni-infer)
