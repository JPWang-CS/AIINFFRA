# openPangu Embedded 7B 两节点部署记录

日期：2026-10-08。目标：11 号机与 4 号机各自完成单机离线推理。

## 执行结果

| 阶段 | 11 号机 | 4 号机 | 验收依据 |
|---|---|---|---|
| 1. 软件版本与依赖 | 完成 | 完成 | 实际 import 版本、插件入口、源码 SHA256 |
| 2. 个人环境与 NPU 运算 | 完成 | 完成 | FP32/BF16 的 8×8 add、matmul 回读误差均为 0，exit 0 |
| 3. 7B 权重 | 完成 | 完成 | 14 个文件逐项校验大小与 SHA256，两台清单一致 |
| 4. 单机离线推理 | 完成 | 完成 | 两条请求取得最终答案、stop、exit 0，运行进程退出 |

两台最终输出 token 序列一致。自我介绍回答与问题对应；算术请求“请计算17加25，并只给出结果。”返回 `42`。

| 目标 | 引擎初始化 | 自我介绍输出 token | 算术输出 token | 算术答案 | finish reason | 退出码 |
|---|---:|---:|---:|---|---|---|
| 11 号机 | 89.382 s | 35 | 4 | `42` | `stop / stop` | `0` |
| 4 号机 | 71.680 s | 35 | 4 | `42` | `stop / stop` | `0` |

上述时间为本轮启动观测。正式性能基线在后续固定负载阶段采集。

## 软件版本

| 组件 | 11 号机 | 4 号机 |
|---|---|---|
| Python | `3.11.6` | 同左 |
| CANN Toolkit / 950 OPS | `9.2.0 / V100R001C12B096` | 同左 |
| torch | `2.9.0` | 同左 |
| torch-npu | `2.9.0.post7.dev20260817` | 同左 |
| vllm | `0.14.0+empty` | 同左 |
| omni-npu | `0.2.0` | 同左 |
| omni-models | `0.1.0` | 同左 |
| transformers | `4.57.6` | 同左 |
| tokenizers | `0.22.2` | 同左 |
| xgrammar | `0.2.3` | 同左 |

vLLM 源码版本为 `0.14.0`，包 metadata 为 `0.14.0+empty`。本轮采用该 vLLM 与 Omni-NPU 后端。镜像中的 `PyTorch-2.10.0` 是环境目录标签，实际 torch 版本为 `2.9.0`。

参考镜像：`registry-cbu.huawei.com/omniai_omniinfer_dev/ai-infra-infer-1.0.3-a5-arm-master-202608271159-daily:0.0.1`；manifest digest 为 `sha256:e54c77fa8f2383d5eb006ba101ea8e6e8be7ee047bbe98f0a0847034c5c9fc8f`。选择性提取并校验 layer 0、5、12、14、16、19、21，在现有自有容器中构造个人 Python 环境。

## 容器与目录

- 容器：`w00939120`；容器入口 `/home/w00939120/work`，物理根 `/data/docker/w00939120`。
- 部署根：`/data/docker/w00939120/pangu-deploy`。
- CANN：`cann/9.2.0/cann-9.2.0`；Python 环境：`envs/pangu7b-image`；原生依赖：`envs/pangu7b-native`。
- 源码：`source/image/opt/vllm`、`source/image/workspace/omniinfer`。
- 模型：`models/openPangu-Embedded-7B-V1.1`；脚本：`scripts/`；结果：`logs/pangu7b/`。
- Ascend 日志：`logs/ascend/`；OmniInfer dump：`logs/omni-npu/dump/`；各类 cache、HOME 和临时目录均显式指向部署根。

入口经 `with-pangu7b.py` → `with-cann-9.2.0.py` → `pangu_runtime_exec.py` 启动。CANN wrapper 使用无 profile/rc 的 shell，显式加载个人 CANN。实际推理主进程与 worker 的 `/proc/PID/maps` 已确认 ACL、HCCL、opapi 来自上述 9.2.0 目录。

NPU 运行身份为自有容器 root。UID 2000 的 ACL init 返回 `500000`、设备查询返回 `507899`；同一容器 root 返回 init 0、设备数 8、finalize 0。当前证据将差异定位到运行身份，具体 driver 权限机制留待单独检查。设备节点和宿主权限保持原状。

## 模型与权重

模型为 `openPangu-Embedded-7B-V1.1`，架构 `PanguEmbeddedForCausalLM`：34 层、hidden 4096、FFN 12800、32 个 Q head / 8 个 KV head、head dim 128、BF16。RoPE theta 为 16000000，配置上下文上限为 32768，本轮运行长度为 1024。

4 个 safetensors 分片合计 `16061839072` bytes；index payload 为 `16061784576` bytes，差值 `54496` bytes 为分片文件头。Windows 本地权重先传至 11 号机，再由 11 号机直传 4 号机；两台最终均独立校验 14 个文件。环境 bundle 采用相同的节点间直传方式，SHA256 为 `c94a1ff7696cd8f8e6965009dbda37334d32770fb5b3ed052734f8302ab4f01d`。

## 最终运行配置

| 参数 | 实际值 |
|---|---|
| 设备 | 每台物理 1 号卡，进程内逻辑设备 0 |
| dtype / TP / 执行模式 | BF16 / TP1 / eager |
| max_model_len / max_num_batched_tokens | 1024 / 1024 |
| gpu_memory_utilization / max_num_seqs | 0.35 / 2 |
| chunked prefill / prefix cache / async scheduling | True / False / False |
| 采样 | temperature 0、seed 0、max_tokens 128 |
| 提示输入 | 用户文本追加 `/no_think` → 官方 chat template → tokenizer 添加一个 BOS |
| 单机 Gloo / vLLM 地址 | `GLOO_SOCKET_IFNAME=lo` / `VLLM_HOST_IP=127.0.0.1` |
| plugins | `omni-npu,omni_custom_models,omni_npu_patches` |

模型入口为 `vllm.model_executor.models.openpangu.PanguEmbeddedForCausalLM`，worker 为 `omni_npu.worker.npu_worker.NPUWorker`。配置分支推导依据为远端 `source/image/opt/vllm/vllm/model_executor/models/openpangu.py`：826–834 行的 MLA/sink 条件均不成立，888–917 行选择 Embedded GQA；470–477、522–525 行构造并拆分 QKV，宽度为 4096+1024+1024=6144。该文件 SHA256 为 `ef6c81b327df0839dec92fd4763a385154b6072ee835f93c9dd1b00dcdc956bf`。

日志显示权重占用约 14.99 GiB，按本轮 0.35 配额计算的可用 KV cache 约 16.91 GiB。日志中的 token 容量为配置估算，本轮实际执行两个顺序请求。

## 问题与调试方法

| 现象 | 检查与处理 | 复验结果 |
|---|---|---|
| OCI 环境提取在 opaque whiteout 处停止 | 按镜像层语义先处理当层 whiteout，再提取当层文件 | 环境构造与两台导入完成 |
| 环境 bundle 中 Python symlink 被路径守卫拒绝 | 仅放行 venv 的 python/python3/python3.11 到系统 Python 3.11，其他链接仍检查目标范围 | 4 号机展开完成，版本一致 |
| `omni_pangu_models` 导入缺少 `PanguSinkAttentionBase` | 沿插件注册核对 7B 实现，选择 native Embedded 路由 | config、worker 与实际生成完成 |
| 通信组初始化每轮约 56 秒 | 单进程 Gloo 对照：默认 55.026 s，指定 lo 为 0.001 s；单机脚本固定 lo | 新 run 的通信初始化继续推进，最终请求完成 |
| 首轮回答偏题，算术返回 17 | 比较官方 generate.py 与实际输入 token，发现缺 BOS；改用模板文本再 tokenizer，断言 BOS 仅出现一次 | 两台输出恢复为对应问题的中文内容 |
| 补 BOS 后 32 token 截在思考段 | 按官方说明添加 `/no_think`，上限调为 128，检查最终 content 与 stop | 两台自我介绍完整，算术返回 42 |
| OmniInfer dump 使用默认 `/var/log` | 补 `OMNI_DUMP_DIR`；11 号机首轮两个本进程诊断文件迁入个人目录并记录 SHA256 | 后续日志确认 dump 根位于部署目录 |
| Transformers 独立参考导入 `LossKwargs` 失败 | 保存导入栈与 exit 1，定位到模型代码与 Transformers 接口差异 | 该参考分支停止在导入阶段；当前离线验收走 vLLM/Omni-NPU |

三轮运行分别保留：首轮缺 BOS 的输出，第二轮 BOS+32 token 的截断输出，最终 BOS+no_think+128 token 的结果。诊断先核对输入 token、模型分支与实际库路径，再逐项改变配置；每次运行独立保留结果。

退出时日志有 resource_tracker semaphore 提示。退出后已复查本轮主进程、worker 和资源跟踪进程均结束，物理 1 号卡无运行进程。远端 shell 禁用 history，并清除本次容器会话的 history。

## 复现与后续

在指定容器内，从部署目录执行：

```sh
set +o history
export HISTFILE=/dev/null
trap 'history -c' EXIT
cd /data/docker/w00939120/pangu-deploy
./scripts/with-pangu7b.py -u ./scripts/inference_7b.py
```

脚本启动前重新校验权重，并检查物理 1 号卡占用。复现脚本、环境入口和 SHA256 在项目 `runtime/`；本轮结果与脱敏控制台日志在项目 `evidence/2026-10-08-seven-b/`。

后续先讨论 HTTP 服务，再确认双机通信网卡与 rank 配置；跨机阶段另设可互通地址。92B 阶段先做 PD 混部，再做 PD 分离，两阶段分别采集 Prefill、Decode、KV、通信和完整请求 trace。

项目证据：[结果索引](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/evidence/2026-10-08-seven-b/README.md)；[复现入口](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/runtime/README.md)。
