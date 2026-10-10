# 盘古部署环境只读快照

## 当前状态（2026-10-08）

两台已完成个人 CANN 9.2.0、Python 环境、基础 NPU 运算、完整 7B 权重校验与单机离线推理。自我介绍取得完整回答；17+25 返回 42；每台两条请求均 stop，退出码 0。完整结果见 [部署记录](./records/2026-10-08-seven-b-execution.md)。

| 项目 | 11 号机 | 4 号机 |
|---|---|---|
| 容器 / 物理工作根 | w00939120 / `/data/docker/w00939120` | 同左 |
| 硬件 / driver | 8×950DT，每卡 98304 MB / 25.1.rc2 | 同左 |
| CANN | 9.2.0 / C12B096，个人 `pangu-deploy/cann/9.2.0/cann-9.2.0` | 同左 |
| Python / torch / torch_npu | 3.11.6 / 2.9.0 / 2.9.0.post7.dev20260817 | 同左 |
| vLLM / Omni-NPU | 0.14.0 / 0.2.0 | 同左 |
| 模型与权重 | openPangu-Embedded-7B-V1.1，14 个文件 SHA256 已核对 | 同左 |
| 单机推理 | BF16 / TP1 / eager / max_model_len 1024 | 同左 |
| 运行身份 / 设备 | 自有容器 root / 物理 1 号卡 | 同左 |
| 退出后复查 | 本轮进程结束，物理 1 号卡无运行进程 | 同左 |

个人 CANN 通过独立 wrapper 显式加载；实际主进程与 worker 的 ACL/HCCL/opapi 路径已核对。后续章节为早期预检快照，保留当时的目录、版本和网络观测。

## 历史快照：范围与时间

- 采样对象：`liteserver-hps-a2c1-11-00001`，仅进入自有容器 `w00939120`；物理工作根守卫为 `/data/docker/w00939120`。
- 本地记录完成时间：2026-10-08 15:22:13 +08:00（远端命令未回显独立时钟）。本轮未访问另外 3 台，不安装、不 source、不 import 框架、不初始化设备、不运行模型或 benchmark。
- 远端 shell 使用 `set +o history`、`HISTFILE=/dev/null`、退出 trap `history -c`；仅清本次容器会话 history，不触及系统审计或其他会话。

## 目录表

| 对象 | 实测结果 |
|---|---|
| `/data/docker/w00939120` | 存在，可作为容器物理工作根；UID/GID `0/0`，aarch64，Huawei Cloud EulerOS 3.0 |
| `/data/docker/w00939120/AscendC/9.2.0.b2/cann-9.2.0` | `exists=false`、`readable=false`、`readlink -e` 无结果 |
| `/data/docker/w00939120/AscendC/9.2.0.b2/cann-9.2.0-beta.2` | `exists=true`、`readable=true`，解析到同一物理路径 |
| `/data/docker/w00939120/torch_env` | 存在，owner `w00939120:w00939120`；含 `bin/lib/include/share/pyvenv.cfg`，无 `conda-meta` |
| `/data/docker/w00939120/torch_wheels` | 存在，owner `w00939120:w00939120` |
| `/data/docker/w00939120/download` | 存在，owner `root:root` |
| `/data/docker/w00939120/pytorch` | 存在；顶层有 `version.txt`、`README.md`；未构建，`git` 不在 PATH，未取得 HEAD |

容器可见 `/data/docker` 为 `2.0T` 总量、约 `1.3T` 已用、约 `608G` 可用、`69%` 使用率。物理 workroot 深度 2 目录实测还包括 `torch_env`、`pytorch`、`AscendC`、`download`、`torch_wheels`。

## CANN 包与组件表

### manifest 原值

`aarch64-linux/ascend_toolkit_install.info`：

- `package_name=Ascend-cann-toolkit`
- `version=9.2.0-beta.2`
- `arch=aarch64`
- `path=/data/w00939120/AscendC/9.2.0.b2/cann-9.2.0-beta.2`

`aarch64-linux/ascend_ops_install.info`：

- `package_name=Ascend-cann-950-ops`
- `version=9.2.0-beta.2`
- `arch=aarch64`
- `path=/data/w00939120/AscendC/9.2.0.b2/cann-9.2.0-beta.2`

### 组件版本原值

- `dvpp`、`hcomm`、`simulator`、`ops_cv`、`pto_isa`、`aoe`、`tbe-tik`、`acl_extend`、`ops_nn`、`pyACL`、`opbase`、`hccl`、`ops_math`：`Version=9.2.0-beta.2`。
- `mstx`、`mindstudio-opprof`、`msboost`、`mindstudio-operator-tools`、`msserviceprofiler`、`ms_fmk_transplt`：`Version=26.2.0`。
- `ascendnpu-ir`：`Version=1.2.0`。
- `version_dir=cann` 出现在相应 CANN 组件记录中。

### 文件与入口

- 物理 `AscendC/9.2.0.b2` 深度 2 未发现 `.run`、`.tar`、`.whl`；因此不能把安装树内部 wheel 称为独立下载包。
- `torch_wheels` 中实际文件包括 `torch-2.10.0+cpu-cp311-cp311-manylinux_2_28_aarch64.whl`（146519440 bytes）和 `torch_npu-2.10.0.post4-cp311-cp311-manylinux_2_28_aarch64.whl`（36262649 bytes），以及依赖 wheel；仅为候选文件，未安装。
- `download` 中实际原始包包括 `Ascend-cann-toolkit_9.2.0-beta.2_linux-aarch64.run`（1378609971 bytes，2026-09-01 10:01:33 +08:00）和 `Ascend-cann-950-ops_9.2.0-beta.2_linux-aarch64.run`（2837526773 bytes，2026-09-01 09:54:41 +08:00）；未执行、未 hash、未打开大包。
- 已存在：`aarch64-linux/lib64/libascendcl.so`、`aarch64-linux/bin/atc`、`opp`。
- 指定位置未找到：`compiler/ccec_compiler/bin/bisheng`、`tools/msprof/bin/msprof`、`opp/.../config/ascend950`。入口文件存在不等于可运行。

## PATH、Python 与现有环境

- 当前 PATH 基础解释器：`/usr/bin/python3`，Python `3.11.6`，`sys.prefix=/usr`、`sys.base_prefix=/usr`；当前 PATH metadata 未发现 `torch`、`torch-npu`、`transformers`、`vllm`。
- `torch_env/bin/python` 和 `python3` 为可执行 symlink，解析到 `/usr/bin/python3.11`；运行 `python -B` 输出 Python `3.11.6`、`sys.prefix=/data/docker/w00939120/torch_env`、`base_prefix=/usr`，metadata 为 `torch=2.13.0+cpu`，`torch-npu/transformers/vllm=absent`。
- 当前 PATH 未发现 `pip3`、`gcc`、`g++`、`cmake`、`git`、`atc`、`bisheng`、`msprof`。`LD_LIBRARY_PATH`、`ASCEND_HOME`、`ASCEND_OPP_PATH`、`ASCEND_TOOLKIT_HOME` 均未设置。
- 这些查询只覆盖当前 PATH 基础解释器和明确的 `torch_env`，不能推断其他 venv、挂载目录或解释器没有对应包。未 import torch/torch_npu。

## 路径阻塞

`set_env.sh` 实际静态字段为：

```text
version_dirpath="/data/w00939120/AscendC/9.2.0.b2/cann-9.2.0-beta.2"
install_dirpath="$(dirname "$version_dirpath")"
```

该引用根与实际存在且可读的物理根 `/data/docker/w00939120/AscendC/9.2.0.b2/cann-9.2.0-beta.2` 不一致；旧 `/data/w00939120/...` 路径不存在。因此当前不能直接 source，CANN 当前生效环境仍未确认。

## 实际有输出的只读命令

```sh
docker exec -w /home/w00939120/work w00939120 bash --noprofile --norc -c '
set +o history
export HISTFILE=/dev/null
trap "history -c" EXIT
p=$(pwd -P)
test "$p" = /data/docker/w00939120 || exit 97
python=/data/docker/w00939120/torch_env/bin/python
"$python" -B - <<'PY'
import sys
import importlib.metadata as metadata
print("executable", sys.executable)
print("version", sys.version.split()[0])
print("prefix", sys.prefix)
print("base_prefix", sys.base_prefix)
names = {d.metadata.get("Name", "").lower(): d.version for d in metadata.distributions()}
print([(name, names.get(name, "absent")) for name in ("torch", "torch-npu", "transformers", "vllm")])
PY
history -c
'
```

该片段实际打印解释器路径、版本、prefix 和 metadata；未 source、未安装、未执行 CANN 工具或模型。

## 下一步与未验证

两台 7B 离线阶段已完成，后续按两节点计划讨论 HTTP 服务与双机通信。

## 模型目录可读性补充

- 当前目标容器可读现有模型目录 `/mnt/sfs_turbo/s3-asset-g-gy-test-a5-exp/pangu_model_zoo/92B/dsa/iter_0011840_mxfp8_npu`；顶层 64 项，50 个 safetensors 分片及 config/index/tokenizer 小文件均可见。该目录是共享挂载来源线索，不等于已确认模型分配或可运行。
- 详细结构、量化和 HF revision 对比见 [model-artifact-comparison.md](./model-artifact-comparison.md)。

## 历史快照：Runtime 适配补充

- 现有容器 driver `version.info`：`Version=25.1.rc2`、`package_version=25.1.rc2`。
- 容器设备节点：`/dev/davinci_manager`、`/dev/hisi_hdc` 可读，`davinci0`–`davinci7` 可见；`/dev/devmm_svm`、`/lib/route.conf`、`/etc/hccl_rootinfo.json`、`/etc/hixlep` 不存在。`/dev/shm` 为 `1.2T`，0 使用。
- 早期匿名 OCI API 请求被 self-signed 证书阻断；随后按用户要求跳过该 registry 的证书验证，已读到两台一致的 manifest/config，并取得代码层。后续结果见 [Registry 检查记录](./records/2026-10-08-registry-and-reference-check.md)。

## 历史评估：CANN 9.1（2026-10-08）

- 仅在 11 号机自有容器内执行官方 OBS 三个 9.1.0 包的 HTTPS `HEAD`；三者均因 DNS `Name or service not known` 失败。
- 因 HEAD 未得到长度，未创建 `/data/docker/w00939120/pangu-deploy` 下的 `.part` 或最终包，未安装、未 source，也未覆盖现有 9.2 beta2 环境。
- 该结果只说明本次容器到 OBS 的解析入口不可用，不能判断官方包是否存在或 CANN 版本兼容性。

## 两节点 7B 预检快照（2026-10-08 16:14）

11 号机与 4 号机的自有容器入口、物理 root、uid/gid、aarch64、Python 3.11.6、driver 25.1.rc2 和 npu-smi 均已核对。两台均为 8×Ascend950DT、每卡 98304 MB HBM、Health OK、采样 util 0 且无 running processes；工作盘可用字节分别为 596644655104 与 633501380608。OBS 与 Hugging Face DNS 各 5 秒超时，HTTP(S)_PROXY 未设置。

11 号机 manifest 为 CANN 9.2.0-beta.2；补充目录清点确认 4 号机 manifest 为 9.2.0，两者均记录旧 `/data/w00939120` 前缀。两台基础 torch_env 分别为 CPU torch 2.13.0 与 2.12.0；torch-npu、transformers、vllm、vllm-ascend 未见。两台 torch_wheels 均有 torch 2.10 CPU 与 torch_npu 2.10.0.post4 文件。共享模型顶层未见 7B；本地 7B 权重尚未传输。

上述预检时，4 号机旧 CANN 根为 `/data/docker/w00939120/AscendC/9.2.0.b2/cann-9.2.0`。本轮两台新安装目录及版本见本文当前状态与统一记录。
