# 两节点 7B 预检记录

## 采样范围

- 时间：2026-10-08 16:14（上海时间）。
- 目标：`liteserver-hps-a2c1-11-00001`、`liteserver-hps-a2c1-4-00001`；两台均为自有 `w00939120` 容器。
- 容器入口：`/home/w00939120/work`；物理 cwd 守卫：`/data/docker/w00939120`。
- 会话：uid/gid `0/0`，架构 `aarch64`，Python `3.11.6`；shell 使用 `HISTFILE=/dev/null`、关闭 history、退出清理本次 history。SSH 会话已关闭。

## 实测快照

| 项目 | 11 号机 | 4 号机 |
|---|---:|---:|
| driver | 25.1.rc2 | 25.1.rc2 |
| npu-smi | exit 0 | exit 0 |
| 设备 | 8×Ascend950DT，Health OK | 8×Ascend950DT，Health OK |
| HBM 容量 | 每卡 98304 MB | 每卡 98304 MB |
| 采样 util / running processes | 0 / 无 | 0 / 无 |
| HBM 使用量 | 4742–4767 MB/卡 | 4573–4765 MB/卡 |
| 工作盘可用字节 | 596644655104 | 633501380608 |
| HTTP(S)_PROXY | 未设置 | 未设置 |
| OBS/HF DNS | 各 5 秒超时 | 各 5 秒超时 |
| PATH 工具 | gcc/g++/cmake/git/pip3 不在 PATH | gcc/g++/cmake/git/pip3 不在 PATH |
| ensurepip | 可用 | 可用 |

## CANN 与 Python

11 号机的 Toolkit/950 OPS manifest 为 `9.2.0-beta.2`，候选根 `/data/docker/w00939120/AscendC/9.2.0.b2/cann-9.2.0-beta.2`；manifest 仍记录旧 `/data/w00939120/...` 前缀。基础 `torch_env` metadata 为 `torch 2.13.0+cpu`，未见 `torch-npu`、`transformers`、`vllm`、`vllm-ascend`。

4 号机实际安装目录为 `/data/docker/w00939120/AscendC/9.2.0.b2/cann-9.2.0`，Toolkit/950 OPS manifest 为 `9.2.0`、内版本 `V100R001C12B096`，manifest 同样记录旧 `/data/w00939120/...` 前缀。`download` 中另有 `9.2.0~weekly.20260902.01` Toolkit（1344324118 bytes）与 950 OPS（2944345633 bytes）。`torch_env` 为 `torch 2.12.0+cpu`，其余四项未见。两台 `torch_wheels` 均可见 `torch2.10.0+cpu` cp311 aarch64（146519440 bytes）和 `torch_npu2.10.0.post4`（36262649 bytes）。

## 7B 文件与本地权重

两台共享 `pangu_model_zoo` 顶层仅见 `92B`、`505B`、`35B`，未见顶层 7B；自有 `pangu-deploy/models` 也未见 7B。本轮未搜索其他共享目录。

本地 `openPangu-Embedded-7B-V1.1` 的 4 分片总大小为 `16061839072` bytes，index 完整；传输状态：尚未传输。

## 过程记录与下一步

早期 helper 在本地认证配方筛选和补丁生成阶段停止；修正后，两台 SSH、容器入口和扩展预检均返回退出码 0。本轮执行只读检查与 HTTP HEAD。下一步确认 driver/ascendhal 与 CANN 9.1 的 A5 配套，再准备离线包并审核安装器参数和写入位置。


## 补充快照

两台 driver `Version/package_version=25.1.rc2`，`ascendhal_version=7.35.23`，`Innerversion=V100R001C10SPC100B203`；兼容字段为 `[V100R001C21],[V100R001C22],[V100R001C23],[V100R001C25],[V100R001C10]` 与 firmware `[9.0.0,9.9.9]`。4 号机已定位 CANN `/data/docker/w00939120/AscendC/9.2.0.b2/cann-9.2.0`，manifest 9.2.0、inner C12B096；11 号机为 beta2、inner C12B086。

Windows Python `ssl.create_default_context()` + `urllib.request` 的 HEAD 均返回 HTTP 200，总 Content-Length `4383208511` bytes。包正文尚未下载。

| 官方包 | Content-Length（bytes） | ETag |
|---|---:|---|
| [Toolkit 9.1.0 aarch64](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-toolkit_9.1.0_linux-aarch64.run) | 1273956884 | 552661d18c6826d5ca74d36c422ce1ff |
| [950 OPS 9.1.0 aarch64](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-950-ops_9.1.0_linux-aarch64.run) | 2535172771 | db6b3dcefddd21990268527af2fd40c8 |
| [NNAL 9.1.0 aarch64](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-nnal_9.1.0_linux-aarch64.run) | 574078856 | d33c70affb829367ce69ef0ff15e4726 |

下载后另行计算 SHA256，并核对发布方提供的校验信息。Windows curl 最初返回 `CRYPT_E_NO_REVOCATION_CHECK`；改用 Python 默认 TLS 后取得以上 HEAD 结果，保留证书验证。

## 可回放的容器检查

以下命令覆盖本轮检查的主要字段。认证信息由既有连接配置提供。

```bash
docker exec -i -w /home/w00939120/work w00939120 bash --noprofile --norc <<'SH'
set +o history
export HISTFILE=/dev/null
trap 'history -c' EXIT
test "$(pwd -P)" = /data/docker/w00939120 || exit 97
LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver npu-smi info
/data/docker/w00939120/torch_env/bin/python -B <<'PY'
import importlib.metadata as metadata
import sys
print(sys.version, sys.prefix)
versions = {d.metadata.get('Name', '').lower(): d.version for d in metadata.distributions()}
print({name: versions.get(name) for name in ('torch', 'torch-npu', 'transformers', 'vllm', 'vllm-ascend')})
PY
SH
```
