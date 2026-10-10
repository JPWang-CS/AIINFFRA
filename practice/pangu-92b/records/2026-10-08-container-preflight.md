# 最新状态摘要（2026-10-08）

- 当前 4 台用户指定候选均已 SSH 认证成功，并在同名自有容器内通过 `pwd -P=/data/docker/w00939120` 守卫；其余设备未探测。
- 本轮最新 NPU 快照的本地命令启动时间为上海时间约 `2026-10-08 15:12:37 +08:00`；远端命令未回显独立时钟，因此不伪造远端精确时间。四台均为 8×`Ascend950DT`，每卡 HBM 原始容量 `98304 MB`；即时 Util/进程差异见后文具名表。
- 初始 `npu-smi` 缺库失败保留为历史调试证据；仅对 `npu-smi` 子进程临时设置既有 driver 目录后成功。未 source CANN、安装、写远端、运行模型或 benchmark。
- 本轮命令实际包含 `set +o history; export HISTFILE=/dev/null`，退出前执行 `history -c`；只影响本次容器 shell，会话未写共享 history。未删除系统审计、其他用户或其他对话历史。

# 2026-10-08 容器盘点前置检查（历史阶段，已被后续当前 4 台验证替代）

- 交接范围：仅 4 台用户指定候选设备；其余设备未探测、未登录。
- 交接信息：已创建自有容器 w00939120，计划入口 /home/w00939120/work；容器内首个停止条件为 pwd -P 必须等于 /data/docker/w00939120。这些是交接线索，尚未在目标机复核。
- 本机认证核对：4 台新表地址均无现有 SSH config hostname 匹配；系统 known_hosts 可读但没有可确认的对应入口；ssh-agent 未提供可用 key。未读取凭据，未运行旧维护脚本。
- 结果：尚未连接任何目标机，未执行 docker exec，无容器身份/架构/OS、npu-smi、型号/卡数/HBM/占用、空间或工具版本证据。
- 停止条件：获得可靠 SSH 认证入口后，才按容器边界逐台选择；登录后先做容器入口与 pwd -P 断言，失败即停止。只读盘点不 source、不安装、不写远端、不运行模型或 benchmark。

## 认证源关联复核（本地，历史阶段，已被后续当前 4 台验证替代）

- 按交接指定 threadId 的唯一 session JSONL 和指定 `preflight_new_a5_identity.py` 做定点 AST/文本解析；未执行旧脚本，未输出认证值，也未恢复到进程环境变量。
- session 中确有静态密码赋值；源码含 4 个 literal 主机、用户名和端口字段。但该 4 个主机与当前新表允许的 4 台设备地址精确交集为 0。
- 为避免把旧入口认证材料用于不同设备，本轮未发起 Paramiko 连接、未进行 TOFU、未执行 `docker exec`；不能把交接源的认证成功推导为当前 4 台成功。
- 当前停止层：新表允许设备与交接认证源没有可证明的同机映射。需获得这 4 台对应入口/映射后，才能按容器 `pwd -P` 守卫继续只读盘点。

## 2026-10-08 容器内首轮只读盘点（当前验证已替代早期“尚未连接”描述）

- 范围严格限定为用户指定 4 台：`liteserver-hps-a2c1-13-00001`、`liteserver-hps-a2c1-5-00001`、`liteserver-hps-a2c1-4-00001`、`liteserver-hps-a2c1-11-00001`；每台各一次 SSH 认证，上海时间约 15:07:04–15:07:17。未探测其余设备。
- 认证材料来自最新交接 turn 的当前 4 台 setup，仅在本地进程内解析和传递；未执行源命令中的宿主脚本，未输出或落盘凭据。首次 host key 仅保存到任务 scratch；未修改全局 known_hosts/config。
- 每台远端唯一入口均为指定容器 `w00939120` 的 `docker exec -w /home/w00939120/work`；容器内 `pwd -P` 均为 `/data/docker/w00939120`，守卫通过。容器内身份 UID/GID 均为 0，架构 `aarch64`，系统为 Huawei Cloud EulerOS 3.0。
- 初始未设置临时 driver 库路径时，容器内 `/usr/local/sbin/npu-smi` 运行失败：缺少 `libcmscbb.so`；该失败已由后续临时子进程库路径复核替代，保留作为调试证据。未 source CANN，未执行模型/benchmark/测试，未安装或写远端文件。
- 初始检查中 `python` 不在容器 PATH；后续确认基础 `python3` 存在，未进行框架 import 或设备初始化。只读查找发现 CANN 候选脚本：第 3 台为 `/home/w00939120/work/AscendC/9.2.0.b2/cann-9.2.0/set_env.sh`，第 4 台为 `/home/w00939120/work/AscendC/9.2.0.b2/cann-9.2.0-beta.2/set_env.sh`；仅确认文件存在，未 source，其他两台未发现输出。
- `/data/docker` 摘要：第 1 台 `2.0T/1.9T/14G/100%`，第 2 台 `2.0T/533G/1.4T/28%`，第 3 台 `2.0T/1.4T/544G/72%`，第 4 台 `2.0T/1.3T/609G/69%`（Size/Used/Avail/Use）。这是容器可见挂载视图，不代表整机资源或可用 NPU。
- 结论边界：四台 SSH 和容器入口已实测成功；NPU 资源仍未知，不能按卡数或空闲量排序。下一步需先讨论如何在容器内补齐运行时库/工具可见性检查，仍不直接修改宿主或安装依赖。


### 实际临时子进程命令（脱敏）

```sh
docker exec -w /home/w00939120/work w00939120 bash --noprofile --norc -c 'set +o history; export HISTFILE=/dev/null; p=$(pwd -P); test "$p" = /data/docker/w00939120 || exit 97; LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver /usr/local/sbin/npu-smi info; history -c'
```

该命令只为 `npu-smi` 子进程设置临时 `LD_LIBRARY_PATH`，没有 source、写配置或替换库。

### 四台具名 NPU 快照

| 设备 | NPU/型号 | HBM 原始容量 | 即时 Util/进程摘要 |
|---|---|---:|---|
| `liteserver-hps-a2c1-13-00001` | 8 × Ascend950DT | 每卡 98304 MB | Util 均 0；HBM 约 4735–4765 MB；NPU 6 有 Python 进程 |
| `liteserver-hps-a2c1-5-00001` | 8 × Ascend950DT | 每卡 98304 MB | NPU 0–3 Util 31–54%、HBM 约 56860–56927 MB 且有 Python 进程；4–7 无进程 |
| `liteserver-hps-a2c1-4-00001` | 8 × Ascend950DT | 每卡 98304 MB | Util 均 0；HBM 约 4765–4766 MB；无进程 |
| `liteserver-hps-a2c1-11-00001` | 8 × Ascend950DT | 每卡 98304 MB | Util 均 0；HBM 约 4518 MB；无进程 |

进程和 Util 是一次容器可见快照，不能换算活跃用户数、长期空闲或整机独占资源。

## 运行时库临时路径复核与 NPU 快照

- 继续只在 4 台同名容器内执行，未 source CANN、未写远端、未安装库。`npu-smi` 的 `ldd` 均报告 `libcmscbb.so` 及若干 driver/mbed 依赖未解析；在限定的既有目录中确认 `/usr/local/Ascend/driver/lib64/common/libcmscbb.so` 存在。
- 仅为 `npu-smi info` 子进程临时设置既有目录：`/usr/local/Ascend/driver/lib64`、`common`、`driver`；未修改 profile、环境文件或系统配置。四台均成功运行 `npu-smi 25.1.rc2`。
- 四台容器视图均为 8 个 `Ascend950DT`，每卡 HBM 原始容量 `98304 MB`。这是本次即时 NPU/逻辑设备快照，不等同于整机独占资源。
- 第 1 台：8 卡 NPU Util 均为 0；HBM 使用约 `4735–4765 MB`；仅 NPU 6 报告一个 Python 进程。第 2 台：NPU 0–3 Util 约 `31–54%`、HBM 约 `56860–56927 MB`，四卡均有 Python 进程；NPU 4–7 Util 为 0、HBM 约 `4768–4770 MB` 且未报告进程。第 3 台：8 卡 Util 均为 0、HBM 约 `4765–4766 MB`、未报告进程。第 4 台：8 卡 Util 均为 0、HBM 约 `4518 MB`、未报告进程。
- 进程表只证明该时刻 NPU 进程可见性，不能换算活跃用户数；无进程也不能证明长期无人使用。卡数是本容器可见视图，不能直接证明整机卡数或可分配性。
- 四台 `python3` 均为 `3.11.6`；使用 `python3 -B` 和 `importlib.metadata` 查询，`torch`、`torch-npu`、`transformers`、`vllm` 均未发现。仅查询当前 PATH 基础 python3 的 metadata；不能推断其他 venv、挂载目录或解释器没有这些包。未 import 框架或初始化设备。
- 依赖边界：`libcmscbb.so` 已定位，但其他 `ldd` 缺失项未在本轮限定目录中逐一定位；当前只证明临时目录组合足以运行 `npu-smi info`，不证明完整 CANN/驱动环境已配置。



## CANN/基础运行时只读盘点（仅目标第 4 台）

- 目标 `liteserver-hps-a2c1-11-00001` 的容器物理守卫 `/data/docker/w00939120` 通过；本轮未访问另外 3 台。
- 可读 CANN 候选为 `/home/w00939120/work/AscendC/9.2.0.b2/cann-9.2.0-beta.2`，`opp/version.info` 实际为 `Version=9.2.0-beta.2`、`version_dir=cann`；同根存在 `ascend_ops_install.info`、`ascend_toolkit_install.info` 和多个组件 `version.info`/`install.info`。另一 `cann-9.2.0` 路径仅完成 readlink，未确认可读。
- 限定目录内发现多个 wheel 文件（含 cann_ops_nn、msprof、mspti、pypto、hccl 等版本线索），但未执行安装；文件存在不等于当前解释器已安装或完整 CANN 可运行。
- 当前基础 `/usr/bin/python3` 为 3.11.6；metadata 未发现 torch、torch-npu、transformers、vllm。PATH 未发现 pip3、gcc、g++、cmake、git、atc、bisheng、msprof；CANN/LD 变量未设置。未 source、未 import torch、未运行模型或 benchmark。
- 本轮只读命令使用 `set +o history`、`HISTFILE=/dev/null`、退出 trap `history -c`；没有清理其他会话或系统日志。

## CANN 精确补查（仅第 4 台）

- 采样仍在同一容器、同一 `/data/docker/w00939120` 守卫内完成；未 source、未安装、未执行二进制工具。`cann-9.2.0` 的 `-e/-d/-r` 均为 0、`readlink -e` 无结果；`cann-9.2.0-beta.2` 三项均为 1，解析到 `/data/docker/w00939120/AscendC/9.2.0.b2/cann-9.2.0-beta.2`。
- `AscendC/9.2.0.b2` 深度 2 内未找到 `.run`、`.tar` 或 `.whl`；已见 wheel 位于 beta.2 安装树内部，不能称独立下载包。
- `ascend_toolkit_install.info` 原值：`package_name=Ascend-cann-toolkit`、`version=9.2.0-beta.2`、`arch=aarch64`、`path=/data/w00939120/AscendC/9.2.0.b2/cann-9.2.0-beta.2`。`ascend_ops_install.info` 原值：`package_name=Ascend-cann-950-ops`，版本/架构/路径相同。
- 组件版本抽样：`dvpp/hcomm/simulator/ops_cv/pto_isa/aoe/tbe-tik/acl_extend/ops_nn/pyACL/opbase/hccl/ops_math=9.2.0-beta.2`，`mstx/mindstudio-opprof/msboost/mindstudio-operator-tools/msserviceprofiler/ms_fmk_transplt=26.2.0`，`ascendnpu-ir=1.2.0`。
- `set_env.sh` 静态前 120 行将 `version_dirpath` 指向上述 beta.2 根，清理/重建 `PATH`、`LD_LIBRARY_PATH`、`PYTHONPATH`，并检查 driver 与 `/etc/ascend_install.info`；未 source，当前 shell 变量状态不变。
- 关键入口实际存在 `aarch64-linux/lib64/libascendcl.so`、`aarch64-linux/bin/atc`、`opp`；指定 bisheng、msprof 和 ascend950 配置路径未找到。workroot 深度 2 目录及深度 3 `pyvenv.cfg` 查询无输出。

## 物理根与 set_env 路径一致性补查（仅第 4 台）

- 实际物理 beta.2 根 `/data/docker/w00939120/AscendC/9.2.0.b2/cann-9.2.0-beta.2` 为 exists=true/readable=true；旧 `/data/w00939120/...` 前缀及 `cann-9.2.0` 均不存在，`readlink -e` 无结果。
- `set_env.sh` 硬编码 `version_dirpath=/data/w00939120/AscendC/9.2.0.b2/cann-9.2.0-beta.2`，与当前物理根不一致；未 source，记录为环境脚本路径迁移阻塞。
- 从物理 `/data/docker/w00939120` 起点有限列目录实际看到 `torch_env`、`pytorch`、`AscendC`、`download`、`torch_wheels`；此前从 `/home/.../work` 软链接起点的空结果不再解释为目录不存在。物理 AscendC/9.2.0.b2 深度 2 未发现 `.run/.tar/.whl`，最多 3 层未发现 `pyvenv.cfg`。

## torch_env、wheel 与原始安装包补查（仅第 4 台）

- 物理目录 `/data/docker/w00939120/torch_env`、`torch_wheels`、`download` 均存在；`torch_env`/`torch_wheels` owner 为 `w00939120:w00939120`，`download` 为 `root:root`。`torch_env/bin/python`、`python3` 解析到 `/usr/bin/python3.11`，`pyvenv.cfg` 存在，`conda-meta` 不存在。
- `torch_env/bin/python -B` 输出 Python 3.11.6，prefix 为 `/data/docker/w00939120/torch_env`、base 为 `/usr`；metadata 为 `torch=2.13.0+cpu`，`torch-npu/transformers/vllm=absent`。未 import 框架或初始化设备。
- `torch_wheels` 实际候选：`torch-2.10.0+cpu-cp311-cp311-manylinux_2_28_aarch64.whl`（146519440 bytes）、`torch_npu-2.10.0.post4-cp311-cp311-manylinux_2_28_aarch64.whl`（36262649 bytes），以及依赖 wheel；均未安装。
- `download` 实际原始安装包：`Ascend-cann-toolkit_9.2.0-beta.2_linux-aarch64.run`（1378609971 bytes，2026-09-01 10:01:33 +08:00）、`Ascend-cann-950-ops_9.2.0-beta.2_linux-aarch64.run`（2837526773 bytes，2026-09-01 09:54:41 +08:00）；未执行或 hash。
- `pytorch` 有 `version.txt`/`README.md`，但 `git` 不在 PATH，未取得 HEAD；不构建、不触网。
