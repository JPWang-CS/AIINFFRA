# CANN 9.2.0 统一路线记录

## 当前状态

- 状态：两台 Toolkit/950 OPS 均 exit 0，已完成显式 source 与动态库加载核验；基础 NPU 运算待执行。
- 目标：11 号机与 4 号机统一 CANN 9.2.0；不使用 `/usr/local` CANN。
- 私有目录：`pangu-deploy`、`packages/cann-9.2.0`、`cann/9.2.0`、`installer-home/cann-9.2.0`、`tmp/cann-9.2.0`、`logs/cann-9.2.0`、`scripts`。本轮目录 owner `2000:2000`、mode `0700`，本轮继承的 access/default ACL 已清除；工作根及 `/data/docker/w00939120` ACL 未改。

## 包证据

4 号机原始包已复制入私有 packages，文件名保留 `weekly.20260902.01`，包内实际版本为 9.2.0、inner `V100R001C12B096`：

- Toolkit：1344324118 bytes，SHA256 `f6c02447b4066fb773f8fa3ed588aee14662ac9728e42f1a8f35c06f564184ea`。
- 950 OPS：2944345633 bytes，SHA256 `c2236cd648dd0e36ff80ab6bb86b6c92a729269c82facd64df1c11fea922a46a`。

同一组包已从 4 号机同步到 11 号机，存放于两台 `/data/docker/w00939120/pangu-deploy/packages/cann-9.2.0`。两台完整文件 SHA256 均与上述值一致。

## 实际安装方法

通过自有容器 `w00939120` 的 `docker exec` 入口进入工作目录，使用 `env -i` 与 `bash --noprofile --norc`，校验物理工作根为 `/data/docker/w00939120`。容器内 Python 启动器以 UID/GID `2000:2000` 启动安装监督进程，两个安装器继承该身份。安装器参数均为 `--install --quiet --install-path=/data/docker/w00939120/pangu-deploy/cann/9.2.0`。

安装器 HOME 为个人 `installer-home/cann-9.2.0`，TMPDIR 为个人 `tmp/cann-9.2.0`，PATH 仅含个人 installer venv、个人 `tools/bin` 与系统命令目录。先安装 Toolkit，再显式加载个人版本目录的 `set_env.sh`，随后安装 950 OPS。安装状态、控制台日志与验证结果统一保存在个人 `logs/cann-9.2.0`。本轮非交互 shell 设置 `HISTFILE=/dev/null`，结束前清理当前会话历史。

## 安装复盘与当前进度（2026-10-08）

- 第一次安装失败：安装器依赖 `flock`，原容器中缺少该命令。未安装系统 RPM；从 openEuler 官方 RPM 静态提取 ARM64 `flock` 到两台个人 `tools/bin`。
- `flock` 证据：来源 [openEuler util-linux RPM](https://repo.openeuler.org/openEuler-22.03-LTS-SP4/OS/aarch64/Packages/util-linux-2.37.2-33.oe2203sp4.aarch64.rpm)，RPM 2440197 bytes，SHA256 `080665a6269c3d69274c2514397d76fead75ae015bc90884b37063d345ef121a`；提取 binary 67768 bytes，SHA256 `2a45d809cb7b8fd99b7d0b24c3696015cb0fc77b840865d70bad2ea3f6390d1d`。两台实测 `contended_exit=1`、`unlocked_exit=0`、`fd_lock_exit=0`、`fd_contended_exit=1`。
- 第二次安装在 GE compiler 阶段失败，日志显示 `python3 -m pip` / `pip3` 不存在。两台已在个人目录创建 `envs/cann-installer`，Python 3.11、pip 23.3.1，仅供 CANN 安装器使用；通过离线 `ensurepip` 准备，设置 `PIP_NO_INDEX=1` 和个人 `PIP_CACHE_DIR`，未激活旧 `torch_env`。
- 11 号机第三次执行完成 Toolkit 和 950 OPS 安装。4 号机在补齐同样依赖后完成安装。
- 安装器声明兼容 `[V100R001C15],[V100R001C23],[V100R001C10],[V100R001C12]`，现场 driver 内版本属于 C10；基础 NPU 运算安排在下一阶段。新增工具和安装器环境均位于个人 `pangu-deploy` 前缀。

## 最终安装与动态库核验（2026-10-08）

- 11 号机 Toolkit/950 OPS：exit 0，耗时 95.66 s / 123.74 s；4 号机：exit 0，耗时 118.47 s / 168.78 s。两台 manifest 均为 9.2.0、inner `V100R001C12B096`，时间戳 `20260902_000324253`。
- 安装树统一为 `/data/docker/w00939120/pangu-deploy/cann/9.2.0/cann-9.2.0`；`cann` 与 `ascend-toolkit/latest` 软链均解析到该个人目录。UID/GID 为 `2000:2000`，版本父目录 mode `0700`，安装树 mode `0750`。
- 使用 `/data/docker/w00939120/pangu-deploy/scripts/with-cann-9.2.0.py COMMAND ARG...` 进入清洁环境，显式 source 个人目录 `set_env.sh`，`BASH_ENV` 为空，不自动加载 shell 配置。两台 `ctypes.CDLL(libascendcl.so)` 与 `ctypes.CDLL(libhccl.so)` 均成功，`/proc/self/maps` 指向个人 CANN；`libascend_hal` 仍来自宿主挂载的 driver 目录。
- 关键库 SHA：`libascendcl.so` `3501e1a5bbfc64597b720d8b9cd9b4c264dbaa245ebcfd8fa8a26cab02c7a5a2`；`libhccl.so` `9a6d2c8809c703e35e61d2af8d3d5d212136f3ad3ae4e9f50c798b2cdf164ac1`。HCCL/HCOMM/OPBASE 与 runtime 版本均为 9.2.0；set_env SHA 两台均为 `0145289004d3ce2f362241b3b3bfb446b764c77fd93dd767ae0ccc4caed73f7d`。
- 每台各自安装前后的 6 项公共文件 SHA256 均保持一致：`/etc/Ascend/ascend_cann_install.info`、`/var/log/ascend_seclog/ascend_toolkit_install.log`、`/var/log/ascend_seclog/ascend_ops_950_install.log`、`/home/w00939120/.bashrc`、`/etc/profile`、`/etc/bashrc`。快照位于 `logs/cann-9.2.0/public-files-before.json`。
- 两台加载动态库时均出现下列提示，原因尚未确定，后续结合设备访问和基础 NPU 运算定位：

```text
Failed to obtain the console log level. The possible causes are as follows:
 1. Different containers share the same device;
 2. Failed to open the character device when interacting with the OS kernel;
 3. Failed to obtain the number of devices.
```

- 本轮执行到安装、显式 source 和动态库加载，尚未执行 NPU 运算或模型推理。两台证据均位于个人 `logs/cann-9.2.0/installation-status.json` 与 `logs/cann-9.2.0/verification.json`。

- 收尾核对：两台 `runtime`/`hccl` 的 `version.info` 均为 9.2.0；`set_env.sh` SHA 两台一致。临时传包 HTTP 进程列表为空，`packages/cann-9.2.0` 下无 `.part` 文件，传输已结束。
