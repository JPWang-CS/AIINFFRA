# GPU 服务器环境记录（2026-10-14）

## 环境与目录

本次安装使用独立工作根，以 `$AIINFFRA_WORK_ROOT` 表示。代码仓库、venv、任务记录、缓存和临时文件都位于该根目录：

```text
$AIINFFRA_WORK_ROOT/
├── AIINFFRA/                         # main 跟踪 origin/main
├── envs/torch-2.7.1-cu128/           # 独立 Python venv
├── runs/<task-id>/                   # 日志、pip 报告、完整 freeze 和退出收据
├── runs/.tmux.sock                   # 私有 tmux socket
├── cache/                            # 安装任务缓存与运行时缓存
└── tmp/                              # 任务和 shell 临时目录
```

先填写已核实的个人允许根和当次 `origin/main` 完整 SHA。工作根必须是允许根的直接子目录：

```bash
export AIINFFRA_ALLOWED_ROOT="/absolute/path/to/verified-personal-root"
export AIINFFRA_WORK_ROOT="$AIINFFRA_ALLOWED_ROOT/aiinffra-learning"
export AIINFFRA_SETUP_SHA="<verified-40-character-origin-main-sha>"
```

进入环境时先设置工作根，再 source 仓库内唯一的启用入口：

```bash
export AIINFFRA_WORK_ROOT=/absolute/path/to/aiinffra-learning
source "$AIINFFRA_WORK_ROOT/AIINFFRA/scripts/activate-gpu-sandbox.sh"
```

helper 检查仓库、venv 与缓存/临时目录边界后启用 venv。PyTorch、Triton、CUDA、pip 和 XDG 缓存写到工作根下的 `cache/runtime/`；`TMPDIR` 使用 `tmp/session-$BASHPID/`。初次 dry-run 使用过账号缓存；限定目录后缓存全部写入工作根，已有缓存未清理。

交互任务使用个人 tmux socket。`-f /dev/null` 不加载用户 tmux 配置；会话从工作根内启动一个不读取 profile/rc 的 Bash，并禁用该任务的 history 文件：

```bash
tmux -f /dev/null -S "$AIINFFRA_WORK_ROOT/runs/.tmux.sock" \
  new-session -d -s gpu-work -c "$AIINFFRA_WORK_ROOT/AIINFFRA" \
  "env -u BASH_ENV HISTFILE=/dev/null HISTSIZE=0 HISTFILESIZE=0 bash --noprofile --norc"
tmux -f /dev/null -S "$AIINFFRA_WORK_ROOT/runs/.tmux.sock" attach -t gpu-work
```

setup 脚本由远端 worker 放在 `runs/` 下。运行时必须显式传入已核实的 `--allowed-root`、`--workspace-root` 和 40 位 `--expected-sha`；脚本只接受全新的仓库与 venv 目标，不覆盖现有目录，也不承诺覆盖重装。每次执行前，应把 `AIINFFRA_SETUP_SHA` 设为当次准备同步的 `origin/main` 完整 SHA。安装命令通过独立 tmux socket 以 detached 会话运行；脚本自身在任务目录保存日志与退出收据：

```bash
tmux -f /dev/null -S "$AIINFFRA_WORK_ROOT/runs/.tmux.sock" \
  new-session -d -s gpu-setup -c "$AIINFFRA_WORK_ROOT" \
  "env -u BASH_ENV HISTFILE=/dev/null HISTSIZE=0 HISTFILESIZE=0 bash --noprofile --norc -c 'bash \"$AIINFFRA_WORK_ROOT/runs/setup-gpu-sandbox.sh\" --allowed-root \"$AIINFFRA_ALLOWED_ROOT\" --workspace-root \"$AIINFFRA_WORK_ROOT\" --expected-sha \"$AIINFFRA_SETUP_SHA\"'"
tmux -f /dev/null -S "$AIINFFRA_WORK_ROOT/runs/.tmux.sock" attach -t gpu-setup
```

## 安装收据

权威任务目录为 `$AIINFFRA_WORK_ROOT/runs/20261014T095214Z-251763/`。其中 `exit.txt` 由 setup 脚本的 EXIT trap 写入，记录 `exit_code=0`，完成时间为 `2026-10-14T09:57:09Z`；执行脚本 SHA256 为 `8cd10963c10814f609531b136639dd00b9e80adbba2a7cfdb91e120be8985c90`。远端仓库位于 `AIINFFRA/`，分支 `main` 跟踪 `origin/main`，HEAD 为 `f63a9e379f198b76f96b224f7e1ce1dfb1c30042`，工作区干净。完整 pip freeze、两份 pip 安装报告、Python smoke、工具版本前后输出及 `pip-check.txt` 保存在任务目录。

系统已有工具版本在安装前后逐项比较一致：Python 3.12.3、NVIDIA driver 570.172.08、nvcc 12.8.61、gcc/g++ 13.3.0。服务器已有 `ncu` 2025.1.1；未发现 `nsys`，本次没有安装系统工具。没有改动 driver、CUDA、编译器或系统 Python。

新 venv 安装了 `torch==2.7.1+cu128`、`triton==3.3.1` 和 `numpy==1.26.4`。`pip check` 返回 `No broken requirements found`。Python smoke 显示 CUDA 可用且有 8 个可见设备；PyTorch wheel 的预编译架构列表为 `['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120', 'compute_120']`。列表没有原生 `sm_89`，含同主版本的 `sm_86` 兼容路径；这不代表 Triton JIT 或所有 kernel 均已验证。

环境检查只查询设备可见性、数量和架构元数据，没有创建 GPU tensor，也没有运行 Triton JIT、课程代码或正式课程实验。容器 overlay 是否跨容器保留尚未验证。

最初选址期间曾出现 3 个误建目录；专用 cleanup 任务退出码为 0，随后确认这些目录已不存在。首次 launcher 因权限返回 126，改用 Bash 脚本启动；主 Agent复核 setup 收据并接手最小远端验收。

PyTorch CUDA 12.8 wheel 来源见[官方 Previous PyTorch Versions 安装说明](https://pytorch.org/get-started/previous-versions/)；`torch` 与 PyPI 包分开安装，固定版本见 [`requirements-gpu-cu128.txt`](../../scripts/requirements-gpu-cu128.txt)。
