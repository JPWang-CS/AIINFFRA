# 盘古模型部署与整网分析实操教程

本教程从 openPangu-Embedded-7B-V1.1 开始，依次验证个人运行环境、单机推理、HTTP 服务、双机通信和双机 TP=2 推理，再进入 92B 的 PD 混部与分离分析。每一步先检查运行条件，再执行固定输入，最后检查输出和资源释放。

下面的 7B 操作适用于 11 号机和 4 号机的 `w00939120` 容器。CANN、Python、源码和模型均已准备在 `/data/docker/w00939120/pangu-deploy`。命令按这套环境编写，软件版本、模型或设备变化时应重新核对条件。

## 1. 连接服务器，确认容器和目录

先在 Codex 的目标连接中打开对应服务器的终端。在目标主机的连接内进入已经创建的个人容器：

~~~sh
docker exec -it -u 0 \
  -w /data/docker/w00939120/pangu-deploy \
  w00939120 \
  env -i PATH=/usr/bin:/bin LANG=C.UTF-8 HISTFILE=/dev/null \
  /bin/bash --noprofile --norc
~~~

`docker exec` 只用于进入现有容器。后面的安装、复制、运行和日志操作均在容器内进行。

进入容器后先执行：

~~~sh
set +o history
history -c
export HISTFILE=/dev/null
trap 'history -c' EXIT
umask 077

PANGU_ROOT=/data/docker/w00939120/pangu-deploy
cd "$PANGU_ROOT"
test "$(pwd -P)" = "$PANGU_ROOT"
test "$(id -u)" = 0
id
pwd -P
uname -m
~~~

预期身份为容器内 root，物理目录为个人部署根，架构为 `aarch64`。使用 `pwd -P` 是为了检查符号链接解析后的实际目录。身份或目录不符时先修正入口。

`HISTFILE=/dev/null` 配合关闭 history，防止当前会话写入命令历史。退出时只清当前 shell 的历史，不删除其他会话的历史文件。每个新终端会话都执行这组设置。

## 2. 理解目录与软件组合

| 对象 | 个人部署根下的路径 |
|---|---|
| CANN Toolkit / 950 OPS | `cann/9.2.0/cann-9.2.0` |
| CANN 原始包 | `packages/cann-9.2.0` |
| Python 运行环境 | `envs/pangu7b-image` |
| 补充原生库 | `envs/pangu7b-native` |
| vLLM 源码 | `source/image/opt/vllm` |
| OmniInfer 源码 | `source/image/workspace/omniinfer` |
| 7B 模型 | `models/openPangu-Embedded-7B-V1.1` |
| 环境入口与单机脚本 | `scripts` |
| 上传控制器的暂存目录 | `staging/<run_id>` |
| HCCL 结果 | `communications/hccl/<run_id>` |
| TP=2 结果 | `runs/tp2-7b/<run_id>` |

已运行的软件组合为 Python 3.11.6、CANN Toolkit/950 OPS 9.2.0（C12B096）、Torch 2.9.0、Torch-NPU 2.9.0.post7.dev20260817、vLLM 0.14.0、Omni-NPU 0.2.0、Transformers 4.57.6。

vLLM 包 metadata 的版本是 `0.14.0+empty`；框架源码记录为 0.14.0。Torch wheel 名称中的 `+cpu` 不表示本教程使用 CPU 推理，实际设备执行由 Torch-NPU 接入。镜像里的环境目录名也不能代替实际软件版本。

容器内检查目录和包 metadata：

~~~sh
test -f "$PANGU_ROOT/cann/9.2.0/cann-9.2.0/set_env.sh"
test -x "$PANGU_ROOT/envs/pangu7b-image/bin/python"
test -d "$PANGU_ROOT/source/image/opt/vllm"
test -d "$PANGU_ROOT/models/openPangu-Embedded-7B-V1.1"

"$PANGU_ROOT/envs/pangu7b-image/bin/python" -I -B - <<'PY'
import importlib.metadata as metadata
import sys
print("python:", sys.version)
print("executable:", sys.executable)
for name in ("torch", "torch-npu", "vllm", "omni-npu",
             "omni-models", "transformers", "tokenizers"):
    print(name, metadata.version(name))
PY
~~~

这一步读取包信息，不初始化 NPU。包缺失或版本不同，先核对解释器和环境来源。

环境构造时，CANN 使用相同的 Toolkit 与 950 OPS 包，先安装 Toolkit，再加载个人 `set_env.sh`，随后安装 OPS。安装器使用 UID/GID 2000:2000 和个人 HOME、TMPDIR、pip 环境，参数为：

~~~text
--install --quiet --install-path=/data/docker/w00939120/pangu-deploy/cann/9.2.0
~~~

个人安装环境补齐了 `flock` 和离线 pip。模型运行使用容器 root；安装身份和运行身份分别检查。Python 与框架来自参考镜像中校验过的代码和依赖层，安装在个人前缀中。

检查个人包是否与这套环境的来源一致：

~~~sh
/usr/bin/python3 -I -B - <<'PY'
import hashlib
from pathlib import Path
folder = Path("/data/docker/w00939120/pangu-deploy/packages/cann-9.2.0")
expected = {
    "f6c02447b4066fb773f8fa3ed588aee14662ac9728e42f1a8f35c06f564184ea":
        ("Toolkit", 1344324118),
    "c2236cd648dd0e36ff80ab6bb86b6c92a729269c82facd64df1c11fea922a46a":
        ("950 OPS", 2944345633),
}
seen = set()
for path in folder.glob("*.run"):
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if sha in expected:
        label, size = expected[sha]
        assert path.stat().st_size == size
        seen.add(sha)
        print(label, path.name, size, sha)
assert seen == set(expected), "个人目录中的 CANN 包与环境来源不一致"
PY
~~~

这组命令只校验已有包。已有环境复现从下一节开始，无需重复安装。

## 3. 显式加载个人 CANN，检查实际库路径

使用绝对路径调用个人入口：

~~~sh
cd "$PANGU_ROOT"
./scripts/with-cann-9.2.0.py /usr/bin/python3 -I -B - <<'PY'
import ctypes
import os
from pathlib import Path

expected = "/data/docker/w00939120/pangu-deploy/cann/9.2.0/cann-9.2.0"
assert os.environ["ASCEND_HOME_PATH"] == expected
assert not os.environ.get("BASH_ENV")
for name in ("libascendcl.so", "libhccl.so"):
    ctypes.CDLL(name)
maps = Path("/proc/self/maps").read_text().splitlines()
paths = sorted({row.split()[-1] for row in maps
                if "libascendcl.so" in row or "libhccl.so" in row})
assert paths and all(path.startswith(expected + "/") for path in paths)
print("ASCEND_HOME_PATH:", os.environ["ASCEND_HOME_PATH"])
print("loaded libraries:", *paths, sep="\n")
PY
~~~

`with-cann-9.2.0.py` 清空继承环境，使用 `bash --noprofile --norc`，只 source 个人版本目录里的 `set_env.sh`。不会自动加载登录 shell 配置。

CANN 动态库应来自个人 9.2.0 安装树。`libascend_hal` 等驱动库使用已挂载的 driver 路径；`npu-smi` 和 `hccn_tool` 是只读查询入口。它们位于 `/usr/local`，与使用公共 CANN Toolkit 是两件事。

库加载异常时依次检查 `set_env.sh` 的真实路径、`ASCEND_HOME_PATH`、`LD_LIBRARY_PATH` 和 `/proc/PID/maps`。文件名存在和环境变量已设置，都不能替代实际库映射检查。

## 4. 校验完整模型文件

模型目录中的 `deployment-manifest.json` 保存 14 个文件的大小和 SHA256。校验时既检查清单，也检查 safetensors index 的分片引用：

~~~sh
cd "$PANGU_ROOT"
/usr/bin/python3 -I -B - <<'PY'
import hashlib
import json
from pathlib import Path

model = Path("/data/docker/w00939120/pangu-deploy/models/openPangu-Embedded-7B-V1.1")
manifest = json.loads((model / "deployment-manifest.json").read_text())
assert len(manifest["files"]) == 14
names = set()
for item in manifest["files"]:
    path = model / item["name"]
    assert path.resolve().is_relative_to(model)
    assert path.stat().st_size == item["size"], item["name"]
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    assert sha == item["sha256"], item["name"]
    names.add(item["name"])
    print("OK", item["name"], item["size"], sha)
index = json.loads((model / "model.safetensors.index.json").read_text())
shards = set(index["weight_map"].values())
assert len(shards) == 4 and shards <= names
print("verified:", len(names), "files;", len(shards), "shards")
PY
~~~

预期全部 14 个文件通过，index 引用 4 个分片。分片合计 16,061,839,072 bytes。两台都校验，不能用一侧传输成功推断另一侧文件完整。大小不符先检查传输；大小一致但 SHA256 不符，重新核对文件来源。

## 5. 单机 NPU 运算与离线推理

### 5.1 检查物理卡

单机脚本固定使用物理 1 号卡，映射为进程内逻辑 `npu:0`。运行前检查：

~~~sh
cd "$PANGU_ROOT"
./scripts/with-cann-9.2.0.py /usr/local/bin/npu-smi info
~~~

应看到 `No running processes found in NPU 1`。有人使用这张卡时，先调整设备安排；本教程的固定设备入口不自动抢占或切换卡。

### 5.2 创建独立运行目录

旧单机脚本写固定日志名。复现时在个人目录创建副本，只替换日志路径，保留原脚本：

~~~sh
cd "$PANGU_ROOT"
PANGU_SINGLE=$(/usr/bin/python3 -I -B - <<'PY'
import ast
import datetime
import secrets
from pathlib import Path

root = Path("/data/docker/w00939120/pangu-deploy")
run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(4)
folder = root / "runs" / "single-7b" / run_id
folder.mkdir(parents=True, exist_ok=False, mode=0o700)
changes = {
    "npu_smoke_7b.py": ("log=prefix/'logs/pangu7b/npu-smoke.json'",
                       "log=pathlib.Path(" + repr(str(folder / "npu-smoke.json")) + ")"),
    "inference_7b.py": ("logs=prefix/'logs/pangu7b'",
                       "logs=pathlib.Path(" + repr(str(folder)) + ")"),
}
for name, (old, new) in changes.items():
    source = (root / "scripts" / name).read_text()
    assert source.count(old) == 1, name
    patched = source.replace(old, new, 1)
    ast.parse(patched)
    (folder / name).write_text(patched)
print(folder)
PY
)
printf '%s\n' "$PANGU_SINGLE"
~~~

如果替换点找不到，说明脚本版本变化，应读取当前脚本后重新确认。日志副本放在新目录，原始文件保持不变。

### 5.3 运行并检查结果

~~~sh
cd "$PANGU_ROOT"
./scripts/with-pangu7b.py -u "$PANGU_SINGLE/npu_smoke_7b.py" \
  > "$PANGU_SINGLE/smoke-console.log" 2>&1
PANGU_SMOKE_EXIT=$?
test "$PANGU_SMOKE_EXIT" -eq 0 || {
  tail -n 60 "$PANGU_SINGLE/smoke-console.log"
  exit "$PANGU_SMOKE_EXIT"
}

./scripts/with-pangu7b.py -u "$PANGU_SINGLE/inference_7b.py" \
  > "$PANGU_SINGLE/inference-console.log" 2>&1
PANGU_INFER_EXIT=$?

/usr/bin/python3 -I -B - "$PANGU_SINGLE" "$PANGU_SMOKE_EXIT" "$PANGU_INFER_EXIT" <<'PY'
import json
import sys
from pathlib import Path
folder = Path(sys.argv[1])
assert sys.argv[2:] == ["0", "0"], sys.argv[2:]
smoke = json.loads((folder / "npu-smoke.json").read_text())
infer = json.loads((folder / "inference.json").read_text())
assert smoke["success"] and infer["success"]
assert smoke["device_count"] == 1
assert all(case["add_max_error"] == case["matmul_max_error"] == 0
           for case in smoke["tests"])
assert len(infer["requests"]) == 2
assert all(case["finish_reason"] == "stop" for case in infer["requests"])
assert infer["requests"][1]["answer_correct"]
print("smoke:", smoke["tests"])
print("answers:", [case["content"] for case in infer["requests"]])
PY
./scripts/with-cann-9.2.0.py /usr/local/bin/npu-smi info
~~~

先看 smoke 的 FP32/BF16 加法和矩阵乘法是否回读正确，再检查模型回答。自我介绍应与盘古身份对应，`17+25` 的最终答案应为 `42`。两个程序退出后，物理 1 号卡应恢复空闲。

单机推理参数为 BF16、TP=1、eager、长度 1024、`max_num_seqs=2`、显存利用率 0.35。单机 Gloo 使用 `lo`，双机使用跨节点网卡。

输入构造的关键代码为：

~~~python
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": question + " /no_think"}],
    tokenize=False,
    add_generation_prompt=True,
)
ids = tokenizer(text, add_special_tokens=True)["input_ids"]
assert ids[0] == tokenizer.bos_token_id
assert ids.count(tokenizer.bos_token_id) == 1
outputs = llm.generate(
    [{"prompt_token_ids": ids}],
    SamplingParams(temperature=0, max_tokens=128, seed=0),
    use_tqdm=False,
)
~~~

模板负责对话格式，tokenizer 添加一个 BOS。`/no_think` 与 128 token 上限用于取得最小请求的完整最终回答。检查最终内容和 `finish_reason`，避免把思考段或长度截断当成请求完成。

## 6. 单机 HTTP 服务

### 6.1 创建服务、监视器和客户端副本

每台先分别运行单机 HTTP，再组合成双机引擎。下面的 `PANGU_NODE` 在 11 号机填 `11`，4 号机填 `4`。

~~~sh
PANGU_NODE=11
cd "$PANGU_ROOT"
PANGU_HTTP=$(/usr/bin/python3 -I -B - <<'PY'
import ast
import datetime
import secrets
from pathlib import Path

root = Path("/data/docker/w00939120/pangu-deploy")
run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(4)
folder = root / "runs" / "http-7b" / run_id
folder.mkdir(parents=True, exist_ok=False, mode=0o700)
old_log = '"logs/http-7b/2026-10-09/attempt-03"'
new_log = repr(str(folder.relative_to(root)))
for name in ("serve_http_7b.py", "monitor_http_7b.py", "verify_http_7b.py"):
    source = (root / "scripts" / "http-7b" / name).read_text()
    assert source.count(old_log) == 1, name
    source = source.replace(old_log, new_log, 1)
    if name == "monitor_http_7b.py":
        old = 'root/"scripts/http-7b/serve_http_7b.py"'
        assert source.count(old) == 1
        source = source.replace(old, "pathlib.Path(" + repr(str(folder / "serve_http_7b.py")) + ")", 1)
    ast.parse(source)
    (folder / name).write_text(source)
print(folder)
PY
)
printf '%s\n' "$PANGU_HTTP"
~~~

三份脚本使用同一个新日志目录，监视器启动新目录中的服务副本。只改日志目录而不改监视器的服务路径，会出现日志分别写入新旧目录的问题。

在启动前检查物理 1 号卡和 API 端口 18081。端口检查覆盖 IPv4、IPv6 的 TCP 监听：

~~~sh
./scripts/with-cann-9.2.0.py /usr/local/bin/npu-smi info
/usr/bin/python3 -I -B - <<'PY'
from pathlib import Path
busy = []
for protocol in ("tcp", "tcp6"):
    for line in Path("/proc/net", protocol).read_text().splitlines()[1:]:
        fields = line.split()
        if fields[3] == "0A" and int(fields[1].split(":")[1], 16) == 18081:
            busy.append((protocol, fields[1]))
assert not busy, ("18081 已被占用", busy)
print("18081: free")
PY
~~~

### 6.2 启动服务并验证

终端 A 留在部署根，前台运行监视器：

~~~sh
cd "$PANGU_ROOT"
/usr/bin/python3 -B "$PANGU_HTTP/monitor_http_7b.py" "$PANGU_NODE"
~~~

终端 B 同样进入自有容器，执行第 1 节的历史和目录设置。把终端 A 输出的目录赋给 `PANGU_HTTP`，设置同一节点编号：

~~~sh
PANGU_NODE=11
PANGU_HTTP='/data/docker/w00939120/pangu-deploy/runs/http-7b/<新运行编号>'
cd "$PANGU_ROOT"
/usr/bin/python3 -I -B - <<'PY'
import time
import urllib.error
import urllib.request

deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen("http://127.0.0.1:18081/health", timeout=2) as reply:
            if reply.status == 200:
                print("health: 200")
                break
    except (OSError, urllib.error.URLError):
        pass
    time.sleep(2)
else:
    raise TimeoutError("服务未就绪；检查新目录中的 console.log")
PY
/usr/bin/python3 -B "$PANGU_HTTP/verify_http_7b.py" "$PANGU_NODE"
~~~

代码中的 `<新运行编号>` 替换为实际目录名。服务仅监听容器内 `127.0.0.1:18081`，客户端也在同一容器执行。

客户端发送的算术请求采用下列字段：

~~~python
request_body = {
    "model": "openPangu-Embedded-7B-V1.1",
    "messages": [{"role": "user", "content": "请计算17加25，并只给出结果。 /no_think"}],
    "temperature": 0,
    "seed": 0,
    "max_tokens": 128,
    "stream": False,
    "add_special_tokens": True,
    "return_token_ids": True,
    "request_id": "由客户端生成的唯一请求编号",
}
~~~

非流式检查回答、prompt/output token、usage、EOS 和 `finish_reason`。流式把 `stream` 改为 `True`，增加 `stream_options={"include_usage": True}`，保存完整 SSE。最后应出现停止状态、usage 和 `[DONE]`。

`stop=["42"]` 用例检查可见文本不再包含 `42`，并且 `stop_reason` 等于 `"42"`。输入 token 里只有一个 BOS，普通与流式算术的 token 序列应一致。完整四组请求由验证脚本自动发送和比较。

### 6.3 正常关闭服务

客户端验证结束后，按监视器保存的 PID 关闭本次服务。先确认 PID 对应新目录中的服务副本和本次进程组：

~~~sh
/usr/bin/python3 -I -B - "$PANGU_HTTP" <<'PY'
import json
import os
import signal
import sys
from pathlib import Path

folder = Path(sys.argv[1]).resolve()
root = Path("/data/docker/w00939120/pangu-deploy")
assert folder.is_relative_to(root / "runs" / "http-7b")
process = json.loads((folder / "process.json").read_text())
pid = process["server_pid"]
argv = Path("/proc", str(pid), "cmdline").read_bytes().split(b"\0")
assert str(folder / "serve_http_7b.py").encode() in argv
assert os.getpgid(pid) == process["pgid"] == pid
os.kill(pid, signal.SIGINT)
print("SIGINT sent:", pid)
PY
~~~

等待终端 A 返回后，读取 `exit.json`：服务退出码应为 0，`timeout` 不应为 true。再查询卡和端口。监视器返回 0 不能代替服务自身的退出码。

## 7. 双机通信：先检查接口，再运行 HCCL

### 7.1 核对两侧接口、设备和 TLS

| 项目 | 11 号机 | 4 号机 |
|---|---|---|
| 网卡 | enp39s0f3 | enp39s0f3 |
| 地址 | 10.83.185.108 | 10.83.185.133 |
| 全局 rank | 0 | 1 |
| 物理设备 → 逻辑设备 | 5 → npu:0 | 5 → npu:0 |
| HCCL TCP 互认端口 | 29581 | 连接 11 号机 |
| HCCL TCPStore 端口 | 29582 | 连接 11 号机 |

容器内查询当前地址：

~~~sh
/usr/bin/python3 -I -B - <<'PY'
import fcntl
import socket
import struct
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    data = fcntl.ioctl(sock.fileno(), 0x8915, struct.pack("256s", b"enp39s0f3"))
print("enp39s0f3:", socket.inet_ntoa(data[20:24]))
PY
./scripts/with-cann-9.2.0.py /usr/local/bin/npu-smi info
./scripts/with-cann-9.2.0.py \
  /usr/local/Ascend/driver/tools/hccn_tool -g -tls -i 5
~~~

地址应与表中一致，物理 5 号卡应无运行进程，两侧查询均为 `tls switch[1]`。这里使用 `-g` 查询，不设置 TLS。设备或网络条件变化时，重新确认分配和配置。

单机 `GLOO_SOCKET_IFNAME=lo` 不能沿用到双机。双机入口使用：

~~~text
GLOO_SOCKET_IFNAME=enp39s0f3
HCCL_SOCKET_IFNAME='=enp39s0f3'
HCCL_IF_IP=本节点地址
VLLM_HOST_IP=本节点地址
~~~

`HCCL_SOCKET_IFNAME` 的值以等号开头，完整值为 `=enp39s0f3`。在 shell 中写成 `export HCCL_SOCKET_IFNAME='=enp39s0f3'`。

### 7.2 在 Windows 本地生成新运行包

成功运行的 controller 保存了选卡、端口、版本校验、双机互认、超时、结果回读和退出检查。复现使用相同主体，每次生成新的 run_id、nonce 和结果目录。

在 Windows PowerShell 中执行：

~~~powershell
$panguPython = 'C:\Users\w00939120\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$panguProject = 'D:\Desktop\Code\CC4Ascend\projects\pangu-92b整网分析'
$panguReplayDir = Join-Path $env:TEMP ('pangu-hccl-' + [guid]::NewGuid().ToString('N'))
& $panguPython "$panguProject\runtime\prepare_replay.py" --phase hccl --output $panguReplayDir
Get-Content -LiteralPath "$panguReplayDir\manifest.json"
~~~

生成器只在本地解析 `CONFIG` 与 `ASSETS`，不导入或执行 controller。两节点共用 run_id 和 nonce。它保留已成功的通信代码，只替换配置中的 `run_id`、`nonce`、`runroot`。

包内包含两份 controller、两份 launch 脚本、结果检查程序和 manifest。`staging` 是上传目录，`config.runroot` 是结果目录，两者用途不同。

### 7.3 上传到两侧个人暂存目录

从 manifest 读取 `run_id` 和 `staging`。在两台自有容器内分别创建该暂存目录：

~~~sh
PANGU_RUN_ID='<生成器输出的run_id>'
PANGU_STAGE="$PANGU_ROOT/staging/$PANGU_RUN_ID"
/usr/bin/python3 -I -B - "$PANGU_STAGE" <<'PY'
import sys
from pathlib import Path
folder = Path(sys.argv[1])
root = Path("/data/docker/w00939120/pangu-deploy")
assert folder.parent == root / "staging"
folder.mkdir(parents=True, exist_ok=False, mode=0o700)
assert folder.resolve().is_relative_to(root)
print(folder)
PY
~~~

使用已有目标连接的文件上传功能，把本地包内文件上传到两侧相同的个人暂存路径。结果目录由 controller 创建，上传时不提前创建 `communications/hccl/<run_id>` 或 `runs/tp2-7b/<run_id>`。

上传后，在两侧暂存目录校验文件：

~~~sh
/usr/bin/python3 -I -B - "$PANGU_STAGE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
folder = Path(sys.argv[1])
manifest = json.loads((folder / "manifest.json").read_text())
assert str(folder) == manifest["staging"]
for name, item in manifest["files"].items():
    path = folder / name
    assert path.stat().st_size == item["size"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], name
print("uploaded files: verified")
PY
~~~

### 7.4 两侧并发执行

先在 11 号机终端运行：

~~~sh
sh "$PANGU_STAGE/launch-11.sh"
~~~

随后立即在 4 号机终端运行：

~~~sh
sh "$PANGU_STAGE/launch-4.sh"
~~~

两台不能按“第一台完全退出，再运行第二台”的方式串行执行。TCP 互认窗口为 90 秒，HCCL 连接超时为 120 秒；另一侧应及时进入同一运行。

launch 使用空环境和非登录 bash，在 `/data/docker/w00939120` 启动 controller。controller 检查这个物理工作根后进入 `pangu-deploy`。它先核对网卡、空闲卡、TLS、端口和库哈希，再创建结果目录并执行通信。

通信主体的典型检查如下；完整运行由 controller 自动执行：

~~~python
tensor = torch.tensor([1, 2] if rank == 0 else [10, 20],
                      dtype=torch.float32, device="npu:0")
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
torch.npu.synchronize()
actual = tensor.cpu()
torch.testing.assert_close(actual, torch.tensor([11, 22], dtype=torch.float32),
                           rtol=0, atol=0)
dist.barrier()
~~~

两侧各完成 4 组 FP32 SUM、3 组 BF16 SUM、1 组 4096 元素 FP32 SUM 和 2 组 AllGather。每组同步、回读并比较预期值；结束时销毁 process group。

`init_process_group` 返回成功后，还需要首个 collective 成功。HCCL 可能在首个集合通信时才建立实际链路，TLS 不一致就是在这一阶段暴露的。

### 7.5 检查完整结果

launch 最后读取 `exit.json`、`postcheck.json` 和本节点 rank 结果。只有通信通过、子进程退出 0、process group 正常销毁、端口释放、卡空闲及文件校验通过，才输出 `"success": true`。

HCCL 的原 controller 最后打印状态，部分失败仍可能使外层解释器返回 0。因此结果检查程序会在状态失败时返回非零。查看日志时始终保留通信程序的 inner exit、控制器状态和外层 shell 退出码。

## 8. 双机 TP=2 推理

### 8.1 生成独立 TP=2 包

HCCL 完成后，在 Windows 本地生成另一个包：

~~~powershell
$panguReplayDir = Join-Path $env:TEMP ('pangu-tp2-' + [guid]::NewGuid().ToString('N'))
& $panguPython "$panguProject\runtime\prepare_replay.py" --phase tp2 --output $panguReplayDir
Get-Content -LiteralPath "$panguReplayDir\manifest.json"
~~~

按第 7 节创建两侧新的暂存目录、上传并校验。TP=2 使用自己的运行编号，不复用 HCCL 目录。

| 参数 | 值与用途 |
|---|---|
| 模型 / dtype | Embedded 7B V1.1 / BF16 |
| tensor_parallel_size / pipeline_parallel_size | 2 / 1 |
| nnodes / node_rank | 2 / 11 号机 0、4 号机 1 |
| executor | mp |
| master | 10.83.185.108:29592 |
| 控制器互认 | 29591 |
| API | 11 号机容器内 127.0.0.1:18091 |
| max_model_len / max_num_batched_tokens | 1024 / 1024 |
| max_num_seqs / memory utilization | 2 / 0.35 |
| eager / chunked prefill | 开启 / 开启 |
| prefix cache / async scheduling | 关闭 / 关闭 |

11 号机运行 API、EngineCore 和 TP rank 0，4 号机运行 headless worker 和 TP rank 1。客户端只向 11 号机 API 发送请求，两侧 worker 共同执行同一模型。

启动入口在导入插件前设置：

~~~python
sys.argv = [str(run_dir / "serve.py"), *argv]
~~~

Omni-NPU 插件直接读取 `sys.argv` 中的模型参数。只把参数交给 parser，而未更新 `sys.argv`，会在插件初始化时出现 `Model type Not Provided`。

运行时创建短的个人临时目录：

~~~python
ipc = root / "tmp" / "tp2" / config["nonce"][:12]
assert ipc.resolve().is_relative_to(root / "tmp")
assert len(str(ipc)) + 37 <= 107
assert not ipc.exists()
ipc.mkdir(parents=True, mode=0o700)
env["TMPDIR"] = str(ipc)
~~~

vLLM 使用临时目录加 UUID 创建 ZMQ Unix socket。目录过长会在 API/engine 通信初始化阶段失败；短目录也位于个人部署根下。

### 8.2 启动、请求和退出

两侧分别并发运行自己的 launch 脚本，形式与 HCCL 相同。controller 自动等待服务 ready，在 11 号机执行四组 HTTP 验证，然后关闭两侧服务并检查退出。它是有限时长的验证入口，运行结束后不保留常驻服务。

| 请求 | 预期检查 |
|---|---|
| 自我介绍 | 完整、与问题对应的盘古介绍，EOS 停止 |
| 非流式 17+25 | 最终答案 42 |
| 流式 17+25 | 最终答案 42，完整 SSE 与 [DONE] |
| stop=["42"] | 可见文本截去 42，stop_reason="42" |

还要检查一次 BOS、usage 和实际 token 数一致，普通与流式算术的 token 序列相同。两侧日志应出现 world size 2、rank 0/1、HCCL 后端和权重加载。

2026-10-09 的运行中，四组请求均通过；两侧 worker 分别记录约 7.51 GiB 的模型加载内存。这个值是框架报告的模型加载占用，完整 HBM 账本还包括 KV、通信、运行时和临时缓冲。

## 9. 沿请求编号追踪运行过程

先从 `client-results.json` 取得 API 请求 ID，再在 11 号机 `stdout.log` 中查找。日志里的 engine ID 可能在 API ID 后追加随机后缀，必须使用实际记录关联。

在容器内把 `PANGU_RESULT` 设为 manifest 中的 `config.runroot`：

~~~sh
PANGU_RESULT='/data/docker/w00939120/pangu-deploy/runs/tp2-7b/<新运行编号>'
/usr/bin/python3 -I -B - "$PANGU_RESULT" <<'PY'
import json
import sys
from pathlib import Path
folder = Path(sys.argv[1])
client = json.loads((folder / "client-results.json").read_text())
lines = (folder / "stdout.log").read_text(errors="replace").splitlines()
for case in client["requests"]:
    api_id = "chatcmpl-" + case["request_id"]
    print("\nCASE", case["case"], "API_ID", api_id)
    for line in lines:
        if api_id in line:
            print(line)
PY
~~~

这个客户端结果文件位于 11 号机。4 号机没有独立客户端结果，检查其 rank、初始化、权重加载和设备日志。

请求分析按以下顺序组织：

1. API 接收：请求编号、输入文本、参数与接收时间。
2. 输入处理：chat template、BOS、prompt token 数和停止条件。
3. 引擎入队：API ID 与 engine ID 的关联、排队与调度记录。
4. 模型执行：worker/rank、Prefill 和 Decode、KV 读写及 TP 通信。
5. 返回输出：采样、输出 token、SSE、EOS/stop 和 usage。
6. 请求结束：最终答案、停止原因、错误状态和资源释放。

普通日志可以确认请求接收、入队和输出。要分析设备执行耗时、P/D 重叠或慢 rank，应进一步采集对应请求的设备 trace。跨主机时间线先核对时钟；客户端等待时间与设备 kernel 时间分别记录。

## 10. 退出与资源检查

所有阶段都保留原失败日志，修正后使用新运行目录。HCCL/TP=2 controller 保存：

~~~text
config.json       本次配置与 rank
preflight.json    启动前条件
stdout.log        程序输出
stderr.log        错误输出
exit.json         真实子进程退出码与结果
postcheck.json    进程、设备、端口及公共文件复查
~~~

HCCL 还保留 rank0/rank1 的数值比较；TP=2 保留客户端 JSON、SSE、服务参数和进程/库映射快照。归档时带上 controller、ASSETS、配置和 SHA256，才能复现同一版本。

退出检查依次核对本次进程已结束、选定端口没有监听、选卡空闲。使用公共文件前后哈希确认检查范围内的文件未变化。停止服务时从本次 `process.json` 或 `exit.json` 取得 PID，核对命令行和进程组；不要按通用进程名批量终止。

退出容器会话前执行：

~~~sh
history -c
exit
~~~

保留个人运行日志和已归档证据。清除命令历史与删除实验记录分别处理。

## 11. 常见问题与检查方法

| 现象 | 优先检查 | 处理方法 |
|---|---|---|
| CANN 版本不同或加载公共库 | wrapper、set_env.sh、进程 maps | 从空环境显式加载个人 9.2.0 |
| ACL 初始化失败 | 运行身份、容器设备访问、原始错误码 | 核对已验证的容器 root 入口，保留权限差异证据 |
| 缺 flock 或安装器 pip | 安装器 PATH、离线依赖 | 在个人工具/安装环境补齐依赖 |
| 回答偏题、算术错误 | 实际 prompt token 和 BOS | 按官方模板构造输入，只添加一次 BOS |
| 只输出思考段 | /no_think、max_tokens、finish_reason | 固定最小提示和 128 token 上限，检查最终内容 |
| 插件导入找不到 PanguSinkAttentionBase | 模型注册与插件列表 | 当前 Embedded 路线使用 native 模型和已验证的三项插件 |
| Model type Not Provided | 插件导入前的 sys.argv | 先设置完整模型 argv，再导入框架 |
| ZMQ IPC 路径过长 | 临时目录加 UUID 后的长度 | 使用个人短 TMPDIR |
| 双机 TCP 超时 | 另一侧是否启动、地址、端口、阶段日志 | 两侧及时并发，TCP 互认置于耗时的 NPU 初始化之前 |
| HCCL 首个 AllReduce 报 EI0016 | 两侧选卡 TLS 状态 | 只读查询，选择空闲且状态一致的卡 |
| 重跑时报目录或 console.log 已存在 | run_id、输出目录 | 创建新编号，不覆盖已有运行 |
| 外层返回 0，但模型失败 | inner exit、state.success、请求结果 | 用结果检查程序和实际响应判定 |
| stop 用例显示空答案 | 请求中的 stop 与 stop_reason | stop=["42"] 会截去停止文本，结合 token 与停止原因核对 |
| /var/log 出现 dump | OMNI_DUMP_DIR | 启动前设为个人目录，检查实际日志路径 |

定位时先提出一个可以检查的问题，保留原输入和原配置，只修改对应条件。每次记录错误发生在哪一阶段、改变了什么、重新运行后看到了什么。初始化成功、健康检查成功和推理结果正确分别验收。

## 12. 进入 92B：只读预检、容量、混部和分离

本节给出当前 92B 只读核对和后续方案。共享权重目录为：

~~~text
/mnt/sfs_turbo/s3-asset-g-gy-test-a5-exp/pangu_model_zoo/92B/dsa/iter_0011840_mxfp8_npu
~~~

在容器内固定个人目录和身份：

~~~sh
cd /data/docker/w00939120/pangu-deploy
test "$(pwd -P)" = /data/docker/w00939120/pangu-deploy
test "$(id -u)" = 0
scripts/with-cann-9.2.0.py /usr/local/bin/npu-smi info
for i in 0 1 2 3 4 5 6 7; do
  scripts/with-cann-9.2.0.py /usr/local/Ascend/driver/tools/hccn_tool -g -tls -i "$i"
done
~~~

这些命令只记录卡状态和 TLS。wrapper 设置的库路径不能再被后续 env -i 清掉。

92B config 的 BOS 为 148899、EOS 为 148902；7B 为 BOS 1、EOS 45892。按 92B 自己的 tokenizer 构造提示，只添加一次 BOS。

下面的代码只读 config、index 和 safetensors header，不读取 tensor payload：

~~~sh
/usr/bin/python3 -I -B - <<'PY'
import hashlib, json, struct
from collections import Counter, defaultdict
from pathlib import Path

MODEL = Path("/mnt/sfs_turbo/s3-asset-g-gy-test-a5-exp/pangu_model_zoo/92B/dsa/iter_0011840_mxfp8_npu").resolve()
CONFIG = MODEL / "config.json"
INDEX = MODEL / "model.safetensors.index.json"
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()
config = json.loads(CONFIG.read_bytes())
index = json.loads(INDEX.read_bytes())
weight_map = index["weight_map"]
refs = sorted(set(weight_map.values()))
assert refs
assert all((MODEL / ref).resolve().is_relative_to(MODEL) for ref in refs)
assert all((MODEL / ref).is_file() for ref in refs)
assert len(weight_map) == len(set(weight_map))
payload = 0
header_sum = 0
file_sum = 0
dtype_bytes = Counter()
layer_bytes = defaultdict(int)
seen = set()
for ref in refs:
    shard = (MODEL / ref).resolve()
    size = shard.stat().st_size
    file_sum += size
    with shard.open("rb") as f:
        raw = f.read(8)
        assert len(raw) == 8
        header_len = struct.unpack("<Q", raw)[0]
        assert 0 < header_len <= 16 * 1024 * 1024
        assert 8 + header_len <= size
        header_sum += 8 + header_len
        assert header_sum <= 32 * 1024 * 1024
        header = json.loads(f.read(header_len))
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        assert key not in seen
        seen.add(key)
        assert weight_map[key] == ref
        lo, hi = meta["data_offsets"]
        assert 0 <= lo <= hi <= size - 8 - header_len
        nbytes = hi - lo
        payload += nbytes
        dtype_bytes[meta["dtype"]] += nbytes
        parts = key.split(".")
        if len(parts) > 2 and parts[:2] == ["model", "layers"]:
            layer_bytes[parts[2]] += nbytes
assert seen == set(weight_map)
assert payload == index["metadata"]["total_size"]
assert payload == file_sum - header_sum
print(json.dumps({"config_sha256": digest(CONFIG),
                  "num_hidden_layers": config.get("num_hidden_layers"),
                  "num_nextn_predict_layers": config.get("num_nextn_predict_layers"),
                  "quantization_config": config.get("quantization_config"),
                  "bos_token_id": config.get("bos_token_id"),
                  "eos_token_id": config.get("eos_token_id"),
                  "index_sha256": digest(INDEX),
                  "architectures": config.get("architectures"),
                  "model_type": config.get("model_type"),
                  "shards": len(refs), "keys": len(seen),
                  "payload_bytes": payload, "header_bytes": header_sum,
                  "file_bytes": file_sum, "dtype_bytes": dict(dtype_bytes),
                  "layer_bytes": dict(layer_bytes)}, ensure_ascii=False))
PY
~~~

当前只读账本覆盖 50 个分片、107169876850 bytes payload（约 99.8097 GiB）、73977 个 key；主模型层 46，MTP 层 46/47/48 合计 11147529216 bytes（约 10.3819 GiB），主模型约 89.4278 GiB。stage4 是 6 个指定文件中 5 个存在导出、platforms/npu_platform.py 缺失；stage5 是 5 个指定文件中 3 个导出、两个预测文件缺失，补丁目录一级仍有 7 个 Python 文件。stage5 vllm_plugin.py:19 显示真实平台类为 omni_npu.platform.NPUPlatform，不能用缺失的旧预测路径判断插件不可用。

模型注册入口为 v1/models/__init__.py:55–57 的 PanguUltraMoEForCausalLM，对应 omni_npu.v1.models.pangu.pangu_ultra_moe；OpenPanguV2ForCausalLM:86–87 是另一条路线。EP、SharedFusedMoE、DSA、MTP 和 MXFP8 的源码行号与 stage manifest 一并保存。两包 metadata 标记 9.1.0.beta1，实际个人 CANN 为 9.2.0；privateopp/vendors 为空，但已有 3 个 custom-op shared objects。二者兼容性仍需专门检查。

### 12.2 候选配置与验收

| 项目 | 讨论值 |
|---|---|
| 设备 | 4号机物理 0/4/6/7，逻辑 0/1/2/3 |
| 并行 | TP1、DP4、EP4；每 rank 64 experts |
| 激活/权重 | BF16 激活、实际 MXFP8 权重 |
| attention/KV | 参考 kv hif8_ds_mla；MLA/DSA 细节以源码和 config 核对 |
| 执行 | eager，MTP 关闭，先不启 graph |
| 长度/批量 | max_model_len 2048、max_num_batched_tokens 2048、max_num_seqs 4 |
| 显存/块 | gpu_memory_utilization 0.6、block 128 |
| 插件 | VLLM_PLUGINS=omni-npu,omni_custom_models,omni_npu_patches |
| patch 目录 | pangu_v2_benchmark、pangu_v2_base、pangu_sink_swa_mla、pangu_v2_hybrid |

参考脚本中的参数不能直接当作当前启动命令。原参考使用 TP1DP8EP8、8 卡、MTP3、graph、max model 30720、benchmark 10240、memory 0.8；当前讨论值改为 TP1DP4EP4、4 卡、MTP 关闭、eager、2048、memory 0.6，block 128 保持。4 卡静态全 checkpoint 估算为 expert_bytes/4 + nonexpert 约 31.6505 GiB/卡；主模型 89.4278 GiB 是文件账本，未计 workspace、KV、通信和 repack。

执行顺序是：依赖与模型配置导入 → 私有算子最小测试 → 模型加载 → 健康请求 → 普通、流式和 stop 请求 → 退出清理。每个失败保留日志，用本次 PID 清理，仅写个人目录。模型实例运行前先讨论并确认拓扑和实时占用。

PD 混部先建立最小请求，覆盖算术 17+25=42 和中文连续文本，分别用实际 tokenizer 构造两种格式；不把 7B 的 no_think 或 token 配置假定为 92B 通用规则。之后再分 Prefill、Decode、排队和通信。PD 分离增加 KV 产生、传输、接收、消费、释放与等待的 request id/rank/batch/step 证据。公平比较固定权重、请求集合、长度、停止条件、缓存状态、总设备预算和测量端。

## 13. 92B 依赖与 A5 最小算子核对

检查顺序为框架导入、模型配置解析、实际算子注册、单卡计算和进程退出。依赖检查不创建模型实例、不初始化 NPU；单卡计算另用独立控制器执行。所有控制器在个人容器内运行，结果写入 `/data/docker/w00939120/pangu-deploy/runs`。

### 13.1 本地生成新的运行包

在 Windows PowerShell 中执行以下命令。生成器只读取本地运行脚本、解析 Python 语法，生成新 `run_id`、nonce、payload 和 manifest；不会连接服务器或启动模型。

~~~powershell
$project = 'D:\Desktop\Code\CC4Ascend\projects\pangu-92b整网分析'
$python = 'C:\Users\w00939120\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$phase = '92b-smoke'
$generator = Join-Path $project 'runtime\92b-smoke\prepare_smoke.py'
$raw = & $python -I -B $generator
if ($LASTEXITCODE -ne 0) { throw '生成运行包失败' }
$manifest = $raw | ConvertFrom-Json
$actual = (Get-FileHash -LiteralPath $manifest.scratch_payload -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $manifest.payload_sha256) { throw 'payload SHA256 不一致' }
$manifest | Select-Object scratch_payload, payload_sha256, evidence, config
~~~

检查依赖时把生成器换成 `runtime\92b-deps\prepare_dependencies.py`。依赖使用原四卡可见配置解析 ModelConfig，记录占用，不执行卡上计算；smoke 只使用物理卡 4。沿用已授权传输入口，把 `scratch_payload` 指向的实际文件上传到个人目录 `pangu-deploy/incoming`。每次使用新包，不重用已经执行的 run_id。

### 13.2 在容器中核对并执行

控制器要求初始目录为 `/data/docker/w00939120`，随后自行切换到 `pangu-deploy`。下列命令在已进入的个人容器内执行，容器 UID 为 0。把文件名和 SHA 替换为新 manifest 的值。

~~~sh
cd /data/docker/w00939120
test "$(pwd -P)" = /data/docker/w00939120
test "$(id -u)" = 0
set +o history
export HISTFILE=/dev/null
trap 'history -c' EXIT
export PANGU_PAYLOAD_NAME='replace-with-generated-payload-name.py'
export PANGU_PAYLOAD_SHA='replace-with-64-character-sha256'
env -i PATH=/usr/bin:/bin LANG=C.UTF-8 PANGU_NODE=4 \
  HOME=/data/docker/w00939120/pangu-deploy/installer-home \
  PANGU_PAYLOAD_NAME="$PANGU_PAYLOAD_NAME" PANGU_PAYLOAD_SHA="$PANGU_PAYLOAD_SHA" \
  /usr/bin/python3 -I -B - <<'PY'
import hashlib, os, pathlib, runpy
root = pathlib.Path('/data/docker/w00939120/pangu-deploy')
assert pathlib.Path.cwd().resolve() == root.parent and os.geteuid() == 0
name = os.environ['PANGU_PAYLOAD_NAME']
assert pathlib.Path(name).name == name and name.startswith('pangu92b_')
payload = (root / 'incoming' / name).resolve()
assert payload.parent == root / 'incoming'
expected = os.environ['PANGU_PAYLOAD_SHA']
assert len(expected) == 64 and all(c in '0123456789abcdef' for c in expected)
assert hashlib.sha256(payload.read_bytes()).hexdigest() == expected
runpy.run_path(str(payload), init_globals={'TASK_TARGET_NODE': '4'})
PY
status=$?
history -c
test "$status" -eq 0
~~~

控制器内部通过 `scripts/with-cann-9.2.0.py` 显式加载个人 CANN，再由 `exec_dependencies.py` 或 `exec_smoke.py` 设置个人 Python、源码路径、可见卡、cache、home、tmp 和日志目录。外层不要再调用 CANN wrapper；也不要把 `env -i` 放在 CANN 加载之后。控制器使用系统 Python 的 `-I -B`；实际框架子进程需要个人 `PYTHONPATH`，使用个人 Python 的 `-u -B`。

### 13.3 检查实际注册和硬件分支

PanguUltraMoE registry 和 `Mxfp8Config` 已完成真实导入及 ModelConfig 解析。初始 metadata 检查把模块当成函数，造成误报；`omni_custom_ops.npu_fused_infer_attention_sink_metadata` 实际为模块，注册名带下划线。实际检查代码为：

~~~python
schema = torch._C._dispatch_find_schema_or_throw(
    'custom::_npu_fused_infer_attention_sink_metadata', '').schema()
print(schema)
~~~

依赖检查脚本已修正该判据，修正版尚未作为完整依赖阶段独立复跑；最终 smoke 已查到上述实际 schema。不要用猜测的别名或不存在的目录判断算子未注册。

已归档的 `mhc_rl.py` 中，`mhc_sinkhorn` 在非 Ascend950 分支调用 custom Sinkhorn，在 Ascend950 分支调用 `torch_npu.npu_mhc_sinkhorn`。最初 custom 调用报缺少 `aclnnManifoldConstrainedHyperConnectionSinkhornEnhance`，随后按当前硬件对应的原生入口完成验证。该次 smoke 直接调用原生算子，模型自身的分支选择仍要在模型启动后核对。

### 13.4 MXFP8 的输入、布局和比较

完整脚本见 [smoke.py](./runtime/92b-smoke/smoke.py)。三组 `(M,K,N)` 为 `(16,128,32)`、`(17,256,64)`、`(31,64,16)`；输入从 `[-2,-1,-0.5,0.5,1,2]` 确定性生成。量化后先按 E8M0 指数解码 scale，验证可无损恢复输入，再比较 NPU quant matmul 与 CPU FP32 matmul 转 BF16 的输出。

以下代码在 smoke 的个人框架子进程中运行，`a_cpu` 为 `(M,K)`，`w_cpu` 为 `(N,K)`，均为 BF16。scale 的分组长度为 32，传入权重布局为 `(K,N)`。

~~~python
a, a_scale = torch_npu.npu_dynamic_mx_quant(
    a_cpu.npu(), dst_type=torch.float8_e4m3fn, scale_alg=1)
w, w_scale = torch_npu.npu_dynamic_mx_quant(
    w_cpu.npu(), dst_type=torch.float8_e4m3fn, scale_alg=1)
weight = w.transpose(0, 1).contiguous()
scale_view = w_scale.reshape(n, k // 64, 2).transpose(0, 1)
scale = torch.empty(tuple(scale_view.shape), dtype=scale_view.dtype, device=scale_view.device)
scale.copy_(scale_view)
output = torch_npu.npu_quant_matmul(
    a, weight, scale.view(torch.int8), pertoken_scale=a_scale.view(torch.int8),
    pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
    scale_dtype=torch_npu.float8_e8m0fnu, group_sizes=[1, 1, 32],
    output_dtype=torch.bfloat16)
torch.npu.synchronize()
actual = output.cpu().float()
expected = (a_cpu.float() @ w_cpu.float().T).to(torch.bfloat16).float()
assert torch.isfinite(actual).all() and torch.equal(actual, expected)
~~~

K=64 时，转置后的某一维长度为 1，单独调用 `.contiguous()` 可能保留原 stride。新建标准布局的 buffer 后 `copy_`，得到 `scale_stride=[32,2,1]`；其余两组为 `[64,2,1]`、`[128,2,1]`。三组实际最大绝对误差均为 0。该用例覆盖布局和可表示输入的最小兼容性，模型权重、完整 attention、MoE 和一般量化误差还要分别检查。

### 13.5 A5 mHC Sinkhorn 和退出检查

~~~python
matrix = torch.ones((2, 4, 4), dtype=torch.float32, device='npu')
outputs = torch_npu.npu_mhc_sinkhorn(matrix, out_flag=0, eps=1e-6, num_iters=20)
torch.npu.synchronize()
assert outputs[0] is not None
actual = outputs[0].cpu().float()
expected = torch.full((2, 4, 4), 0.25)
assert torch.isfinite(actual).all()
assert torch.allclose(actual, expected, atol=1e-5, rtol=0)
details = [{'absent': True} if item is None else {'shape': list(item.shape)}
           for item in outputs]
~~~

`out_flag=0` 的两个辅助返回值实际为 `None`，记录其缺省状态，不读取 `.shape`。均匀输入主输出的最大绝对误差为 `2.384185791015625e-07`。该结果只覆盖均匀矩阵的最小用例。

最终运行 `20261009T105122Z-3821c1d6` 的外层远端命令和子进程退出码均为 0，`success=true`。实际加载的 ACL、HCCL、HCOMM、opapi 库均来自个人 CANN 9.2.0。退出后本次进程组为空，物理卡 4 空闲，检查范围内五个公共文件哈希一致。结果、stdout/stderr、maps 和前后检查见 [smoke 证据](./evidence/2026-10-09-92b-smoke/README.md)。

## 14. 四卡 HTTP 准备与资源检查

当前已确认的原配置为 4 号机物理卡 `0,4,6,7`，逻辑卡 `0,1,2,3`；TP1、DP4、EP4，BF16 激活和实际 MXFP8 权重。服务采用 eager，关闭 MTP/speculative，长度和 batched tokens 均为 2048，max seqs 为 4，memory utilization 为 0.6，block size 为 128，KV dtype 为 `hif8_ds_mla`。API 和 DP 端口为 18092、18192，单机通信使用回环地址。

`prepare_http.py` 默认生成原配置，接受4个不同的0–7物理卡编号。用户已授权按实时空闲状态选卡；控制器仍逐次核对占用、TLS和端口。`candidate=true` 只表示设备组不同于原0/4/6/7。当前选定2/4/5/7，在本地生成新包：

~~~powershell
$project = 'D:\Desktop\Code\CC4Ascend\projects\pangu-92b整网分析'
$python = 'C:\Users\w00939120\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$devices = '2,4,5,7'
$raw = & $python -I -B (Join-Path $project 'runtime\92b-http\prepare_http.py') --devices $devices
if ($LASTEXITCODE -ne 0) { throw '生成 HTTP 运行包失败' }
$manifest = $raw | ConvertFrom-Json
$actual = (Get-FileHash -LiteralPath $manifest.scratch_payload -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $manifest.payload_sha256) { throw 'payload SHA256 不一致' }
$manifest | Select-Object scratch_payload, payload_sha256, evidence, config
~~~

沿用 13.2 的上传、SHA 核对和容器入口。控制器会在实际启动前重新读取设备进程表、TLS 和端口占用，任一选卡有进程便停止。空闲快照不能当作资源预留，也不终止已有占用进程。`20261009T105907Z-0d96d8ad`、`20261009T110312Z-c32464a1` 两次原配置运行均因物理卡 0 占用而停在 preflight，未启动 server、模型加载、health 或客户端请求。随后实际检查2/4/6/7时发现卡6占用，改用2/4/5/7启动；后续结果见14.1和14.2。

启动后按模型加载、health、模型列表、普通请求、流式请求、stop 请求和退出依次检查。[serve.py](./runtime/92b-http/serve.py) 与 [verify.py](./runtime/92b-http/verify.py) 已完成本地语法和参数核对，实际 HTTP 路径尚未运行。92B 自身的 tokenizer 已包含 BOS 148899，EOS 为 148902；请求固定 `add_special_tokens=false` 和 `chat_template_kwargs={"thinking": false}`，核对 BOS 只出现一次，不复用 7B 的 `/no_think`。

客户端准备了中文模型介绍、`17+25=42` 普通/流式一致性和 `stop=["42"]` 四组请求，保存请求 ID、原 JSON、SSE、token、usage、finish/stop reason 和接收时间。服务结束后仅清理本次进程组，再核对端口、选卡、公共文件哈希和当前会话历史。详细记录见 [92B 依赖与最小算子记录](./execution-record-2026-10-09-92b-deps.md)。

### 14.1 冷加载与等待时限

2/4/6/7 的实际启动检查中，6号卡已有进程，服务未启动。全8卡只读查询后改用2/4/5/7；八卡TLS均开启。选卡完成后仍由启动前检查放行，不改设备TLS，也不终止占用进程。

2/4/5/7 初次运行 `20261009T112127Z-e4728033` 已建立API、DP Coordinator、四个EngineCore与DP/EP worker，开始读取50个共享分片。DP/EP rank 0–3、TP rank 0、HCCL backend和 `Mxfp8Config` 已出现在实际日志中。加载到7/50时达到外层420秒时限，控制器先TERM、再KILL本次进程组。最终server exit=-9，记录forced_cleanup；选卡恢复空闲、端口释放、公共文件哈希一致。该次没有健康请求或推理结果。

源码中的 `VLLM_ENGINE_READY_TIMEOUT_S` 默认600秒，外层原420秒窗口更短。当前运行包显式设置 `startup_timeout_s=2400`，`exec_http.py` 把同一值写入个人子进程的 `VLLM_ENGINE_READY_TIMEOUT_S`。传输入口的等待上限同时覆盖模型加载、245秒客户端阶段和清理；当前3200秒入口副本只修改等待常量，原入口保留，前后SHA见 `transport.json`。这些时间限制只调整本次运行的观察窗口。

新运行 `20261009T112949Z-43fbdca6` 使用相同模型和推理参数，在新目录重新加载。实际服务配置记录 `speculative_config=null`、私有CANN路径和2400秒等待值。本次已完成权重加载，随后在启动profiling的DP同步处失败。最终结果见14.2，原始记录写入 [92B HTTP记录](./execution-record-2026-10-09-92b-http.md)。

启动日志按以下顺序判读：

|阶段|核对内容|
|---|---|
|API和插件|模型argv先于插件导入；实际平台为NPU；使用个人Python和源码|
|模型配置|注册类为PanguUltraMoE；量化mxfp8；MTP关闭；长度与批量正确|
|进程和通信|四个EngineCore/Worker、DP/EP rank、TP rank、backend和实际库映射|
|模型实例和权重|真实模型/quant类、专家分配、分片进度与加载完成记录|
|执行准备|权重后处理、KV分配、设备warmup或profile_run及其错误|
|服务和请求|health、模型列表、完整答案、BOS/token/usage、SSE和停止原因|
|退出|本次进程组、端口、选卡、公共文件和会话历史|

先找最终异常，再回到异常前最后一个完成的阶段。插件覆盖、缺少可选GPU包等提示与致命异常分别记录。权重读取仍有进度时，不能仅用尚未ready判断设备故障；若进度停止，再结合本次worker的CPU/I/O、设备状态和原始错误定位。

### 14.2 权重加载完成后的 DP 同步失败

运行 `20261009T112949Z-43fbdca6` 使用物理2/4/5/7。50个分片读取完成，四个DP/EP rank均在19:55:23报告模型加载完成，每个rank模型占用26.76GiB，加载约1376.9秒。随后在KV可用内存探测所调用的启动profiling中失败，服务没有达到ready，四组HTTP请求均未执行。

实际调用栈为：

~~~text
NPUWorker.determine_available_memory
  -> model_runner.profile_run
  -> NPUModelRunner._dummy_run
  -> _determine_batch_execution_and_padding
  -> coordinate_batch_across_dp
  -> patched _synchronize_dp_ranks
  -> patched _run_ar
  -> event.synchronize
  -> aclrtSynchronizeEvent: 507018 / AICPU exception
~~~

`patch_dp_utils.py:121–137` 创建shape=(5,4)、dtype=int32的CPU pinned张量，按rank填入原始token数、padding后token数、ubatch/padding标志和图模式。专用流依次执行H2D、HCCL AllReduce、D2H和event同步，张量共80字节。该分支使用专门的HCCL group，`hccl_op_expansion_mode=2`、`hccl_buffer_size=20`。当前栈和设备日志证明此次执行到这一同步路径。

rank0在19:55:25.109记录 `userDevId=2` 超出范围 `[0,1)`；随后报 `HcclLaunchAicpuKernel`、`libscatter_aicpu_kernel.so` 和507018。设备4/5/7的日志含同一group的kernel success，设备2日志停在启动后的notify处理。设备2和其余卡的同名函数源码行号也不同。上述现象用于设计下一次检查；设备号映射为何不一致、设备侧库是否存在差异，仍待最小复现确认。

本次外层runner退出0，远端控制器和server退出1，`success=false`，未发生强制清理。本次进程组为空，18092/18192释放，物理2/4/5/7均无运行进程，五个公共文件哈希一致，个人运行脚本未变。物理0的既有进程保留。实际进程映射中的ACL、HCCL、HCOMM和opapi均来自个人CANN9.2.0。

原始stdout/stderr、退出、postcheck、实际库映射和设备日志保存在 [本次证据目录](./evidence/2026-10-09-92b-http/20261009T112949Z-43fbdca6/manifest.json)。只读导出的同步源码和四张卡设备日志均按size/SHA核对。首次批量日志回收的内层JSON不完整，原输出保留；后续分文件回收结果单独保存，见 `hccl-source/collection-checks.json`、`device-log-checks.json` 和 `rank0-run-log-checks.json`。

### 14.3 下一步：四卡 DP 同步最小复现

先按 [DP同步诊断方案](./dp-sync-diagnosis-plan.md) 核对worker可见设备与逻辑编号，再分别验证默认HCCL和AICPU展开模式的80字节AllReduce。输入固定为5×4 int32，每个rank仅填写自己的列，CPU侧直接构造汇总结果，逐元素比较输出。随后复现专用流的H2D、AllReduce、D2H和event顺序。该诊断尚未执行。

每个测试独立建目录并记录卡组、每个rank的实际device/env、group选项、结果、耗时、库映射和退出检查。先取得小张量通信结果，再决定是否调整个人配置或重启92B。四卡加载已经完成，下一次整网启动仍要重新检查资源并使用新run_id。
## 15. 性能测量工具与口径

### 15.1 测量层次与触发方式

正式模型性能采用 ACS 工具报告与昇腾 profiler 采集；自写 `time`、`perf_counter`、`monotonic` 只用于控制器超时、轮询或诊断，不进入性能表。历史 23 分钟是模型加载日志，不是推理性能。

ACS 1.4.1 的 `prof` 报告提供请求层 TTFT、TPOT、E2E、吞吐和失败率。`--trace` 生成 ACS 请求时间线，不能替代设备采集。请求层基线应先关闭设备 profiler，再以少量固定请求采集设备 trace。ACS `--profile` 只是请求服务端 `/start_profile` 和 `/stop_profile`；当前 7B 证据没有证明这条服务路由已打通，92B 尚未 ready。

镜像 worker 的 `npu_worker.py:393-429` 显示 `VLLM_TORCH_PROFILER_DIR` 启用 `torch_npu.profiler.profile`，记录 CPU、NPU、Level1、PipeUtilization，并用 TensorBoard handler 输出。`worker.py:276-285` 负责 start/stop，`330-389` 按请求/token 条件触发。`envs.py:371` 的 `OMNI_PROFILE_TOKEN_THRESHOLD` 默认是 `None`；设为非 None 时，`profile()` 在 `worker.py:279-281` 直接返回，不能与自动 token 触发模式混用。自动停止条件是 `profile_step > stop_step`，不是固定采集五步。`VLLM_SERVICE_PROFILER_DIR` 不是本 worker 读取的启动变量。7B 需要先验证 API→engine→worker 的真实路由，再做设备采集。

### 15.2 msprof、帮助命令与查看工具

已核对的私有工具路径为：

```text
/data/docker/w00939120/pangu-deploy/cann/9.2.0/cann-9.2.0/tools/profiler/bin/msprof
```

本次实际执行的只读帮助命令如下；它只验证工具和参数，不启动采集：

```bash
R=/data/docker/w00939120/pangu-deploy
cd "$R"
scripts/with-cann-9.2.0.py /usr/bin/env HOME="$R/runs/profiler-audit-home" TMPDIR="$R/runs/profiler-audit-tmp" MSPROF_OUTPUT_PATH="$R/runs/profiler-audit-out" "$R/cann/9.2.0/cann-9.2.0/tools/profiler/bin/msprof" --help
```

所有目录都在个人运行根 `R` 下。帮助输出确认了采集、`--parse`、`--query`、`--export`、`--analyze`、`--output`、`--task-time`、`--aicpu` 和 `--runtime-api` 等选项；实际采集命令仍需按当前服务进程和帮助文本逐项确定。

设备采集完成后用 MindStudio Insight 查看时间线，用 `msprof-analyze` 辅助分析。官方 950 CCU 说明对 slow-rank、slow-link 和 communication 分析有适用限制；缺少这些数据只能记为未采集，不能解释为没有通信。`msServiceProfiler` 与 `acs-service-profiler` 是不同工具；当前官方产品表中 950 对 msServiceProfiler 为 No，本次不选该路线，ACS-service-profiler 是否适用于本环境尚未核对。

### 15.3 口径、顺序与证据

TTFT、TPOT、E2E 和吞吐必须记录测量端与统计范围。ACS 的 `AVG_TPOT(s)` 与从第二 token 计算的 `AVG_TPOT_SEC(s)` 分开记录。CPU API、NPU kernel 和通信事件可能重叠，不能把各 kernel duration 相加当请求时延；跨进程或跨主机 trace 保存 request id、rank、batch/step、时钟来源和偏移，时钟未对齐时不直接相减。

执行顺序为：基线（关闭 profiler）→少量固定请求与 `--trace`→torch_npu profiler 或 msprof 采集→Insight/`msprof-analyze` 查看→在相同输入、输出长度、并发、batch、缓存、EOS、卡组和资源条件下比较混部与分离。分离还要记录 KV connector、KV 传输字节、网络事件、P/D 独立 trace 和资源分配。当前证据只覆盖 7B 请求层短输入基线与工具核对，92B 性能和设备 trace 均未完成。

工具与证据：[性能测量口径](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/performance-measurement.md)、[profiler 审计证据](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/evidence/2026-10-09-profiler-tool-audit/README.md)、[CANN msprof 950采集说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta1/devaids/Profiling/atlasprofiling_16_0011.html)、[CANN torch profiler说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta1/devaids/Profiling/atlasprofiling_16_0033.html)、[msprof-analyze Advisor](https://raw.githubusercontent.com/Ascend/msprof-analyze/master/docs/zh/user_guide/advisor_instruct.md)。以上官方资料访问日期为 2026-10-09。

- [ACS 模型服务与 profiler 示例](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_llm_infer_5910029.html)
- [msServiceProfiler 服务调优说明](https://raw.githubusercontent.com/Ascend/msserviceprofiler/master/docs/en/msserviceprofiler_serving_tuning_instruct.md)

## 16. DP 同步诊断与私有配置回退

92B 启动阶段先用 80 字节通信合同缩小问题范围，再回到整网启动。矩阵固定为 `shape=(5,4)`、`dtype=int32`：每个 rank 只填写自己的列，CPU 侧独立构造四列求和结果，逐元素比较设备返回值。最小复现的核心逻辑如下：

```python
import torch
inputs = [[16, 32, 0, 1, 0], [17, 33, 1, 1, 0],
          [18, 34, 0, 1, 0], [19, 35, 1, 0, 0]]
expected = torch.tensor(inputs, dtype=torch.int32).sum(dim=0)
local = torch.zeros((5, 4), dtype=torch.int32)
local[:, rank] = torch.tensor(inputs[rank], dtype=torch.int32)
dist.all_reduce(local, group=group)
assert torch.equal(local.sum(dim=1), expected)
```

已保存的 `cases-2-complete` 覆盖默认 HCCL、显式 AI_CPU 和 AI_CPU 专用流；设备 2/4/5/7 每组 4 rank、每组执行三次，结果均为 exit 0。`actual-1` 是未改源码函数体 AST、合成 DP coordinator 与真实 group 的组合验证；`actual-4567` 覆盖物理 4/5/6/7 的 original 路径。这些结果只证明最小同步合同，不证明完整模型启动或服务请求。

回退配置使用完整原始 JSON，只修改一个字段。原始文件为 `evidence/2026-10-09-dp-sync/config-1/model-extra-original.json`，将 `model_parallel_config.enable_aicpu_dp_sync` 改为 `false`，保存为个人 run 内的 `model-extra.json`，再通过 `OMNI_CUSTOM_MODEL_CONFIG_PATH` 加载。生成包命令：

```powershell
python D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/runtime/92b-http/prepare_http.py --dp-sync original --devices 4,5,6,7
```

启动前重新检查设备占用、TLS、端口和原始配置 SHA；配置回退只作用于个人 run，原始模型配置不覆盖。运行时使用私有 CANN 9.2.0、TP1/DP4/EP4、eager、MXFP8/BF16、MTP 关闭。

整网复现时，先检查每个 rank 的 device/env 映射，再记录模型加载、DP 同步、KV 初始化、health 和请求结果。退出后核对进程、端口、选卡和公共文件哈希。进行中的启动状态留在记录和分析状态，教程只保留可复现的操作和验收边界。

证据见 [DP 同步证据](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/evidence/2026-10-09-dp-sync/README.md) 和 [DP 同步记录](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/execution-record-2026-10-09-dp-sync.md)。

## 17. 自定义算子包与 92B 请求前置检查

92B 首请求缺少 `aclnnAiInfraKvQuantSparseFlashAttentionV2` 及其 `GetWorkspaceSize` 时，先区分 Python wrapper、算子包注册和底层动态库。Python 模块能导入，只说明上层入口存在；它不等于 OPP、custom op 动态库和 aclnn symbol 已加载。固定镜像 layer14 的 `libcust_opapi.so` SHA256 为 `0aca4613eac2f5dfe0b4060cd052bd5a1db45319e2ceac1b926d06faff2ce727`，实际定义了两个目标 symbol。

选择性提取使用已核对的 layer14 产物，不执行镜像脚本，也不覆盖公共目录。压缩 bundle 为 19,400,961 bytes，SHA256 为 `f2cd3a6ce9b9a4e8fb98d4621bcee103c1c793991fb941f70be0405cd4b08155`；展开后的 vendor 目录包含 335 个文件，逐文件大小和 SHA256 由 vendor manifest 校验，manifest SHA256 为 `02e382470feef1e3cd0e2403e059db7161150e51b36a9891414af9de2635e77d`。bundle 与 manifest 的本地来源记录见 transfer manifest；它不是 controller 在远端读取的文件。

node4 的 loader-06 已在 `custom-ops/<id>/overlay/vendor` 完成加载，退出码 0，两个目标 symbol 的 `dlsym` 均非空；CANN 的 `libopapi`/`libnnopbase` 来自私有 CANN 9.2，vendor 来自该 overlay。前期 01–05 的脚本构造或解析失败只记录为工具过程，不能写成包 ABI 失败。loader 通过也不等于模型请求通过。

先在本地生成新包，并从 JSON 输出读取 `scratch_payload` 与 `evidence`，每次使用新 run id：

```powershell
$PY = 'C:/Users/w00939120/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$PREP = 'D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/runtime/92b-http/prepare_http.py'
$manifest = & $PY $PREP --devices 4,5,6,7 --dp-sync original --custom-opp 20261009T-layer14-vendor-01 | ConvertFrom-Json
$PAYLOAD = $manifest.scratch_payload
$CAPTURE = 'D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/evidence/2026-10-09-custom-ops/new-capture/' + $manifest.config.run_id
New-Item -ItemType Directory -Force $CAPTURE | Out-Null
```

随后沿用已授权传输入口运行 `scratch_payload`；具体认证材料由既有入口内部读取，不写入命令行或日志。运行前确认新 capture 目录、payload SHA、设备占用、TLS 和端口；不复用旧 run。controller 的执行顺序是：在 wrapper 的清洁环境完成 CANN 加载，再设置自定义 `ASCEND_CUSTOM_OPP_PATH` 和自有 `LD_LIBRARY_PATH`；bundle 解包到个人 `custom-ops/<id>/overlay`，按 vendor manifest 逐文件校验，最后检查 `/proc/<pid>/maps` 中 `libcust_opapi.so` 来自 overlay、`libopapi`/`libnnopbase` 来自个人 CANN。`exec_http.py` 的实际入口和 controller 资产见 [92B HTTP runtime](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/runtime/92b-http/README.md)。

以上只证明自定义动态库的选择性注入和 symbol 加载。HTTP health、模型加载、首请求、DP 同步和退出清理必须读取新 capture 的实际 JSON 与日志后分别判定。

证据见 [custom-ops 证据](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/evidence/2026-10-09-custom-ops/README.md) 和 [custom-ops 记录](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/execution-record-2026-10-09-custom-ops.md)。

## 18. MLA Metadata 组合 vendor 的准备与验收

当 KvQuant symbol 已补齐而 intro 继续在 MLA 路径缺少 `aclnnAiInfraAttentionPioneerMetadata*` 时，要把 inference 与 training 两套 vendor 一起核对。node11 的真实 ELF 证据表明 Metadata symbol 位于 training 库，库 SHA256 为 `c3c63a91a3cba2ac25e0cf60c5ad31590ee07ba53902a9fdd066003858812480`。包名或 Python 模块名不能代替接口检查。

本次组合包为 `custom-ops/20261009T-layer14-vendor-combo-01`：完整 1101 files、502151939 bytes；bundle 为 186852590 bytes，SHA256 `29629f7cc7ebccdcbad12944d06e9070224d7907a9816ad8ecbcb04d310c03f1`；manifest 311742 bytes，SHA256 `1e83d061f58276bc977dadc78fcecaea831b647fe7d036f6799aa84e6403ea24`。两个 vendor 均来自个人 overlay，所依赖的 `libopapi`/`libnnopbase` 来自私有 CANN 9.2。node4 的 combo-loader-02 exit 0，26 个目标 symbol 均非空、missing 为空。

生成新准备包时固定使用四卡、DP 回退、组合 custom-ops manifest 和两个 vendor 名称：

```powershell
$PY = 'C:/Users/w00939120/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$PREP = 'D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/runtime/92b-http/prepare_http.py'
$MANIFEST = 'D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/evidence/2026-10-09-custom-ops/20261009T-layer14-transfer-03/manifest.json'
& $PY $PREP --devices 4,5,6,7 --dp-sync original --custom-opp 20261009T-layer14-vendor-combo-01 --custom-opp-manifest $MANIFEST --custom-vendors omni_custom_transformer,omni_training_custom_transformer
```

启动前核对新 run 的 payload/evidence 路径、bundle 与 manifest SHA、四张卡占用、TLS 和端口。controller 的端口检查实际使用 `SO_REUSEADDR` 的 bind/listen 与 `connect_ex`，不能把它描述为 `/proc/net/tcp` 解析。loader 通过后仍需分别核对模型加载、health、models、intro 请求和退出清理。本轮成功 run `20261009T134358Z-cda0d887` 已完成四组 HTTP 验收，结果见第19节。

证据见 [custom-ops 证据](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/evidence/2026-10-09-custom-ops/README.md) 和 [组合 vendor 记录](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/execution-record-2026-10-09-custom-ops-combo.md)。

## 19. 组合 vendor 下的 92B HTTP 验收

组合 inference/training vendor 通过 loader 后，使用新 run 逐项验收 HTTP，不把 health 200 单独当作请求成功。复现入口先生成新 run，再把 capture 交给已授权的远端传输入口：

```powershell
$py = 'C:/Users/w00939120/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$prep = 'D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/runtime/92b-http/prepare_http.py'
$remote = 'D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/runtime/dp-sync/run_remote.py'
$manifest = 'D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/evidence/2026-10-09-custom-ops/20261009T-layer14-transfer-03/manifest.json'
$runInfo = & $py -X utf8 -B $prep --devices '4,5,6,7' --dp-sync original --custom-opp '20261009T-layer14-vendor-combo-01' --custom-opp-manifest $manifest --custom-vendors 'omni_custom_transformer,omni_training_custom_transformer' | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw "prepare_http.py failed: $LASTEXITCODE" }
$capture = Join-Path $runInfo.evidence 'capture'
if (Test-Path -LiteralPath $capture) { throw "capture already exists: $capture" }
& $py -X utf8 -B $remote $runInfo.scratch_payload $capture
if ($LASTEXITCODE -ne 0) { throw "run_remote.py failed: $LASTEXITCODE" }
```

这里的 `capture` 由复现入口创建，命令本身不预先 `New-Item`；运行前仍需重新检查空卡、TLS、端口、四个 cases 和提示词 BOS。上述命令使用本轮成功配置的准备入口，具体认证材料由已授权传输入口内部读取。

run `20261009T134358Z-cda0d887` 的配置为物理 4/5/6/7、TP1/DP4/EP4、eager、MXFP8/BF16、MTP 关闭和私有 CANN 9.2.0。四个 cases 均返回 HTTP 200：intro 返回盘古文本（prompt 19、completion 28）；普通算术和流式算术均返回 42，token `[19,17,148902]`；prompt 26、completion 3；显式 `stop=['42']` 返回空文本，`stop_reason=42`，token `[19,17]`。BOS 仅一次且为 148899，EOS 为 148902，SSE `[DONE]`、usage 和请求 ID 均已核对。

退出验收同时检查 server/client/runner/remote exit=0、`state.success=true`、own 进程为空、4–7 卡 idle、18092/18192 端口可用和公共文件哈希一致。该 run 证明组合 vendor 下的 92B 请求链路可完成；正式性能、NPU trace 和 PD 分离仍需按后续测量口径单独采集。

证据见 [组合 vendor 记录](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/execution-record-2026-10-09-custom-ops-combo.md) 和 [custom-ops 证据](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/evidence/2026-10-09-custom-ops/README.md)。
