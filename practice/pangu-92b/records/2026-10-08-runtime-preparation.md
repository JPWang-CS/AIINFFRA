# CANN 9.1 准备记录

- 采样日期：2026-10-08（上海时间；本地记录时间与远端命令输出按实际会话保留）。
- 范围：仅 11 号机的 `w00939120` 自有容器；入口为已授权的容器 exec，容器内物理 cwd 守卫 `/data/docker/w00939120`。
- 远端 shell 使用 `set +o history`、`HISTFILE=/dev/null` 和退出时 `history -c`；未输出凭据，也未将凭据写入命令、日志或新增文件。
- 目标个人前缀：`/data/docker/w00939120/pangu-deploy`；本轮未创建该前缀下的包文件或目录。

## 官方 HEAD 结果

以 Python 标准库 HTTPS `HEAD`，TLS 校验保持开启，并带官方文档 Referer，检查：

1. `Ascend-cann-toolkit_9.1.0_linux-aarch64.run`
2. `Ascend-cann-950-ops_9.1.0_linux-aarch64.run`
3. `Ascend-cann-nnal_9.1.0_linux-aarch64.run`

三项结果均为：`URLError: <urlopen error [Errno -2] Name or service not known>`。这是 DNS/网络解析层失败；没有得到 HTTP status、Content-Length 或 ETag，不能据此判断 URL 资源不存在。

## 本轮边界

未下载、未生成 `.part`、未计算包 SHA256、未运行安装器、未 source CANN、未安装 Python 依赖、未运行模型或 benchmark。镜像 registry 路线已按用户决定弃用；本记录仅保留官方包准备失败证据。后续需先恢复容器到官方 OBS 的 DNS/网络路径，再重新执行 HEAD，HEAD 通过后可在已授权个人前缀继续流式下载；安装步骤另行讨论。


## DNS 范围诊断

同一容器、同一物理 cwd 守卫下，`socket.getaddrinfo` 对 `github.com`、`huggingface.co`、`ascend-repo.obs.cn-east-2.myhuaweicloud.com` 均返回 `gaierror: [Errno -2] Name or service not known`。这说明本次容器外网 DNS 解析整体不可用；未修改 `/etc/resolv.conf`、代理或证书，未重试 registry。
