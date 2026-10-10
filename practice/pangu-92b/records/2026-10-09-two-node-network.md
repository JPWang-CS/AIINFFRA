# 2026-10-09 双机网络盘点

在 11 号机和 4 号机的自有容器内完成只读网络盘点。CPU 候选接口均为 enp39s0f3，地址分别为 10.83.185.108/24、10.83.185.133/24，MTU 1500，默认网关 10.83.185.1。候选端口 29571 在本次内核端口表中未绑定，尚未预留。

个人 CANN 9.2.0 wrapper 下，两台物理 1 号卡均返回 17 个 UB 端口 UP，以及 ipourma0–3 的地址和 udma/EID。最新占用快照中，两台物理 1 号卡均无进程，执行测试前仍须复查。hccn 的 vnic 查询返回 248、ub_connect 返回 10、net_health 返回 Init，原始结果已保留。

容器缺少 ip、ss、rdma、ibv_devinfo；通过 Python 只读 ioctl 和 /proc/net 补齐地址、路由及绑定端口。首轮查询误用 bash -lc，随后停用，改用无登录 shell 的 Python 和私有 wrapper 复查；首轮无单独原始 stdout，不作为当前证据。

本轮未执行主动跨机探测、服务、模型或 collective。下一步为 CPU TCP/Gloo 最小通信验证，之后确定 HCCL 用例与 UB/URMA 选路。所有操作仍限定在自有容器和个人部署目录，退出时处理当前 shell 指令历史。

- [完整记录](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/execution-record-2026-10-09-network.md)
- [网络矩阵与原始证据](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/evidence/2026-10-09-two-node-network/README.md)
