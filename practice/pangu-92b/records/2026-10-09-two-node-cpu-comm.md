# 2026-10-09 双机 CPU/Gloo 记录

2026-10-09 11:52:24–11:52:29（上海时间），11 号机与 4 号机在自有容器、个人运行目录内完成一次 CPU 双节点验证。TCP 29571 完成 nonce/rank/run id/values 互认；独立 TCPStore 29572 建立 Gloo world size 2。三组 AllReduce 与 barrier 均通过，两个子进程退出码均为 0，选定端口释放。检查的 5 个公共配置路径的存在状态、大小与 SHA256 前后一致。

输入与输出为：`[1,2]+[10,20]=[11,22]`、`[-3,0]+[1,1]=[-2,1]`、`[0,5]+[-4,-5]=[-4,0]`。运行时使用私有 CANN 9.2.0 入口、Torch `2.9.0+cpu`、单线程和 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`；未加载 `torch_npu`，未执行 NPU collective/HCCL。父进程正常等待并回收子进程，120 秒 watchdog 未触发。

证据与调用顺序见 [项目记录](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/execution-record-2026-10-09-cpu-comm.md) 和 [AIINFFRA 记录索引](./README.md)。


