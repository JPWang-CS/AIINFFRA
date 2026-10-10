# 双节点 HCCL 实操记录

2026-10-09，11 号机与 4 号机在自有容器、个人 CANN 9.2.0 下完成 HCCL world size 2 的数据通信验证。每台使用物理 5 号卡，映射为逻辑 `npu:0`；CPU 接口为 enp39s0f3。

四组 FP32 SUM、三组 BF16 SUM、一组 4096 元素 FP32 SUM、两组 AllGather 全部通过。4096 元素逐项比较的最大误差为 0。两侧退出码 0，销毁 process group、释放选定端口、卡恢复空闲，五个公共文件哈希前后一致。

前两轮分别暴露启动时序和 TLS 配置问题。首轮将 TCP 互认放在 NPU 初始化之后，等待窗口在另一侧完成初始化之前结束；修正为先互认，再初始化。第二轮 process group 已建立，但第一个 AllReduce 在 HCCL 通信域初始化时返回 EI0016：11 的物理 1 开启 TLS，4 的物理 1 关闭 TLS。只读盘点全部卡后，选择两侧 TLS 都开启且空闲的物理 5，未修改设备 TLS 或公共配置。

该过程保留了四个独立验收点：root info 接口返回、process group 建立、首个设备 collective、全部回读数据比较。排查同类问题时，按真实执行位置区分阶段；先核对两个 rank 的设备选择、TLS、接口和实际库路径，再扩大测试规模。

- [完整运行记录](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/execution-record-2026-10-09-hccl.md)
- [原始输出与源码](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/evidence/2026-10-09-hccl/README.md)

后续 7B TP2 推理、92B 与 PD 分析各自保留运行记录。
