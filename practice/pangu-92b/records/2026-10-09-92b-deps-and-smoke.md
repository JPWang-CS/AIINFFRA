# 2026-10-09 92B 依赖与最小算子记录

4 号机个人容器完成 Torch/Torch-NPU、自定义包、PanguUltraMoE registry、Mxfp8Config 导入和 ModelConfig 解析。依赖阶段未初始化 NPU 或创建模型实例。20261009T103041Z-d9b19dfa 原判据把 metadata 模块当成 callable，阶段退出 1；真实注册名为 custom::_npu_fused_infer_attention_sink_metadata。判据已经修正，但修正版完整依赖阶段尚未独立复跑。最终单卡 smoke 查到了真实 dispatcher schema。

物理卡 4 的最终 smoke 20261009T105122Z-3821c1d6 完成三组 MXFP8 quant matmul 与 A5 原生 mHC Sinkhorn。三组 (M,K,N) 为 (16,128,32)、(17,256,64)、(31,64,16)，最大绝对误差均为 0；均匀 Sinkhorn (2,4,4) 主输出与 0.25 的最大误差为 2.384185791015625e-07。外层远端命令/子进程退出码均为 0，success=true；加载库来自个人 CANN9.2.0，退出后本次进程组为空，物理卡4空闲，五个公共文件哈希一致。

调试记录保留三项修正：K64 scale 使用新标准布局 buffer 再 copy_；A5 走 torch_npu.npu_mhc_sinkhorn，失败的 custom 路径属于非950分支；out_flag=0 的辅助返回值为 None，检查主输出并记录辅助缺省状态。最小用例不包含完整模型、attention/MoE 或一般 Sinkhorn 精度。

原四卡 0/4/6/7 的 HTTP 尝试 20261009T105907Z-0d96d8ad、20261009T110312Z-c32464a1 均在启动前因物理卡0存在进程停止，没有92B模型加载、HTTP响应或性能结果。20261009T110602Z-21102cb1 是本地准备的2/4/6/7候选，仅有payload和manifest，未执行。设备分配确认后使用新包，重新检查实时占用和TLS。

- [依赖证据](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/evidence/2026-10-09-92b-deps/README.md)
- [最小算子证据](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/evidence/2026-10-09-92b-smoke/README.md)
- [HTTP准备与失败记录](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/evidence/2026-10-09-92b-http/README.md)
- [教程](../tutorial.md#13-92b-依赖与-a5-最小算子核对)
