# 2026-10-09 custom-ops 检查记录

## 结论

固定镜像 layer14 的 `libcust_opapi` 已确认定义 `aclnnKvQuantSparseFlashAttentionV2` 与 `aclnnKvQuantSparseFlashAttentionV2GetWorkspaceSize` 两个目标 symbol。node4 的 loader-06 在个人目录完成加载，退出码 0，两个 `dlsym` 均非空；私有 CANN 9.2 的 `libopapi`/`libnnopbase` 与个人 vendor overlay 的来源分别记录在 maps 中。

完整 vendor 为 335 files / 73,871,433 bytes，bundle 为 19,400,961 bytes，bundle SHA256 `f2cd3a6ce9b9a4e8fb98d4621bcee103c1c793991fb941f70be0405cd4b08155`；`libcust_opapi.so` SHA256 `0aca4613eac2f5dfe0b4060cd052bd5a1db45319e2ceac1b926d06faff2ce727`。前 01–05 测试为构造/解析过程，最终 06 才是有效 loader 证据。

本轮只证明自定义动态库的选择性注入和 symbol 加载，未证明 92B 模型请求成功。运行包命令为：

```powershell
python D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/runtime/92b-http/prepare_http.py --devices 4,5,6,7 --dp-sync original --custom-opp 20261009T-layer14-vendor-01
```

- [custom-ops 证据目录](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/evidence/2026-10-09-custom-ops/README.md)
- [DP 同步与首请求记录](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/execution-record-2026-10-09-dp-sync.md)


## 2026-10-09 run132908Z-cc018d8e

新 run `20261009T132908Z-cc018d8e` 已完成权重读取、DP 同步、KV 初始化和 HTTP health/models 200。四个 worker 的 `libcust_opapi.so` 均从个人 custom-ops overlay 加载；此前 KvQuant symbol 缺口已解决。首个 intro 请求仍返回 HTTP 500，新的缺口为 `aclnnAiInfraAttentionPioneerMetadata*`。实际调用栈进入 `npu_mla.py:_forward_prefill_standard:1987` → `_apply_attention:1090` → `_apply_sink_attention_a5:1379` → `attention/backends/utils.py:377` → `torch.ops`。

server exit=0、client exit=1、controller exit=1；own 进程为空，selected cards idle，公共文件哈希一致。记录中的 18092 bind 为 false、18192 为 true，下一次运行前复核 TCP LISTEN；不据此写端口残留或已释放。host worker 正在一次性核对 MLA/DSA 所有 custom symbol 缺口。
