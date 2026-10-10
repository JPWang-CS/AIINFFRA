# 2026-10-09 custom-ops 组合 vendor 记录

node11 的真实 ELF 证据确认 Metadata symbol 位于 training 库，SHA256 `c3c63a91a3cba2ac25e0cf60c5ad31590ee07ba53902a9fdd066003858812480`。组合包 `20261009T-layer14-vendor-combo-01` 包含 inference 与 training vendor：1101 files、502151939 bytes；bundle 186852590 bytes，SHA256 `29629f7cc7ebccdcbad12944d06e9070224d7907a9816ad8ecbcb04d310c03f1`；manifest 311742 bytes，SHA256 `1e83d061f58276bc977dadc78fcecaea831b647fe7d036f6799aa84e6403ea24`。

node4 combo-loader-02 exit 0，26 symbols 非空，missing=[]，两套库 maps 来自个人 overlay 与私有 CANN9.2。新 run `20261009T134358Z-cda0d887` 已完成：四个 HTTP cases 返回 200，server/client/runner/remote 均为 0，状态 success=true。controller 端口检查为 SO_REUSEADDR bind/listen 后 connect_ex。

- [custom-ops 证据](D:/Desktop/Code/CC4Ascend/projects/pangu-92b整网分析/evidence/2026-10-09-custom-ops/README.md)


## HTTP 复验结果

组合 vendor 与私有 CANN 9.2 均从预期目录加载。intro 返回盘古文本（prompt 19、completion 28）；算术普通与流式均返回 42，token 为 `[19,17,148902]`，prompt 26、completion 3；显式 stop=42 返回空文本，stop_reason=42，token 为 `[19,17]`。BOS 仅一次且为 148899，EOS 为 148902，SSE `[DONE]` 和 usage 均已核对。server/client/runner/remote exit 均为 0，own 进程为空，4–7 卡 idle，18092/18192 为 true，公共文件哈希一致。

本次结果覆盖 custom-op 组合下的 92B HTTP 请求验证；正式性能、NPU trace 和 PD 分离仍待后续阶段。

## 最终资源回收检查

`evidence/2026-10-09-custom-ops/final-resource-check/resource-check.json` 的最终只读回收显示：npu-smi exit 0，物理 4/5/6/7 的 HBM 使用分别为 4752/4748/4747/4751 MB，均无运行进程；这是回收时的瞬时占用，不写成预留资源。原 run PGID 85739 的成员为空。18092 和 18192 均可 bind/listen，`connect_ex=111`，没有残留服务连接。

