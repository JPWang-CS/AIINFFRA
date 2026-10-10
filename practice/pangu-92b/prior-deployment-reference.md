# 既有部署经验参考

## 结论

历史记录证明 `openPangu-Embedded-7B-V1.1` 曾在 NPU 上完成一次离线单请求生成，可借鉴排查顺序和证据组织方式。它不是盘古92B证据，也不是 HTTP 常驻服务、PD 混部/分离或正式 benchmark 证明。

记录日期为 2026-08-21。本地索引位于 [course-source-index.json](D:/Desktop/Code/Learn/AIINFFRA/.codex/course-source-index.json:902)，其中 `pangu_case` 和证据边界见 `:902-917`。索引声明的 success-tail 路径和 SHA256（`:911-912`）已于 2026-10-08 实际核对，文件 SHA256 为：

`B3FE23C4889810CD954D6C1E2FAB9E012A96D1B3C19BB1C79414FB7BBE22C09D`

与索引一致。

## 两份成功证据要分开看

原始成功运行日志：[pangu-tp1-smoke.log](D:/Desktop/Code/CC4Ascend/projects/pangu-7B整网分析/log/pangu-tp1-smoke.log)。

- `:11` 解析到 `PanguEmbeddedForCausalLM`。
- `:29` 记录成功 run 的 vLLM V1 `0.23.0`、TP1、PP1、BF16、NPU、`max_seq_len=1024`、eager、prefix/chunked cache 等配置。
- `:58` 记录权重加载 `14.9828 GB`；`:61` 记录可用 KV cache `21.44 GiB`。原样保留 GB/GiB，不混换。
- `:83-87` 记录单个 prompt 完成和中文输出。
- `:88-93` 记录 SIGTERM、abort teardown 及 engine core unexpected cleanup。

尾部核验文件：[2026-08-21-pangu-embedded-7b-tp1-success-tail.log](D:/Desktop/Code/CC4Ascend/projects/pangu-7B整网分析/logs/2026-08-21-pangu-embedded-7b-tp1-success-tail.log)。其 `:1-3` 是生成完成和中文输出，`:4-9` 是退出 cleanup。两份文件不能合并成多轮稳定性结果；日志中的 tokens/s 估算也不等于 TTFT 或 TPOT。

vllm-ascend `0.23.0rc1` 与 A2 标签来自历史总结表 [调用链路分析-TP4静态分析旧稿.md](D:/Desktop/Code/CC4Ascend/projects/pangu-7B整网分析/调用链路分析-TP4静态分析旧稿.md:1254)，不是这组日志里的独立 `pip` 或 `npu-smi` 探针。目标环境必须现场复核版本和硬件。

## 失败与证据边界

attempt-1 日志：[2026-08-21-pangu-embedded-7b-tp1-attempt-1-layer-prefix-error.log](D:/Desktop/Code/CC4Ascend/projects/pangu-7B整网分析/logs/2026-08-21-pangu-embedded-7b-tp1-attempt-1-layer-prefix-error.log:185)。真正阻断层在 `:185-194`：`layer_fn(prefix=...)` 与 `PanguEmbeddedModel.__init__` 内 lambda 的签名不兼容，最终为 `unexpected keyword argument 'prefix'`。

当前本地没有找到对应补丁、attempt-2 日志或明确修复记录。因此这里只能得出“适配类与框架构造接口需要版本兼容核对”的方法结论，不能编造具体改动或声称补丁已验证。Triton 导入告警和 `vllm_ascend_C` 缺失也出现在完整 smoke 日志（`:6`、`:30/:60`）；它们在这一次记录中不是最终阻断，但不证明其他模型或 fast path 可用。eager、slow tokenizer、custom ops disabled、prefix/chunked cache 都会影响性能解释。

## 可复用步骤

| 历史经验 | 92B 本次动作 | 必须重验 |
|---|---|---|
| 单卡小请求先确认模型能加载和生成 | 先按容量确认可行的最小拓扑，再发最小请求 | 总显存、权重驻留、KV、临时/通信 buffer、P/D 资源 |
| 模型类曾与 `prefix` 构造接口不兼容 | 将模型 config、适配类、框架和插件版本绑定后再启动 | 构造签名、模型注册、实际 adapter、版本/commit |
| 权重和 KV 有运行时记录 | 分开记录权重、KV、激活、临时和通信账本 | 原始加载日志、设备内存、KV 容量与请求长度 |
| 告警、阻断、输出、退出状态不同层次 | 分别记录告警、最深阻断、请求结果、服务存活和退出 | 原始日志、请求状态、退出码（仅实际退出时） |
| 功能 smoke 不能推出性能 | 按功能→多例稳定性→benchmark→profile 逐层增加证据 | 重复次数、负载、指标口径、profile 开销 |
| 7B 使用 TP1 仅是历史事实 | 92B 不预设 TP1、框架、精度或版本 | 模型结构、容量和最小可行拓扑 |
| TP1 离线生成未提供 PD 分离证据 | 分离时额外记录 KV connector、网络和 P/D 独立 trace | KV 传输/等待、P/D 请求链和完整返回 |

## 当前空白

本轮核查的索引与日志中，尚未找到完整环境创建/安装命令、独立硬件探针、成功 run 的完整可复现脚本与 exit code、模型适配修复补丁、HTTP 服务证据或多卡 PD 证据。这里表述为当前未找到或证据不完整，不断言它们绝不存在。跨仓绝对链接和行号只适用于当前本地工作区；连接远端或进入容器后需重新核查路径、版本和可读性。本文不复制账号、IP、凭据或整段环境命令。
