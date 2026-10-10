# Registry 与参考材料检查

## 参考材料

目录：`CC4Ascend/projects/pangu-92b整网分析/参考`。`参考1.md` 与 `参考2.md` 内容相同，SHA256 均为 `39182371672A9D61A2AAAE3FC05A3FCBBF9C2AD5A72302FD201B8BD904A4BA22`。材料描述 A5 92B MXFP8 hybrid，脚本 `/workspace/omniinfer/components/omni-models/tools/scripts/openpangu_v2/openpangu_v2_92B/openpangu_v2_92B_a5_hybrid.sh`（第4行），服务 benchmark 脚本 `/workspace/omniinfer/components/omni-models/omni_models/models/pangu_v2/start_server/acs-bench-a5.sh`（第19行）。第9–13行是 perf/acc 入口；第25–26行观察 Prefill 与从第二 token 起的 AVG_TPOT；第28–45行是 72/576 并发和 OpenAI-compatible profiling；第50行是 `/v1/completions` 单请求。56/72/576 仅属 92B 参考实验参数，不是 7B 配置。

`acs_bench-1.4.1-py3-none-any.whl` SHA256 `632B0D5E6698FF465D46701BB45F4740EEEA69FA031525668E5F9409F675F6C2`，元数据为 `acs-bench 1.4.1`，入口 `acs-bench=acs_benchmark.cli.entrance:benchmark_cli`；源码包含 OpenAI backend、请求数、并发、output length、TTFT、TPOT、吞吐字段。`LongBench500.zip` SHA256 `344A590F75F31DBD8F9263FE0F358D02801EC91B331D10F4D80CC2E7E882529E`，条目为 `LongBench500/131072.json`、`8192.json`、`8192_single.json`；文件名不作为实际上下文长度证据。

参考材料未提供 Registry 登录信息或 CA 证书路径。

## Registry metadata

目标镜像：`registry-cbu.huawei.com/omniai_omniinfer_dev/ai-infra-infer-1.0.3-a5-arm-master-202608271159-daily:0.0.1`。检查对象为 11 号机、4 号机的自有 `w00939120` 容器，物理工作根均为 `/data/docker/w00939120`。

两台容器的 DNS 与 TCP 443 可达；默认 TLS verify_code 18，错误为 self-signed。按用户明确要求，本次客户端 context 跳过证书验证后，两台 TLS 1.2 成功；GET `/v2/` 返回 HTTP 401、`Docker-Distribution-API-Version: registry/2.0`，Bearer realm 与 registry 同 host，scheme 为 `http`。客户端仍以 HTTPS 443 访问同 host token endpoint，匿名 token HTTP 200；指定 tag manifest HTTP 200、config HTTP 200。未执行 `docker login`，未把 SSH 认证材料用于 registry。

两台 tag digest 一致：`sha256:e54c77fa8f2383d5eb006ba101ea8e6e8be7ee047bbe98f0a0847034c5c9fc8f`；平台 `linux/arm64`，22 层，压缩总字节 `17627299325`。匿名授权已用于 manifest、config 和第 21 号代码层读取。

容器内 `curl` 可用，docker/skopeo/crane 不在 PATH；私有配置未发现该 registry 授权项。没有修改全局 TLS 或 daemon 配置。Windows 默认 TLS 对照经代理返回 504；不作为容器可达性结论。

## 镜像 metadata 与当前动作

ENV 显示 `Python3.11` site-packages 路径和 `ENV_NAME=PyTorch-2.10.0`，仅为镜像环境名，未据此确认实际 torch 版本。CANN 路径为 `/usr/local/Ascend/ascend-toolkit/latest`，精确 package metadata 待核。buildhistory 22 层与 manifest 对齐；第9层 COPY `/workspace/dist/codes` 压缩 581900391 bytes，第16层 RUN 涉及 omniinfer 499469066 bytes，第21层 RUN 涉及 omniinfer 与 proxy 脚本 1117029039 bytes，层 digest 为 `sha256:d582623f6e34c5478354f1b5dade5bb6f5ec5345aeaa3fa4677c065780ec7558`。

11 号机已完成单一代码层下载与静态源码提取。压缩层保存在 `/data/docker/w00939120/pangu-deploy/registry/ai-infra-infer-1.0.3-a5/blobs`，提取文件保存在同级 `files-layer21`。本轮执行范围为镜像与源码检查，环境安装和模型启动待方案讨论后进行。

## 单层代码核对

11 号机已下载并校验 manifest 对应的单一 21 号代码层：压缩层大小 `1117029039` bytes，耗时 15.66s，SHA256 与 manifest 匹配（`d582623f6e34c5478354f1b5dade5bb6f5ec5345aeaa3fa4677c065780ec7558`）。静态 tar 扫描 16074 entries，未 extractall、未执行；四个选定源码文件已复制到本地 scratch 并逐一核对 bytes/SHA。

有限 source trace：`omni_models/models/__init__.py:45-68` 的 `register_model()` 先调用 processor/configuration 注册，再将 `PanguEmbeddedForCausalLM` 注册到 `omni_models.models.pangu.openpangu:PanguEmbeddedForCausalLM`；7B 的注册语句位于 `:65-68`，位于前面 MoE 注册的条件块之外。实际插件 entry point 与 processor/configuration 导入依赖仍需结合包 metadata 核对。

`openpangu.py:1807-1814` 定义 `OpenPanguEmbeddedModel(OpenPanguModelBase)` 与 `PanguEmbeddedForCausalLM(OpenPanguEmbeddedModel)`；`OpenPanguModel:1435-1469` 通过配置构造层，`prefix` lambda 接收 prefix。2026-10-08 重新读取本地 7B config：architecture 为 `PanguEmbeddedForCausalLM`，dtype BF16，34 层，hidden 4096，MLP 12800，Q heads 32 / KV heads 8；config SHA256 为 `1F1124C339C014111A425F5A8BCCA0165C5C8CA0AF2F9F0141303A9FA5FC9C6E`。当前源码与模型 architecture 名称对应，安装后的导入、权重加载和 NPU 推理尚待执行。

92B hybrid 脚本 `openpangu_v2_92B_a5_hybrid.sh:38-45` source 三个 CANN/custom-op 环境文件并设置 `VLLM_PLUGINS`、`OMNI_NPU_PATCHES_DIR`、`OMNI_NPU_VLLM_PATCHES`；`:58-75` 以 `vllm serve` 启动，启用 BF16、chunked prefill、MP、TP、EP、DP、MTP 和 `hif8_ds_mla` KV cache。`acs-bench-a5.sh:3-12` 使用 OpenAI backend、448 requests/concurrency、output 1024 和 3 个 MTP tokens；这些是 92B 参考脚本参数，7B 仍拟 BF16/TP1/eager/1024/32。

脚本的关键公共依赖位于 `:38-44`：`/usr/local/Ascend/cann/set_env.sh`、`omni_custom_transformer` 和 `omni_training_custom_transformer` 两个 vendor 环境，以及 `VLLM_PLUGINS="omni-npu,omni_custom_models,omni_npu_patches"`。脚本 `:56` 还包含清理 `/root/.cache/vllm/torch_compile_cache/` 的命令。后续 7B 脚本将把缓存和日志定位到自有工作根，并逐项确定所需插件与算子依赖。

选定源码相对根为 `files-layer21/workspace/omniinfer/components/omni-models`：

| 文件 | SHA256 |
|---|---|
| `omni_models/models/__init__.py` | `2b7839ea3e09bb90afd0ebc5d405686b24f447b998fd1fc1713ee3da26c7b8bd` |
| `omni_models/models/pangu/openpangu.py` | `a9751aec1d3306dd316e99953c90a518c97d3242d40db77e4ac9fec961051bf8` |
| `tools/scripts/openpangu_v2/openpangu_v2_92B/openpangu_v2_92B_a5_hybrid.sh` | `93d8e8601bc43bd63801340f7b627280dd05ee75ef542095fa8dee61b8ddfdde` |
| `omni_models/models/pangu_v2/start_server/acs-bench-a5.sh` | `7d7d93d342089d1217535611730b2793a58576a1ccbb0d72aafdacca41ca705e` |

当前建议：先比较镜像实际 OmniInfer/vLLM 栈与自建 0.23 候选，再确定 7B 安装命令；不将 0.23 或 CANN 9.1 锁为唯一方案。
