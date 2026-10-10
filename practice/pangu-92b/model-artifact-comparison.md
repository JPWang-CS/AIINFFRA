# openPangu-2.0-Flash 与现有 92B 权重对比

## 范围与结论

- 核对时间：2026-10-08；远端仅访问 `liteserver-hps-a2c1-11-00001` 的自有容器，物理目录守卫通过；未访问其他主机。
- 现有权重目录：`/mnt/sfs_turbo/s3-asset-g-gy-test-a5-exp/pangu_model_zoo/92B/dsa/iter_0011840_mxfp8_npu`。目录存在、可读；顶层计数 64 项，含 50 个 safetensors 分片、index、config、tokenizer 文件和本地自定义代码。未复制、下载、运行模型或读取 tensor payload。
- 结论：现有目录与 HF `openpangu/openPangu-2.0-Flash` 是**同系列/结构相近，但不是同一字节发布权重**。远端明确是 mxfp8 量化变体；HF pinned revision 的 config/index/tokenizer 小文件均与远端不同，分片总量也不同。因此当前应保留并核对现有权重，不能因名称相近直接宣称与 HF 权重相同或直接复用 HF 量化/启动结论。

## HF 固定来源

- API：`https://huggingface.co/api/models/openpangu/openPangu-2.0-Flash`
- 访问日期：2026-10-08；API 返回 commit SHA：`5b78fc70a06dd27c5c4c5e918c46c49531d357b4`。
- 仅读取该 revision 的 `config.json`、`tokenizer_config.json`、`model.safetensors.index.json`；未下载任何权重。HF index：37587 个 weight map 条目、50 个分片、`total_size=200272648050`。

## 结构与量化矩阵

| 字段 | 远端现有目录 | HF pinned revision |
|---|---|---|
| architectures | `PanguUltraMoEForCausalLM` | `OpenPanguV2ForCausalLM` |
| model_type | `openpangu_v2` | `openpangu_v2` |
| hidden/layers/heads | 2560 / 46 / 48 | 2560 / 46 / 48 |
| routed/shared experts | 256 / 1 | 256 / 1 |
| experts per token | 8 | 8 |
| dtype | config `bfloat16` | `bfloat16` |
| quantization | `mxfp8`，weight `float8_e4m3fn`，block 32，scale `e8m0`/`uint8` | config 未提供对应 `quantization_config` |
| transformers_version | `4.48.2` | `5.0.0` |
| max position | 524288 | 524288 |

结构字段高度相近，但 architecture 名、transformers 版本和量化配置差异是实质证据；不能视作同一发布目录。

## 小文件与分片证据

| 文件 | 远端大小 / SHA256 | HF pinned 大小 / SHA256 | 判定 |
|---|---:|---:|---|
| `config.json` | 3323 / `80412608474b7d5356829dab0535fd940338f61c2495e9ea3b93a2b719ab9011` | 1852 / `af58889e31ec191b81cc56fc9e8ed67e5e10790d75992ea05714306ede74a` | 不同 |
| `tokenizer_config.json` | 163317 / `7ce015ab1e4d01e2bc3fcc7fa336754fd2be392c580a810186c8e4aa91aef092` | 163323 / `5253c845a1f4a62428e7521c6215fe552a877c9331e0e86b2f2a4327814acdd2` | 不同 |
| `model.safetensors.index.json` | 6259140 / `9daa669d706c709cee306616320ce8ea0334bb5d7439db551fb9f289dc83b53e` | 3216951 / `893556af125d43a1305affd0741034566d01f9014aafa229cf92dee1ac1bda9a` | 不同 |

远端 index 有 73977 个 weight map 条目、50 个分片、`metadata.total_size=107169876850`；50 个 header 有界读取后，`sum(8+header_len)=9428296`，`file_sum-header_sum=107169876850`，且 `data_offsets` 长度和也为 `107169876850`，与 index 精确一致。9,428,296 是 safetensors header 开销，不是模型数据异常。index 引用的 50 个分片均存在。50 个 safetensors 均只读取 8 字节 header 长度和有界 header（单个 ≤16 MiB、总 header ≤32 MiB），未读取 payload。dtype 统计为 `F8_E4M3=36390`、`U8=36390`、`BF16=1150`、`F32=47`；代表项包括 `model.layers.0.mlp.down_proj.weight_scale`（U8，[2560,288]）、`gate_proj.weight_scale`（U8，[9216,80]）及 `model.layers.2.mlp.e_score_correction_bias`（F32，[256]）。

tokenizer 小文件大小/hash 不同，不能直接认为 tokenizer 字节一致；本轮未比较完整 tokenizer payload 与所有分片 hash；但 header/offset 账本已闭合。

## 可复用结论与未验证项

- 当前目录可作为本机后续模型确认的候选来源；“相同”只能在完整分片 hash 与官方同 revision LFS hash 全部一致时成立，本轮证据明确不满足。
- 远端目录通过容器挂载可见，但共享路径的来源、权限长期稳定性和实际分配仍待确认。
- 未验证 torch_npu/vLLM/vLLM-Ascend 兼容矩阵、CANN 路径迁移问题、模型启动、精度、服务、PD 混部/分离或性能。
