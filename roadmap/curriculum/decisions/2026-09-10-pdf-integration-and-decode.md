# 2026-09-10：两份 PDF 融合与 Decode/PagedAttention 边界

## 范围与来源

本记录只保存本地资料核对、课程映射和实现边界，不外传讲义全文或附件，不记录讲义底部的内部社区原始链接。课程正文只写经主 agent 核对后可独立解释的机制；没有原始实验身份的加速数字不进入实测表。

| 本地来源 | 核对结果 | 课程用途 |
|---|---|---|
| 大模型推理实践.pdf | 84 页、16 章；作为主题地图和错误表，不把讲义所有断言当作官方事实 | ch2/12 映射性能与模型分析；ch8 Fusion；ch9 Decode；ch10 量化；ch7/13 MoE/系统；ch11/16 sampling/投机 |
| DeepSeek_V41_Tech_Report.pdf | 实际标题为 DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression，共 51 页 | §2 p7-14 与 §3.2 p18-20 支撑 Decode 的地址管理/结构压缩/精度三维联系；p22、§2.4.4/§4.2.1做容量手算案例 |

训练、scaling law 和 RL 留作扩展，不抢算子主干；Single-Pass mHC p12-13 依赖模型修改才能减少遍历，已融入 Fusion 正文。论文线继续独立，不能把本记录写成作者代码审读结论。

## 讲义错误与正文校准

| 位置 | 核对结论 | 处理 |
|---|---|---|
| p8 | TTFT 不只是 prefill，还包括 queue、tokenize、transfer 等边界；最大并发不能从容量直接绝对推出，容量只是上界，SLO 有效并发需压测 | 正文写测量边界，不写绝对并发公式 |
| p39 | Megatron SP/CP 表述需官方核对 | 不照搬，暂不作为机制结论 |
| p40 | 常规 FlashAttention 主要是 QK-softmax-PV，不把 QKV projection 包含进 FA；浮点通常不结合，例 (1e20 + -1e20)+3.14 与 1e20+(-1e20+3.14)；determinism 不是 idempotence | Prefill/Decode 正文按真实算子边界和数值顺序写 |
| p52 | 稀疏选择减少当前 query 的部分读取和计算，不代表未选 KV 可以从持久状态中永久删除 | Decode 正文区分有效读取、存储容量和可见集合 |
| p53 | “连续浮点”只是近似说法，浮点也是离散表示；tensor/channel/token/group 不是全序粒度层级 | 不使用绝对层级叙述 |
| p54 | 权重/KV 的容量收益要按组成项分别核算，不能把两个比例相乘 | 不引入整模乘性收益 |
| p55 | 同页出现 weight > activation > KV 与 KV > weight > activation 两种优先级 | 正文不固定优先级 |
| p61 | DSpark 称独立 V2/V3 草稿，与报告 §2.4.3 不一致 | 不照抄、不泛化 |
| p83 | 闲时投机的边际成本不等于无条件为零 | 作为投机边界，不写零成本定律 |

## MLA 与 DeepSeek-V4.1-Flash 数值边界

讲义 p49 的 MLA cache 公式把 latent 512 与 RoPE 64 分别乘 2，导致约 137 KiB/token 的错误印象。主 agent 核对 DeepSeek-V3 官方 inference/model.py：absorbed 路径真实 register_buffer 只有 latent KV 与 RoPE cache，且代码验证两者共用；在无 TP/metadata 的数据模型中，61 层的正确账本是

61*(512+64)*2 = 70272 B/token。

这只是缓存数据模型示例，不是完整部署容量。

报告 §2 的机制按页码使用：p9 CED 的 decoder globalKV 来自 encoder 最后 hidden 的层特定 projection，本地 SWA 仍分层；p10-11 CSA2 Full 生成 mainKV/indexerK/indexerQ/topK，Reindex 重用前 Full mainKV/indexerK 但新 indexerQ/topK，Reuse 还重用最近 index producer topK 而无 indexerQ，各模式保留各自 mainQ 与 SWA KV；p14 mainKV 为 E2M1 + 每16 channel 一个 E4M3 scale、4.5 bit/element，indexerK 为不同的 MXFP4，SWA 为 FP8，先 dequant 不等于原生 FP4 MMA；p19-20 global 持久 SSD/host cache 与 minute-TTL SWA pool 分开，bounded replay 只重放末尾 window，论文明确是近似而非数学等价 cache miss 恢复。

报告 p22 的容量复算：3 个 encoder Full 组各 compression 2，加 decoder 1 个 Full compression 1，因此 global entry 数约 2.5S；main512d 为 256+32=288 B/entry，indexerK128d 为 64+4=68 B/entry，合计

(288+68)*2.5 = 890 B/token。

该数值是根据报告 §2.4.4/§4.2.1 的 CPU 复算，不是实验测量；不含 SWA、页 padding、索引输出、allocator 和 TP 副本，也不是全模型显存。NVFP4 完整格式的第二层 global scale 以 NVIDIA 公开主源为准，不能反推报告 mainKV 已使用完整 NVFP4。

## Decode 实现边界

新章固定 q[B,Hq,D]、独立 cache_k/cache_v[P,T,Hkv,D]、lengths[B]、table[B,max_pages]，明确 kvhead=qhead//(Hq/Hkv)。CPU 模拟器先独立生成 dense logical K/V，再 scatter 到随机非连续物理页，分别测试 dense、paged、split (m,l,U) merge，覆盖 ragged lengths、GQA、页尾、空段、-1/>=P 坏页。

Triton 示例是 FP32 simple correctness baseline：单 Query page-table 读取、append 后 token 边界、独立 K/V pool、tl.sum(q*k)、普通 tl.range，不声称 FlashAttention-2 性能。host-only fakeTensor 合同覆盖合法 read/append、output alias、new/cache alias、cache 与 q/metadata 写入别名、batch 内共享 target、zero Hkv 和 trusted 超容量；ownership 只对传入 batch 可见引用负责，不能替代全局 allocator/COW 或 stream event 管理。

LeetGPU 题面边界：

公开题库核对快照：AlphaGPU/leetgpu-challenges 的 main commit 为 dff541e5856ddaee4c93e23c0d7489c125fd6ea3（2026-09-10 查询）。教程保留平台入口与公开题面链接；后续提交仍需核对平台当时的输入输出合同。

- #6 Softmax Attention 的连续题面为 Q[M,d],K/V[N,d],out[M,d]；M=1 可作为单 Query/长 KV 连续数学基础。
- #80 Grouped Query Attention 验证多头 GQA 映射，但它是同长无 batch 题。
- 公开题库树未见 paged/decode 命名题；两题都不验证 page table、append 或 COW。因此本章先做基础题归档，再做分页 CPU/GPU 扩展，不伪造平台成绩。

GPU 不可用时只报告静态检查、py_compile、CPU 数学和 host 合同；benchmark 先对自己的 benchmark shape 完成独立 logical reference correctness，再分别报告预分配 wrapper、输出分配、预先 gather dense、gather+attention，并带 device、shape、dtype、软件版本和计时边界。

## 实际落地与接续

- 新增完整算子第六章 Decode/PagedAttention，课程站编号 12；保留既有 1–11 页入口。
- 性能基础章新增时间戳、单 token TPOT 边界、P95 反例、Amdahl 和 MFU 口径；Fusion 章新增 Single-Pass mHC 的公式、依赖反例和流量手算。
- 原始 PDF、用户 solutions/reference、PATH/NOW 未修改；新增教材不计为用户阅读或实践完成。
- 一个 Luna worker 写入初稿后因额度限制停止。主 agent 接手修正与验证，未启动替代 worker。先前错误涉及 append 越界、tile 广播、别名检查、COW 页内偏移、PDF 相对路径和正文导航截断；修正后才生成新网页。
- 核查重点是讲义与推理有关的章节、报告 §2、§3.2、§4.2.1；没有把训练/后训练的所有实验或作者全部代码记为已精读。
- 后续仍需详细展开量化算子、MoE、Sampling/KV 辅助算子，以及独立的极致性能、量化算法、模型分析和系统章节。这些不是本次新建正文。

## 验证结果

正文引用方式补充：用户要求不使用“某日期的真实 sweep”“值得追问”等历史叙事代替来源。GEMM 与性能章已改为具体代码、实验分析和 s2/s3 trace 链接；日期留在原始记录及文件名中，正文以实验对象和配置命名。没有删除原始记录、性能数字或改写用户源码。Decode 继续融入讲义的容量预算与量化组成项，以及报告的 CED 公式、CSA2 来源追踪和 bounded replay 反例；新增 cache_accounting.py 核对这些推导。

主 agent 实际运行：CPU paged/dense/split 检查、host-only 合同检查、mHC/缓存算术、时间戳/P95/Amdahl 均通过；另抽查 10 个随机页映射种子。Python compileall 通过。GPU harness 在本机返回 CUDA/PyTorch 不可用并跳过，不计 GPU 编译或执行通过。

站点构建为 12 章，850 个公式、31 个原始源码摘录通过构建检查；本地链接审计和 DOM 导航回归通过，含 11↔12、侧栏折叠、搜索和直接锚点。HTTP 访问第 12 页返回 200。DOM 测试不等于像素级浏览器验收；没有新增 GPU 性能数据。
