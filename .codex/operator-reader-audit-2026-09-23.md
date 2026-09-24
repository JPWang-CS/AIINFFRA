# 算子与优化九章：读者走读与校订记录

此文件记录教材维护与验证证据，不进入课程网页，也不代表学习进度。

## 走读范围与问题

- 阅读顺序：访存与布局 → 归约、Softmax 与归一化 → GEMM → Activation/Fusion → Prefill Attention → Decode/PagedAttention → 量化 → MoE → Sampling/KV。
- 旧版长章节常把平台题目和服务器实验集中到章尾；读者在学习具体算子时，需要跨过后续主题才能找到对应练习。改为在具体算子的公式、形状、地址与代码之后说明正确性题面、可观察的优化旋钮和真实设备记录位置。
- GEMM 与量化篇幅很长；保留硬件细节、关键推导、用户源码与性能证据，删去重复的学习流程口号，不把“简洁”理解为省略成立条件。
- 前台不设算子总序或源码跳转页。代码片段、公式及必要图解在当前正文解释；原始文件和精确页码只作为后台校订依据。

## 来源校核

- `downloads/cuda-programming-guide.pdf`，Release 13.3：PDF 页 72–74 的 §2.3.4.1 说明 32B global-memory transaction 与 warp 地址合并；页 81–83 说明 shared-memory 转置的同步和 32×33 padding 对 bank conflict 的影响。硬件说法还须按所用架构及代码布局限定。
- `downloads/大模型推理实践.pdf`，PDF 页 49–60 涉及 KV 分层/卸载、量化、投机验证与 MTP。页 49 对 MLA 缓存使用 `2×512+2×64` 的账本不能直接套到共享 latent 的吸收实现；缓存须按实际持久化张量重新计数。页 52 的稀疏选择不等于历史 KV 一律被永久删除。页 56、58 的 Decode 流量和投机收益表述也依赖工作集、草稿开销、接受率与部署条件；课程不能引用为无条件结论。
- `downloads/DeepSeek_V41_Tech_Report.pdf` 的 §2.2–2.3 包含 CED、CSA2、跨层 KV/索引复用；这些结构改变了缓存与访问合同，不与普通 paged dense decode 混为一谈。
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 与 [Triton 官方矩阵乘法教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) 用于核对算子实现的硬件/编译语义；[AIInfraGuide 学习路线](https://caomaolufei.github.io/AIInfraGuide/guides/ai-infra%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/) 只参考由概览进入专题、专题内连续解释与练习的组织方式，不复制未经核实的硬件或性能说法。
- LeetGPU 题面与仓库中的 `solve`/kernel 是不同证据；需区分题目实际合同、已有本地示例、平台原样提交和服务器实测。

## 独立可复核的 CPU 检查

2026-09-23 使用仓库 `.venv` 运行：`03-gemm/examples/cpu_gemm_checks.py`、`05-prefill-attention/examples/cpu_attention_checks.py`、`06-decode-paged-attention/examples/cpu_paged_decode_checks.py`、`02-reduction-and-norm/examples/cpu_semantics.py`、`04-activation-and-fusion/examples/cpu_semantics.py`、`07-quantized-operators/examples/semantics.py` 与 `quantization_lab.py --demo`、`08-moe/examples/semantics.py`、`09-sampling-kv/examples/semantics.py`，均退出 0。CPU 语义及算术对照不证明 Triton/CUDA 编译、LeetGPU 通过或真实 GPU 性能。

## 网页与兼容性验收

九章校订完成后，`node build.cjs` 构建 39 篇主课、102 篇附属页、42 项本地资源；1734 处公式渲染、171 段原样源码摘录通过。`node check.cjs --content-only`、`node editorial-audit.cjs`、`node theme.test.cjs`、`node paper-reader.test.cjs`、`node navigation.test.cjs`、`node font-anchor.test.cjs` 和 `git -c core.autocrlf=false -c core.whitespace=cr-at-eol diff --check` 通过。最后一处 KV 逻辑元素系数修正后重新构建，再运行 `node link-audit.cjs`：103 个 HTML 页面、5278 个本地链接/资源目标、4069 个片段锚点和 1102 个附属文档目录链接通过。

新练习位置：GEMM 基础 tile 后为 Matrix Multiplication；量化的 block-scale 寻址后为 Weight Dequantization，W8A8 算术后为 INT8 Quantized MatMul；Prefill/Decode、MoE、Sampling 的平台题紧贴对应数据路径。题面验证范围与扩展 CPU/GPU 实验分开写；没有把平台题外的 paged/ragged/COW、完整 MoE 或模型级量化声称为平台成绩。服务器表为空，不改原始用户源码与历史结果。

题面来源抽查：正文引用的 13 个 AlphaGPU 官方挑战目录均以对应公开 `challenge.py` 的只读 HEAD 请求确认存在；课程正文只保留目录和 LeetGPU 编辑器链接，不链接 `.py` 原文件。平台 URL 是公开网页入口，不等于提交成功或在线编辑器可运行。

旧锚点在 `.codex/course-section-anchors.json` 中重定向：访存 7/15→7/31、7/26→7/32、已删除总问答 7/29→7/28；归约 8/22→8/3；融合 10/20–25→10/30–35；量化原章末 LeetGPU 13/15→13/35。浏览器中实际打开 7/15、8/22、10/20、10/25、13/15，均显示对应章节和展开侧栏；13/15 落在前移的 Weight Dequantization 题面前。顶部 GPU/算子切换及当前章节记忆也已在浏览器中验证。

量化页旧题目锚点落到第一道前移题的目的位置；INT8 MatMul 有自己的同页新小节，读者不必跳回章末。静态与浏览器检查验证站点路由和正文呈现，不验证 CUDA 编译或真实 GPU 性能。
