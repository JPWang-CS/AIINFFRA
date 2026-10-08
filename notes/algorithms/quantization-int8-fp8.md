# 数值格式：INT8 / FP8 量化基础

> 量化类 · 从表示、校准到 kernel 路径理解推理加速

---

## 解决了什么问题

LLM 推理同时受权重容量、内存流量、矩阵计算和数值误差约束。量化把权重、激活或 KV cache 映射到较低位宽的表示，并保存还原所需的 scale（以及某些方案的 zero-point）。它可能减少存储和传输，也可能引入 scale 读取、解包、反量化与额外转换；是否加速要由具体矩阵形状、硬件指令、校准数据和请求阶段验证。

先区分三个层次：表示格式决定可编码的数值；量化算法决定 scale、粒度和误差补偿；运行时 kernel 决定这些编码是否直接进入矩阵乘，还是先解包到更高精度。训练后量化（PTQ）以已训练模型为起点，不进行完整的量化感知训练；具体方法可能需要校准数据、权重重构或等价变换，也有 data-free 变体。量化感知训练（QAT）则在训练中把量化误差纳入优化；两者不是位宽名称。

## 数值格式对比

| 格式 | 位宽 | 表示范围/编码 | 计算语义 | 硬件支持 |
|------|:---:|---------|:---:|---------|
| FP32 | 32-bit | IEEE 浮点 | 高精度基线 | 由设备支持情况决定 |
| FP16 | 16-bit | IEEE 浮点，范围较窄 | 浮点乘加 | 由 GPU、库和 Tensor Core 路径决定 |
| **BF16** | 16-bit | 1 符号位、8 指数位、7 尾数位 | 浮点乘加 | 由 GPU、库和累加类型决定 |
| **INT8** | 8-bit | 有符号整数编码 | 整数乘加，通常 INT32 累加 | 由 GPU、库、对齐和指令路径决定 |
| **FP8 (E4M3)** | 8-bit | 指数/尾数编码，常见最大有限值 448 | 浮点乘加，需 scale | 由 GPU 代际与库 recipe 决定 |
| **FP8 (E5M2)** | 8-bit | 指数/尾数编码，常见最大有限值 57344 | 浮点乘加，需 scale | 由 GPU 代际与库 recipe 决定 |

**关键区别**：
- **FP16 vs BF16**：两者都是 16-bit 浮点，但指数与尾数分配不同；范围和舍入误差影响溢出、精度及 kernel 选择，不能只用“训练/推理”二分。
- **INT8 vs FP8**：INT8 需要 scale/zero-point 把实数映射到整数；FP8 通过指数和尾数表达范围，也仍需 scale 管理张量动态范围。是否使用 Tensor Core 取决于 GPU 代际、库、维度对齐、累加类型和布局。

## INT8 量化原理

### 对称量化（Symmetric）

$$
\begin{aligned}
\text{FP16\_val} \in [-\alpha, \alpha] &\;\to\; \text{INT8\_val} \in [-127, 127] \\
\text{scale} &= \frac{\alpha}{127} \\
\text{INT8\_val} &= \operatorname{round}\left(\frac{\text{FP16\_val}}{\text{scale}}\right) \\
\text{FP16\_val} &\approx \text{INT8\_val} \times \text{scale}
\end{aligned}
$$

这里按一个量化组计算 scale；per-tensor、per-channel 或其他分组需要分别保存对应的 scale。公式假设有限输入且范围非零；实际实现还需明确舍入规则、饱和范围，并处理全零组。

### 非对称量化（Asymmetric）

$$
\begin{aligned}
\text{FP16\_val} \in [\beta, \gamma] &\;\to\; \text{INT8\_val} \in [-128, 127] \\
\text{scale} &= \frac{\gamma - \beta}{255} \\
\text{zero\_point} &= \operatorname{round}\left(\frac{-\beta}{\text{scale}}\right) - 128 \\
\text{INT8\_val} &= \operatorname{round}\left(\frac{\text{FP16\_val}}{\text{scale}}\right) + \text{zero\_point}
\end{aligned}
$$

需要保存 scale 与 zero-point。实际编码还要裁剪到合法整数范围，并处理常量组；校准范围通常应包含零，以便零点具有明确表示。是否适合采用非对称格式，还要考虑模型分布及 kernel 对 zero-point 的支持。

### 量化粒度
- **Per-tensor**：整个 tensor 一个 scale（最粗，精度最低）
- **Per-channel** (权重)：每个输出通道一个 scale（常用，精度和开销平衡好）
- **Per-token** (激活)：每个 token 一个 scale；它与 per-group/per-block 是不同维度，实际粒度要按张量布局和 kernel 定义。

## FP8 量化

FP8 方案在不同 GPU 代际和库中有不同支持；常见 recipe 使用两种编码：

| 格式 | 指数位 | 尾数位 | 范围 | 适用 |
|------|:---:|:---:|------|------|
| **E4M3** | 4 | 3 | ±448 | **前向**（精度优先） |
| **E5M2** | 5 | 2 | ±57344 | **梯度**（范围优先） |

FP8 通常不需要 zero-point（浮点编码自带符号），但仍需要 **scale** 把输入动态范围映射到 FP8 可表示范围。E4M3/E5M2 的前向或反向用途是常见配方，不是所有模型和库的硬性规定；应记录格式、累加类型与 scale 更新策略。

**动态 scale**：可以在当前张量、延迟的历史 amax，或块级统计上更新 scale。当前值更贴近输入但会增加同步/计算，延迟统计减少开销却要承担分布变化带来的溢出风险。

## 性能与精度如何测量

硬件峰值和权重文件大小只能给出上限或容量线索，不能直接推出模型吞吐。应固定模型、输入/输出长度、batch、设备、库版本和 warmup/repeat，分别测量量化/解包、GEMM、完整层、整网和请求级 TTFT/TPOT；同时记录 scale、临时 buffer、显存峰值和实际低精度指令路径。

矩阵维度和布局决定 Tensor Core 是否高效。NVIDIA 文档指出，INT8、FP16 等精度对关键维度有对齐偏好；实际收益还取决于量化开销是否被大矩阵工作摊薄。因此，同一格式在 Prefill 与 Decode、不同 batch 或不同形状上可能有相反结果。

精度评估也要按任务和校准流程固定条件。至少比较输出误差、困惑度或下游任务指标，并记录异常 token、长上下文和代表性激活；不能把某个模型的百分点差异推广到所有模型。

逐层 round-to-nearest 只是基线；AWQ 使用激活统计选择需要保护的权重通道，GPTQ 用近似二阶信息做逐层误差补偿，SmoothQuant 通过等价缩放把激活量化难度迁移到权重。三者的校准数据、粒度、位宽和 kernel 不同，质量与速度必须分别复测。

## 在 Ascend 的对应

跨设备迁移时可复用量化数学，但不能把 dtype 名称或转换 API 当作执行路径等价。需要重新核对 scale 的布局、累加精度、饱和/舍入规则、硬件指令和端到端测量。

## 与我何干

**C 线推理系统（算子线）**：vLLM / TensorRT-LLM 都支持 INT8/FP8 推理，你要知道：
- 量化权重怎么加载（`*.safetensors` 里存的是 INT8 + scale）
- kernel 怎么调（`cutlass::gemm<int8>` vs `<half>`）
- 精度怎么验证（对比 FP16 baseline 的困惑度）

**理论线下一步**：[AWQ](remaining-theory-primer.md#c3-awq)（权重量化算法，比 naive scale 更聪明）、[GPTQ](remaining-theory-primer.md#c2-gptq)（另一种权重量化）

**求职追问段落见下方独立小节。**

## 求职追问与示范回答

以下是教学示例，不代表个人实测：
- “INT8 量化怎么落到 kernel？”——先选对称或非对称映射与粒度，离线或运行时得到 scale/zero-point；kernel 读取编码值并以目标累加精度完成矩阵乘，再按输出 scale 还原。验证时要检查饱和、舍入、误差和生成代码，而不是只看权重文件变小。
- “PTQ 和 QAT 怎么选？”——PTQ 以已训练模型为起点，可包含校准、权重重构或等价变换；QAT 在训练中让模型适应量化误差，代价是训练或微调。选择取决于可用数据、部署格式和质量预算。
- “FP8 一定比 INT8 快吗？”——不一定。FP8 需要 scale 管理，INT8 需要整数映射；硬件、矩阵对齐、解包/转换和阶段形状决定实际吞吐，应以同条件 kernel 与端到端测量回答。

## 代码示例（PyTorch 伪代码）

```python
# 对称量化（per-tensor）
def quantize_int8(x_fp16):
    alpha = x_fp16.float().abs().max()
    # 零张量没有动态范围；用 1 作为无损的占位 scale，避免除零。
    scale = torch.where(alpha > 0, alpha / 127.0, torch.ones_like(alpha))
    x_int8 = torch.round(x_fp16.float() / scale).clamp(-127, 127).to(torch.int8)
    return x_int8, scale

def dequantize_int8(x_int8, scale):
    return (x_int8.float() * scale).to(torch.float16)

# 实际 GEMM 的简化流程（X、W 都先映射到整数，scale 另行保存）
X_int8, scale_x = quantize_int8(X_fp16)
W_int8, scale_w = quantize_int8(W_fp16)
Y_int32 = torch.matmul(X_int8.to(torch.int32), W_int8.to(torch.int32))
Y_fp16 = (Y_int32.float() * (scale_x * scale_w)).to(torch.float16)
```

这是量化数学的教学伪码，不代表 `torch.matmul` 已选择 native INT8 kernel。真实 CUDA/Tensor Core 路径会按布局、scale 粒度和库 API 打包输入，并在 INT32/更高精度累加后完成缩放。

## 扩展主题

- **KV Cache 量化**：Attention 的 KV Cache 也能量化（省显存），但要小心精度
- **混合精度**：敏感层保持 FP16，不敏感层 INT8（逐层校准）
- **动态量化 vs 静态量化**：动态是运行时算 scale（灵活但慢），静态是离线校准（快但需要代表性数据）

---

*下一步：[AWQ](remaining-theory-primer.md#c3-awq)（activation-aware 权重量化，解决"哪些层该保持高精度"）*
