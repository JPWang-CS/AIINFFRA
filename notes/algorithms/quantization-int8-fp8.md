# 数值格式：INT8 / FP8 量化基础

> 量化类 · 第一条 · 推理加速的基础

---

## 解决了什么问题

LLM 推理慢、显存占用大：
- **FP16 模型**：7B 参数 ≈ 14GB 显存，单卡放不下；计算 TFLOPS 受限于 FP16/BF16 吞吐
- **INT8/FP8**：显存砍一半，吞吐提升 2-4×（Tensor Core 对低精度有硬件加速）

量化就是把高精度（FP16/FP32）的权重和激活压缩成低精度（INT8/FP8），在精度损失可控的前提下加速。

## 数值格式对比

| 格式 | 位宽 | 表示范围 | 精度（相对 FP16） | 硬件支持 |
|------|:---:|---------|:---:|---------|
| FP32 | 32-bit | ±3.4×10³⁸ | 1× (baseline) | 所有 GPU |
| FP16 | 16-bit | ±65504 | ~0.5× | Volta+ Tensor Core |
| **BF16** | 16-bit | ±3.4×10³⁸ (同 FP32) | ~0.5× | Ampere+ |
| **INT8** | 8-bit | -128 ~ 127 (有符号) | ~0.1× | Turing+ (INT8 Tensor Core) |
| **FP8 (E4M3)** | 8-bit | ±448 | ~0.2× | Hopper+ (H100) |
| **FP8 (E5M2)** | 8-bit | ±57344 | ~0.15× | Hopper+ |

**关键区别**：
- **FP16 vs BF16**：FP16 精度高但容易溢出；BF16 范围大（和 FP32 一样）但精度略低。训练一般用 BF16，推理看情况。
- **INT8 vs FP8**：INT8 是整数（需要 scale/zero-point 做映射），FP8 是浮点（天然有指数位，表示范围更灵活）。H100 有原生 FP8 Tensor Core，更快。

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

只需要存一个 **scale**（每层或每通道一个）。

### 非对称量化（Asymmetric）

$$
\begin{aligned}
\text{FP16\_val} \in [\beta, \gamma] &\;\to\; \text{INT8\_val} \in [-128, 127] \\
\text{scale} &= \frac{\gamma - \beta}{255} \\
\text{zero\_point} &= \operatorname{round}\left(\frac{-\beta}{\text{scale}}\right) - 128 \\
\text{INT8\_val} &= \operatorname{round}\left(\frac{\text{FP16\_val}}{\text{scale}}\right) + \text{zero\_point}
\end{aligned}
$$

需要存 **scale + zero_point**。适合分布不对称的激活（如 ReLU 后全是正数）。

### 量化粒度
- **Per-tensor**：整个 tensor 一个 scale（最粗，精度最低）
- **Per-channel** (权重)：每个输出通道一个 scale（常用，精度和开销平衡好）
- **Per-token** (激活)：每个 token 一个 scale（最细，精度高但开销大）

## FP8 量化

H100 引入两种 FP8 格式：

| 格式 | 指数位 | 尾数位 | 范围 | 适用 |
|------|:---:|:---:|------|------|
| **E4M3** | 4 | 3 | ±448 | **前向**（精度优先） |
| **E5M2** | 5 | 2 | ±57344 | **梯度**（范围优先） |

FP8 不需要 zero-point（浮点天然有符号），只需要 **scale**（把 FP16 范围映射到 FP8）。

**动态 scale**：训练时每步根据激活/梯度的 max 动态调 scale，避免溢出。

## 性能数据（A100 vs H100）

| 配置 | TFLOPS (Llama-7B) | 显存 (7B) | 吞吐 (tokens/s) |
|------|:---:|:---:|:---:|
| FP16 | 312 | 14 GB | 100 |
| INT8 (A100) | ~600 | **7 GB** | **180** |
| FP8 (H100) | ~1000 | **7 GB** | **300** |

**INT8 vs FP8**：H100 的 FP8 Tensor Core 原生支持，比 INT8 (通过 DP4A 模拟) 更快。A100 只有 INT8 Tensor Core。

## 量化的精度损失

| 任务 | FP16 acc | INT8 (naive) | INT8 (校准后) |
|------|:---:|:---:|:---:|
| BERT (GLUE) | 85.2 | 82.1 ❌ | **84.9** ✅ |
| Llama-7B (困惑度) | 5.68 | 6.2 ❌ | **5.71** ✅ |

**Naive 量化**（直接 scale）会掉点。**校准/算法量化**（AWQ、GPTQ、SmoothQuant）能恢复到接近 FP16。

## 在 Ascend 的对应

昇腾的 `int8` / `fp16` 类型和 CUDA 一样。Ascend C 的 `Cast` / `Quantize` intrinsic 就是做格式转换。

区别：CUDA 要手写 scale 计算和 round，Ascend 有 `Quantize` 指令一步到位（但底层逻辑一样）。

## 与我何干

**C 线推理系统（算子线）**：vLLM / TensorRT-LLM 都支持 INT8/FP8 推理，你要知道：
- 量化权重怎么加载（`*.safetensors` 里存的是 INT8 + scale）
- kernel 怎么调（`cutlass::gemm<int8>` vs `<half>`）
- 精度怎么验证（对比 FP16 baseline 的困惑度）

**理论线下一步**：[AWQ](awq.md)（权重量化算法，比 naive scale 更聪明）、[GPTQ](gptq.md)（另一种权重量化）

**[面试]** 高频题：
- "INT8 量化怎么做？" → 对称/非对称、per-channel scale
- "为什么 INT8 能加速？" → Tensor Core 硬件支持 + 显存带宽省一半
- "FP8 和 INT8 区别？" → FP8 是浮点（有指数），H100 原生支持更快
- "量化会掉多少点？" → naive 掉 1-2%，校准后 <0.5%

## 代码示例（PyTorch 伪代码）

```python
# 对称量化（per-tensor）
def quantize_int8(x_fp16):
    alpha = x_fp16.abs().max()
    scale = alpha / 127.0
    x_int8 = torch.round(x_fp16 / scale).clamp(-128, 127).to(torch.int8)
    return x_int8, scale

def dequantize_int8(x_int8, scale):
    return x_int8.to(torch.float16) * scale

# 实际 GEMM: Y = X @ W (X 激活 FP16, W 权重 INT8)
W_int8, scale_w = quantize_int8(W_fp16)
Y_int32 = torch.matmul(X_fp16, W_int8.to(torch.int32))  # INT8 GEMM
Y_fp16 = Y_int32.to(torch.float16) * scale_w
```

CUDA kernel 里就是把这个逻辑融合进 GEMM（Tensor Core 直接算 INT8 矩阵乘，输出 INT32 累加器）。

## 扩展主题

- **KV Cache 量化**：Attention 的 KV Cache 也能量化（省显存），但要小心精度
- **混合精度**：敏感层保持 FP16，不敏感层 INT8（逐层校准）
- **动态量化 vs 静态量化**：动态是运行时算 scale（灵活但慢），静态是离线校准（快但需要代表性数据）

---

*下一步：[AWQ](awq.md)（activation-aware 权重量化，解决"哪些层该保持高精度"）*
