# 可选 CUDA 深钻菜单

> B 级够用不用看这页。想要面试强信号、或想把某个算子钻到峰值再回来。
> 基础层（naive / tiled / online / warp-reduce）和进度都在 [PATH.md](../PATH.md) 算子线。
> LeetGPU 题号用作 baseline/正确性验证。完整题库 → [notes/cuda/leetgpu-challenges.md](../notes/cuda/leetgpu-challenges.md)

---

## 怎么用

1. 每次只选一个算子。
2. 在 B 级版本上逐步加一层优化。
3. 每层都要有 correctness check + 性能数字。
4. 跑不过、没数字就不算完成。

### 完成标准

- [ ] 代码在 `solutions/`
- [ ] 正确性和 reference 对齐
- [ ] 有 GFLOPS 或 GB/s
- [ ] 能说出这一层优化省了什么
- [ ] 能用 Nsight 看到瓶颈变化

### 示例：GEMM vec4

```text
B 级：float 标量 load
进阶：float4 一次读 4 个 float
验证：看 kernel 指令数和带宽是否改善
```

---

## GEMM 进阶

> 基础 naive/tiled 见 PATH A2/A3。下面是 tiled 之后往峰值钻的层：

| 层 | 技术要点 | 验证 |
|------|---------|------|
| vec4 | `float4` 128-bit vectorized load，减少访存指令 | |
| double buffer | 双 shared memory buffer，copy-compute overlap（≈ Ascend pipe 流水） | |
| tensor core | `mma.sync`，FP16→FP32，warp-level matrix fragment | `22_gemm` / `57_fp16_batched_matmul` |

**目标**：tensor core GEMM 达硬件峰值 70%+

## Softmax 进阶

> 基础 3-pass/online/warp-reduce 见 PATH A4。

| 层 | 技术要点 |
|------|---------|
| fused | softmax 结果直接喂下一步（如 attention 内），不写回 HBM |

**目标**：online softmax 的 bandwidth utilization > 80%

## Norm（LayerNorm / RMSNorm）

> ⭐ 整个 Norm 在 Triton-first 路径里是 bonus（现实 LLM 用 RMSNorm）。料：[reference/cuda/layernorm/layernorm.cu](../reference/cuda/layernorm/layernorm.cu)

| 层 | 技术要点 | 验证 |
|------|---------|------|
| warp reduce | warp shuffle + shared memory cross-warp reduce | `50_rms_normalization` |
| fused residual | RMSNorm(x) + x 一次完成，省一次 memory round-trip | |

**目标**：融合版比 PyTorch 原生快 2×+

## Flash Attention 进阶

> 基础（读懂 tiled + online softmax）见 PATH A5。下面是手写到能跑、再往上：

| 层 | 技术要点 | 验证 |
|------|---------|------|
| causal | causal mask 跳过无效 tile | `53_casual_attention` |
| multi-head | B×H×N×d，batch + head 维并行 | `12_multi_head_attention` |

**目标**：seq_len=4096 比 naive attention 快 5×+，显存降至 O(N)

## GQA Attention

| 层 | 技术要点 | 验证 |
|------|---------|------|
| GQA | 从 MHA 扩展，K/V head 数 < Q head 数 | `80_grouped_query_attention` |

> GQA 的机制理解（非手写）在理论线"注意力演进"——这里是想手写优化时的进阶。

---

*这是静态的进阶清单，不追踪进度（进度在 [PATH.md](../PATH.md)）。手写这些任意一层，对面试都是强信号。*