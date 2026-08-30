# Triton 实现

> 位置：这里保存用户的可执行版本和明确标记的 `WIP` 快照；当前单元的题目、代码快照和验收入口见 [Lesson 06](../../lessons/06-triton-intro.md)。
> 计划入口：[roadmap/ai-infra-curriculum.md](../../roadmap/ai-infra-curriculum.md) M2

## 当前进度

| 文件 | 功能 | LeetGPU 原始代码 | 验收 |
|------|------|------|------|
| [`vector_add.py`](./vector_add.py) | Triton Vector Add | **原始 LeetGPU `solve` 尚未单独归档**；当前文件是本地验证/benchmark wrapper | `GPU_VALIDATED`：LeetGPU 通过；AutoDL RTX 3090，Triton 840.1 GB/s，`torch.add` 843.0 GB/s |
| [`matmul_leetgpu.py`](./matmul_leetgpu.py) | Triton tiled GEMM：LeetGPU 最终原始 `solve`/kernel 归档 | `LEETGPU_PASS`：LeetGPU #02，Triton，2026-08-28；SuccessPublicTrace | A100-80GB，24.54 ms，55.3th percentile |
| [`matmul_leetgpu_wip.py`](./matmul_leetgpu_wip.py) | Triton tiled GEMM：历史平台代码快照 | 历史 `WIP`：默认 TF32 精度失败案例，保留用于复盘 | 4×4 case 最大绝对误差 `0.1275177001953125`；IEEE 版本后平台通过 |
| [`matmul.py`](./matmul.py) | Triton tiled GEMM：服务器验证版 | 基于 MatMul 逻辑的本地适配，不是平台原始归档 | `GPU_VALIDATED` baseline：RTX 3090 正确性通过；s3/s2 Nsight Systems P0-lite 已完成；P0–P8 deferred |
| `fused_softmax.py` | Triton Fused Softmax | 正确性 + 提速 |
| `flash_attention.py` | Triton Flash Attention | 对比 PyTorch ref + 显存/速度 |
| `gqa.py` / `fused_mlp.py` | 模型结构组件 | 正确性 + autotune |

### MatMul LeetGPU 入口

当前 MatMul 题目是 [LeetGPU Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication)，题目规格见 [02_Matrix_Multiplication.md](https://github.com/HaoyangPing0324/LeetGPU/blob/main/problems/02_Matrix_Multiplication.md)。要求是 FP32、row-major、A(M×N) × B(N×K) = C(M×K)，性能形状为 M=8192、N=6144、K=4096。平台原始代码已归档为 [`matmul_leetgpu.py`](./matmul_leetgpu.py)，服务器适配版继续独立记录真实性能。

LeetGPU 最终代码与服务器适配版分开保存：[`matmul_leetgpu.py`](./matmul_leetgpu.py) 是通过后原样归档的原始平台 `solve`/kernel；[`matmul.py`](./matmul.py) 是服务器正确性和性能验证版。历史 [`matmul_leetgpu_wip.py`](./matmul_leetgpu_wip.py) 保留默认 TF32 导致精度失败的过程证据。

### MatMul LeetGPU 归档结果

```text
题目: #02 Matrix Multiplication
语言: Triton
状态: LEETGPU_PASS
SuccessPublicTrace: A100-80GB，2026-08-28 22:23:16，24.54 ms，55.3th percentile
精度修正: tl.dot(..., input_precision='ieee')
失败证据: 历史 WIP 默认 TF32 在 4×4 case 的最大绝对误差为 0.1275177001953125
```

`matmul_leetgpu.py` 以 `matmul_leetgpu_wip.py` 为精确基线，仅增加归档说明，并将 `tl.dot(tile_a, tile_b)` 指定为 IEEE 输入精度；kernel 和 `solve` 逻辑没有其他变化。

### Vector Add benchmark

```text
vector_add (2026-08-23)
- N=2^25, BLOCK_SIZE=256, RTX 3090 (AutoDL)
- 正确性: assert_close OK (N=1/256/257/1000/2^20)
- 带宽: 840.1 GB/s (torch.add: 843.0 GB/s)
- 结论: 内存带宽瓶颈，BLOCK_SIZE=256 最优；Triton 达到 torch.add 的 99.7%
```

> 说明：本次 benchmark 验证了 kernel 在 AutoDL 上的正确性和性能；代码归属与运行验证分开记录。

### MatMul benchmark

```text
matmul (2026-08-26)
- GPU: NVIDIA GeForce RTX 3090 (AutoDL)
- shape: M=8192, N=6144, K=4096
- 正确性: M/N/K = (1,1,1), (64,32,64), (65,33,67), (257,513,129) 全部 OK
- 精度口径: FP32，tl.dot(input_precision="ieee")，PyTorch allow_tf32=False
- Triton: 24.681 ms，16,706.0 GFLOPS
- torch.mm: 17.120 ms，24,083.3 GFLOPS
- 对比: Triton 为 torch.mm 的约 69.4%，耗时约慢 1.44x
- 结论: 服务器版正确性通过；初始 tile/config 仍有调优空间
```

> 这组数字属于服务器适配版，状态为 `GPU_VALIDATED`；LeetGPU 原始版本已另行归档为 `LEETGPU_PASS`。MatMul 当前按 baseline 阶段性收口；P0–P8 极致优化延期至 [GPU 优化篇](../../roadmap/gpu-foundations.md#matmul-优化债务池-deferred-backlog)，不改变本页既有 benchmark 数字。

### MatMul 配置 sweep（2026-08-26）

固定 IEEE FP32、GPU 和 shape，只比较 tile、warp、stage：

| 配置 | 结果 | 相对 `torch.mm` |
|---|---:|---:|
| `64×32×64, w4, s3` | 24.924 ms / 16,542.7 GFLOPS | 68.8% |
| `128×32×64, w4, s3` | 22.298 ms / 18,491.3 GFLOPS | 76.9% |
| `128×32×128, w4, s3` | 28.354 ms / 14,541.8 GFLOPS | 60.4% |
| `128×64×128, w4, s3` | 编译失败：shared memory 需要 131,072 B，硬件上限 101,376 B | — |
| `128×32×256, w8, s3` | **22.033 ms / 18,713.5 GFLOPS** | **77.8%** |
| `128×32×256, w8, s2` | Nsight Systems 60 次 mean：22.362 ms / 18,438.6 GFLOPS | 同次 CUTLASS 的 74.6% |

结论：`BLOCK_M=128, BLOCK_N=32, BLOCK_K=256, num_warps=8, num_stages=3` 是当前测试集最佳配置；相对 baseline 64×32×64，耗时下降约 11.6%，GFLOPS 提升约 13.1%。`BLOCK_N=64` 的配置因 shared memory 超限，说明 tile 和 stages 不能只看算力，还要受片上资源约束。

详细硬件机制、profiler 观察项和后续 P0–P4 优化顺序见：[MatMul 性能分析记录](../../notes/triton/matmul-performance-analysis.md)。

### MatMul Nsight Systems P0-lite（2026-08-30）

AutoDL 禁止 NCU hardware counters，因此使用 Nsight Systems 对同一 `128×32×256, w8` 配置只改变 `num_stages`：

| 配置 | Triton mean / median（60 次大 shape） | Reg/Trd | DymSMem | 同次 CUTLASS mean | 结论 |
|---|---:|---:|---:|---:|---|
| s3 | 21.208 / 21.163 ms | 255 | 0.098 MB | 16.716 ms | 当前更快 |
| s2 | 22.362 / 22.317 ms | 255 | 0.049 MB | 16.682 ms | shared memory 减半但慢 5.44% |

s2 没有降低 Reg/Trd，因此静态寄存器约束仍近似为 1 block/SM；减少 pipeline stage 没有换来驻留改善，反而损失延迟隐藏。完整分析与原始证据：[P0-lite 分析](../../notes/triton/matmul-nsys-p0-lite-2026-08-30.md) · [s3 raw](../../notes/triton/logs/2026-08-29-matmul-k256-s3-nsys.txt) · [s2 raw](../../notes/triton/logs/2026-08-30-matmul-k256-s2-nsys.txt)。

MatMul 当前 baseline 出口：RTX 3090 最佳 20.830 ms / 19,794.1 GFLOPS / `torch.mm` 80.3%。剩余 NCU counters、PTX/SASS、spill/occupancy、多 shape 回归和完整 P0–P8 闭环已延期至 GPU 优化篇。

## 规则

1. 先看完原理，直接在 LeetGPU 题目编辑器从题目模板开始写，不直接复制 `reference/triton/`。
2. 两个验收章节：**LeetGPU 正确性与代码归档 → 服务器真实性能**；默认先过 LeetGPU，平台不可运行时允许用独立服务器适配版验证，但必须分开记录状态。
3. 每个文件开头写一行说明：算子、版本、关键优化。
4. 跑不通、没有数字的文件不要标完成；Agent 草稿一律不算完成。

## 验证命令

```bash
python vector_add.py
python matmul.py
python fused_softmax.py
python flash_attention.py
```

> LeetGPU 在线判题优先（支持 Triton）；本机无 GPU 时解释器只用来验正确性，不产出性能结论。

无 GPU 时可用（脚本检测不到 CUDA 会自动切）：

```bash
TRITON_INTERPRET=1 python xxx.py
```

本机 Windows 无 GPU 环境（2026-08-10 已搭好）：

```powershell
py -m venv .venv   # 用 Python 3.12
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install triton-windows numpy --index-url https://pypi.org/simple
.venv\Scripts\python.exe vector_add.py
```

## 参考

- [Triton 语法速查](../../notes/triton/triton-cheatsheet.md)
- [reference/triton](../../reference/triton/)
- [Lesson 06](../../lessons/06-triton-intro.md)
