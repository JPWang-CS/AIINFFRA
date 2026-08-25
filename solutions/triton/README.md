# Triton 实现

> 位置：这里保存用户的可执行版本和明确标记的 `WIP` 快照；当前单元的题目、代码快照和验收入口见 [Lesson 06](../../lessons/06-triton-intro.md)。
> 计划入口：[roadmap/ai-infra-curriculum.md](../../roadmap/ai-infra-curriculum.md) M2

## 当前进度

| 文件 | 功能 | LeetGPU 原始代码 | 验收 |
|------|------|------|------|
| [`vector_add.py`](./vector_add.py) | Triton Vector Add | **原始 LeetGPU `solve` 尚未单独归档**；当前文件是本地验证/benchmark wrapper | `GPU_VALIDATED`：LeetGPU 通过；AutoDL RTX 3090，Triton 840.1 GB/s，`torch.add` 843.0 GB/s |
| [`matmul.py`](./matmul.py) | Triton tiled GEMM | `WIP`：LeetGPU 编写中，尚未通过，暂无完成版 | 正确性、GFLOPS 待验收 |
| `fused_softmax.py` | Triton Fused Softmax | 正确性 + 提速 |
| `flash_attention.py` | Triton Flash Attention | 对比 PyTorch ref + 显存/速度 |
| `gqa.py` / `fused_mlp.py` | 模型结构组件 | 正确性 + autotune |

### MatMul LeetGPU 入口

当前 MatMul 先做 [LeetGPU Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication)，题目规格见 [02_Matrix_Multiplication.md](https://github.com/HaoyangPing0324/LeetGPU/blob/main/problems/02_Matrix_Multiplication.md)。要求是 FP32、row-major、A(M×N) × B(N×K) = C(M×K)，性能形状为 M=8192、N=6144、K=4096。LeetGPU 通过后，再把代码同步到本地，最后在真实 GPU 上记录 GFLOPS。

### Vector Add benchmark

```text
vector_add (2026-08-23)
- N=2^25, BLOCK_SIZE=256, RTX 3090 (AutoDL)
- 正确性: assert_close OK (N=1/256/257/1000/2^20)
- 带宽: 840.1 GB/s (torch.add: 843.0 GB/s)
- 结论: 内存带宽瓶颈，BLOCK_SIZE=256 最优；Triton 达到 torch.add 的 99.7%
```

> 说明：本次 benchmark 验证了 kernel 在 AutoDL 上的正确性和性能；代码归属与运行验证分开记录。

## 规则

1. 先看完原理，直接在 LeetGPU 题目编辑器从题目模板开始写，不直接复制 `reference/triton/`。
2. 两个验收章节：**LeetGPU 正确性与代码归档 → 服务器真实性能**；LeetGPU 未通过或原始代码未归档时，不进入服务器。
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
