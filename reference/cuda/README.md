# Reference — CUDA Kernels（代码库）

> **参考实现，看不抄。** 这是独立代码库,有自己的索引(就是本页)。
> 学的时候对着 [lessons/](../../lessons/) 从空文件自己写，写完跑通存进 [solutions/](../../solutions/)。
> 笔记速查在 [notes/cuda/](../../notes/cuda/)，学习路径在 [PATH.md](../../PATH.md)。

## 算子实现

| 文件 | 算子 | 要点 | 对应课 |
|------|------|------|--------|
| [vector_add.cu](vector_add.cu) | Vector Add | grid-stride loop、cudaEvent 计时、CPU 校验 | [01](../../lessons/01-cuda-basics.md) |
| [gemm/gemm.cu](gemm/gemm.cu) | GEMM | naive + tiled（TILE=32），GFLOPS 输出 | [02](../../lessons/02-gemm-naive.md)·[03](../../lessons/03-gemm-tiled.md) |
| [softmax/softmax.cu](softmax/softmax.cu) | Softmax | 3-pass naive + online，warp/block reduce | [04](../../lessons/04-softmax.md) |
| [layernorm/layernorm.cu](layernorm/layernorm.cu) | LayerNorm | warp reduce 求 mean/var，gamma/beta | ⭐ bonus（理论线 Norm） |
| [flash_attention/flash_attn.cu](flash_attention/flash_attn.cu) | Flash Attention | BR=BC=32，online softmax，causal | [05](../../lessons/05-flash-attn-reading.md) |

## 工具 / 头文件

| 文件 | 内容 |
|------|------|
| [include/cuda_utils.h](include/cuda_utils.h) · [.cu](include/cuda_utils.cu) | `CUDA_CHECK` 宏 + `random_fill` / `compare_arrays` |
| [include/activations.cuh](include/activations.cuh) | relu/sigmoid/silu/gelu/swiglu/geglu（料，给未来 fused MLP 用） |
| [CMakeLists.txt](CMakeLists.txt) | 建 5 个目标，都链 `kernel_utils` |

## 编译

```bash
cd reference/cuda && mkdir -p build && cd build
cmake .. && make
./gemm    # 跑某个算子
```

> LeetGPU 不需要这套（在线跑）。本地有 GPU 时用 CMake 编译对照。

## 说明

- **LayerNorm / activations 是 bonus**：现阶段 Triton-first 路径用不到（RMSNorm 被砍、fused MLP 还早），留着当料。要学时理论线会链过来。
- GEMM 参考用的是 `C = A × Bᵀ` 布局，和 LeetGPU `2_matrix_multiplication`（`C = A × B`）不同——对照时注意。
