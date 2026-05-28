# Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations

**Authors**: Tillet, Kung, Cox (Harvard)
**Venue**: MAPS@PLDI 2019 | **优先级**: P0 | **状态**: ✅ | **日期**: 2026-05-28

---

## 解决了什么问题

手写 CUDA kernel 太费劲 —— 每次都要操心 thread/warp 分配、shared memory bank conflict、memory coalescing、occupancy、register pressure。能否像写 for-loop 一样写 GPU kernel 同时不丢性能？

## 怎么解决的

**Block-level 编程模型**：程序员只指定每个 block 对 tile 的操作，编译器自动处理 thread 层细节。

1. **编程抽象**: 操作 tile（矩阵块）而非单个 thread
2. **自动优化**: 编译器自动做 shared memory promotion、coalescing、bank conflict avoidance
3. **Autotuning**: 网格搜索 BLOCK_SIZE / num_warps / num_stages 等超参
4. **编译器栈**: Triton IR → MLIR → LLVM IR → PTX → SASS（理论上可扩展非 NVIDIA 后端）

## 关键数据

| 指标 | PyTorch (cuBLAS) | Triton (autotuned) | 手写 CUDA (expert) |
|------|:---:|:---:|:---:|
| GEMM FP16 性能 | 1× (baseline) | **~0.95×** | ~1.05× |
| Softmax 性能 | 1× | **~1.2×** (fused) | - |
| 开发时间 | 短（调库） | **中** | 长（数天到数周） |
| 代码行数 | - | ~50-100 行 | ~200-500 行 |

## 与我何干

> Triton 之于 CUDA，就像 C 之于汇编。日常 90% 的 kernel 都应该用 Triton 写，只在需要 warp-level 精细控制时才回退到 CUDA。面试中 "Triton 的编程模型和 CUDA 有什么区别" 是高频题。
