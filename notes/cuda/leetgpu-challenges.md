# LeetGPU 题库索引

> 来源：https://github.com/AlphaGPU/leetgpu-challenges
> 平台：https://leetgpu.com
> 更新：2026-06-06

---

## 5 算子路线图 ↔ LeetGPU 映射

```
我们的算子        LeetGPU 题                         文件夹                   难度
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GEMM            General Matrix Multiplication     22_gemm                  Medium
  ├─ batched    Batched Matrix Multiplication     30_batched_matrix_mul    Medium
  ├─ FP16       FP16 Batched MatMul               57_fp16_batched_matmul   Medium
  ├─ INT8       INT8 Quantized MatMul             32_int8_quantized_matmul Medium
  └─ INT4       INT4 Weight-Only Quantized MatMul 81_int4_matmul           Medium
Softmax         Softmax                           5_softmax                Medium
  ├─ reduction  Reduction                         4_reduction              Medium
  └─ +attn      Softmax Attention                 6_softmax_attention      Medium
RMSNorm         RMS Normalization                 50_rms_normalization     Medium
  └─ batchnorm  Batch Normalization               40_batch_normalization   Medium
Flash Attn      Multi-Head Attention              12_multi_head_attention  Hard
  ├─ causal     Causal Self-Attention             53_casual_attention      Hard
  ├─ linear     Linear Self-Attention             56_linear_attention      Hard
  └─ sliding    Sliding Window Self-Attention     59_sliding_window_attn   Hard
GQA             Grouped Query Attention           80_grouped_query_attention Medium
```

---

## 核心 5 题详细规格

### 1. GEMM — `22_gemm`

```
签名: solve(half* A, half* B, half* C, int M, int N, int K, float alpha, float beta)
计算: C = α·(A×B) + β·C
精度: FP16 in, FP32 accumulate, FP16 out
约束: 16 ≤ M,N,K ≤ 4096
性能测试: M=N=K=1024
atol/rtol: 0.05
```

**Ascend 对照**：Cube Unit `mmad` 直接做矩阵乘。CUDA v0 阶段需手动 dot product，Tensor Core (`mma.sync`) 是 v4 的事。

### 2. Softmax — `5_softmax`

```
签名: solve(float* input, float* output, int N)
计算: softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max)
约束: 1 ≤ N ≤ 500,000
性能测试: N = 500,000
```

**关键**：必须用 max trick 防溢出。`[面试]` online softmax 是 Flash Attention 的前置。

### 3. RMSNorm — `50_rms_normalization`

```
签名: solve(float* input, float gamma, float beta, float* output, int N, float eps)
计算: rms = sqrt(mean(x²) + ε), y = γ·x/rms + β
约束: 1 ≤ N ≤ 100,000, ε = 1e-5
性能测试: N = 100,000
```

**关键**：两次遍历（一次算 rms，一次 normalize）。warp shuffle reduce 是优化核心。

### 4. Multi-Head Attention — `12_multi_head_attention` (Hard)

```
签名: solve(float* Q, float* K, float* V, float* output, int N, int d_model, int h)
计算: Concat(head_1,...,head_h), head_i = softmax(Q_i·K_i^T/√d_k)·V_i
       d_k = d_model / h
约束: 1 ≤ N ≤ 10,000, 2 ≤ d_model ≤ 1,024, d_model % h == 0
性能测试: N=1024, d_model=1024
```

### 5. GQA — `80_grouped_query_attention`

```
签名: solve(float* Q, float* K, float* V, float* output,
            int num_q_heads, int num_kv_heads, int seq_len, int head_dim)
计算: group_size = num_q_heads / num_kv_heads
      每个 group 内 query heads 共享 KV head
约束: num_kv_heads ≤ num_q_heads ≤ 64, num_q_heads % num_kv_heads == 0
      1 ≤ seq_len ≤ 4,096, 8 ≤ head_dim ≤ 256 (multiples of 8)
性能测试: num_q_heads=32, num_kv_heads=8, seq_len=1024, head_dim=128
```

---

## 全部题目索引（按难度+编号）

### Easy

| # | 文件夹 | 题目 | 签名关键参数 |
|---|--------|------|-------------|
| 1 | `1_vector_add` | Vector Addition | `(A, B, C, N)` |
| 2 | `2_matrix_multiplication` | Matrix Multiplication | `(A, B, C, M, N, K)` |
| 3 | `3_matrix_transpose` | Matrix Transpose | `(input, output, M, N)` |
| 7 | `7_color_inversion` | Color Inversion | |
| 8 | `8_matrix_addition` | Matrix Addition | `(A, B, C, N)` |
| 9 | `9_1d_convolution` | 1D Convolution | `(input, kernel, output, N, K)` |
| 19 | `19_reverse_array` | Reverse Array | `(A, N)` |
| 21 | `21_relu` | ReLU | `(A, N)` |
| 23 | `23_leaky_relu` | Leaky ReLU | |
| 24 | `24_rainbow_table` | Rainbow Table | |
| 31 | `31_matrix_copy` | Matrix Copy | `(A, B, N)` |
| 41 | `41_simple_inference` | Simple Inference | |
| 52 | `52_silu` | SiLU | |
| 54 | `54_swiglu` | SwiGLU | |
| 62 | `62_value_clipping` | Value Clipping | |
| 63 | `63_interleave` | Interleave Arrays | |
| 65 | `65_geglu` | GEGLU | |
| 66 | `66_rgb_to_grayscale` | RGB to Grayscale | |
| 68 | `68_sigmoid` | Sigmoid | |

### Medium

| # | 文件夹 | 题目 | 签名关键参数 |
|---|--------|------|-------------|
| 4 | `4_reduction` | Reduction | `(input, output, N)` |
| 5 | `5_softmax` | Softmax | `(input, output, N)` |
| 6 | `6_softmax_attention` | Softmax Attention | `(Q, K, V, output, N, d)` |
| 10 | `10_2d_convolution` | 2D Convolution | |
| 11 | `11_3d_convolution` | 3D Convolution | |
| 13 | `13_histogramming` | Histogramming | |
| 16 | `16_prefix_sum` | Prefix Sum | |
| 17 | `17_dot_product` | Dot Product | `(A, B, result, N)` |
| 18 | `18_sparse_matrix_vector_mul` | SpMV | |
| 22 | `22_gemm` | **GEMM (FP16)** | `(A, B, C, M, N, K, α, β)` |
| 25 | `25_categorical_crossentropy` | Categorical Cross Entropy | |
| 27 | `27_mean_squared_error` | MSE | |
| 28 | `28_gaussian_blur` | Gaussian Blur | |
| 29 | `29_top_k_selection` | Top K Selection | |
| 30 | `30_batched_matrix_mul` | Batched MatMul | `(A, B, C, BATCH, M, N, K)` |
| 32 | `32_int8_quantized_matmul` | INT8 Quantized MatMul | |
| 33 | `33_ordinary_least_squares` | OLS | |
| 34 | `34_logistic_regression` | Logistic Regression | |
| 35 | `35_monte_carlo_integration` | Monte Carlo | |
| 37 | `37_matrix_power` | Matrix Power | |
| 38 | `38_nearest_neighbor` | Nearest Neighbor | |
| 40 | `40_batch_normalization` | BatchNorm | |
| 42 | `42_2d_max_pooling` | 2D Max Pooling | |
| 43 | `43_count_array_element` | Count Array Element | |
| 44 | `44_count_2d_array_element` | Count 2D Array | |
| 45 | `45_count_3d_array_element` | Count 3D Array | |
| 47 | `47_subarray_sum` | Subarray Sum | |
| 48 | `48_2d_subarray_sum` | 2D Subarray Sum | |
| 49 | `49_3d_subarray_sum` | 3D Subarray Sum | |
| 50 | `50_rms_normalization` | **RMSNorm** | `(input, γ, β, output, N, ε)` |
| 51 | `51_max_subarray_sum` | Max Subarray Sum | |
| 55 | `55_attn_w_linear_bias` | ALiBi | |
| 57 | `57_fp16_batched_matmul` | FP16 Batched MatMul | |
| 58 | `58_fp16_dot_product` | FP16 Dot Product | |
| 60 | `60_top_p_sampling` | Top-p Sampling | |
| 61 | `61_rope_embedding` | **RoPE** | `(Q, cos, sin, output, M, D)` |
| 64 | `64_weight_dequantization` | Weight Dequant | |
| 67 | `67_moe_topk_gating` | MoE Top-K Gating | |
| 69 | `69_jacobi_stencil_2d` | 2D Jacobi Stencil | |
| 70 | `70_segmented_prefix_sum` | Segmented Prefix Sum | |
| 71 | `71_parallel_merge` | Parallel Merge | |
| 72 | `72_stream_compaction` | Stream Compaction | |
| 75 | `75_sparse_dense_matmul` | SpMM | |
| 76 | `76_adder_transformer` | Adder Transformer | |
| 78 | `78_2d_fft` | 2D FFT | |
| 80 | `80_grouped_query_attention` | **GQA** | `(Q,K,V,out,n_qh,n_kvh,seq,dh)` |
| 81 | `81_int4_matmul` | INT4 MatMul (W4A16) | |
| 82 | `82_linear_recurrence` | Linear Recurrence | |
| 84 | `84_swiglu_mlp_block` | **SwiGLU MLP** | `(x,W_g,W_u,W_d,out,M,dm,df)` |
| 85 | `85_lora_linear` | LoRA Linear | |
| 87 | `87_speculative_decoding` | Speculative Decoding | |
| 90 | `90_causal_depthwise_conv1d` | Causal Conv1D | |
| 92 | `92_decaying_causal_attention` | Decaying Causal Attn | |
| 94 | `94_ssm_selective_scan` | SSM Selective Scan (Mamba) | |
| 96 | `96_int8_kv_cache_attention` | INT8 KV-Cache Attn | |

### Hard

| # | 文件夹 | 题目 | 签名关键参数 |
|---|--------|------|-------------|
| 12 | `12_multi_head_attention` | **Multi-Head Attention** | `(Q,K,V,out,N,d_model,h)` |
| 14 | `14_multi_agent_sim` | Multi-Agent Simulation | |
| 15 | `15_sorting` | Sorting | |
| 20 | `20_kmeans_clustering` | K-Means | |
| 36 | `36_radix_sort` | Radix Sort | |
| 39 | `39_fast_fourier_transform` | FFT | |
| 46 | `46_bfs_shortest_path` | BFS Shortest Path | |
| 53 | `53_casual_attention` | **Causal Self-Attention** | `(Q,K,V,out,N,d_model,h)` |
| 56 | `56_linear_attention` | Linear Self-Attention | |
| 59 | `59_sliding_window_attn` | Sliding Window Attn | |
| 73 | `73_all_pairs_shortest_paths` | All-Pairs Shortest Paths | |
| 74 | `74_gpt2_block` | GPT-2 Transformer Block | |
| 93 | `93_llama_transformer_block` | **Llama Transformer Block** | `(x,out,weights,cos,sin,seq)` |

---

## 刷题路线（按难度/主题聚类）

> 不绑周次——这是题目地图，按你算子线进度随用随取。学习节奏看 [PATH.md](../../PATH.md) / [NOW.md](../../NOW.md)。

```
入门 (Easy)        1_vector_add → 8_matrix_addition → 2_matrix_multiplication
                   4_reduction, 17_dot_product, 16_prefix_sum (parallel pattern)

GEMM              2_matrix_multiplication (naive→tiled) → 22_gemm (Medium)
                   30_batched_matmul, 57_fp16_batched_matmul
                   ⭐ 22_gemm Tensor Core 版（可选深钻 → roadmap/leetgpu-ladder.md）

Softmax/采样       5_softmax → 6_softmax_attention
                   29_top_k_selection, 60_top_p_sampling

Norm              50_rms_normalization, 40_batch_normalization
                   61_rope_embedding (高频面试加分)

Attention         12_multi_head_attention (Hard), 53_casual_attention (Hard)
                   tiled + online softmax 版 → 完整 Flash Attn 自实现
                   56_linear_attention, 59_sliding_window_attn

GQA/MLP           80_grouped_query_attention, 84_swiglu_mlp_block (Llama FFN)

LLM 整块 (Bonus)   67_moe_topk_gating, 94_ssm_selective_scan
                   85_lora_linear, 87_speculative_decoding
                   74_gpt2_block, 93_llama_transformer_block
```

> Tensor Core / v2+ 优化层是 ⭐ 可选，见 [roadmap/leetgpu-ladder.md](../../roadmap/leetgpu-ladder.md)。

---

## 通用 starter 模板模式

所有 LeetGPU CUDA 题遵循统一模板：

```cpp
#include <cuda_runtime.h>

// kernel 函数 — 你来写
__global__ void my_kernel(...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: 你的实现
}

// solve 函数 — 签名不可改，LeetGPU 直接调用它
extern "C" void solve(/* 题目定义的参数 */) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    my_kernel<<<blocksPerGrid, threadsPerBlock>>>(...);
    cudaDeviceSynchronize();
}
```

**关键规则**：
- `solve` 是入口，签名不能改
- 所有指针参数都是 device pointer（已分配好）
- 不用 `cudaMalloc`/`cudaFree` — 平台管理显存
- 禁止外部库（WMMA 例外，GEMM v4 可用）
- 结果写入 output/C 指针即可
