# LeetGPU 刷题路线

> [leetgpu.com](https://leetgpu.com) — 浏览器内写 CUDA/Triton kernel，在线编译运行，无需 GPU
> 共 75 题：Easy 19 | Medium 44 | Hard 12

---

## Phase 1: CUDA 基础 (Week 1-10) — 必刷 18 题

### W1-2 GPU 架构 + 编程模型
| # | 题目 | 难度 | 学什么 |
|---|------|------|--------|
| 1 | 1_vector_add | Easy | grid/block/thread 模型，第一个 kernel |
| 2 | 31_matrix_copy | Easy | global memory 读写，coalescing 概念 |
| 3 | 19_reverse_array | Easy | stride access pattern |

### W3-4 内存管理
| # | 题目 | 难度 | 学什么 |
|---|------|------|--------|
| 4 | 3_matrix_transpose | Easy | shared memory + bank conflict |
| 5 | 8_matrix_addition | Easy | 二维 grid 组织 |
| 6 | 9_1d_convolution | Easy | shared memory tiling + halo region |

### W5-6 经典 ML 算子
| # | 题目 | 难度 | 学什么 |
|---|------|------|--------|
| 7 | 2_matrix_multiplication | Easy | naive GEMM → tiled GEMM |
| 8 | 21_relu | Easy | element-wise kernel 模板 |
| 9 | 68_sigmoid | Easy | 数学函数在 GPU 上的使用 |
| 10 | 52_silu | Easy | 融合算子思想 |
| 11 | 54_swiglu | Easy | 多输入融合 |
| 12 | 65_geglu | Easy | GELU 变体对比 |

### W7-8 Reduce + Softmax
| # | 题目 | 难度 | 学什么 |
|---|------|------|--------|
| 13 | 4_reduction | Medium | warp shuffle + shared memory 多级归约 |
| 14 | 5_softmax | Medium | online softmax：max-subtract + exp + sum-reduce |
| 15 | 50_rms_normalization | Medium | RMSNorm kernel，大模型标配 |
| 16 | 40_batch_normalization | Medium | BatchNorm 的并行化策略 |

### W9-10 GEMM 深入 + Profiling
| # | 题目 | 难度 | 学什么 |
|---|------|------|--------|
| 17 | 22_gemm | Medium | FP32 tiled GEMM，shared memory double buffering |
| 18 | 30_batched_matrix_multiplication | Medium | batched GEMM，多 batch 并行策略 |

---

## Phase 2: Triton + Attention (Week 11-18) — 必刷 16 题

### W11-12 Triton 入门
| # | 题目 | 难度 | 学什么 |
|---|------|------|--------|
| 1 | 1_vector_add (Triton) | Easy | block-level 编程模型 vs CUDA |
| 2 | 2_matrix_multiplication (Triton) | Easy | Triton 的 tiling 表示法 |
| 3 | 21_relu (Triton) | Easy | element-wise 的 Triton 写法 |

### W13-14 融合算子 + Attention 基础
| # | 题目 | 难度 | 学什么 |
|---|------|------|--------|
| 4 | 5_softmax (Triton) | Medium | online softmax + fused scale |
| 5 | 6_softmax_attention | Medium | naive attention forward |
| 6 | 86_transposed_softmax | Medium | 不同维度 softmax，理解 layout 影响 |
| 7 | 61_rope_embedding | Medium | RoPE 旋转编码，LLaMA 标配 |

### W15-16 Attention 全线
| # | 题目 | 难度 | 学什么 |
|---|------|------|--------|
| 8 | 55_attn_w_linear_bias | Medium | attention + bias 融合 |
| 9 | 80_grouped_query_attention | Medium | GQA — vLLM 默认使用的 attention 模式 |
| 10 | 12_multi_head_attention | **Hard** | **完整 MHA 前向+反向** |
| 11 | 53_casual_attention | Hard | causal mask 优化 |

### W17-18 前沿 Attention 变体
| # | 题目 | 难度 | 学什么 |
|---|------|------|--------|
| 12 | 59_sliding_window_attn | Hard | Mistral 同款滑动窗口 |
| 13 | 56_linear_attention | Hard | linear attention 原理 |
| 14 | 32_int8_quantized_matmul | Medium | INT8 量化 GEMM |
| 15 | 81_int4_matmul | Medium | INT4 量化 GEMM |
| 16 | 64_weight_dequantization | Medium | 反量化 kernel |

---

## Phase 3: 推理系统相关 (Week 19-30) — 选刷 8 题

vLLM 源码学习期间，按需回刷这些题加深理解：

| # | 题目 | 难度 | 与 vLLM 的关联 |
|---|------|------|---------------|
| 1 | 29_top_k_selection | Medium | top-k sampling |
| 2 | 60_top_p_sampling | Medium | nucleus sampling |
| 3 | 67_moe_topk_gating | Medium | MoE 的 top-k gating |
| 4 | 17_dot_product | Medium | attention score 计算 |
| 5 | 85_lora_linear | Medium | LoRA 推理时的 fused kernel |
| 6 | 82_linear_recurrence | Medium | Mamba/SSM 类模型基础 |
| 7 | 74_gpt2_block | **Hard** | 完整 Transformer block → 微型推理引擎 |
| 8 | 41_simple_inference | Easy | 最简推理 pipeline 理解 |

---

## 题目完成追踪

状态：⏳ 待做 | 🚧 进行中 | ✅ 完成

### Phase 1 进度
| # | 题目 | LeetGPU ID | 难度 | 状态 | 日期 | 耗时 |
|---|------|-----------|------|------|------|------|
| 1 | Vector Add | 1_vector_add | Easy | ⏳ | - | - |
| 2 | Matrix Copy | 31_matrix_copy | Easy | ⏳ | - | - |
| 3 | Reverse Array | 19_reverse_array | Easy | ⏳ | - | - |
| 4 | Matrix Transpose | 3_matrix_transpose | Easy | ⏳ | - | - |
| 5 | Matrix Addition | 8_matrix_addition | Easy | ⏳ | - | - |
| 6 | 1D Convolution | 9_1d_convolution | Easy | ⏳ | - | - |
| 7 | Matrix Multiplication | 2_matrix_multiplication | Easy | ⏳ | - | - |
| 8 | ReLU | 21_relu | Easy | ⏳ | - | - |
| 9 | Sigmoid | 68_sigmoid | Easy | ⏳ | - | - |
| 10 | SiLU | 52_silu | Easy | ⏳ | - | - |
| 11 | SwiGLU | 54_swiglu | Easy | ⏳ | - | - |
| 12 | GeGLU | 65_geglu | Easy | ⏳ | - | - |
| 13 | Reduction | 4_reduction | Medium | ⏳ | - | - |
| 14 | Softmax | 5_softmax | Medium | ⏳ | - | - |
| 15 | RMS Normalization | 50_rms_normalization | Medium | ⏳ | - | - |
| 16 | Batch Normalization | 40_batch_normalization | Medium | ⏳ | - | - |
| 17 | GEMM | 22_gemm | Medium | ⏳ | - | - |
| 18 | Batched Matmul | 30_batched_matrix_multiplication | Medium | ⏳ | - | - |

### Phase 2 进度
| # | 题目 | LeetGPU ID | 难度 | 状态 | 日期 | 耗时 |
|---|------|-----------|------|------|------|------|
| 1-3 | Triton 入门三题 | 1_vector_add, 2_matmul, 21_relu | Easy | ⏳ | - | - |
| 4 | Softmax (Triton) | 5_softmax | Medium | ⏳ | - | - |
| 5 | Softmax Attention | 6_softmax_attention | Medium | ⏳ | - | - |
| 6 | Transposed Softmax | 86_transposed_softmax | Medium | ⏳ | - | - |
| 7 | RoPE Embedding | 61_rope_embedding | Medium | ⏳ | - | - |
| 8 | Attn w/ Linear Bias | 55_attn_w_linear_bias | Medium | ⏳ | - | - |
| 9 | Grouped Query Attn | 80_grouped_query_attention | Medium | ⏳ | - | - |
| 10 | Multi-Head Attention | 12_multi_head_attention | Hard | ⏳ | - | - |
| 11 | Causal Attention | 53_casual_attention | Hard | ⏳ | - | - |
| 12 | Sliding Window Attn | 59_sliding_window_attn | Hard | ⏳ | - | - |
| 13 | Linear Attention | 56_linear_attention | Hard | ⏳ | - | - |
| 14 | INT8 Quantized Matmul | 32_int8_quantized_matmul | Medium | ⏳ | - | - |
| 15 | INT4 Matmul | 81_int4_matmul | Medium | ⏳ | - | - |
| 16 | Weight Dequantization | 64_weight_dequantization | Medium | ⏳ | - | - |

### Phase 3 进度
| # | 题目 | LeetGPU ID | 难度 | 状态 | 日期 | 耗时 |
|---|------|-----------|------|------|------|------|
| 1 | Top-K Selection | 29_top_k_selection | Medium | ⏳ | - | - |
| 2 | Top-P Sampling | 60_top_p_sampling | Medium | ⏳ | - | - |
| 3 | MoE TopK Gating | 67_moe_topk_gating | Medium | ⏳ | - | - |
| 4 | Dot Product | 17_dot_product | Medium | ⏳ | - | - |
| 5 | LoRA Linear | 85_lora_linear | Medium | ⏳ | - | - |
| 6 | Linear Recurrence | 82_linear_recurrence | Medium | ⏳ | - | - |
| 7 | GPT-2 Block | 74_gpt2_block | Hard | ⏳ | - | - |
| 8 | Simple Inference | 41_simple_inference | Easy | ⏳ | - | - |
