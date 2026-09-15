#include "../include/cuda_utils.h"
#include <cstdio>
#include <cmath>

// ============================================================
// LayerNorm Reference
//
// 【算子是什么】y = (x - mean) / sqrt(var + eps) * gamma + beta
//   沿 D 维度归一化每行，保持均值=0 方差=1，再 affine transform
// 【在模型里干嘛】每个 Transformer block 两次：Attention 前 + FFN 前
//   - 稳定训练：防止激活值漂移/爆炸
//   - LLaMA 用 RMSNorm（去掉 mean 减法），更快，效果接近
// 【什么模型用】GPT (LayerNorm)、LLaMA (RMSNorm)、BERT (LayerNorm)
//   几乎所有 Transformer 架构都有归一化层（只是 Norm 类型不同）
// ============================================================

// Step 1: compute mean and variance per row (warp-level reduce)
__global__ void layernorm_forward(const float *input, const float *gamma,
                                  const float *beta, float *output,
                                  int B, int D, float eps) {
  int row = blockIdx.x;
  if (row >= B) return;

  int tid = threadIdx.x;
  int stride = blockDim.x;
  extern __shared__ float s_buf[]; // [D] for temp storage, flexible usage

  // --- Compute mean ---
  float mean = 0.0f;
  for (int j = tid; j < D; j += stride) {
    mean += input[row * D + j];
  }

  // Warp reduce sum → block reduce
  for (int offset = 16; offset > 0; offset /= 2) {
    mean += __shfl_down_sync(0xffffffff, mean, offset);
  }
  // Broadcast mean from lane 0 of each warp, then across warps
  // Simplified: use shared memory for block-level reduction
  int warp_id = tid / 32;
  int lane = tid % 32;
  int num_warps = (blockDim.x + 31) / 32;

  if (lane == 0) s_buf[warp_id] = mean;
  __syncthreads();

  if (tid < num_warps) {
    float block_mean = s_buf[tid];
    for (int w = 0; w < num_warps; w++) {
      block_mean += s_buf[w];
    }
    // block_mean now has sum of partial sums from each warp → divide by D
    s_buf[0] = block_mean / D;
    s_buf[1] = 0.0f; // will hold variance
  }
  __syncthreads();

  float fmean = s_buf[0];

  // --- Compute variance ---
  float var = 0.0f;
  for (int j = tid; j < D; j += stride) {
    float diff = input[row * D + j] - fmean;
    var += diff * diff;
  }

  for (int offset = 16; offset > 0; offset /= 2) {
    var += __shfl_down_sync(0xffffffff, var, offset);
  }

  if (lane == 0) s_buf[warp_id] = var;
  __syncthreads();

  if (tid < num_warps) {
    float block_var = s_buf[tid];
    for (int w = 0; w < num_warps; w++) {
      block_var += s_buf[w];
    }
    s_buf[1] = block_var / D; // final variance
  }
  __syncthreads();

  float fvar = s_buf[1];
  float inv_std = rsqrtf(fvar + eps);

  // --- Normalize + scale + shift ---
  for (int j = tid; j < D; j += stride) {
    float val = (input[row * D + j] - fmean) * inv_std;
    float g = gamma ? gamma[j] : 1.0f;
    float b = beta ? beta[j] : 0.0f;
    output[row * D + j] = val * g + b;
  }
}

// ============================================================
// CPU reference
// ============================================================
void layernorm_cpu(const float *input, const float *gamma, const float *beta,
                   float *output, int B, int D, float eps) {
  for (int i = 0; i < B; i++) {
    float mean = 0.0f;
    for (int j = 0; j < D; j++) mean += input[i * D + j];
    mean /= D;

    float var = 0.0f;
    for (int j = 0; j < D; j++) {
      float diff = input[i * D + j] - mean;
      var += diff * diff;
    }
    var /= D;

    float inv_std = 1.0f / sqrtf(var + eps);
    for (int j = 0; j < D; j++) {
      float g = gamma ? gamma[j] : 1.0f;
      float b = beta ? beta[j] : 0.0f;
      output[i * D + j] = (input[i * D + j] - mean) * inv_std * g + b;
    }
  }
}

// ============================================================
// Main
// ============================================================
int main() {
  const int B = 256;
  const int D = 1024;
  const float eps = 1e-5f;
  const size_t bytes_in = B * D * sizeof(float);
  const size_t bytes_param = D * sizeof(float);

  float *h_in = (float *)malloc(bytes_in);
  float *h_gamma = (float *)malloc(bytes_param);
  float *h_beta = (float *)malloc(bytes_param);
  float *h_out = (float *)malloc(bytes_in);
  float *h_ref = (float *)malloc(bytes_in);

  random_fill(h_in, B * D);
  random_fill(h_gamma, D);
  random_fill(h_beta, D);

  float *d_in, *d_gamma, *d_beta, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes_in));
  CUDA_CHECK(cudaMalloc(&d_gamma, bytes_param));
  CUDA_CHECK(cudaMalloc(&d_beta, bytes_param));
  CUDA_CHECK(cudaMalloc(&d_out, bytes_in));

  CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes_in, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, bytes_param, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_beta, h_beta, bytes_param, cudaMemcpyHostToDevice));

  int threads = 256;
  size_t shm_bytes = (threads + 31) / 32 * 2 * sizeof(float);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  layernorm_forward<<<B, threads, shm_bytes>>>(d_in, d_gamma, d_beta, d_out, B, D, eps);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes_in, cudaMemcpyDeviceToHost));

  layernorm_cpu(h_in, h_gamma, h_beta, h_ref, B, D, eps);
  float err = compare_arrays(h_out, h_ref, B * D);

  printf("LayerNorm: B=%d D=%d\n", B, D);
  printf("  Time: %.3f ms\n", ms);
  printf("  Max error: %.2e\n", err);

  CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_gamma));
  CUDA_CHECK(cudaFree(d_beta)); CUDA_CHECK(cudaFree(d_out));
  free(h_in); free(h_gamma); free(h_beta); free(h_out); free(h_ref);
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));

  return err < 1e-3 ? 0 : 1;
}
