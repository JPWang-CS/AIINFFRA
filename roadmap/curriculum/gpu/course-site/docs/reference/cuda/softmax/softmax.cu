#include "../include/cuda_utils.h"
#include <cstdio>
#include <cmath>
#include <cfloat>

// ============================================================
// Softmax Reference — naive 3-pass + online 1-pass
//
// 【算子是什么】softmax(x_i) = exp(x_i-max(x)) / Σ_j exp(x_j-max(x))
// 【在模型里干嘛】Attention 归一化——把 raw scores 变成概率分布
// 【什么模型用】所有 Transformer 的 self-attention / cross-attention
// ============================================================
__global__ void softmax_naive(const float *input, float *output,
                              int B, int D) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= B) return;

  // Pass 1: find max
  float max_val = -FLT_MAX;
  for (int j = 0; j < D; j++) {
    max_val = fmaxf(max_val, input[row * D + j]);
  }

  // Pass 2: exp sum
  float sum = 0.0f;
  for (int j = 0; j < D; j++) {
    sum += expf(input[row * D + j] - max_val);
  }

  // Pass 3: normalize
  for (int j = 0; j < D; j++) {
    output[row * D + j] = expf(input[row * D + j] - max_val) / sum;
  }
}

// ============================================================
// Kernel 2: Online softmax (single-pass per row, warp-level reduce)
// Each warp handles one row.
// ============================================================
__global__ void softmax_online(const float *input, float *output,
                               int B, int D) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int stride = blockDim.x;

  if (row >= B) return;

  // Online softmax: maintain running max and sum
  float max_val = -FLT_MAX;
  float sum = 0.0f;

  // Pass 1: online max + sum
  for (int j = tid; j < D; j += stride) {
    float val = input[row * D + j];
    if (val > max_val) {
      // Rescale old sum when max updates
      sum = sum * expf(max_val - val);
      max_val = val;
    }
    sum += expf(val - max_val);
  }

  // Warp-level reduce for max and sum
  // (simplified: block-level reduction using shared memory)
  __shared__ float s_max[32], s_sum[32];
  int warp_id = tid / 32;
  int lane = tid % 32;
  int num_warps = (blockDim.x + 31) / 32;

  // Warp reduce max
  float w_max = max_val;
  for (int offset = 16; offset > 0; offset /= 2) {
    float other = __shfl_down_sync(0xffffffff, w_max, offset);
    if (w_max < other) w_max = other;
  }

  // Warp reduce sum (with rescaling based on global max)
  float w_sum = sum;
  for (int offset = 16; offset > 0; offset /= 2) {
    w_sum += __shfl_down_sync(0xffffffff, w_sum, offset);
  }

  if (lane == 0) {
    s_max[warp_id] = w_max;
    s_sum[warp_id] = w_sum;
  }
  __syncthreads();

  // Merge across warps (single-thread for block)
  if (tid < num_warps) {
    // Find global max across warps
    float g_max = s_max[tid];
    for (int w = 0; w < num_warps; w++) {
      if (s_max[w] > g_max) g_max = s_max[w];
    }
    // Rescale and sum
    float total = 0.0f;
    for (int w = 0; w < num_warps; w++) {
      total += s_sum[w] * expf(s_max[w] - g_max);
    }
    s_max[0] = g_max;
    s_sum[0] = total;
  }
  __syncthreads();

  float g_max = s_max[0];
  float g_sum = s_sum[0];

  // Pass 2: normalize
  for (int j = tid; j < D; j += stride) {
    output[row * D + j] = expf(input[row * D + j] - g_max) / g_sum;
  }
}

// ============================================================
// CPU reference
// ============================================================
void softmax_cpu(const float *input, float *output, int B, int D) {
  for (int i = 0; i < B; i++) {
    float max_val = -FLT_MAX;
    for (int j = 0; j < D; j++) {
      max_val = fmaxf(max_val, input[i * D + j]);
    }
    float sum = 0.0f;
    for (int j = 0; j < D; j++) {
      sum += expf(input[i * D + j] - max_val);
    }
    for (int j = 0; j < D; j++) {
      output[i * D + j] = expf(input[i * D + j] - max_val) / sum;
    }
  }
}

// ============================================================
// Main
// ============================================================
int main() {
  const int B = 1024; // rows (batch)
  const int D = 2048; // columns (dim)
  const size_t bytes = B * D * sizeof(float);

  float *h_in = (float *)malloc(bytes);
  float *h_out_naive = (float *)malloc(bytes);
  float *h_out_online = (float *)malloc(bytes);
  float *h_ref = (float *)malloc(bytes);

  random_fill(h_in, B * D);

  float *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  // ---- Naive softmax ----
  int block_naive = 256;
  int grid_naive = (B + block_naive - 1) / block_naive;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  softmax_naive<<<grid_naive, block_naive>>>(d_in, d_out, B, D);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms_naive = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));
  CUDA_CHECK(cudaMemcpy(h_out_naive, d_out, bytes, cudaMemcpyDeviceToHost));

  // ---- Online softmax ----
  int block_online = 256;
  // One block per row
  CUDA_CHECK(cudaEventRecord(start));
  softmax_online<<<B, block_online>>>(d_in, d_out, B, D);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms_online = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms_online, start, stop));
  CUDA_CHECK(cudaMemcpy(h_out_online, d_out, bytes, cudaMemcpyDeviceToHost));

  // Verify
  softmax_cpu(h_in, h_ref, B, D);
  float err_naive = compare_arrays(h_out_naive, h_ref, B * D);
  float err_online = compare_arrays(h_out_online, h_ref, B * D);

  printf("Softmax: B=%d D=%d\n", B, D);
  printf("  Naive:  %.3f ms, err=%.2e\n", ms_naive, err_naive);
  printf("  Online: %.3f ms, err=%.2e\n", ms_online, err_online);

  CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
  free(h_in); free(h_out_naive); free(h_out_online); free(h_ref);
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
