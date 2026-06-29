// ============================================================
// Vector Add — element-wise addition: C[i] = A[i] + B[i]
//
// 【在模型里干嘛】Residual connections、bias addition、LayerNorm/RMSNorm 内部步骤
// 【什么模型用】所有 Transformer（LLaMA/GPT/BERT/Mistral）的 skip connection
// ============================================================
#include "cuda_utils.h"
#include <cstdio>
#include <vector>

// ---------- CUDA kernel ----------
// Each thread handles one element.
// gridDim.x * blockDim.x threads total, stride across data if N > total threads.
__global__ void vector_add_kernel(const float *a, const float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < n; i += stride) {
    c[i] = a[i] + b[i];
  }
}

// ---------- CPU reference ----------
void vector_add_cpu(const float *a, const float *b, float *c, int n) {
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

// ---------- Main ----------
int main() {
  const int N = 1 << 24; // 16M elements, ~64 MB per array
  const int bytes = N * sizeof(float);
  const int threads_per_block = 256;
  const int blocks = (N + threads_per_block - 1) / threads_per_block;

  // Allocate host memory (pageable for now; pinned memory later)
  float *h_a = (float *)malloc(bytes);
  float *h_b = (float *)malloc(bytes);
  float *h_c = (float *)malloc(bytes);
  float *h_ref = (float *)malloc(bytes);

  random_fill(h_a, N);
  random_fill(h_b, N);

  // Allocate device memory
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

  // Launch kernel + time it
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  vector_add_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, N);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  // Copy result back
  CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

  // Verify
  vector_add_cpu(h_a, h_b, h_ref, N);
  float max_diff = compare_arrays(h_c, h_ref, N);

  // Bandwidth: 3 arrays read/write = 3 * bytes transferred
  float gb_per_sec = (3.0f * bytes) / (ms / 1000.0f) / 1e9f;

  printf("Vector Add: N=%d, blocks=%d, threads=%d\n", N, blocks, threads_per_block);
  printf("  Time: %.3f ms\n", ms);
  printf("  Bandwidth: %.2f GB/s\n", gb_per_sec);
  printf("  Max error: %e\n", max_diff);

  // Cleanup
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  free(h_a); free(h_b); free(h_c); free(h_ref);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return max_diff < 1e-5 ? 0 : 1;
}
