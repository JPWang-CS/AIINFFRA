#include "../include/cuda_utils.h"
#include <cstdio>
#include <cmath>

// ============================================================
// Kernel 1: Naive GEMM — C = A × Bᵀ  (A: M×K, B: N×K)
// One thread per output element. No shared memory.
// ============================================================
__global__ void gemm_naive(const float *A, const float *B, float *C,
                           int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
      sum += A[row * K + k] * B[col * K + k]; // B is N×K, transposed access
    }
    C[row * N + col] = sum;
  }
}

// ============================================================
// Kernel 2: Tiled GEMM — C = A × Bᵀ, shared memory tiling
// Each block computes a TILE×TILE sub-matrix of C.
// A_tile and B_tile are loaded into shared memory cooperatively.
// ============================================================
#define TILE 32

__global__ void gemm_tiled(const float *A, const float *B, float *C,
                           int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  float sum = 0.0f;

  // Loop over tiles of K dimension
  for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
    // Cooperative load A tile
    int a_k = t * TILE + threadIdx.x;
    if (row < M && a_k < K)
      As[threadIdx.y][threadIdx.x] = A[row * K + a_k];
    else
      As[threadIdx.y][threadIdx.x] = 0.0f;

    // Cooperative load B tile (B is N×K, transposed)
    int b_k = t * TILE + threadIdx.y;
    if (col < N && b_k < K)
      Bs[threadIdx.y][threadIdx.x] = B[col * K + b_k];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    // Compute partial dot product over this tile
    for (int k = 0; k < TILE; k++) {
      sum += As[threadIdx.y][k] * Bs[threadIdx.x][k];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

// ============================================================
// CPU reference
// ============================================================
void gemm_cpu(const float *A, const float *B, float *C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[j * K + k];
      }
      C[i * N + j] = sum;
    }
  }
}

// ============================================================
// Main
// ============================================================
int main() {
  const int M = 512, N = 512, K = 512;
  const size_t bytes_A = M * K * sizeof(float);
  const size_t bytes_B = N * K * sizeof(float);
  const size_t bytes_C = M * N * sizeof(float);

  float *h_A = (float *)malloc(bytes_A);
  float *h_B = (float *)malloc(bytes_B);
  float *h_C_naive = (float *)malloc(bytes_C);
  float *h_C_tiled = (float *)malloc(bytes_C);
  float *h_ref = (float *)malloc(bytes_C);

  random_fill(h_A, M * K);
  random_fill(h_B, N * K);

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
  CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (M + 15) / 16);

  // ---- Naive GEMM ----
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms_naive = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));
  CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, bytes_C, cudaMemcpyDeviceToHost));

  // ---- Tiled GEMM ----
  dim3 block_t(TILE, TILE);
  dim3 grid_t((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

  CUDA_CHECK(cudaEventRecord(start));
  gemm_tiled<<<grid_t, block_t>>>(d_A, d_B, d_C, M, N, K);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms_tiled = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms_tiled, start, stop));
  CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, bytes_C, cudaMemcpyDeviceToHost));

  // Verify
  gemm_cpu(h_A, h_B, h_ref, M, N, K);
  float err_naive = compare_arrays(h_C_naive, h_ref, M * N);
  float err_tiled = compare_arrays(h_C_tiled, h_ref, M * N);

  float gflops_naive = (2.0f * M * N * K) / (ms_naive / 1000.0f) / 1e9f;
  float gflops_tiled = (2.0f * M * N * K) / (ms_tiled / 1000.0f) / 1e9f;

  printf("GEMM: M=%d N=%d K=%d\n", M, N, K);
  printf("  Naive: %.3f ms, %.2f GFLOPS, err=%.2e\n", ms_naive, gflops_naive, err_naive);
  printf("  Tiled: %.3f ms, %.2f GFLOPS, err=%.2e\n", ms_tiled, gflops_tiled, err_tiled);

  // Cleanup
  CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
  free(h_A); free(h_B); free(h_C_naive); free(h_C_tiled); free(h_ref);
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
