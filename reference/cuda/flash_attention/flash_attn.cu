#include "../include/cuda_utils.h"
#include <cstdio>
#include <cmath>
#include <cfloat>

// ============================================================
// Simplified Flash Attention — forward pass, causal optional
//
// Key ideas (from Flash Attention paper):
// 1. Tiling: Q split into blocks (Br), K/V split into blocks (Bc)
// 2. Online softmax: track running m (max), l (exp sum) per Q block
// 3. Incremental rescaling: when max updates, rescale old accumulator
//
// Q: B×H×D   K: B×H×D   V: B×H×D   (query, key, value)
// O: B×H×D     output = softmax(QK^T/√d) × V
//
// This simplified version:
//  - B=1, H=1 (single head for clarity)
//  - Square: seq_len = N, head_dim = d
//  - Br = Bc = TILE constant
// ============================================================

#define BR 32 // tile size for Q rows
#define BC 32 // tile size for K/V rows

__global__ void flash_attn_forward(const float *Q, const float *K, const float *V,
                                    float *O, int N, int d, bool causal) {
  // Block index: which row-tile of Q this block handles
  int q_start = blockIdx.x * BR;
  int q_end = min(q_start + BR, N);

  int tid = threadIdx.x;
  int lane = tid % 32; // within-warp lane

  // Shared memory layout:
  // Q_tile: [BR][d] — loaded once before K/V loop
  // K_tile: [BC][d] — reloaded each inner loop iteration
  // V_tile: [BC][d] — reloaded each inner loop iteration
  extern __shared__ float smem[];
  float *Q_tile = smem;
  float *K_tile = Q_tile + BR * d;
  float *V_tile = K_tile + BC * d;
  // smem size: BR*d + BC*d + BC*d = (BR + 2*BC) * d

  // ---- Load Q tile (all threads cooperate) ----
  for (int i = tid; i < BR * d; i += blockDim.x) {
    int qr = i / d;
    int qc = i % d;
    int global_row = q_start + qr;
    if (q_start + qr < N) {
      Q_tile[i] = Q[global_row * d + qc];
    }
  }
  __syncthreads();

  // Online softmax state per Q row (registers or shared mem for the block)
  // Using registers: one thread handles one Q row
  // For simplicity, use shared memory arrays
  float *m_i = smem + (BR + 2 * BC) * d; // [BR] running max
  float *l_i = m_i + BR;                  // [BR] running exp sum
  // Accumulator O is kept in registers per-row then written back

  // Each thread handles one Q row and all K/V columns
  int qr = threadIdx.x; // thread = Q row (assumes blockDim.x >= BR)
  if (qr >= (q_end - q_start)) return;

  int global_qr = q_start + qr;

  // Initialize running state for this Q row
  float m = -FLT_MAX;
  float l = 0.0f;

  // Output accumulator (kept in registers, written back at end)
  float acc[128]; // assume d <= 128 for register storage
  if (d > 128) return; // safety guard — could use loop for larger d
  for (int j = 0; j < d; j++) acc[j] = 0.0f;

  // ---- Loop over K/V tiles ----
  for (int kv_start = 0; kv_start < N; kv_start += BC) {
    int kv_end = min(kv_start + BC, N);

    // Load K tile
    for (int i = tid; i < BC * d; i += blockDim.x) {
      int kr = i / d;
      int kc = i % d;
      int global_row = kv_start + kr;
      if (global_row < N) {
        K_tile[i] = K[global_row * d + kc];
      }
    }

    // Load V tile
    for (int i = tid; i < BC * d; i += blockDim.x) {
      int vr = i / d;
      int vc = i % d;
      int global_row = kv_start + vr;
      if (global_row < N) {
        V_tile[i] = V[global_row * d + vc];
      }
    }
    __syncthreads();

    int kv_len = kv_end - kv_start;

    // Compute attention scores for this Q row vs all K rows in tile
    for (int j = 0; j < kv_len; j++) {
      int global_kr = kv_start + j;

      // Causal mask: if Q row < K row → skip
      if (causal && global_qr < global_kr) continue;

      // Dot product: Q[qr] · K[j] / sqrt(d)
      float score = 0.0f;
      for (int k = 0; k < d; k++) {
        score += Q_tile[qr * d + k] * K_tile[j * d + k];
      }
      score /= sqrtf((float)d);

      // Online softmax update
      float m_new = fmaxf(m, score);
      float p = expf(score - m_new);

      // Rescale accumulator
      float scale = expf(m - m_new);
      for (int k = 0; k < d; k++) {
        acc[k] = acc[k] * scale + p * V_tile[j * d + k];
      }

      l = l * scale + p;
      m = m_new;
    }
    __syncthreads();
  }

  // ---- Write output ----
  for (int k = 0; k < d; k++) {
    O[global_qr * d + k] = acc[k] / l;
  }
}

// ============================================================
// CPU reference: standard attention
// ============================================================
void attention_cpu(const float *Q, const float *K, const float *V,
                   float *O, int N, int d, bool causal) {
  float scale = 1.0f / sqrtf((float)d);

  for (int i = 0; i < N; i++) {
    // Compute max for stability
    float max_val = -FLT_MAX;
    for (int j = 0; j < N; j++) {
      if (causal && i < j) continue;
      float score = 0.0f;
      for (int k = 0; k < d; k++) score += Q[i * d + k] * K[j * d + k];
      score *= scale;
      max_val = fmaxf(max_val, score);
    }

    // Exp sum
    float sum = 0.0f;
    float *scores = (float *)malloc(N * sizeof(float));
    for (int j = 0; j < N; j++) {
      if (causal && i < j) { scores[j] = 0.0f; continue; }
      float score = 0.0f;
      for (int k = 0; k < d; k++) score += Q[i * d + k] * K[j * d + k];
      score *= scale;
      scores[j] = expf(score - max_val);
      sum += scores[j];
    }

    // Weighted sum
    for (int k = 0; k < d; k++) {
      float val = 0.0f;
      for (int j = 0; j < N; j++) {
        val += scores[j] * V[j * d + k];
      }
      O[i * d + k] = val / sum;
    }
    free(scores);
  }
}

// ============================================================
// Main
// ============================================================
int main() {
  const int N = 128; // sequence length
  const int d = 64;  // head dimension (≤128 for register storage)
  const bool causal = true;
  const size_t bytes = N * d * sizeof(float);

  float *h_Q = (float *)malloc(bytes);
  float *h_K = (float *)malloc(bytes);
  float *h_V = (float *)malloc(bytes);
  float *h_O = (float *)malloc(bytes);
  float *h_ref = (float *)malloc(bytes);

  random_fill(h_Q, N * d);
  random_fill(h_K, N * d);
  random_fill(h_V, N * d);

  float *d_Q, *d_K, *d_V, *d_O;
  CUDA_CHECK(cudaMalloc(&d_Q, bytes));
  CUDA_CHECK(cudaMalloc(&d_K, bytes));
  CUDA_CHECK(cudaMalloc(&d_V, bytes));
  CUDA_CHECK(cudaMalloc(&d_O, bytes));

  CUDA_CHECK(cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice));

  // Shared memory: Q_tile[BR*d] + K_tile[BC*d] + V_tile[BC*d] + m[BR] + l[BR]
  size_t smem = (BR * d + BC * d + BC * d) * sizeof(float) + BR * 2 * sizeof(float);

  int grid = (N + BR - 1) / BR;
  int block = BR; // one thread per Q row

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  flash_attn_forward<<<grid, block, smem>>>(d_Q, d_K, d_V, d_O, N, d, causal);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaMemcpy(h_O, d_O, bytes, cudaMemcpyDeviceToHost));

  // Verify
  attention_cpu(h_Q, h_K, h_V, h_ref, N, d, causal);
  float err = compare_arrays(h_O, h_ref, N * d);

  printf("Flash Attention (simplified): N=%d d=%d causal=%d\n", N, d, causal);
  printf("  Time: %.3f ms\n", ms);
  printf("  Max error: %.2e\n", err);

  CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
  CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
  free(h_Q); free(h_K); free(h_V); free(h_O); free(h_ref);
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));

  return err < 1e-2 ? 0 : 1;
}
