// ============================================================
// gemm_fp16_naive.cu — 手写版本（2026-06-22）
// C = alpha * A × B + beta * C，half 精度，LeetGPU 接口
// ============================================================
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>

__global__ void kernel(const half* A, const half* B, half* C,
                       int M, int N, int K, float alpha, float beta) {
    int m = blockDim.x * blockIdx.x + threadIdx.x;
    int n = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = m * N + n;
    if ((n < N) && (m < M)) {
        float sum = (beta == 0.0f) ? 0 : __half2float(C[idx]);
        sum = sum * beta;
        for (int k = 0; k < K; k++) {
            sum += alpha * (__half2float(A[m * K + k]) *
                            __half2float(B[k * N + n]));
        }
        C[idx] = sum;  // implicit float→half，better: __float2half_rn(sum)
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C,
                       int M, int N, int K, float alpha, float beta) {
    dim3 blockDim(16, 16);
    dim3 gridDim((M + 15) / 16, (N + 15) / 16);
    kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}
