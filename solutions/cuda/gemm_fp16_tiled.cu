// ============================================================
// gemm_fp16_tiled.cu — 手写版本（2026-06-22）
// C = alpha * A × B + beta * C，half 精度，shared memory tiling
// TILE=32，LeetGPU 接口
// ============================================================
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>

constexpr int tileLen = 32;

__global__ void kernel(const half* A, const half* B, half* C,
                       int M, int N, int K, float alpha, float beta) {
    int m = blockIdx.x * tileLen + threadIdx.x;
    int n = blockIdx.y * tileLen + threadIdx.y;
    float sum = 0.0f;

    __shared__ half As[tileLen][tileLen];
    __shared__ half Bs[tileLen][tileLen];

    for (int t = 0; t < (K + tileLen - 1) / tileLen; ++t) {
        // A tile: threadIdx.y walks K, threadIdx.x walks M
        int aK = t * tileLen + threadIdx.y;
        As[threadIdx.x][threadIdx.y] = (m < M && aK < K)
            ? A[m * K + aK] : __float2half_rn(0.0f);

        // B tile: threadIdx.x walks K, threadIdx.y walks N
        int bK = t * tileLen + threadIdx.x;
        Bs[threadIdx.x][threadIdx.y] = (n < N && bK < K)
            ? B[bK * N + n] : __float2half_rn(0.0f);

        __syncthreads();

        for (int k = 0; k < tileLen; k++) {
            sum += __half2float(As[threadIdx.x][k]) *
                   __half2float(Bs[k][threadIdx.y]);
        }

        __syncthreads();
    }

    if (m < M && n < N) {
        C[m * N + n] = __float2half_rn(
            alpha * sum + beta * __half2float(C[m * N + n]));
    }
}

extern "C" void solve(const half* A, const half* B, half* C,
                       int M, int N, int K, float alpha, float beta) {
    dim3 blockDim(tileLen, tileLen);
    dim3 gridDim((M + tileLen - 1) / tileLen,
                 (N + tileLen - 1) / tileLen);
    kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}
