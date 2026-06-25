#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA ERROR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

constexpr int TILE = 32;

__global__ void naive_k(const half* A, const half* B, half* C, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || m >= M) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += __half2float(A[m * K + k]) * __half2float(B[k * N + n]);
    C[m * N + n] = __float2half_rn(sum);
}

__global__ void tiled_k(const half* A, const half* B, half* C, int M, int N, int K) {
    int m = blockIdx.x * TILE + threadIdx.x;
    int n = blockIdx.y * TILE + threadIdx.y;
    float sum = 0.0f;
    __shared__ half As[TILE][TILE];
    __shared__ half Bs[TILE][TILE];
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aK = t * TILE + threadIdx.y;
        As[threadIdx.x][threadIdx.y] = (m < M && aK < K) ? A[m * K + aK] : __float2half_rn(0.0f);
        int bK = t * TILE + threadIdx.x;
        Bs[threadIdx.x][threadIdx.y] = (n < N && bK < K) ? B[bK * N + n] : __float2half_rn(0.0f);
        __syncthreads();
        for (int k = 0; k < TILE; k++)
            sum += __half2float(As[threadIdx.x][k]) * __half2float(Bs[k][threadIdx.y]);
        __syncthreads();
    }
    if (m < M && n < N)
        C[m * N + n] = __float2half_rn(sum);
}

int main() {
    const int M = 2048, N = 2048, K = 2048;
    size_t sA = M * K * sizeof(half), sB = K * N * sizeof(half), sC = M * N * sizeof(half);

    printf("=== GEMM fp16 M=N=K=%d ===\n", M);

    half *h_A = (half*)malloc(sA), *h_B = (half*)malloc(sB);
    half *h_Cn = (half*)malloc(sC), *h_Ct = (half*)malloc(sC), *h_Ccpu = (half*)malloc(sC);
    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half_rn((float)rand()/RAND_MAX - 0.5f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half_rn((float)rand()/RAND_MAX - 0.5f);
    memset(h_Cn, 0, sC); memset(h_Ct, 0, sC); memset(h_Ccpu, 0, sC);

    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sA));
    CUDA_CHECK(cudaMalloc(&d_B, sB));
    CUDA_CHECK(cudaMalloc(&d_C, sC));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sB, cudaMemcpyHostToDevice));
    printf("[init] GPU mem allocated\n");

    dim3 nb(16, 16), ng((N+15)/16, (M+15)/16);
    dim3 tb(TILE, TILE), tg((M+TILE-1)/TILE, (N+TILE-1)/TILE);

    // CPU ref
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0;
            for (int k = 0; k < K; k++)
                s += __half2float(h_A[m*K+k]) * __half2float(h_B[k*N+n]);
            h_Ccpu[m*N+n] = __float2half_rn(s);
        }
    printf("[cpu] done\n");

    // warmup
    CUDA_CHECK(cudaMemset(d_C, 0, sC));
    naive_k<<<ng, nb>>>(d_A, d_B, d_C, M, N, K);
    tiled_k<<<tg, tb>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[warmup] done\n");

    // naive timed
    CUDA_CHECK(cudaMemset(d_C, 0, sC));
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 10; i++) naive_k<<<ng, nb>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms_n; cudaEventElapsedTime(&ms_n, s, e); ms_n /= 10;
    CUDA_CHECK(cudaMemcpy(h_Cn, d_C, sC, cudaMemcpyDeviceToHost));

    float max_n = 0;
    for (int i = 0; i < M*N; i++) {
        float err = fabsf(__half2float(h_Cn[i]) - __half2float(h_Ccpu[i]));
        if (err > max_n) max_n = err;
    }
    float gflops_n = (2.0f * M * N * K) / (ms_n / 1000) / 1e9;
    printf("[N] time:%.3fms GFLOPS:%.1f err:%.6f\n", ms_n, gflops_n, max_n);

    // tiled timed
    CUDA_CHECK(cudaMemset(d_C, 0, sC));
    cudaEventRecord(s);
    for (int i = 0; i < 10; i++) tiled_k<<<tg, tb>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms_t; cudaEventElapsedTime(&ms_t, s, e); ms_t /= 10;
    CUDA_CHECK(cudaMemcpy(h_Ct, d_C, sC, cudaMemcpyDeviceToHost));

    float max_t = 0;
    for (int i = 0; i < M*N; i++) {
        float err = fabsf(__half2float(h_Ct[i]) - __half2float(h_Ccpu[i]));
        if (err > max_t) max_t = err;
    }
    float gflops_t = (2.0f * M * N * K) / (ms_t / 1000) / 1e9;
    printf("[T] time:%.3fms GFLOPS:%.1f err:%.6f\n", ms_t, gflops_t, max_t);

    printf("\n=== RESULT ===\n");
    printf("Naive: %.1f GFLOPS\n", gflops_n);
    printf("Tiled: %.1f GFLOPS\n", gflops_t);
    printf("Speedup: %.1fx\n", gflops_t / gflops_n);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_Cn); free(h_Ct); free(h_Ccpu);
    return (max_n < 1e-3 && max_t < 1e-3) ? 0 : 1;
}
