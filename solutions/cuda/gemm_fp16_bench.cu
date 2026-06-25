// ============================================================
// gemm_fp16_bench.cu — 4090 GEMM benchmark
// 对比 naive vs tiled fp16 GEMM，M=N=K=2048
// 编译：nvcc -arch=sm_89 -o gemm_bench gemm_fp16_bench.cu
// ============================================================
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

constexpr int TILE = 32;

// =================== Naive Kernel（你的写法）===================
__global__ void gemm_fp16_naive(const half* A, const half* B, half* C,
                                 int M, int N, int K, float alpha, float beta) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N && m < M) {
        float sum = (beta == 0.0f) ? 0.0f : (beta * __half2float(C[m * N + n]));
        for (int k = 0; k < K; k++) {
            sum += alpha * (__half2float(A[m * K + k]) * __half2float(B[k * N + n]));
        }
        C[m * N + n] = __float2half_rn(sum);
    }
}

// =================== Tiled Kernel（你的写法）===================
__global__ void gemm_fp16_tiled(const half* A, const half* B, half* C,
                                 int M, int N, int K, float alpha, float beta) {
    int m = blockIdx.x * TILE + threadIdx.x;
    int n = blockIdx.y * TILE + threadIdx.y;
    float sum = 0.0f;

    __shared__ half As[TILE][TILE];
    __shared__ half Bs[TILE][TILE];

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aK = t * TILE + threadIdx.y;
        As[threadIdx.x][threadIdx.y] = (m < M && aK < K)
            ? A[m * K + aK] : __float2half_rn(0.0f);

        int bK = t * TILE + threadIdx.x;
        Bs[threadIdx.x][threadIdx.y] = (n < N && bK < K)
            ? B[bK * N + n] : __float2half_rn(0.0f);

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
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

// =================== CPU 参考 ====================
void gemm_cpu(const half* A, const half* B, half* C,
              int M, int N, int K, float alpha, float beta) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += __half2float(A[m * K + k]) * __half2float(B[k * N + n]);
            }
            C[m * N + n] = __float2half_rn(alpha * sum + beta * __half2float(C[m * N + n]));
        }
    }
}

// =================== 工具函数 ====================
float max_error(const half* a, const half* b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(__half2float(a[i]) - __half2float(b[i]));
        if (err > max_err) max_err = err;
    }
    return max_err;
}

float run_kernel(void (*kernel)(const half*, const half*, half*, int, int, int, float, float),
                 const half* d_A, const half* d_B, half* d_C,
                 int M, int N, int K, float alpha, float beta,
                 dim3 grid, dim3 block, int warmup, int iters,
                 float* ms_out) {
    // warmup
    for (int i = 0; i < warmup; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    cudaDeviceSynchronize();

    // timed runs
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(ms_out, start, stop);
    *ms_out /= iters;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return *ms_out;
}

// =================== 主函数 ====================
int main() {
    // 参数
    const int M = 2048, N = 2048, K = 2048;
    const float alpha = 1.0f, beta = 0.0f;
    const size_t size_A = M * K * sizeof(half);
    const size_t size_B = K * N * sizeof(half);
    const size_t size_C = M * N * sizeof(half);

    printf("=== GEMM fp16 Benchmark (M=N=K=%d) ===\n", M);
    printf("alpha=%.1f, beta=%.1f\n\n", alpha, beta);

    // 分配 host 内存
    half *h_A = (half*)malloc(size_A);
    half *h_B = (half*)malloc(size_B);
    half *h_C_naive = (half*)malloc(size_C);
    half *h_C_tiled = (half*)malloc(size_C);
    half *h_C_cpu   = (half*)malloc(size_C);

    // 随机初始化
    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half_rn((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half_rn((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < M * N; i++) h_C_naive[i] = h_C_tiled[i] = h_C_cpu[i] = __float2half_rn(0.0f);

    // 分配 device 内存
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A); cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMalloc(&d_B, size_B); cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMalloc(&d_C, size_C);

    dim3 naiveBlock(16, 16);
    dim3 naiveGrid((N + 15) / 16, (M + 15) / 16);
    dim3 tiledBlock(TILE, TILE);
    dim3 tiledGrid((M + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // CPU 参考
    gemm_cpu(h_A, h_B, h_C_cpu, M, N, K, alpha, beta);
    printf("[CPU] reference done\n");

    // ---- Naive ----
    printf("\n--- Naive ---\n");
    cudaMemset(d_C, 0, size_C);
    float ms_naive;
    run_kernel(gemm_fp16_naive, d_A, d_B, d_C, M, N, K, alpha, beta,
               naiveGrid, naiveBlock, 3, 5, &ms_naive);
    cudaMemcpy(h_C_naive, d_C, size_C, cudaMemcpyDeviceToHost);

    float gflops_naive = (2.0f * M * N * K) / (ms_naive / 1000.0f) / 1e9f;
    printf("time: %.3f ms, GFLOPS: %.1f\n", ms_naive, gflops_naive);
    printf("max error: %.6f\n", max_error(h_C_naive, h_C_cpu, M * N));

    // ---- Tiled ----
    printf("\n--- Tiled (TILE=32) ---\n");
    cudaMemset(d_C, 0, size_C);
    float ms_tiled;
    run_kernel(gemm_fp16_tiled, d_A, d_B, d_C, M, N, K, alpha, beta,
               tiledGrid, tiledBlock, 3, 5, &ms_tiled);
    cudaMemcpy(h_C_tiled, d_C, size_C, cudaMemcpyDeviceToHost);

    float gflops_tiled = (2.0f * M * N * K) / (ms_tiled / 1000.0f) / 1e9f;
    printf("time: %.3f ms, GFLOPS: %.1f\n", ms_tiled, gflops_tiled);
    printf("max error: %.6f\n", max_error(h_C_tiled, h_C_cpu, M * N));

    // ---- 结论 ----
    printf("\n=== 结论 ===\n");
    printf("Naive:  %.1f GFLOPS\n", gflops_naive);
    printf("Tiled:  %.1f GFLOPS\n", gflops_tiled);
    printf("加速比: %.1fx\n", gflops_tiled / gflops_naive);
    printf("RTX 4090 FP16 理论峰值: ~330 TFLOPS (Tensor Core)\n");
    printf("（当前实测会远低于峰值——GPU cuda core 路径，非 Tensor Core 加速）\n");

    // 清理
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tiled); free(h_C_cpu);

    return (max_error(h_C_tiled, h_C_cpu, M * N) < 1e-3) ? 0 : 1;
}
