// ============================================================
// gemm_fp16_tiled.cu — tiled GEMM with shared memory（2026-06-25）
// C = alpha * A × B + beta * C，half 精度，shared memory tiling，TILE=32
//
// 【算子是什么】矩阵乘法 + shared memory tiling 优化
//   - 和 naive 算的一样，但通过把 A/B 切成 32×32 tile 搬到 shared memory，
//     减少 HBM 重复读取。每个 tile 在片上复用，减少了 K/TILE 倍 HBM 访存。
// 【在模型里干嘛】同 naive GEMM——所有 Linear/FFN 层。
//   - 区别：当 K 维度大（如 FFN 的 d_ff=11008），tiling 能显著减少 HBM 访存
//   - 但在 4090 大 L2(72MB) 上，小规模(K<8K) naive 的 cache 命中率够高，tiling 不总是赢
// 【什么模型用】LLaMA/DeepSeek/Qwen 的推理引擎（TensorRT-LLM/vLLM 内部用 tiled GEMM）
//
// 代码归档。讲解见 ../../lessons/03-gemm-tiled.md
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

    // 每个线程先搬运暂存
    __shared__ half As[tileLen][tileLen];
    __shared__ half Bs[tileLen][tileLen];

    // 遍历 K
    for (int t = 0; t < (K + tileLen - 1) / tileLen; ++t) {
        // 每个线程搬运自己负责的那块
        // A 矩阵按列步进
        int aK = t * tileLen + threadIdx.y;
        As[threadIdx.x][threadIdx.y] = (m < M && aK < K)
            ? A[m * K + aK] : __float2half_rn(0.0f);

        // B 矩阵按行步进
        int bK = t * tileLen + threadIdx.x;
        Bs[threadIdx.x][threadIdx.y] = (n < N && bK < K)
            ? B[bK * N + n] : __float2half_rn(0.0f);

        // 同步
        __syncthreads();

        // 计算
        for (int k = 0; k < tileLen; k++) {
            sum += __half2float(As[threadIdx.x][k]) *
                   __half2float(Bs[k][threadIdx.y]);
        }

        // 同步
        __syncthreads();
    }

    if (m < M && n < N) {
        C[m * N + n] = __float2half_rn(
            alpha * sum + beta * __half2float(C[m * N + n]));
    }
}


// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C,
                       int M, int N, int K, float alpha, float beta) {
    dim3 blockDim(tileLen, tileLen);              // Thread: 32x32=1024
    dim3 gridDim((M + tileLen - 1) / tileLen,
                 (N + tileLen - 1) / tileLen);
    kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}
