// ============================================================
// gemm_fp16_naive.cu — 手写版本（2026-06-22）
// C = alpha * A × B + beta * C，half 精度，LeetGPU 接口
//
// 【算子是什么】矩阵乘法 GEMM，FP16 精度，支持 BLAS 标准公式
// 【在模型里干嘛】所有 Linear/FFN 层。FP16 是推理默认精度——7B 模型从 14GB→7GB。
//   Attention: Q = x@W_Q, K = x@W_K, V = x@W_V, output = concat(heads)@W_O
//   FFN: gate = x@W_gate, up = x@W_up, down = (gate⊙up)@W_down
// 【什么模型用】LLaMA 2/3 (FP16 推理)、GPT-3/4、所有 HuggingFace 半精度模型
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
