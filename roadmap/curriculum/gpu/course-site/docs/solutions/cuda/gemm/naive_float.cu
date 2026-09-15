// ============================================================
// GEMM naive — LeetGPU `2_matrix_multiplication`（✅ 2026-06-16 通过）
//
// 【算子是什么】矩阵乘法：C[M×K] = A[M×N] × B[N×K]，row-major，float32
// 【在模型里干嘛】所有 Linear/FFN 层的底层计算。Transformer 的 QKV projection、
//   attention score(QK^T)、output projection、FFN gate/up/down 都是 GEMM。
// 【什么模型用】LLaMA/GPT/BERT/DeepSeek/Mistral/Qwen…所有神经网络。
//
// 自己写的版本。讲解见 ../../lessons/02-gemm-naive.md
//
// 关键点：
//   - 2D grid，blockIdx.x→K 维，blockIdx.y→M 维
//   - 每线程独占一个输出 C[idx]，用 = 不用 +=（+= 依赖 C 清零，是埋雷）
//   - 瓶颈：A 被读 K 次、B 被读 M 次，算术强度≈0.25 → memory-bound
//   - 下一步优化：shared memory tiling（→ ../../lessons/03-gemm-tiled.md）
// ============================================================

#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C,
                                             int M, int N, int K) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;  // K 维度索引
    int m = blockDim.y * blockIdx.y + threadIdx.y;  // M 维度索引
    int idx = m * K + k;

    if ((k < K) && (m < M)) {
        float sum = 0;
        for (int n = 0; n < N; n++) {
            sum += A[m * N + n] * B[n * K + k];
        }
        C[idx] = sum;
    }
}

extern "C" void solve(const float* A, const float* B, float* C,
                      int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + 15) / 16, (M + 15) / 16);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
