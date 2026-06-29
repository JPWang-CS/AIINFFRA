#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#define CUDA_CHECK(e) do{cudaError_t r=e;if(r){printf("ERR %s\n",cudaGetErrorString(r));exit(1);}}while(0)

// ============================================================
// bench_transpose.cu — Naive vs Tiled Matrix Transpose
//
// 【算子是什么】矩阵转置: out[j][i] = in[i][j]
// 【在模型里干嘛】Attention 中的 QK^T 需要 K 的转置、batch/seq 维度交换、
//   multi-head 的 reshape+transpose 操作（[B,N,H,D]↔[B,H,N,D]）
// 【什么模型用】所有 Transformer——LLaMA/GPT/DeepSeek/Mistral
//   - Q@K^T 中 K 需要转置（N×d → d×N）
//   - Flash Attention 的分块策略本质是 transpose + tiling 的组合
// ============================================================
__global__ void transpose_naive(const float* in, float* out, int N) {
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (x >= N || y >= N) return;
    out[x * N + y] = in[y * N + x];
}

// Tiled: shared memory square tile, read coalesced + write coalesced
template<int TILE>
__global__ void transpose_tiled(const float* in, float* out, int N) {
    __shared__ float tile[TILE][TILE+1];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    int x_in  = blockIdx.y * TILE + threadIdx.x;  // reading from
    int y_in  = blockIdx.x * TILE + threadIdx.y;  // transposed block

    if (x_in < N && y_in < N)
        tile[threadIdx.y][threadIdx.x] = in[y_in * N + x_in];
    else
        tile[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();

    if (x < N && y < N)
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];
}

int main() {
    int N = 4096;
    size_t sz = N * N * sizeof(float);
    printf("=== Matrix Transpose N=%d (%d MB) ===\n", N, (int)(sz/1e6));

    float *h_in = (float*)malloc(sz), *h_out = (float*)malloc(sz), *h_ref = (float*)malloc(sz);
    srand(42);
    for (int i = 0; i < N*N; i++) h_in[i] = (float)rand() / RAND_MAX;

    // CPU reference
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) h_ref[j*N+i] = h_in[i*N+j];

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, sz)); CUDA_CHECK(cudaMalloc(&d_out, sz));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice));

    dim3 nb(32, 32), ng((N+31)/32, (N+31)/32);

    // correctness
    CUDA_CHECK(cudaMemset(d_out, 0, sz));
    transpose_naive<<<ng, nb>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost));
    float en = 0; for (int i = 0; i < N*N; i++) { float e = fabsf(h_out[i]-h_ref[i]); if(e>en) en=e; }
    printf("Naive err:  %.6f %s\n", en, en<1e-4?"PASS":"FAIL");

    CUDA_CHECK(cudaMemset(d_out, 0, sz));
    dim3 tb(32, 32), tg((N+31)/32, (N+31)/32);
    transpose_tiled<32><<<tg, tb>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost));
    float et = 0; for (int i = 0; i < N*N; i++) { float e = fabsf(h_out[i]-h_ref[i]); if(e>et) et=e; }
    printf("Tiled err:  %.6f %s\n", et, et<1e-4?"PASS":"FAIL");

    // warmup
    transpose_naive<<<ng, nb>>>(d_in, d_out, N);
    transpose_tiled<32><<<tg, tb>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // benchmark
    int iters = 20;
    cudaEvent_t st, en; cudaEventCreate(&st); cudaEventCreate(&en);

    cudaEventRecord(st);
    for (int i = 0; i < iters; i++) transpose_naive<<<ng, nb>>>(d_in, d_out, N);
    cudaEventRecord(en); cudaEventSynchronize(en);
    float mn; cudaEventElapsedTime(&mn, st, en); mn /= iters;
    double bw_n = sz * 2.0 / (mn / 1000) / 1e9;

    cudaEventRecord(st);
    for (int i = 0; i < iters; i++) transpose_tiled<32><<<tg, tb>>>(d_in, d_out, N);
    cudaEventRecord(en); cudaEventSynchronize(en);
    float mt; cudaEventElapsedTime(&mt, st, en); mt /= iters;
    double bw_t = sz * 2.0 / (mt / 1000) / 1e9;

    printf("\n=== RESULT ===\n");
    printf("Naive: %.3f ms  %.1f GB/s  (read coalesced, write stride=%d = terrible)\n", mn, bw_n, N);
    printf("Tiled: %.3f ms  %.1f GB/s  (both coalesced via shared mem)\n", mt, bw_t);
    printf("Speedup: %.2fx\n\n", mn / mt);
    printf("Why transpose works (unlike GEMM tiled):\n");
    printf("  - Naive write is stride-N (N=%d = %d bytes apart): each warp write is %d separate transactions\n", N, N*4, 32);
    printf("  - Tiled buffers in shared memory, writes coalesced\n");
    printf("  - This bandwidth gap is NOT fixable by L2 (write pattern fundamentally bad)\n");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out); free(h_ref);
    return 0;
}
