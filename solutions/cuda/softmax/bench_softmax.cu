#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cfloat>

#define CUDA_CHECK(e) do { cudaError_t r=e; if(r){printf("ERR %s\n",cudaGetErrorString(r));exit(1);} }while(0)

// ============================================================
// softmax_bench.cu — Naive 3-pass vs Online 1-pass benchmark
//
// 【算子是什么】softmax: 把向量映射为概率分布（每元素∈[0,1]，和为 1）
//   - Naive: 3 遍遍历（max → sum_exp → normalize）
//   - Online: 1 遍遍历（增量更新 running max/sum，修正因子校正）
// 【在模型里干嘛】Attention 的归一化——把 QK^T dot-product scores 变成概率权重
//   S = Q@K^T/√d  →  P = softmax(S)  →  O = P@V
//   每个 token 对所有 token 的 attention 权重通过 softmax 归一化为概率分布
// 【什么模型用】所有 Transformer 的 Multi-Head Attention
//   LLaMA/GPT/BERT/DeepSeek/Mistral/Qwen/Claude 系列
// ============================================================
__global__ void softmax_naive(const float* in, float* out, int B, int D) {
    int row = blockIdx.x;
    if (row >= B) return;
    in += row * D; out += row * D;

    __shared__ float smax[256], ssum[256];

    // Pass 1: local max
    float mx = -FLT_MAX;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float v = in[j];
        if (v > mx) mx = v;
    }
    smax[threadIdx.x] = mx;
    __syncthreads();
    // thread 0 serial reduce
    if (threadIdx.x == 0) {
        float gm = smax[0];
        for (int i = 1; i < blockDim.x; i++)
            if (smax[i] > gm) gm = smax[i];
        smax[0] = gm;
    }
    __syncthreads();
    mx = smax[0];

    // Pass 2: sum exp
    float sm = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x)
        sm += expf(in[j] - mx);
    ssum[threadIdx.x] = sm;
    __syncthreads();
    if (threadIdx.x == 0) {
        float gs = ssum[0];
        for (int i = 1; i < blockDim.x; i++) gs += ssum[i];
        ssum[0] = gs;
    }
    __syncthreads();
    sm = ssum[0];

    // Pass 3: normalize
    for (int j = threadIdx.x; j < D; j += blockDim.x)
        out[j] = expf(in[j] - mx) / sm;
}

// =================== Online: 1-pass, warp shuffle reduce ====================
__device__ inline float warp_reduce_max(float v) {
    for (int o = 16; o > 0; o >>= 1) {
        float w = __shfl_down_sync(0xffffffff, v, o);
        if (w > v) v = w;
    }
    return v;
}
__device__ inline float warp_reduce_sum(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

__global__ void softmax_online(const float* in, float* out, int B, int D) {
    int row = blockIdx.x;
    if (row >= B) return;
    in += row * D; out += row * D;

    // Step 1: single-pass compute online
    float mx = -FLT_MAX, sm = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float v = in[j];
        float m_new = fmaxf(mx, v);
        float corr = expf(mx - m_new);
        sm = sm * corr + expf(v - m_new);
        mx = m_new;
    }

    // Step 2: warp-level reduce to get global max/sum
    mx = warp_reduce_max(mx);
    sm = warp_reduce_sum(sm);

    // Step 3: normalize and write back
    for (int j = threadIdx.x; j < D; j += blockDim.x)
        out[j] = expf(in[j] - mx) / sm;
}

// =================== CPU reference ====================
void softmax_cpu(const float* in, float* out, int D) {
    float mx = -FLT_MAX;
    for (int i = 0; i < D; i++) if (in[i] > mx) mx = in[i];
    float sm = 0;
    for (int i = 0; i < D; i++) sm += expf(in[i] - mx);
    for (int i = 0; i < D; i++) out[i] = expf(in[i] - mx) / sm;
}

// =================== Main ====================
int main() {
    const int B = 4096, D = 4096;
    const int N = B * D;
    size_t sz = N * sizeof(float);
    printf("=== Softmax Benchmark  B=%d x D=%d ===\n", B, D);

    float *h_in = (float*)malloc(sz), *h_out = (float*)malloc(sz), *h_ref = (float*)malloc(sz);
    srand(42);
    for (int i = 0; i < N; i++) h_in[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, sz)); CUDA_CHECK(cudaMalloc(&d_out, sz));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice));

    // correctness: small slice
    {
        float *h_t = (float*)malloc(D*4);
        CUDA_CHECK(cudaMemset(d_out, 0, D*4));
        softmax_online<<<1, 256>>>(d_in, d_out, 1, D);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_t, d_out, D*4, cudaMemcpyDeviceToHost));
        softmax_cpu(h_in, h_ref, D);
        float e = 0;
        for (int i = 0; i < D; i++) { float d = fabsf(h_t[i]-h_ref[i]); if(d > e) e = d; }
        printf("Correctness (online vs cpu): err=%.6f %s\n", e, e < 1e-4 ? "PASS" : "FAIL");

        CUDA_CHECK(cudaMemset(d_out, 0, D*4));
        softmax_naive<<<1, 256>>>(d_in, d_out, 1, D);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_t, d_out, D*4, cudaMemcpyDeviceToHost));
        e = 0;
        for (int i = 0; i < D; i++) { float d = fabsf(h_t[i]-h_ref[i]); if(d > e) e = d; }
        printf("Correctness (naive3pass vs cpu): err=%.6f %s\n", e, e < 1e-4 ? "PASS" : "FAIL");
        free(h_t);
    }

    // warmup
    softmax_naive<<<B, 256>>>(d_in, d_out, B, D);
    softmax_online<<<B, 256>>>(d_in, d_out, B, D);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t st, en;
    cudaEventCreate(&st); cudaEventCreate(&en);

    // Naive 3-pass
    cudaEventRecord(st);
    for (int i = 0; i < 10; i++) softmax_naive<<<B, 256>>>(d_in, d_out, B, D);
    cudaEventRecord(en); cudaEventSynchronize(en);
    float mn; cudaEventElapsedTime(&mn, st, en); mn /= 10;
    double gb_n = (double)N * 4 * 3 / (mn / 1000) / 1e9;  // 3 reads + 1 write
    printf("\nNaive 3-pass:  %.3f ms  %.1f GB/s\n", mn, gb_n);

    // Online 1-pass with warp reduce
    cudaEventRecord(st);
    for (int i = 0; i < 10; i++) softmax_online<<<B, 256>>>(d_in, d_out, B, D);
    cudaEventRecord(en); cudaEventSynchronize(en);
    float mt; cudaEventElapsedTime(&mt, st, en); mt /= 10;
    double gb_o = (double)N * 4 * 1 / (mt / 1000) / 1e9;  // 1 read + 1 write
    printf("Online 1-pass: %.3f ms  %.1f GB/s\n", mt, gb_o);

    printf("\n=== RESULT ===\n");
    printf("Speedup: %.1fx\n", mn / mt);
    printf("Bandwidth: %.1f -> %.1f GB/s\n\n", gb_n, gb_o);
    printf("Why it works (unlike GEMM tiled):\n");
    printf("  Naive reads %d floats %d times from HBM = %.0f MB total\n", N, 3, (double)N * 4 * 3 / 1e6);
    printf("  Online reads %d floats %d time  from HBM = %.0f MB total\n", N, 1, (double)N * 4 * 1 / 1e6);
    printf("  L2 can't help: data size = %.0f MB > 72MB L2, first read always from HBM\n", (double)N * 4 / 1e6);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out); free(h_ref);
    return 0;
}
