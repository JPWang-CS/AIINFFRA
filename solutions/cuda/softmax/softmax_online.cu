// LeetGPU 5_softmax — online softmax (2-pass, fused max+sum)
// 优化 ①：把 findMax + countSum 合成一个 kernel
// 结构：Kernel 1 出 (partial_max, partial_sum) → host 用 online 修正公式 merge → Kernel 2 normalize
//
// vs naive 3-pass:
//   3 kernel → 2 kernel
//   input HBM 读 3× → 2×
//   省一次 kernel launch + 一次全量 HBM 读

#include <cuda_runtime.h>

__global__ void online_kernel(const float* input, float* partial_max,
                               float* partial_sum, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // ── Step A: block 内 reduce 找 local max ──
    __shared__ float smax[256];
    float val = (idx < N) ? input[idx] : -INFINITY;
    smax[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }

    float block_max = smax[0];
    __syncthreads();

    // ── Step B: 用 block_max 算局部指数和 ──
    // 注意：sum 是相对于 block_max 算的，不是全局 max
    // 越界线程 val=-INF → exp(-INF)=0，不污染 sum
    __shared__ float ssum[256];
    ssum[tid] = (idx < N) ? expf(val - block_max) : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_max[blockIdx.x] = block_max;
        partial_sum[blockIdx.x] = ssum[0];
    }
}

__global__ void normalize_kernel(const float* input, float* output,
                                  float global_max, float global_sum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = expf(input[idx] - global_max) / global_sum;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerGrid;

    float *partial_max, *partial_sum;
    cudaMalloc(&partial_max, blocksPerGrid * sizeof(float));
    cudaMalloc(&partial_sum, blocksPerGrid * sizeof(float));

    float* h_max = (float*)malloc(blocksPerGrid * sizeof(float));
    float* h_sum = (float*)malloc(blocksPerGrid * sizeof(float));

    // ── Kernel 1: 一趟出 (partial_max, partial_sum) ──
    online_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, partial_max, partial_sum, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_max, partial_max, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sum, partial_sum, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    // ── Host merge: online 修正公式 ──
    // global_max = max(所有 block_max)
    // global_sum = Σ partial_sum[i] × exp(partial_max[i] - global_max)
    // 每个 block 的 sum 是相对于自己的 block_max 算的，merge 时乘修正因子
    float global_max = -INFINITY;
    for (int i = 0; i < blocksPerGrid; i++) {
        global_max = fmaxf(global_max, h_max[i]);
    }

    float global_sum = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++) {
        global_sum += h_sum[i] * expf(h_max[i] - global_max);
    }

    // ── Kernel 2: normalize ──
    normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, global_max, global_sum, N);
    cudaDeviceSynchronize();

    free(h_max);
    free(h_sum);
    cudaFree(partial_max);
    cudaFree(partial_sum);
}
