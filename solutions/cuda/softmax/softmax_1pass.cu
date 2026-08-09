// LeetGPU 5_softmax — softmax 1-pass true online (Flash Attention 的心脏写法)
// 优化 3)：per-thread 同步维护 (m, s) pair，block 内 tree-merge 归并.
// 结构：Kernel 1 一趟读 input 出 (partial_max, partial_sum) → host 修正 merge → Kernel 2 normalize
//
// vs naive 3-pass (softmax_naive.cu):
//   reduce 阶段从 3 趟 HBM 读降到 1 趟（normalize 仍要 1 读 1 写，总流量 2R+1W）
// vs 2-pass fused (softmax_online.cu):
//   省掉"先 block 内 reduce 出 max、再相对 max 算 sum"的第二趟：
//   每个 thread 边扫边维护 (m, s)，一次 tree-merge 直接出 (max, sum)
//   每个 thread 扫自己 block chunk 内的元素 (stride = blockDim.x),
//   (m, s) 全程在寄存器里，只有最后 tree-merge 才进 shared memory
//
// 关键公式:
//   scan:  m_new = max(m, v);  s = s·exp(m - m_new) + exp(v - m_new)
//   merge: s_new = s_a·exp(m_a - m_new) + s_b·exp(m_b - m_new)
//   merge 满足交换律 + 结合律 → 可以上 tree reduce
//   哨兵：空 thread 的 m = -INF，merge 时直接返回另一边，避免 -INF - (-INF) = NaN

#include <cuda_runtime.h>

// merge 两个 (m, s) pair，输出可原地写回 smax[tid] / ssum[tid]:
// ma/sa/mb/sb 都是值传递，函数体内先读完再写，不存在读写冲突.
// 输出用指针不用引用：device 函数不支持引用参数.
__device__ __forceinline__ void merge_pair(float ma, float sa,
                                           float mb, float sb,
                                           float* m_out, float* s_out) {
    if (ma == -INFINITY) { *m_out = mb; *s_out = sb; return; }
    if (mb == -INFINITY) { *m_out = ma; *s_out = sa; return; }
    float m_new = fmaxf(ma, mb);
    *m_out = m_new;
    *s_out = sa * expf(ma - m_new) + sb * expf(mb - m_new);
}

// 一趟读 input：per-thread online scan + block 内 tree-merge，出 (partial_max, partial_sum)
__global__ void maxsum_kernel(const float* input, float* partial_max,
                              float* partial_sum, int N) {
    int tid = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x + tid;
    int block_end = min((blockIdx.x + 1) * blockDim.x, N);

    // -- per-thread online scan：只扫自己 block 的 chunk，stride = blockDim.x --
    // 注意上界是 block_end，不是 N；否则每个 block 都会扫到全局，重复计算.
    float m = -INFINITY;
    float s = 0.0f;
    for (int i = block_start; i < block_end; i += blockDim.x) {
        float v = input[i];
        float m_new = fmaxf(m, v);
        s = s * expf(m - m_new) + expf(v - m_new);
        m = m_new;
    }

    // -- block 内 tree-merge (m, s) pair --
    __shared__ float smax[256];
    __shared__ float ssum[256];
    smax[tid] = m;
    ssum[tid] = s;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            merge_pair(smax[tid], ssum[tid],
                       smax[tid + stride], ssum[tid + stride],
                       &smax[tid], &ssum[tid]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_max[blockIdx.x] = smax[0];
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
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *partial_max, *partial_sum;
    cudaMalloc(&partial_max, blocksPerGrid * sizeof(float));
    cudaMalloc(&partial_sum, blocksPerGrid * sizeof(float));

    float* h_max = (float*)malloc(blocksPerGrid * sizeof(float));
    float* h_sum = (float*)malloc(blocksPerGrid * sizeof(float));

    // -- Kernel 1: 一趟出 (partial_max, partial_sum) --
    maxsum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, partial_max, partial_sum, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_max, partial_max, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sum, partial_sum, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    // -- Host merge: online 修正公式 --
    // global_max = max(所有 block_max)
    // global_sum = Σ partial_sum[i] × exp(partial_max[i] - global_max)
    float global_max = -INFINITY;
    for (int i = 0; i < blocksPerGrid; i++) {
        global_max = fmaxf(global_max, h_max[i]);
    }

    float global_sum = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++) {
        global_sum += h_sum[i] * expf(h_max[i] - global_max);
    }

    // -- Kernel 2: normalize --
    normalize_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, global_max, global_sum, N);
    cudaDeviceSynchronize();

    free(h_max);
    free(h_sum);
    cudaFree(partial_max);
    cudaFree(partial_sum);
}
