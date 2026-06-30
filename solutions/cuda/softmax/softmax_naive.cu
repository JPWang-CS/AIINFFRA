// LeetGPU 5_softmax — 3-pass naive (跨 block 归约)
// 结构：Kernel 1 找全局 max → Kernel 2 算全局 sum → Kernel 3 normalize

#include <cuda_runtime.h>

__global__ void findMax_kernel(const float* input, float* partial_max, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    __shared__ float inputMax[256];
    inputMax[tid] = (idx < N) ? input[idx] : -INFINITY;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            inputMax[tid] = fmaxf(inputMax[tid], inputMax[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_max[blockIdx.x] = inputMax[0];
    }
}

__global__ void countSum_kernel(const float* input, float* partial_sum, float global_max, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    __shared__ float inputSum[256];
    float val = (idx < N) ? expf(input[idx] - global_max) : 0.0f;
    inputSum[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            inputSum[tid] += inputSum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sum[blockIdx.x] = inputSum[0];
    }
}

__global__ void softmax_kernel(const float* input, float* output, float global_max, float global_sum, int N) {
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

    float* partial_host = (float*)malloc(blocksPerGrid * sizeof(float));

    // Step 1: 找全局 max
    findMax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, partial_max, N);
    cudaDeviceSynchronize();
    cudaMemcpy(partial_host, partial_max, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    float global_max = -INFINITY;
    for (int i = 0; i < blocksPerGrid; i++) {
        global_max = fmaxf(global_max, partial_host[i]);
    }

    // Step 2: 算全局 sum
    countSum_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, partial_sum, global_max, N);
    cudaDeviceSynchronize();
    cudaMemcpy(partial_host, partial_sum, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    float global_sum = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++) {
        global_sum += partial_host[i];
    }

    // Step 3: normalize
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, global_max, global_sum, N);
    cudaDeviceSynchronize();

    free(partial_host);
    cudaFree(partial_max);
    cudaFree(partial_sum);
}
