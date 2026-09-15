#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); \
        std::exit(1); \
    } \
} while (0)

__global__ void produce(int* values, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] = 2 * i;
}

__global__ void consume(int* values, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] += 1;
}

int main() {
    int count = 0;
    cudaError_t init = cudaGetDeviceCount(&count);
    if (init != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device (%s)\n", cudaGetErrorString(init));
        return 0;
    }
    CUDA_CHECK(cudaSetDevice(0));
    int supported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, 0));
    if (!supported) {
        std::puts("SKIP: stream-ordered memory pools are unsupported");
        return 0;
    }
    constexpr int n = 1025;
    constexpr size_t bytes = n * sizeof(int);
    cudaStream_t producer, consumer, reclaimer;
    cudaEvent_t ready, done;
    CUDA_CHECK(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&reclaimer, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    int* host = nullptr;
    int* device = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&host), bytes));

    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&device), bytes, producer));
    produce<<<(n + 255) / 256, 256, 0, producer>>>(device, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ready, producer));

    CUDA_CHECK(cudaStreamWaitEvent(consumer, ready, 0));
    consume<<<(n + 255) / 256, 256, 0, consumer>>>(device, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, consumer));
    CUDA_CHECK(cudaEventRecord(done, consumer));

    CUDA_CHECK(cudaStreamWaitEvent(reclaimer, done, 0));
    CUDA_CHECK(cudaFreeAsync(device, reclaimer));
    CUDA_CHECK(cudaStreamSynchronize(reclaimer));
    // This synchronization transitively covers produce, consume and the D2H copy.
    for (int i = 0; i < n; ++i) {
        if (host[i] != 2 * i + 1) {
            std::fprintf(stderr, "FAIL: index=%d actual=%d expected=%d\n",
                         i, host[i], 2 * i + 1);
            return 1;
        }
    }
    CUDA_CHECK(cudaFreeHost(host));
    CUDA_CHECK(cudaEventDestroy(done));
    CUDA_CHECK(cudaEventDestroy(ready));
    CUDA_CHECK(cudaStreamDestroy(reclaimer));
    CUDA_CHECK(cudaStreamDestroy(consumer));
    CUDA_CHECK(cudaStreamDestroy(producer));
    std::puts("PASS: cross-stream allocation, data dependency, copy and reclamation");
    return 0;
}
