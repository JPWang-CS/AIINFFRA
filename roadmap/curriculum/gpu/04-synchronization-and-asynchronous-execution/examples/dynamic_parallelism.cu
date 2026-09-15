#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

__global__ void child_grid(int *output, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = 7 * i + 1;
}

__global__ void parent_grid(int *output, int count, int child_blocks,
                            int *child_launch_status) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        child_grid<<<child_blocks, 128>>>(output, count);
        // Preserve the device-runtime launch result for the host.
        *child_launch_status = static_cast<int>(cudaGetLastError());
    }
}

int main() {
    constexpr int count = 257;
    constexpr int threads = 128;
    const int child_blocks = (count + threads - 1) / threads;
    int *device_output = nullptr;
    int *device_launch_status = nullptr;
    std::vector<int> host_output(count, -777);  // sentinel catches missing child work
    constexpr int kStatusSentinel = -1;
    int host_child_launch_status = kStatusSentinel;

    cudaError_t status = cudaSetDevice(0);
    if (status == cudaSuccess)
        status = cudaMalloc(&device_output, count * sizeof(int));
    if (status == cudaSuccess)
        status = cudaMalloc(&device_launch_status, sizeof(int));
    if (status == cudaSuccess)
        status = cudaMemcpy(device_output, host_output.data(),
                            count * sizeof(int), cudaMemcpyHostToDevice);
    if (status == cudaSuccess)
        status = cudaMemcpy(device_launch_status, &kStatusSentinel,
                            sizeof(int), cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
        parent_grid<<<1, 1>>>(device_output, count, child_blocks,
                              device_launch_status);
        status = cudaGetLastError();
    }
    // Parent completion includes its nested children. Read the separate device
    // launch status even if child execution reports another asynchronous error.
    if (status == cudaSuccess) {
        const cudaError_t execution_status = cudaDeviceSynchronize();
        const cudaError_t read_status = cudaMemcpy(
            &host_child_launch_status, device_launch_status, sizeof(int),
            cudaMemcpyDeviceToHost);
        if (read_status != cudaSuccess) status = read_status;
        else if (host_child_launch_status == kStatusSentinel)
            status = cudaErrorUnknown;
        else if (host_child_launch_status != static_cast<int>(cudaSuccess))
            status = static_cast<cudaError_t>(host_child_launch_status);
        else
            status = execution_status;
    }
    if (status == cudaSuccess)
        status = cudaMemcpy(host_output.data(), device_output,
                            count * sizeof(int), cudaMemcpyDeviceToHost);

    if (status == cudaSuccess) {
        for (int i = 0; i < count; ++i) {
            if (host_output[i] != 7 * i + 1) {
                std::fprintf(stderr, "mismatch at %d\n", i);
                status = cudaErrorUnknown;
                break;
            }
        }
    }
    if (device_output != nullptr) {
        const cudaError_t free_status = cudaFree(device_output);
        if (status == cudaSuccess) status = free_status;
    }
    if (device_launch_status != nullptr) {
        const cudaError_t free_status = cudaFree(device_launch_status);
        if (status == cudaSuccess) status = free_status;
    }
    if (status != cudaSuccess) {
        if (host_child_launch_status != kStatusSentinel &&
            host_child_launch_status != static_cast<int>(cudaSuccess))
            std::fprintf(stderr, "child launch failed: %s\n",
                         cudaGetErrorString(
                             static_cast<cudaError_t>(host_child_launch_status)));
        std::fprintf(stderr, "CUDA/verification error: %s\n",
                     cudaGetErrorString(status));
        return 1;
    }
    std::puts("PASS: parent-child grid output verified");
    return 0;
}
