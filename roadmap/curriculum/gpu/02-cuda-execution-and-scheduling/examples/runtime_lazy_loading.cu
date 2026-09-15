#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        const cudaError_t error__ = (call);                                   \
        if (error__ != cudaSuccess) {                                         \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__,     \
                         #call, cudaGetErrorString(error__));                 \
            return EXIT_FAILURE;                                              \
        }                                                                     \
    } while (false)

__global__ void add_one(const int* input, int* output, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i] + 1;
}

}  // namespace

int main(int argc, char** argv) {
    bool preload = false;
    if (argc == 2 && std::strcmp(argv[1], "--preload") == 0) {
        preload = true;
    } else if (argc > 1) {
        std::printf("usage: %s [--preload]\n", argv[0]);
        return argc == 2 && std::strcmp(argv[1], "--help") == 0
                   ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    constexpr int n = 257;
    constexpr int threads = 128;
    const int blocks = (n + threads - 1) / threads;
    std::vector<int> input(n), output(n, -1);
    for (int i = 0; i < n; ++i) input[i] = i * 3;
    int* device_input = nullptr;
    int* device_output = nullptr;

    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status == cudaErrorNoDevice ||
        (count_status == cudaSuccess && device_count == 0)) {
        std::fprintf(stderr, "SKIP: CUDA runtime reports no device\n");
        return 77;
    }
    CUDA_CHECK(count_status);

    // This explicit no-op allocation forces Runtime/context initialization;
    // it does not, by itself, prove every kernel module has been loaded.
    CUDA_CHECK(cudaFree(nullptr));
    CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(int),
                          cudaMemcpyHostToDevice));

    if (preload) {
        cudaFuncAttributes attributes{};
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, add_one));
        std::printf("kernel preloaded: registers=%d\n", attributes.numRegs);
    } else {
        std::printf("first kernel launch is the first use of add_one\n");
    }

    add_one<<<blocks, threads>>>(device_input, device_output, n);
    // Immediate check reports launch/configuration errors.  Execution faults
    // are observed at the stream synchronization boundary below.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(output.data(), device_output, n * sizeof(int),
                          cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (output[i] != input[i] + 1) {
            std::fprintf(stderr, "mismatch at %d: got %d expected %d\n",
                         i, output[i], input[i] + 1);
            ok = false;
            break;
        }
    }
    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_input));
    if (!ok) return EXIT_FAILURE;
    std::printf("PASS: %d elements; module loading mode is selected before process start\n", n);
    return EXIT_SUCCESS;
}
