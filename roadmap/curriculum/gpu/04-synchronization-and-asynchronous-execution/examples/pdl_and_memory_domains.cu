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
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__,      \
                         #call, cudaGetErrorString(error__));                 \
            return EXIT_FAILURE;                                              \
        }                                                                     \
    } while (false)

__global__ void primary_kernel(int* produced, int* independent, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) produced[i] = 2 * i + 5;
    __syncthreads();

    // Every thread in every CTA reaches this point.  The trigger permits the
    // dependent grid to begin; it does not publish produced[] to that grid.
    cudaTriggerProgrammaticLaunchCompletion();

    if (i < n) independent[i] = i + 7;
}

__global__ void secondary_kernel(const int* produced,
                                 const int* independent,
                                 int* output, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    volatile int* preamble_output = output;
    if (i < n) preamble_output[i] = i + 13;

    // Every secondary thread waits before reading data written by primary.
    cudaGridDependencySynchronize();

    if (i < n) {
        output[i] = preamble_output[i] + produced[i] + independent[i];
    }
}

__global__ void mark_domain(int* values, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] = i;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--help") == 0) {
        std::printf("usage: %s\n", argv[0]);
        return EXIT_SUCCESS;
    }
    if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--help]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int device = 0;
    cudaDeviceProp properties{};
    const cudaError_t device_status = cudaGetDevice(&device);
    if (device_status == cudaErrorNoDevice) {
        std::fprintf(stderr, "SKIP: CUDA runtime reports no device\n");
        return 77;
    }
    CUDA_CHECK(device_status);
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (properties.major < 9) {
        std::fprintf(stderr, "SKIP: PDL overlap requires compute capability 9.0+\n");
        return 77;
    }

    constexpr int n = 4099;
    constexpr int threads = 128;
    const int blocks = (n + threads - 1) / threads;
    int *produced = nullptr, *independent = nullptr, *output = nullptr;
    std::vector<int> host(n, -1);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMalloc(&produced, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&independent, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&output, n * sizeof(int)));

    cudaLaunchAttribute pdl_attribute{};
    pdl_attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    pdl_attribute.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t secondary_config{};
    secondary_config.gridDim = dim3(blocks, 1, 1);
    secondary_config.blockDim = dim3(threads, 1, 1);
    secondary_config.dynamicSmemBytes = 0;
    secondary_config.stream = stream;
    secondary_config.attrs = &pdl_attribute;
    secondary_config.numAttrs = 1;

    primary_kernel<<<blocks, threads, 0, stream>>>(produced, independent, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaLaunchKernelEx(&secondary_config, secondary_kernel,
                                  produced, independent, output, n));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(host.data(), output, n * sizeof(int),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        const int expected = (i + 13) + (2 * i + 5) + (i + 7);
        if (host[i] != expected) {
            std::fprintf(stderr, "PDL mismatch at %d: %d != %d\n",
                         i, host[i], expected);
            return EXIT_FAILURE;
        }
    }
    std::printf("PDL correctness PASS on %s; no overlap is assumed\n",
                properties.name);

    int domain_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&domain_count,
                                     cudaDevAttrMemSyncDomainCount, device));
    std::printf("memory synchronization domains reported: %d\n", domain_count);
    if (domain_count > 1) {
        cudaLaunchAttribute domain_attributes[2]{};
        domain_attributes[0].id = cudaLaunchAttributeMemSyncDomain;
        domain_attributes[0].val.memSyncDomain = cudaLaunchMemSyncDomainRemote;
        domain_attributes[1].id = cudaLaunchAttributeMemSyncDomainMap;
        domain_attributes[1].val.memSyncDomainMap.default_ = 0;
        domain_attributes[1].val.memSyncDomainMap.remote = 1;

        cudaLaunchConfig_t domain_config{};
        domain_config.gridDim = dim3(blocks, 1, 1);
        domain_config.blockDim = dim3(threads, 1, 1);
        domain_config.stream = stream;
        domain_config.attrs = domain_attributes;
        domain_config.numAttrs = 2;
        CUDA_CHECK(cudaLaunchKernelEx(&domain_config, mark_domain, output, n));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(host.data(), output, n * sizeof(int),
                              cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i) {
            if (host[i] != i) {
                std::fprintf(stderr, "domain launch mismatch at %d\n", i);
                return EXIT_FAILURE;
            }
        }
        std::printf("remote logical-domain launch correctness PASS\n");
    } else {
        std::printf("remote logical-domain launch skipped: fewer than two domains\n");
    }

    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(independent));
    CUDA_CHECK(cudaFree(produced));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
