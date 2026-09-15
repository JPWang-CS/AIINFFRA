#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>

int main() {
    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) != cudaSuccess) {
        std::fprintf(stderr, "cudaRuntimeGetVersion failed\n");
        return EXIT_FAILURE;
    }
    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuInit failed; check the installed NVIDIA driver\n");
        return EXIT_FAILURE;
    }
    int driver_api_version = 0;
    if (cuDriverGetVersion(&driver_api_version) != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuDriverGetVersion failed\n");
        return EXIT_FAILURE;
    }

    std::printf("headers CUDA_VERSION=%d, linked runtime=%d, driver API=%d\n",
                CUDA_VERSION, runtime_version, driver_api_version);

    constexpr int kRequiredAbi = 11020;  // CUDA 11.2 ABI for cuMemAllocAsync.
    if (driver_api_version < kRequiredAbi) {
        std::printf("driver API is older than requested ABI %d\n", kRequiredAbi);
        return 77;
    }

    PFN_cuMemAllocAsync_v11020 allocate_async = nullptr;
    CUdriverProcAddressQueryResult symbol_status{};
    const CUresult lookup = cuGetProcAddress(
        "cuMemAllocAsync",
        &allocate_async,
        kRequiredAbi,
        CU_GET_PROC_ADDRESS_DEFAULT,
        &symbol_status);
    if (lookup != CUDA_SUCCESS || allocate_async == nullptr) {
        std::fprintf(stderr,
                     "cuGetProcAddress failed: CUresult=%d symbolStatus=%d functionFound=%s\n",
                     static_cast<int>(lookup), static_cast<int>(symbol_status),
                     allocate_async ? "yes" : "no");
        return EXIT_FAILURE;
    }

    std::puts("resolved cuMemAllocAsync using PFN_cuMemAllocAsync_v11020 / ABI 11020");
    return EXIT_SUCCESS;
}
