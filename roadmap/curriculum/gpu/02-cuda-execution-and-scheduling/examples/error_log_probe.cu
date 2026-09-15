#include <cuda.h>

#include <cstdio>
#include <cstdlib>

int main() {
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(result, &name);
        std::fprintf(stderr, "cuInit: %s\n", name ? name : "unknown");
        return EXIT_FAILURE;
    }

    // The CUDA Programming Guide documents this invalid call as a source of
    // the Error Log message "buffer cannot be NULL".
    size_t invalid_capacity = 0;
    result = cuLogsDumpToMemory(nullptr, nullptr, &invalid_capacity, 0);
    if (result != CUDA_ERROR_INVALID_VALUE) {
        const char* name = nullptr;
        cuGetErrorName(result, &name);
        std::fprintf(stderr, "expected CUDA_ERROR_INVALID_VALUE, got %s\n",
                     name ? name : "unknown");
        return EXIT_FAILURE;
    }

    char log_buffer[25600] = {};
    size_t bytes_written = sizeof(log_buffer);
    result = cuLogsDumpToMemory(nullptr, log_buffer, &bytes_written, 0);
    if (result != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(result, &name);
        std::fprintf(stderr, "cuLogsDumpToMemory: %s\n", name ? name : "unknown");
        return EXIT_FAILURE;
    }
    if (bytes_written != 0) std::printf("%s", log_buffer);
    return EXIT_SUCCESS;
}
