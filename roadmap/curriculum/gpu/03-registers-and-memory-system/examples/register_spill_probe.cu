#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t error__ = (call);                                        \
        if (error__ != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n",                  \
                         __FILE__, __LINE__, cudaGetErrorString(error__));    \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (false)

// The array is thread-private by CUDA semantics.  Whether it is scalarized
// into registers or placed in local memory is a compiler decision.
__global__ void dynamic_local_array_kernel(const float* input, float* output,
                                           int n, int steps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float values[64];
    #pragma unroll 1
    for (int j = 0; j < 64; ++j) {
        values[j] = input[i] + static_cast<float>(j);
    }

    float result = 0.0f;
    #pragma unroll 1
    for (int step = 0; step < steps; ++step) {
        const int slot = (step * 13 + threadIdx.x) & 63;
        values[slot] = fmaf(values[slot], 1.0001f, 0.25f);
        result += values[slot];
    }
    output[i] = result;
}

// Constant-index accesses provide a comparison point for the compiler.  The
// two kernels intentionally do not have the same array size: the useful
// question is how the generated register/local allocation changes when the
// index is statically visible, not whether the outputs have equal arithmetic.
__global__ void static_index_array_kernel(const float* input, float* output,
                                          int n, int steps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float values[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) values[j] = input[i] + static_cast<float>(j);
    float result = 0.0f;
    for (int step = 0; step < steps; ++step) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            values[j] = fmaf(values[j], 1.0001f, 0.25f);
            result += values[j];
        }
    }
    output[i] = result;
}

float dynamic_reference(float input, int lane, int steps) {
    float values[64];
    for (int j = 0; j < 64; ++j) values[j] = input + static_cast<float>(j);
    float result = 0.0f;
    for (int step = 0; step < steps; ++step) {
        const int slot = (step * 13 + lane) & 63;
        values[slot] = fmaf(values[slot], 1.0001f, 0.25f);
        result += values[slot];
    }
    return result;
}

float static_reference(float input, int steps) {
    float values[8];
    for (int j = 0; j < 8; ++j) values[j] = input + static_cast<float>(j);
    float result = 0.0f;
    for (int step = 0; step < steps; ++step) {
        for (float& value : values) {
            value = std::fma(value, 1.0001f, 0.25f);
            result += value;
        }
    }
    return result;
}

int main() {
    constexpr int n = 256;
    const int steps = 128;
    std::vector<float> host_input(n);
    std::vector<float> host_output(n, 0.0f);
    for (int i = 0; i < n; ++i) host_input[i] = 0.01f * static_cast<float>(i + 1);

    float* input = nullptr;
    float* output = nullptr;
    CUDA_CHECK(cudaMalloc(&input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(input, host_input.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(output, 0, n * sizeof(float)));

    dynamic_local_array_kernel<<<1, n>>>(input, output, n, steps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_output.data(), output, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float dynamic_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float ref = dynamic_reference(host_input[i], i, steps);
        if (!std::isfinite(host_output[i]) || !std::isfinite(ref)) {
            dynamic_error = std::numeric_limits<float>::infinity();
            break;
        }
        dynamic_error = std::max(dynamic_error, std::fabs(host_output[i] - ref));
    }

    CUDA_CHECK(cudaMemset(output, 0, n * sizeof(float)));
    static_index_array_kernel<<<1, n>>>(input, output, n, steps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_output.data(), output, n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float static_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float ref = static_reference(host_input[i], steps);
        if (!std::isfinite(host_output[i]) || !std::isfinite(ref)) {
            static_error = std::numeric_limits<float>::infinity();
            break;
        }
        static_error = std::max(static_error, std::fabs(host_output[i] - ref));
    }

    std::printf("dynamic max_abs_error=%.8g\n", dynamic_error);
    std::printf("static  max_abs_error=%.8g\n", static_error);
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));
    return (std::isfinite(dynamic_error) && std::isfinite(static_error) &&
            dynamic_error <= 1.0e-5f && static_error <= 1.0e-5f)
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
