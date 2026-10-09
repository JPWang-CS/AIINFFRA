#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t error__ = (call);                                        \
        if (error__ != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n",                  \
                         __FILE__, __LINE__, cudaGetErrorString(error__));    \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (false)

__global__ void dependent_chain_kernel(const float* input, float* output,
                                       int n, int steps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = input[i];
    for (int step = 0; step < steps; ++step) {
        x = fmaf(x, 1.000001f, 0.00001f);
    }
    output[i] = x;
}

__global__ void independent_accumulators_kernel(const float* input,
                                                float* output, int n,
                                                int steps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x0 = input[i];
    float x1 = input[i] + 0.1f;
    float x2 = input[i] + 0.2f;
    float x3 = input[i] + 0.3f;
    for (int step = 0; step < steps; ++step) {
        x0 = fmaf(x0, 1.000001f, 0.00001f);
        x1 = fmaf(x1, 1.000001f, 0.00001f);
        x2 = fmaf(x2, 1.000001f, 0.00001f);
        x3 = fmaf(x3, 1.000001f, 0.00001f);
    }
    output[i] = 0.25f * (x0 + x1 + x2 + x3 - 0.6f);
}

__device__ __forceinline__ float even_path(float x) {
    return fmaf(x, 1.25f, 0.5f);
}

__device__ __forceinline__ float odd_path(float x) {
    return fmaf(x, 0.75f, -0.25f);
}

__global__ void divergent_branch_kernel(const float* input, float* output,
                                        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if ((i & 1) == 0) {
        output[i] = even_path(input[i]);
    } else {
        output[i] = odd_path(input[i]);
    }
}

__global__ void predicated_select_kernel(const float* input, float* output,
                                         int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float even_value = even_path(input[i]);
    const float odd_value = odd_path(input[i]);
    output[i] = ((i & 1) == 0) ? even_value : odd_value;
}

void reference_chain(const std::vector<float>& input,
                     std::vector<float>& output, int steps) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        for (int step = 0; step < steps; ++step) {
            x = std::fma(x, 1.000001f, 0.00001f);
        }
        output[i] = x;
    }
}

void reference_independent(const std::vector<float>& input,
                           std::vector<float>& output, int steps) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        float x0 = input[i];
        float x1 = input[i] + 0.1f;
        float x2 = input[i] + 0.2f;
        float x3 = input[i] + 0.3f;
        for (int step = 0; step < steps; ++step) {
            x0 = std::fma(x0, 1.000001f, 0.00001f);
            x1 = std::fma(x1, 1.000001f, 0.00001f);
            x2 = std::fma(x2, 1.000001f, 0.00001f);
            x3 = std::fma(x3, 1.000001f, 0.00001f);
        }
        output[i] = 0.25f * (x0 + x1 + x2 + x3 - 0.6f);
    }
}

void reference_branch(const std::vector<float>& input,
                      std::vector<float>& output) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = (i & 1) ? (input[i] * 0.75f - 0.25f)
                            : (input[i] * 1.25f + 0.5f);
    }
}

float max_abs_error(const std::vector<float>& actual,
                    const std::vector<float>& expected) {
    float error = 0.0f;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            return std::numeric_limits<float>::infinity();
        }
        error = std::max(error, std::fabs(actual[i] - expected[i]));
    }
    return error;
}

template <typename Launch>
float measure(Launch launch, float* device_output,
              std::vector<float>& host_output, int repeats) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(host_output.data(), device_output,
                          host_output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms / static_cast<float>(repeats);
}

}  // namespace

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 1 << 20;
    const int steps = argc > 2 ? std::atoi(argv[2]) : 200;
    const int repeats = argc > 3 ? std::atoi(argv[3]) : 50;
    if (n <= 0 || n > std::numeric_limits<int>::max() - 255 ||
        steps <= 0 || steps > std::numeric_limits<int>::max() / 4 ||
        repeats <= 0) return EXIT_FAILURE;

    std::vector<float> input(n), output(n), reference(n);
    for (int i = 0; i < n; ++i) input[i] = 0.001f * static_cast<float>(i % 1000);

    float* device_input = nullptr;
    float* device_output = nullptr;
    CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    bool all_ok = true;
    auto run = [&](const char* name, int source_fmas, auto launch, auto reference_fn) {
        reference_fn(input, reference);
        const float ms = measure(launch, device_output, output, repeats);
        const float error = max_abs_error(output, reference);
        const bool ok = std::isfinite(error) && error <= 1.0e-4f &&
                        std::isfinite(ms) && ms > 0.0f;
        all_ok = all_ok && ok;
        std::printf("%-14s %.4f ms max_abs_error=%.8g %s", name, ms,
                    error, ok ? "PASS" : "FAIL");
        if (source_fmas > 0 && ms > 0.0f) {
            // Source-level arithmetic count, not an instruction counter.
            const double gflops = 2.0 * n * source_fmas / (ms * 1.0e6);
            std::printf(" source_FMA_GFLOP_s=%.3f", gflops);
        }
        std::printf("\n");
    };

    run("dependent", steps, [&] {
        dependent_chain_kernel<<<blocks, threads>>>(device_input, device_output,
                                                     n, steps);
    }, [&](const auto& a, auto& b) { reference_chain(a, b, steps); });

    run("independent", 4 * steps, [&] {
        independent_accumulators_kernel<<<blocks, threads>>>(
            device_input, device_output, n, steps);
    }, [&](const auto& a, auto& b) { reference_independent(a, b, steps); });

    run("divergent", 0, [&] {
        divergent_branch_kernel<<<blocks, threads>>>(device_input, device_output,
                                                      n);
    }, [&](const auto& a, auto& b) { reference_branch(a, b); });

    run("predicated", 0, [&] {
        predicated_select_kernel<<<blocks, threads>>>(
            device_input, device_output, n);
    }, [&](const auto& a, auto& b) { reference_branch(a, b); });

    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
