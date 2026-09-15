#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

constexpr int kTile = 32;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t error__ = (call);                                        \
        if (error__ != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n",                  \
                         __FILE__, __LINE__, cudaGetErrorString(error__));    \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (false)

__global__ void transpose_naive_kernel(const float* input, float* output,
                                       int rows, int cols) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        // input is rows x cols; output is cols x rows.
        output[col * rows + row] = input[row * cols + col];
    }
}

template <bool Padded>
__global__ void transpose_tiled_kernel(const float* input, float* output,
                                       int rows, int cols) {
    // Every thread executes both the load phase and the barrier.  The padded
    // layout changes only the physical shared-memory row stride.
    __shared__ float tile[kTile][Padded ? kTile + 1 : kTile];

    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
    const bool input_valid = row < rows && col < cols;

    tile[threadIdx.y][threadIdx.x] =
        input_valid ? input[row * cols + col] : 0.0f;

    __syncthreads();

    // The logical transposed coordinate is (col, row).  A thread whose
    // transposed output is outside the rectangle simply skips the store; it
    // does not leave the block before the first barrier.
    const int output_row = blockIdx.x * kTile + threadIdx.y;
    const int output_col = blockIdx.y * kTile + threadIdx.x;
    if (output_row < cols && output_col < rows) {
        output[output_row * rows + output_col] =
            tile[threadIdx.x][threadIdx.y];
    }
}

void transpose_cpu(const std::vector<float>& input, std::vector<float>& output,
                   int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            output[col * rows + row] = input[row * cols + col];
        }
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
float run_and_measure(Launch launch, float* device_output, std::size_t output_size,
                      std::vector<float>& host_output, int repeats) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(host_output.data(), device_output,
                          output_size * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms / static_cast<float>(repeats);
}

bool report(const char* name, float ms, float error, int rows, int cols) {
    const double bytes = 2.0 * static_cast<double>(rows) * cols * sizeof(float);
    const double gb_per_second = bytes / (static_cast<double>(ms) * 1.0e6);
    constexpr float kTolerance = 1.0e-6f;
    const bool pass = std::isfinite(error) && error <= kTolerance;
    std::printf("%-8s  %.4f ms  %.3f GB/s  max_abs_error=%.8g  %s\n",
                name, ms, gb_per_second, error, pass ? "PASS" : "FAIL");
    return pass;
}

}  // namespace

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 37;
    const int cols = argc > 2 ? std::atoi(argv[2]) : 65;
    const int repeats = argc > 3 ? std::atoi(argv[3]) : 50;
    if (rows <= 0 || cols <= 0 || repeats <= 0 ||
        rows > std::numeric_limits<int>::max() - kTile ||
        cols > std::numeric_limits<int>::max() - kTile) {
        std::fprintf(stderr, "usage: %s [rows] [cols] [repeats]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (static_cast<std::int64_t>(rows) * cols >
        static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "rows*cols is too large for this example\n");
        return EXIT_FAILURE;
    }
    const std::size_t input_size = static_cast<std::size_t>(rows) * cols;
    const std::size_t output_size = static_cast<std::size_t>(cols) * rows;
    std::vector<float> host_input(input_size);
    std::vector<float> host_reference(output_size, 0.0f);
    std::vector<float> host_output(output_size, 0.0f);
    for (std::size_t i = 0; i < input_size; ++i) {
        host_input[i] = static_cast<float>(((i % 101) * 17 + 3) % 101) * 0.125f;
    }
    transpose_cpu(host_input, host_reference, rows, cols);

    float* device_input = nullptr;
    float* device_output = nullptr;
    CUDA_CHECK(cudaMalloc(&device_input, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_output, output_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_input, host_input.data(),
                          input_size * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 naive_block(32, 8);
    const dim3 naive_grid((cols + naive_block.x - 1) / naive_block.x,
                          (rows + naive_block.y - 1) / naive_block.y);
    const dim3 tiled_block(kTile, kTile);
    const dim3 tiled_grid((cols + kTile - 1) / kTile,
                          (rows + kTile - 1) / kTile);

    const float naive_ms = run_and_measure(
        [&] { transpose_naive_kernel<<<naive_grid, naive_block>>>(
                  device_input, device_output, rows, cols); },
        device_output, output_size, host_output, repeats);
    bool all_ok = report("naive", naive_ms,
                         max_abs_error(host_output, host_reference), rows, cols);

    const float unpadded_ms = run_and_measure(
        [&] { transpose_tiled_kernel<false><<<tiled_grid, tiled_block>>>(
                  device_input, device_output, rows, cols); },
        device_output, output_size, host_output, repeats);
    all_ok = report("tile32", unpadded_ms,
                    max_abs_error(host_output, host_reference), rows, cols) && all_ok;

    const float padded_ms = run_and_measure(
        [&] { transpose_tiled_kernel<true><<<tiled_grid, tiled_block>>>(
                  device_input, device_output, rows, cols); },
        device_output, output_size, host_output, repeats);
    all_ok = report("tile33", padded_ms,
                    max_abs_error(host_output, host_reference), rows, cols) && all_ok;

    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
