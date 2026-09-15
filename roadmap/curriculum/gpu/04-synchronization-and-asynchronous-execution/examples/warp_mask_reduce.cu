#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                              \
    if (status != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(status));                               \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (false)

__global__ void one_warp_sum(const float* input, float* output, int active_count) {
  const int lane = threadIdx.x;
  const bool active = lane < active_count;

  // All 32 lanes execute every shuffle with the same full mask. Invalid lanes
  // contribute the identity value, so no shuffle reads a non-participating
  // source lane. This is the simplest correct tail reduction.
  float value = active ? input[lane] : 0.0f;

  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  if (lane == 0) {
    *output = value;
  }
}

}  // namespace

int main() {
  const int active_count = 19;
  std::vector<float> host_input(active_count);
  for (int i = 0; i < active_count; ++i) {
    host_input[i] = static_cast<float>(i + 1);
  }

  float* device_input = nullptr;
  float* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, 32 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, host_input.data(),
                        active_count * sizeof(float), cudaMemcpyHostToDevice));

  one_warp_sum<<<1, 32>>>(device_input, device_output, active_count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float gpu_sum = 0.0f;
  CUDA_CHECK(cudaMemcpy(&gpu_sum, device_output, sizeof(float),
                        cudaMemcpyDeviceToHost));
  const float cpu_sum = std::accumulate(host_input.begin(), host_input.end(), 0.0f);
  std::printf("active_lanes=%d gpu_sum=%.3f cpu_sum=%.3f\n",
              active_count, gpu_sum, cpu_sum);

  CUDA_CHECK(cudaFree(device_output));
  CUDA_CHECK(cudaFree(device_input));
  const float error = std::abs(gpu_sum - cpu_sum);
  return std::isfinite(gpu_sum) && std::isfinite(cpu_sum) &&
                 std::isfinite(error) && error < 1e-5f
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
