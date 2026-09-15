#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace {

constexpr int kThreads = 256;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                              \
    if (status != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(status));                               \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (false)

__global__ void block_sum(const float* input, float* partial, int n) {
  __shared__ float tile[kThreads];
  const int tid = threadIdx.x;
  const int index = blockIdx.x * blockDim.x + tid;

  // Tail lanes still execute the barrier protocol. They contribute the
  // identity value instead of returning early.
  tile[tid] = index < n ? input[index] : 0.0f;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      tile[tid] += tile[tid + stride];
    }
    // The barrier separates this phase from the next one. After stride==1
    // there is no later shared-memory phase, so the final barrier is optional.
    if (stride > 1) {
      __syncthreads();
    }
  }

  if (tid == 0) {
    partial[blockIdx.x] = tile[0];
  }
}

}  // namespace

int main() {
  const int n = 100003;  // deliberately not divisible by the block size
  std::vector<float> host_input(n);
  for (int i = 0; i < n; ++i) {
    host_input[i] = 0.25f + static_cast<float>((i * 17) % 101) * 0.01f;
  }

  const int blocks = (n + kThreads - 1) / kThreads;
  float* device_input = nullptr;
  float* device_partial = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_partial, blocks * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, host_input.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  block_sum<<<blocks, kThreads>>>(device_input, device_partial, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> host_partial(blocks);
  CUDA_CHECK(cudaMemcpy(host_partial.data(), device_partial,
                        blocks * sizeof(float), cudaMemcpyDeviceToHost));
  const double gpu_sum = std::accumulate(host_partial.begin(), host_partial.end(), 0.0);
  const double cpu_sum = std::accumulate(host_input.begin(), host_input.end(), 0.0);
  const double error = std::abs(gpu_sum - cpu_sum);
  const double tolerance = 1e-4 + 1e-6 * std::abs(cpu_sum);
  const bool valid = std::isfinite(gpu_sum) && std::isfinite(cpu_sum) &&
                     std::isfinite(error) && error <= tolerance;
  std::printf("n=%d blocks=%d gpu_sum=%.7f cpu_sum=%.7f abs_error=%.7g\n",
              n, blocks, gpu_sum, cpu_sum, error);

  CUDA_CHECK(cudaFree(device_partial));
  CUDA_CHECK(cudaFree(device_input));
  return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
