#include <cuda/atomic>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

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

__global__ void publish_message(int* payload, int* ready) {
  __shared__ int shared_payload[4];
  __shared__ int shared_ready;
  if (threadIdx.x == 0) {
    shared_ready = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    shared_payload[0] = 40;
    shared_payload[1] = 2;
    shared_payload[2] = 0;
    shared_payload[3] = 0;
    cuda::atomic_ref<int, cuda::thread_scope_block> flag(shared_ready);
    flag.store(1, cuda::memory_order_release);
  }
  // The consumer is in a different warp. This example requires a modern
  // device/runtime with CUDA C++ atomics and independent thread scheduling
  // (compile/run it for compute capability 7.0 or newer).
  if (threadIdx.x == 32) {
    cuda::atomic_ref<int, cuda::thread_scope_block> flag(shared_ready);
    while (flag.load(cuda::memory_order_acquire) != 1) {
    }
    payload[0] = shared_payload[0] + shared_payload[1];
    *ready = 1;
  }
}

}  // namespace

int main() {
  int* device_payload = nullptr;
  int* device_ready = nullptr;
  CUDA_CHECK(cudaMalloc(&device_payload, 4 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&device_ready, sizeof(int)));
  CUDA_CHECK(cudaMemset(device_ready, 0, sizeof(int)));

  publish_message<<<1, 64>>>(device_payload, device_ready);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int result = 0;
  int ready = 0;
  CUDA_CHECK(cudaMemcpy(&result, device_payload, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&ready, device_ready, sizeof(int), cudaMemcpyDeviceToHost));
  std::printf("ready=%d result=%d\n", ready, result);

  CUDA_CHECK(cudaFree(device_ready));
  CUDA_CHECK(cudaFree(device_payload));
  return ready == 1 && result == 42 ? EXIT_SUCCESS : EXIT_FAILURE;
}
