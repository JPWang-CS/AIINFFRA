#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { \
  const cudaError_t status = (call); \
  if (status != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (false)

// Both kernels require disjoint source/destination allocations.
__global__ void scalar_copy(const float* __restrict__ src,
                            float* __restrict__ dst, std::size_t n) {
  const std::size_t tid = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t step = std::size_t(gridDim.x) * blockDim.x;
  for (std::size_t i = tid; i < n; i += step) dst[i] = src[i];
}

__global__ void aligned_copy(const float* __restrict__ src,
                             float* __restrict__ dst, std::size_t n) {
  const std::size_t tid = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t step = std::size_t(gridDim.x) * blockDim.x;
  const bool aligned = ((reinterpret_cast<std::uintptr_t>(src) |
                         reinterpret_cast<std::uintptr_t>(dst)) & 15u) == 0;
  if (!aligned) {
    for (std::size_t i = tid; i < n; i += step) dst[i] = src[i];
    return;
  }
  const std::size_t vectors = n / 4;
  const float4* src4 = reinterpret_cast<const float4*>(src);
  float4* dst4 = reinterpret_cast<float4*>(dst);
  for (std::size_t v = tid; v < vectors; v += step) dst4[v] = src4[v];
  for (std::size_t i = vectors * 4 + tid; i < n; i += step) dst[i] = src[i];
}

bool run_case(std::size_t n, int offset, int repeats) {
  std::vector<float> input(n), result(n);
  for (std::size_t i = 0; i < n; ++i) input[i] = float(i % 1024) * 0.25f;
  float *src_base = nullptr, *dst_base = nullptr;
  CUDA_CHECK(cudaMalloc(&src_base, (n + 4) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dst_base, (n + 4) * sizeof(float)));
  float* src = src_base + offset;
  float* dst = dst_base + offset;
  CUDA_CHECK(cudaMemcpy(src, input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  const unsigned blocks = unsigned((n + 255) / 256);
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  bool ok = true;
  for (int mode = 0; mode < 2; ++mode) {
    auto launch = [&] {
      if (mode == 0) scalar_copy<<<blocks, 256>>>(src, dst, n);
      else aligned_copy<<<blocks, 256>>>(src, dst, n);
    };
    CUDA_CHECK(cudaMemset(dst, 0xff, n * sizeof(float)));
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result.data(), dst, n * sizeof(float), cudaMemcpyDeviceToHost));
    bool case_ok = true;
    for (std::size_t i = 0; i < n; ++i) {
      // Input is finite; NaN and unwritten values also fail this comparison.
      if (result[i] != input[i]) { case_ok = false; break; }
    }
    ok = ok && case_ok;
    if (!case_ok) {
      std::fprintf(stderr, "FAIL n=%zu offset=%d mode=%d\n", n, offset, mode);
      continue;
    }
    for (int i = 0; i < 5; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float ms = total_ms / repeats;
    const double effective_gbps = ms > 0 ? 2.0 * n * sizeof(float) / (ms * 1e6) : 0;
    std::printf("n=%zu offset=%d mode=%s PASS %.6f ms %.3f effective_GB_s\n",
                n, offset, mode == 0 ? "scalar" : "aligned-or-fallback", ms, effective_gbps);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(src_base));
  CUDA_CHECK(cudaFree(dst_base));
  return ok;
}

int main() {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device=%s cc=%d.%d; same-buffer warm-cache repeats\n",
              prop.name, prop.major, prop.minor);
  bool ok = true;
  for (std::size_t n : {std::size_t(1), std::size_t(3), std::size_t(4),
                       std::size_t(5), std::size_t(257), std::size_t(1 << 22)}) {
    ok = run_case(n, 0, 30) && ok;
    ok = run_case(n, 1, 30) && ok;
  }
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
