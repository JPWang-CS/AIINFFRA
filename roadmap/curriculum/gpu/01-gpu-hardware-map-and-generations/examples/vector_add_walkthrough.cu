#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// 教学示例：展示显式分配、复制、启动、检查和释放的完整路径。
// 正常路径检查 CUDA 返回值；异常清理保留原始异常，不覆盖它。
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t error__ = (call);                                         \
    if (error__ != cudaSuccess) {                                               \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                               std::to_string(__LINE__) + ": " +               \
                               cudaGetErrorString(error__));                    \
    }                                                                            \
  } while (false)

__global__ void vector_add_kernel(const float* a, const float* b, float* c,
                                  std::size_t n) {
  // 一个 thread 处理一个逻辑元素；block 是调度/资源分配单位。
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

void vector_add_cpu(const std::vector<float>& a, const std::vector<float>& b,
                    std::vector<float>* c) {
  if (a.size() != b.size() || a.size() != c->size()) {
    throw std::invalid_argument("CPU vectors have different sizes");
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    (*c)[i] = a[i] + b[i];
  }
}

int main(int argc, char** argv) {
  try {
    const std::size_t n = argc > 1 ? std::stoull(argv[1]) : 1000;
    if (n == 0) {
      throw std::invalid_argument("N must be greater than zero");
    }
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
      throw std::invalid_argument("N * sizeof(float) overflows size_t");
    }

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
      throw std::runtime_error("no CUDA-capable device is available");
    }
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp device{};
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
    constexpr unsigned int block_size = 256;
    const std::size_t grid_size = n / block_size + (n % block_size != 0);
    if (grid_size > static_cast<std::size_t>(device.maxGridSize[0])) {
      throw std::invalid_argument("N is too large for this 1D grid");
    }

    std::vector<float> h_a(n), h_b(n), h_c(n), h_reference(n);
    for (std::size_t i = 0; i < n; ++i) {
      h_a[i] = static_cast<float>((i % 97) * 0.25);
      h_b[i] = static_cast<float>((i % 53) * -0.5);
    }
    vector_add_cpu(h_a, h_b, &h_reference);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    try {
      const std::size_t bytes = n * sizeof(float);
      CUDA_CHECK(cudaMalloc(&d_a, bytes));
      CUDA_CHECK(cudaMalloc(&d_b, bytes));
      CUDA_CHECK(cudaMalloc(&d_c, bytes));

      // 这里使用默认 stream 的普通 cudaMemcpy，展示阶段顺序：H2D → kernel。
      // 不把 cudaMemcpy 的“同步”泛化到所有方向和 pageable/pinned 情况。
      CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

      vector_add_kernel<<<static_cast<unsigned int>(grid_size), block_size>>>(
          d_a, d_b, d_c, n);
      CUDA_CHECK(cudaGetLastError());       // 检查 launch 配置/提交错误
      CUDA_CHECK(cudaDeviceSynchronize());  // 检查执行期异步错误

      // D2H 返回后，本例中的 h_c 已可由 CPU 读取并比较。
      CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

      double max_abs_error = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(h_c[i])) {
          throw std::runtime_error("GPU output contains a non-finite value");
        }
        max_abs_error = std::max(
            max_abs_error,
            static_cast<double>(std::fabs(h_c[i] - h_reference[i])));
      }
      if (max_abs_error != 0.0) {
        throw std::runtime_error("GPU result differs from CPU reference: " +
                                 std::to_string(max_abs_error));
      }

      std::cout << "device=" << device.name << " N=" << n
                << " block=" << block_size << " grid=" << grid_size
                << " max_abs_error=" << max_abs_error << " PASS\n";

      CUDA_CHECK(cudaFree(d_c));
      d_c = nullptr;
      CUDA_CHECK(cudaFree(d_b));
      d_b = nullptr;
      CUDA_CHECK(cudaFree(d_a));
      d_a = nullptr;
    } catch (...) {
      // 释放已成功分配的 device memory；不要让错误路径把显存永久留给进程。
      if (d_c != nullptr) cudaFree(d_c);
      if (d_b != nullptr) cudaFree(d_b);
      if (d_a != nullptr) cudaFree(d_a);
      throw;
    }
  } catch (const std::exception& error) {
    std::cerr << "FAILED: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
