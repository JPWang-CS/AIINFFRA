#include "cpp_device_ops.cuh"
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

template <typename T>
__host__ __device__ constexpr T twice(T value) { return value + value; }

__host__ __device__ int apply_host_device_lambda(int value) {
    auto double_value = [] __host__ __device__ (int x) { return twice(x); };
    return double_value(value);
}

struct DeviceSpan {
    int* data;
    int size;
    __host__ __device__ int load(int i) const { return data[i]; }
    __host__ __device__ void store(int i, int x) const { data[i] = x; }
};

static_assert(std::is_trivially_copyable<DeviceSpan>::value, "plain kernel descriptor");
static_assert(std::is_trivially_destructible<DeviceSpan>::value, "no async destructor");

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t n) : pointer_(nullptr) {
        const cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&pointer_), n * sizeof(int));
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() noexcept {
        if (pointer_) {
            const cudaError_t e = cudaFree(pointer_);
            if (e != cudaSuccess) std::fprintf(stderr, "cudaFree: %s\n", cudaGetErrorString(e));
        }
    }
    DeviceSpan view(int n) const { return DeviceSpan{pointer_, n}; }
    int* data() const { return pointer_; }
private:
    int* pointer_;
};

__global__ void apply_static_shared(DeviceSpan in, DeviceSpan out) {
    // Contract: this kernel is launched with exactly 128 threads per block.
    __shared__ int staging[128];
    const int lane = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + lane;
    staging[lane] = i < in.size ? in.load(i) : 0;
    __syncthreads();
    if (i < in.size) out.store(i, scale_in_other_tu(apply_host_device_lambda(staging[lane])));
}

__global__ void apply_dynamic_shared(DeviceSpan in, DeviceSpan out) {
    extern __shared__ int staging[];
    const int lane = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + lane;
    staging[lane] = i < in.size ? in.load(i) : 0;
    __syncthreads();
    if (i < in.size) out.store(i, scale_in_other_tu(apply_host_device_lambda(staging[lane])));
}

static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
    }
}

int main(int argc, char** argv) {
  try {
    constexpr int n = 5, threads = 128;
    const bool use_static = argc > 1 && std::string(argv[1]) == "--static";
    const std::vector<int> h_in{1, 2, 3, 4, 5};
    std::vector<int> h_out(n, -1);
    {
        DeviceBuffer d_in(n), d_out(n);
        check(cudaMemcpy(d_in.data(), h_in.data(), n * sizeof(int), cudaMemcpyHostToDevice), "H2D");
        const DeviceSpan in = d_in.view(n), out = d_out.view(n);
        const int blocks = (n + threads - 1) / threads;
        if (use_static) apply_static_shared<<<blocks, threads>>>(in, out);
        else apply_dynamic_shared<<<blocks, threads, threads * sizeof(int)>>>(in, out);
        check(cudaGetLastError(), "launch");
        check(cudaDeviceSynchronize(), "kernel completion");
        check(cudaMemcpy(h_out.data(), d_out.data(), n * sizeof(int), cudaMemcpyDeviceToHost), "D2H");
        for (int i = 0; i < n; ++i) {
            if (h_out[i] != h_in[i] * 6) return EXIT_FAILURE;
        }
        std::puts(use_static ? "static shared: PASS" : "dynamic shared: PASS");
        // Owners leave scope after the explicit GPU completion boundary.
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "CUDA C++ example failed: %s\n", error.what());
    return EXIT_FAILURE;
  }
}
