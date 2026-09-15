// CUDA C pipeline primitive demonstration. Correctness-only; not a performance benchmark.
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t status__ = (call); \
    if (status__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status__)); \
        throw std::runtime_error(cudaGetErrorString(status__)); \
    } \
} while (0)

template <int STAGES>
__global__ void cp_async_ring_kernel(const float* in, float* out, int count) {
    static_assert(STAGES >= 1 && STAGES <= 3, "STAGES must be in [1, 3]");
    __shared__ __align__(16) float buffer[STAGES][128];
    const int lane = threadIdx.x;
    const int total_tiles = (count + 127) / 128;
    const int initial = total_tiles < STAGES ? total_tiles : STAGES;

#if __CUDA_ARCH__ >= 800
    for (int t = 0; t < initial; ++t) {
        const int idx = t * 128 + lane;
        const bool valid = idx < count;
        const float* source = valid ? (in + idx) : in;
        __pipeline_memcpy_async(&buffer[t % STAGES][lane], source, 4, valid ? 0 : 4);
        __pipeline_commit();
    }
    for (int t = 0; t < total_tiles; ++t) {
        const int remaining = total_tiles - t;
        if (remaining < STAGES) __pipeline_wait_prior(0);
        else __pipeline_wait_prior(STAGES - 1);
        __syncthreads();

        const int idx = t * 128 + lane;
        const int next_lane = (lane + 1) % 128;
        const float self = buffer[t % STAGES][lane];
        const float next = buffer[t % STAGES][next_lane];
        if (idx < count) out[idx] = self + next;

        __syncthreads();
        const int refill = t + STAGES;
        if (refill < total_tiles) {
            const int refill_idx = refill * 128 + lane;
            const bool valid = refill_idx < count;
            const float* source = valid ? (in + refill_idx) : in;
            __pipeline_memcpy_async(&buffer[refill % STAGES][lane], source, 4, valid ? 0 : 4);
            __pipeline_commit();
        }
    }
#else
    (void)in; (void)out; (void)count;
    asm volatile("trap;");
#endif
}

template <int STAGES>
static void run_case(int count) {
    std::vector<float> h_in(count), h_out(count, -1.0f), expected(count);
    for (int i = 0; i < count; ++i) h_in[i] = static_cast<float>(i + 1);
    for (int i = 0; i < count; ++i) expected[i] = h_in[i] + ((i / 128) * 128 + ((i % 128) + 1) % 128 < count
        ? h_in[(i / 128) * 128 + ((i % 128) + 1) % 128] : 0.0f);
    float *d_in = nullptr, *d_out = nullptr;
    try {
        CUDA_CHECK(cudaMalloc(&d_in, count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, count * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), count * sizeof(float), cudaMemcpyHostToDevice));
        cp_async_ring_kernel<STAGES><<<1, 128>>>(d_in, d_out, count);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, count * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < count; ++i)
            if (!std::isfinite(h_out[i]) || h_out[i] != expected[i]) throw std::runtime_error("cp.async result mismatch");
        CUDA_CHECK(cudaFree(d_in)); d_in = nullptr;
        CUDA_CHECK(cudaFree(d_out)); d_out = nullptr;
    } catch (...) { cudaFree(d_in); cudaFree(d_out); throw; }
    std::printf("cp.async STAGES=%d count=%d: OK\n", STAGES, count);
}

int main() {
    try {
        int device = 0; cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDevice(&device)); CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        std::printf("GPU %d: %s (CC %d.%d)\n", device, prop.name, prop.major, prop.minor);
        if (prop.major < 8) { std::fprintf(stderr, "cp.async requires CC >= 8.0\n"); return 2; }
        run_case<1>(1); run_case<1>(128); run_case<1>(129); run_case<1>(1025);
        run_case<2>(1); run_case<2>(128); run_case<2>(129); run_case<2>(1025);
        run_case<3>(1); run_case<3>(128); run_case<3>(129); run_case<3>(1025);
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
    return 0;
}
