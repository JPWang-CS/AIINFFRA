// Correctness-only WMMA example.  It is not a benchmark or a claim of platform performance.
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <vector>

using namespace nvcuda;

#define CUDA_CHECK(call) do { \
    cudaError_t status__ = (call); \
    if (status__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status__)); \
        throw std::runtime_error(cudaGetErrorString(status__)); \
    } \
} while (0)

__global__ void wmma_padded_kernel(const half* A, const half* B, float* C,
                                   int M, int N, int K) {
#if __CUDA_ARCH__ >= 700
    __shared__ __align__(32) half As[256];
    __shared__ __align__(32) half Bs[256];
    __shared__ __align__(32) float Cs[256];
    const int lane = threadIdx.x;
    const int tile_m = blockIdx.y * 16;
    const int tile_k = blockIdx.x * 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int n0 = 0; n0 < N; n0 += 16) {
        // Every lane participates in both loads; out-of-range axes are zero-filled.
        for (int x = lane; x < 256; x += 32) {
            const int r = x / 16;
            const int c = x % 16;
            const int a_m = tile_m + r;
            const int a_n = n0 + c;
            const int b_n = n0 + r;
            const int b_k = tile_k + c;
            As[x] = (a_m < M && a_n < N) ? A[a_m * N + a_n] : __float2half(0.0f);
            Bs[x] = (b_n < N && b_k < K) ? B[b_n * K + b_k] : __float2half(0.0f);
        }
        __syncwarp();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, As, 16);
        wmma::load_matrix_sync(b_frag, Bs, 16);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
        __syncwarp();
    }

    wmma::store_matrix_sync(Cs, acc, 16, wmma::mem_row_major);
    __syncwarp();
    for (int x = lane; x < 256; x += 32) {
        const int r = x / 16;
        const int c = x % 16;
        const int out_m = tile_m + r;
        const int out_k = tile_k + c;
        if (out_m < M && out_k < K) C[out_m * K + out_k] = Cs[x];
    }
#endif
}

static void run_case(int M, int N, int K) {
    const size_t a_count = static_cast<size_t>(M) * N;
    const size_t b_count = static_cast<size_t>(N) * K;
    const size_t c_count = static_cast<size_t>(M) * K;
    std::vector<half> hA(a_count), hB(b_count);
    std::vector<double> reference(c_count, 0.0);
    for (size_t i = 0; i < a_count; ++i) hA[i] = __float2half(static_cast<float>((static_cast<int>(i % 11) - 5) / 8.0));
    for (size_t i = 0; i < b_count; ++i) hB[i] = __float2half(static_cast<float>((static_cast<int>(i % 13) - 6) / 8.0));
    for (int m = 0; m < M; ++m) for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            reference[m * K + k] += static_cast<double>(__half2float(hA[m * N + n])) * __half2float(hB[n * K + k]);

    half *dA = nullptr, *dB = nullptr;
    float *dC = nullptr;
    try {
        CUDA_CHECK(cudaMalloc(&dA, a_count * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dB, b_count * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dC, c_count * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), a_count * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), b_count * sizeof(half), cudaMemcpyHostToDevice));
        const dim3 grid((K + 15) / 16, (M + 15) / 16);
        wmma_padded_kernel<<<grid, 32>>>(dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> actual(c_count);
        CUDA_CHECK(cudaMemcpy(actual.data(), dC, c_count * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < c_count; ++i) {
            if (!std::isfinite(actual[i]) || std::fabs(actual[i] - reference[i]) > 1e-3 + 1e-3 * std::fabs(reference[i]))
                throw std::runtime_error("WMMA result mismatch");
        }
        CUDA_CHECK(cudaFree(dA)); dA = nullptr;
        CUDA_CHECK(cudaFree(dB)); dB = nullptr;
        CUDA_CHECK(cudaFree(dC)); dC = nullptr;
    } catch (...) {
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        throw;
    }
    std::printf("WMMA case %dx%dx%d: OK\n", M, N, K);
}

int main() {
    try {
        int device = 0; cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDevice(&device)); CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        std::printf("GPU %d: %s (CC %d.%d)\n", device, prop.name, prop.major, prop.minor);
        if (prop.major < 7) { std::fprintf(stderr, "WMMA requires CC >= 7.0\n"); return 2; }
        run_case(1, 1, 1); run_case(16, 16, 16); run_case(17, 19, 21); run_case(33, 65, 7);
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
    return 0;
}
