#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <mma.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace wmma = nvcuda::wmma;
namespace ptx = cuda::ptx;
using Barrier = cuda::barrier<cuda::thread_scope_block>;
constexpr int Tile = 16;
constexpr unsigned Mask = 0xffffffffu;
void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error));
        std::exit(1);
    }
}
void driver_check(CUresult error) {
    if (error != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(error, &message);
        std::fprintf(stderr, "Driver: %s\n", message ? message : "unknown error");
        std::exit(1);
    }
}
CUtensorMap make_map(half* data, int rows, int cols) {
    alignas(64) CUtensorMap map{};
    const uint64_t dimensions[2] = {uint64_t(cols), uint64_t(rows)};
    const uint64_t strides[1] = {uint64_t(cols) * sizeof(half)};
    const uint32_t box[2] = {Tile, Tile}, element_strides[2] = {1, 1};
    driver_check(cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, data, dimensions, strides,
        box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

#if __CUDA_ARCH__ >= 900
__device__ __forceinline__ Barrier::arrival_token submit_ab(
    const CUtensorMap* Amap, const CUtensorMap* Bmap, half* a, half* b,
    Barrier& ready, bool leader, int m0, int n0, int k0) {
    if (leader) {
        const int32_t ac[2] = {n0, m0}, bc[2] = {k0, n0};
        auto* handle = cuda::device::barrier_native_handle(ready);
        ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                                 a, Amap, ac, handle);
        ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                                 b, Bmap, bc, handle);
        // One arrival from the leader, accounting for BOTH complete tiles.
        return cuda::device::barrier_arrive_tx(ready, 1, 2 * 256 * sizeof(half));
    }
    return ready.arrive();
}
#endif

template<int Stages>
__global__ void gemm(const __grid_constant__ CUtensorMap Amap,
                     const __grid_constant__ CUtensorMap Bmap,
                     float* C, int Np, int Kp) {
#if __CUDA_ARCH__ >= 900
    static_assert(Stages == 1 || Stages == 2);
    __shared__ __align__(128) half a[Stages][256];
    __shared__ __align__(128) half b[Stages][256];
    __shared__ __align__(32) float out[256];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ Barrier ready[Stages];
    if (threadIdx.x == 0)
        for (int s = 0; s < Stages; ++s) init(&ready[s], 32);
    __syncthreads();
    const bool leader = ptx::elect_sync(Mask);
    Barrier::arrival_token tokens[Stages];
    const int m0 = blockIdx.y * Tile, k0 = blockIdx.x * Tile;
    const int rounds = Np / Tile;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    if constexpr (Stages == 2) {
        tokens[0] = submit_ab(&Amap, &Bmap, a[0], b[0], ready[0], leader, m0, 0, k0);
        ready[0].wait(std::move(tokens[0]));
        __syncwarp(Mask);
    }
    for (int t = 0; t < rounds; ++t) {
        const int current = t % Stages;
        if constexpr (Stages == 1) {
            tokens[0] = submit_ab(&Amap, &Bmap, a[0], b[0], ready[0], leader, m0, t * Tile, k0);
            ready[0].wait(std::move(tokens[0]));
            __syncwarp(Mask);
        } else if (t + 1 < rounds) {
            const int next = (t + 1) % Stages;
            tokens[next] = submit_ab(&Amap, &Bmap, a[next], b[next], ready[next],
                                     leader, m0, (t + 1) * Tile, k0);
        }
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
        wmma::load_matrix_sync(af, a[current], Tile);
        wmma::load_matrix_sync(bf, b[current], Tile);
        wmma::mma_sync(acc, af, bf, acc);
        __syncwarp(Mask);
        if constexpr (Stages == 2) {
            if (t + 1 < rounds) {
                const int next = (t + 1) % Stages;
                ready[next].wait(std::move(tokens[next]));
                __syncwarp(Mask);
            }
        }
    }
    wmma::store_matrix_sync(out, acc, Tile, wmma::mem_row_major);
    __syncwarp(Mask);
    for (int i = threadIdx.x; i < 256; i += 32)
        C[size_t(m0 + i / Tile) * Kp + k0 + i % Tile] = out[i];
#else
    asm volatile("trap;");
#endif
}

void launch(int stages, const CUtensorMap& Amap, const CUtensorMap& Bmap,
            float* C, int Mp, int Np, int Kp) {
    const dim3 grid(Kp / Tile, Mp / Tile);
    if (stages == 1) gemm<1><<<grid, 32>>>(Amap, Bmap, C, Np, Kp);
    else gemm<2><<<grid, 32>>>(Amap, Bmap, C, Np, Kp);
}

bool run_case(int M, int N, int K, bool benchmark) {
    const int Mp = (M + 15) / 16 * 16;
    const int Np = (N + 15) / 16 * 16;
    const int Kp = (K + 15) / 16 * 16;
    const size_t count = size_t(Mp) * Kp;
    std::vector<half> A(size_t(Mp) * Np, __float2half(0));
    std::vector<half> B(size_t(Np) * Kp, __float2half(0));
    std::vector<double> reference(count, 0);
    std::vector<float> result(count + 16, -9876.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            A[size_t(m) * Np + n] = __float2half(float((m * 3 + n) % 17 - 8) / 16.0f);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            B[size_t(n) * Kp + k] = __float2half(float((n + k * 5) % 19 - 9) / 16.0f);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                reference[size_t(m) * Kp + k] += double(__half2float(A[size_t(m) * Np + n])) * __half2float(B[size_t(n) * Kp + k]);

    half *dA = nullptr, *dB = nullptr;
    float* dC = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&dA), A.size() * sizeof(half)));
    check(cudaMalloc(reinterpret_cast<void**>(&dB), B.size() * sizeof(half)));
    check(cudaMalloc(reinterpret_cast<void**>(&dC), result.size() * sizeof(float)));
    check(cudaMemcpy(dA, A.data(), A.size() * sizeof(half), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dB, B.data(), B.size() * sizeof(half), cudaMemcpyHostToDevice));
    const CUtensorMap Amap = make_map(dA, Mp, Np);
    const CUtensorMap Bmap = make_map(dB, Np, Kp);
    bool ok = true;
    for (int stages : {1, 2}) {
        std::fill(result.begin(), result.end(), -9876.0f);
        check(cudaMemcpy(dC, result.data(), result.size() * sizeof(float), cudaMemcpyHostToDevice));
        launch(stages, Amap, Bmap, dC, Mp, Np, Kp);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(result.data(), dC, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
        bool valid = true;
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(result[i]) || std::abs(result[i] - reference[i]) > 1e-3 + 1e-3 * std::abs(reference[i])) {
                std::fprintf(stderr, "FAIL stages=%d index=%zu actual=%g expected=%g\n", stages, i, result[i], reference[i]);
                valid = false; break;
            }
        }
        for (size_t i = count; i < result.size(); ++i) {
            if (result[i] != -9876.0f) { valid = false; std::fprintf(stderr, "FAIL output guard\n"); break; }
        }
        std::printf("M=%d N=%d K=%d padded=%dx%dx%d stages=%d: %s\n",
                    M, N, K, Mp, Np, Kp, stages, valid ? "PASS" : "FAIL");
        ok = ok && valid;
    }
    if (benchmark && ok) {
        constexpr int Warmup = 10, Iterations = 100;
        cudaEvent_t start, stop;
        check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
        for (int stages : {1, 2}) {
            for (int i = 0; i < Warmup; ++i) launch(stages, Amap, Bmap, dC, Mp, Np, Kp);
            check(cudaGetLastError()); check(cudaDeviceSynchronize());
            check(cudaEventRecord(start));
            for (int i = 0; i < Iterations; ++i) launch(stages, Amap, Bmap, dC, Mp, Np, Kp);
            check(cudaGetLastError()); check(cudaEventRecord(stop));
            check(cudaEventSynchronize(stop));
            float total_ms = 0; check(cudaEventElapsedTime(&total_ms, start, stop));
            const double ms = total_ms / Iterations;
            std::printf("stages=%d mean_ms=%.6f valid_TFLOPS=%.4f padded_TFLOPS=%.4f warmup=%d iterations=%d\n",
                        stages, ms, (2.0*M*N*K)/(ms*1e9), (2.0*Mp*Np*Kp)/(ms*1e9), Warmup, Iterations);
        }
        check(cudaEventDestroy(stop)); check(cudaEventDestroy(start));
    }
    check(cudaFree(dC)); check(cudaFree(dB)); check(cudaFree(dA));
    return ok;
}

int main(int argc, char** argv) {
    bool benchmark = false;
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::puts("Usage: tma_wmma_pipeline [--bench]\n"
                  "One-warp, FP16 input / FP32 output, serial vs double-buffer TMA schedule.\n"
                  "Default: small correctness. --bench: also validates and times 256x512x256.\n"
                  "Timing excludes padding, allocation, copies and validation; not a library comparison.");
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) benchmark = true;
    else if (argc != 1) return 2;
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || (status == cudaSuccess && count == 0)) {
        std::puts("SKIP: no CUDA device"); return 77;
    }
    check(status); check(cudaSetDevice(0)); driver_check(cuInit(0));
    cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 9) { std::puts("SKIP: TMA path needs supported CC9.0+"); return 77; }
    int runtime = 0, driver = 0;
    check(cudaRuntimeGetVersion(&runtime)); check(cudaDriverGetVersion(&driver));
    std::printf("GPU=%s CC=%d.%d runtime=%d driver=%d block=32 tile=16x16x16\n",
                prop.name, prop.major, prop.minor, runtime, driver);
    const int shapes[][3] = {{1,1,1}, {16,16,16}, {17,19,21}, {33,65,7}, {32,48,32}};
    bool ok = true;
    for (const auto& s : shapes) ok = run_case(s[0], s[1], s[2], false) && ok;
    if (benchmark && ok) ok = run_case(256, 512, 256, true);
    return ok ? 0 : 1;
}
