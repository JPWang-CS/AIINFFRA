#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace ptx = cuda::ptx;
using barrier_t = cuda::barrier<cuda::thread_scope_block>;
constexpr int Tile = 1024;
constexpr int Threads = 128;

#define CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, \
                     #call, cudaGetErrorString(error)); \
        std::exit(1); \
    } \
} while (0)

__global__ void tma_roundtrip(const int* input, int* output, int tiles) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    __shared__ alignas(16) int buffer[Tile];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier_t full;
    if (threadIdx.x == 0) init(&full, blockDim.x);
    __syncthreads();

    // All lanes participate in the shuffle; only warp 0 performs election.
    const unsigned warp = __shfl_sync(0xffffffffu, threadIdx.x / 32, 0);
    const bool leader = warp == 0 && ptx::elect_sync(0xffffffffu);
    for (int t = 0; t < tiles; ++t) {
        const int offset = t * Tile;
        if (leader) {
            // This overload accounts for the transfer automatically.
            cuda::memcpy_async(buffer, input + offset,
                               cuda::aligned_size_t<16>(sizeof(buffer)), full);
        }
        auto token = full.arrive();
        full.wait(std::move(token));

        for (int i = threadIdx.x; i < Tile; i += blockDim.x) {
            buffer[i] += 1;
        }
        // Every writer publishes its own generic-proxy shared writes.
        ptx::fence_proxy_async(ptx::space_shared);
        __syncthreads();

        if (leader) {
            ptx::cp_async_bulk(ptx::space_global, ptx::space_shared,
                               output + offset, buffer, sizeof(buffer));
            ptx::cp_async_bulk_commit_group();
            ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>{});
        }
        // Publish the issuing thread's source-read completion to the whole CTA.
        // Only after this point may the next iteration overwrite buffer.
        __syncthreads();
    }
#else
    // Never silently substitute an empty kernel for the unsupported device path.
    asm volatile("trap;");
#endif
}

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::puts("Usage: tma_roundtrip [--help]\n"
                  "Checks padded 1-D TMA roundtrips, including repeated slot reuse.\n"
                  "CUDA Toolkit 13.3 headers; compile for sm_90 or a supported newer target.\n"
                  "This is a correctness example, not an overlap benchmark.");
        return 0;
    }
    if (argc != 1) return 2;
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || (status == cudaSuccess && count == 0)) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    CHECK(status);
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 9) {
        std::puts("SKIP: bulk TMA needs a supported CC 9.0+ target");
        return 77;
    }
    int runtime = 0, driver = 0;
    CHECK(cudaRuntimeGetVersion(&runtime));
    CHECK(cudaDriverGetVersion(&driver));
    std::printf("GPU=%s CC=%d.%d runtime=%d driver=%d tile=%d threads=%d\n",
                prop.name, prop.major, prop.minor, runtime, driver, Tile, Threads);

    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const int sizes[] = {1, 1023, 1024, 1025, 3077};
    bool passed = true;
    for (int n : sizes) {
        const int tiles = (n + Tile - 1) / Tile;
        const int padded = tiles * Tile;
        const int guard = 16;
        std::vector<int> input(padded, -71);
        std::vector<int> output(padded + guard, -999);
        for (int i = 0; i < n; ++i) input[i] = (i % 257) - 128;
        int *device_in = nullptr, *device_out = nullptr;
        CHECK(cudaMalloc(reinterpret_cast<void**>(&device_in), padded * sizeof(int)));
        CHECK(cudaMalloc(reinterpret_cast<void**>(&device_out), output.size() * sizeof(int)));
        // Synchronous copies are outside the experiment; pageable host vectors are fine.
        CHECK(cudaMemcpy(device_in, input.data(), padded * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(device_out, output.data(), output.size() * sizeof(int), cudaMemcpyHostToDevice));
        tma_roundtrip<<<1, Threads, 0, stream>>>(device_in, device_out, tiles);
        CHECK(cudaGetLastError());
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaMemcpy(output.data(), device_out, output.size() * sizeof(int), cudaMemcpyDeviceToHost));
        bool case_ok = true;
        for (int i = 0; i < padded + guard; ++i) {
            const int expected = i < padded ? input[i] + 1 : -999;
            if (output[i] != expected) {
                std::fprintf(stderr, "FAIL n=%d i=%d got=%d expected=%d\n",
                             n, i, output[i], expected);
                case_ok = false;
                break;
            }
        }
        CHECK(cudaFree(device_out));
        CHECK(cudaFree(device_in));
        passed = passed && case_ok;
        std::printf("n=%d padded=%d tiles=%d: %s\n", n, padded, tiles,
                    case_ok ? "PASS" : "FAIL");
    }
    CHECK(cudaStreamDestroy(stream));
    return passed ? 0 : 1;
}
