#include <cuda_runtime.h>

#include <cooperative_groups.h>
#include <cuda/ptx>

#include <cstdio>
#include <vector>

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

namespace {
constexpr int kCount = 1'000'003;
constexpr int kThreads = 256;
constexpr int kBlocks = (kCount + kThreads - 1) / kThreads;
constexpr int kOutputSentinel = -777;

bool check(cudaError_t status, const char *where) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(status));
    return false;
}

bool unsupported(cudaError_t status) {
    return status == cudaErrorNotSupported ||
           status == cudaErrorCallRequiresNewerDriver ||
           status == cudaErrorInsufficientDriver ||
           status == cudaErrorInitializationError ||
           status == cudaErrorNoKernelImageForDevice;
}

__device__ void process_tile(int bx, const int *input, int *output,
                             int *visits, int count, int *steal_count) {
    const int i = bx * blockDim.x + threadIdx.x;
    if (i < count) {
        atomicAdd(&visits[i], 1);
        output[i] = 3 * input[i] + 1;
    }
    if (threadIdx.x == 0 && bx != blockIdx.x)
        atomicAdd(steal_count, 1);
}

__global__ void clc_vector_transform(const int *input, int *output,
                                    int *visits, int count,
                                    int *steal_count) {
    __shared__ uint4 cancel_result;
    __shared__ uint64_t cancel_barrier;
    int phase = 0;

    if (cg::thread_block::thread_rank() == 0)
        ptx::mbarrier_init(&cancel_barrier, 1);
    __syncthreads();

    int bx = blockIdx.x;
    while (true) {
        __syncthreads();
        if (cg::thread_block::thread_rank() == 0) {
            ptx::fence_proxy_async_generic_sync_restrict(
                ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);
            cg::invoke_one(cg::coalesced_threads(), [&] {
                ptx::clusterlaunchcontrol_try_cancel(
                    &cancel_result, &cancel_barrier);
            });
            ptx::mbarrier_arrive_expect_tx(
                ptx::sem_relaxed, ptx::scope_cta, ptx::space_shared,
                &cancel_barrier, sizeof(uint4));
        }

        // Useful tile work proceeds while the asynchronous cancellation is pending.
        process_tile(bx, input, output, visits, count, steal_count);

        while (!ptx::mbarrier_try_wait_parity(
            ptx::sem_acquire, ptx::scope_cta, &cancel_barrier, phase)) {}
        phase ^= 1;

        const bool success =
            ptx::clusterlaunchcontrol_query_cancel_is_canceled(cancel_result);
        if (!success) break;  // Never decode an index after a failed request.

        bx = ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(
            cancel_result);
        ptx::fence_proxy_async_generic_sync_restrict(
            ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
    }
}
}  // namespace

int main() {
    int device_count = 0;
    int device = 0;
    int *device_input = nullptr;
    int *device_output = nullptr;
    int *device_visits = nullptr;
    int *device_steal_count = nullptr;
    std::vector<int> input(kCount);
    std::vector<int> output(kCount, kOutputSentinel);
    std::vector<int> visits(kCount, 0);
    int steal_count = 0;
    int result = 1;
    auto cleanup_check = [&](cudaError_t cleanup_status, const char *where) {
        if (cleanup_status != cudaSuccess) {
            std::fprintf(stderr, "cleanup %s: %s\n", where,
                         cudaGetErrorString(cleanup_status));
            if (result == 0) result = 1;
        }
    };

    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice || unsupported(status) ||
        (status == cudaSuccess && device_count == 0)) {
        std::puts("SKIP: no CUDA device");
        return 77;
    }
    if (status != cudaSuccess) {
        if (unsupported(status)) {
            std::puts("SKIP: CUDA runtime/driver is unavailable");
            return 77;
        }
        (void)check(status, "cudaGetDeviceCount");
        return 1;
    }
    status = cudaSetDevice(device);
    if (unsupported(status)) {
        std::puts("SKIP: CUDA runtime/driver is unavailable");
        return 77;
    }
    if (!check(status, "cudaSetDevice")) return 1;

    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    if (unsupported(status)) {
        std::puts("SKIP: device properties are unavailable");
        return 77;
    }
    if (!check(status, "cudaGetDeviceProperties")) return 1;
    if (properties.major < 10) {
        std::puts("SKIP: Cluster Launch Control requires CC 10.0+");
        return 77;
    }

    for (int i = 0; i < kCount; ++i) input[i] = i % 113;
    do {
        if (!check(cudaMalloc(&device_input, kCount * sizeof(int)), "cudaMalloc input"))
            break;
        if (!check(cudaMalloc(&device_output, kCount * sizeof(int)), "cudaMalloc output"))
            break;
        if (!check(cudaMalloc(&device_visits, kCount * sizeof(int)), "cudaMalloc visits"))
            break;
        if (!check(cudaMalloc(&device_steal_count, sizeof(int)), "cudaMalloc steal_count"))
            break;

        if (!check(cudaMemcpy(device_input, input.data(), kCount * sizeof(int),
                              cudaMemcpyHostToDevice), "H2D input")) break;
        if (!check(cudaMemcpy(device_output, output.data(), kCount * sizeof(int),
                              cudaMemcpyHostToDevice), "initialize output sentinel")) break;
        if (!check(cudaMemcpy(device_visits, visits.data(), kCount * sizeof(int),
                              cudaMemcpyHostToDevice), "initialize visit counts")) break;
        if (!check(cudaMemcpy(device_steal_count, &steal_count, sizeof(int),
                              cudaMemcpyHostToDevice), "initialize steal count")) break;

        clc_vector_transform<<<kBlocks, kThreads>>>(
            device_input, device_output, device_visits, kCount,
            device_steal_count);
        status = cudaGetLastError();
        if (unsupported(status)) {
            std::puts("SKIP: runtime/driver does not support this CLC image");
            result = 77;
            break;
        }
        if (!check(status, "CLC kernel launch")) break;
        status = cudaDeviceSynchronize();
        if (unsupported(status)) {
            std::puts("SKIP: runtime/driver does not support CLC execution");
            result = 77;
            break;
        }
        if (!check(status, "cudaDeviceSynchronize")) break;

        if (!check(cudaMemcpy(output.data(), device_output, kCount * sizeof(int),
                              cudaMemcpyDeviceToHost), "D2H output")) break;
        if (!check(cudaMemcpy(visits.data(), device_visits, kCount * sizeof(int),
                              cudaMemcpyDeviceToHost), "D2H visit counts")) break;
        if (!check(cudaMemcpy(&steal_count, device_steal_count, sizeof(int),
                              cudaMemcpyDeviceToHost), "D2H steal count")) break;

        for (int i = 0; i < kCount; ++i) {
            if (visits[i] != 1 || output[i] != 3 * input[i] + 1) {
                std::fprintf(stderr,
                             "FAIL: index %d visits=%d output=%d expected=%d\n",
                             i, visits[i], output[i], 3 * input[i] + 1);
                goto cleanup;
            }
        }
        result = 0;
    } while (false);

cleanup:
    if (device_steal_count != nullptr)
        cleanup_check(cudaFree(device_steal_count), "cudaFree steal_count");
    if (device_visits != nullptr) cleanup_check(cudaFree(device_visits), "cudaFree visits");
    if (device_output != nullptr) cleanup_check(cudaFree(device_output), "cudaFree output");
    if (device_input != nullptr) cleanup_check(cudaFree(device_input), "cudaFree input");
    if (result == 0)
        std::printf("PASS: all %d elements processed once; CLC steals=%d\n",
                    kCount, steal_count);
    return result;
}
