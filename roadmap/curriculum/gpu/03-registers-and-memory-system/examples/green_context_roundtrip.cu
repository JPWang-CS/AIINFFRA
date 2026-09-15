#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <vector>

namespace {
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
           status == cudaErrorInvalidResourceConfiguration;
}

__global__ void transform(const int *input, int *output, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = 3 * input[i] + 5;
}
}  // namespace

int main() {
    constexpr int count = 257;
    constexpr int threads = 128;
    const int blocks = (count + threads - 1) / threads;
    std::vector<int> host_input(count);
    std::vector<int> host_output(count, -777);
    for (int i = 0; i < count; ++i) host_input[i] = i;

    int result = 1;
    int device_count = 0;
    int device = 0;
    cudaExecutionContext_t green = nullptr;
    cudaStream_t stream = nullptr;
    int *device_input = nullptr;
    int *device_output = nullptr;

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
        result = 77;
        goto cleanup;
    }
    if (status != cudaSuccess) {
        if (unsupported(status)) {
            std::puts("SKIP: CUDA runtime/driver is unavailable");
            result = 77;
        } else {
            (void)check(status, "cudaGetDeviceCount");
        }
        goto cleanup;
    }
    status = cudaSetDevice(device);
    if (unsupported(status)) {
        std::puts("SKIP: CUDA runtime/driver is unavailable");
        result = 77;
        goto cleanup;
    }
    if (!check(status, "cudaSetDevice")) goto cleanup;

    {
        cudaDevResource all_sms{};
        status = cudaDeviceGetDevResource(
            device, &all_sms, cudaDevResourceTypeSm);
        if (unsupported(status)) {
            std::puts("SKIP: Green Context SM resources are unsupported");
            result = 77;
            goto cleanup;
        }
        if (!check(status, "cudaDeviceGetDevResource")) goto cleanup;
        if (all_sms.sm.smCount < 2 || all_sms.sm.minSmPartitionSize == 0) {
            std::puts("SKIP: device cannot form a non-empty SM partition");
            result = 77;
            goto cleanup;
        }

        unsigned int group_count = 1;
        cudaDevResource selected[1]{};
        cudaDevResource remainder{};
        const unsigned int min_count = std::max(
            all_sms.sm.minSmPartitionSize, all_sms.sm.smCount / 2);
        status = cudaDevSmResourceSplitByCount(
            selected, &group_count, &all_sms, &remainder, 0, min_count);
        if (unsupported(status)) {
            std::puts("SKIP: this device/driver cannot split the SM resource");
            result = 77;
            goto cleanup;
        }
        if (!check(status, "cudaDevSmResourceSplitByCount")) goto cleanup;
        if (group_count != 1 || selected[0].sm.smCount >= all_sms.sm.smCount) {
            std::puts("SKIP: requested non-overlapping SM subset is unavailable");
            result = 77;
            goto cleanup;
        }

        cudaDevResourceDesc_t descriptor{};
        status = cudaDevResourceGenerateDesc(&descriptor, selected, group_count);
        if (unsupported(status)) {
            std::puts("SKIP: Green Context descriptors are unsupported");
            result = 77;
            goto cleanup;
        }
        if (!check(status, "cudaDevResourceGenerateDesc")) goto cleanup;

        status = cudaGreenCtxCreate(&green, descriptor, device, 0);
        if (unsupported(status)) {
            std::puts("SKIP: Green Context creation is unsupported");
            result = 77;
            goto cleanup;
        }
        if (!check(status, "cudaGreenCtxCreate")) goto cleanup;

        // cudaStreamDefault is valid here and has non-blocking Green Context semantics.
        status = cudaExecutionCtxStreamCreate(
            &stream, green, cudaStreamDefault, 0);
        if (unsupported(status)) {
            std::puts("SKIP: Green Context streams are unsupported");
            result = 77;
            goto cleanup;
        }
        if (!check(status, "cudaExecutionCtxStreamCreate")) goto cleanup;

        cudaDevResource context_sms{};
        cudaDevResource stream_sms{};
        status = cudaExecutionCtxGetDevResource(
            green, &context_sms, cudaDevResourceTypeSm);
        if (unsupported(status)) {
            std::puts("SKIP: Green Context resource queries are unsupported");
            result = 77;
            goto cleanup;
        }
        if (!check(status, "cudaExecutionCtxGetDevResource")) goto cleanup;
        status = cudaStreamGetDevResource(
            stream, &stream_sms, cudaDevResourceTypeSm);
        if (unsupported(status)) {
            std::puts("SKIP: Green stream resource queries are unsupported");
            result = 77;
            goto cleanup;
        }
        if (!check(status, "cudaStreamGetDevResource")) goto cleanup;
        if (context_sms.sm.smCount != selected[0].sm.smCount ||
            stream_sms.sm.smCount != selected[0].sm.smCount) {
            std::fprintf(stderr, "FAIL: requested %u SMs, context has %u, stream has %u\n",
                         selected[0].sm.smCount, context_sms.sm.smCount,
                         stream_sms.sm.smCount);
            goto cleanup;
        }

        std::printf("Green Context resource target: %u / %u SMs\n",
                    stream_sms.sm.smCount, all_sms.sm.smCount);
    }

    if (!check(cudaMalloc(&device_input, count * sizeof(int)), "cudaMalloc input"))
        goto cleanup;
    if (!check(cudaMalloc(&device_output, count * sizeof(int)), "cudaMalloc output"))
        goto cleanup;
    if (!check(cudaMemcpyAsync(device_input, host_input.data(),
                              count * sizeof(int), cudaMemcpyHostToDevice, stream),
               "H2D input on Green Context stream")) goto cleanup;
    if (!check(cudaMemcpyAsync(device_output, host_output.data(),
                              count * sizeof(int), cudaMemcpyHostToDevice, stream),
               "initialize output sentinel on Green Context stream")) goto cleanup;

    transform<<<blocks, threads, 0, stream>>>(device_input, device_output, count);
    if (!check(cudaGetLastError(), "Green Context kernel launch")) goto cleanup;
    // Keep both host vectors alive until all same-stream H2D copies and the
    // following kernel have completed. Pageable sources are not promised to
    // transfer asynchronously, but stream order still defines the dependency.
    if (!check(cudaStreamSynchronize(stream), "Green Context stream synchronize"))
        goto cleanup;
    if (!check(cudaMemcpy(host_output.data(), device_output, count * sizeof(int),
                          cudaMemcpyDeviceToHost), "D2H")) goto cleanup;

    for (int i = 0; i < count; ++i) {
        const int expected = 3 * i + 5;
        if (host_output[i] != expected) {
            std::fprintf(stderr, "FAIL: index %d got %d expected %d\n",
                         i, host_output[i], expected);
            goto cleanup;
        }
    }
    result = 0;

cleanup:
    if (stream != nullptr) {
        cleanup_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        cleanup_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    }
    if (green != nullptr)
        cleanup_check(cudaExecutionCtxDestroy(green), "cudaExecutionCtxDestroy");
    if (device_output != nullptr) cleanup_check(cudaFree(device_output), "cudaFree output");
    if (device_input != nullptr) cleanup_check(cudaFree(device_input), "cudaFree input");
    if (result == 0)
        std::puts("PASS: Green Context resource query and output verified");
    return result;
}
