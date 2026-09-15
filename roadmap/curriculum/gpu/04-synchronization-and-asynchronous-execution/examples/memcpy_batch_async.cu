#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdio>

namespace {
constexpr std::size_t kCopies = 4;
constexpr std::size_t kPinnedCopies = 3;
constexpr std::size_t kElements = 16;
constexpr std::size_t kBytes = kElements * sizeof(int);

bool check(cudaError_t status, const char *where) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(status));
    return false;
}
}  // namespace

int main() {
    std::array<int *, kPinnedCopies> pinned{};
    std::array<void *, kCopies> device{};
    std::array<const void *, kCopies> srcs{};
    std::array<const void *, kCopies> dsts{};
    std::array<std::size_t, kCopies> sizes{};
    std::array<int, kElements> ephemeral{};
    cudaStream_t stream = nullptr;
    bool ok = check(cudaSetDevice(0), "cudaSetDevice");

    if (ok) ok = check(cudaStreamCreateWithFlags(
                         &stream, cudaStreamNonBlocking),
                       "cudaStreamCreateWithFlags");

    for (std::size_t i = 0; ok && i < kPinnedCopies; ++i) {
        ok = check(cudaMallocHost(reinterpret_cast<void **>(&pinned[i]), kBytes),
                   "cudaMallocHost");
        if (!ok) break;
        ok = check(cudaMalloc(&device[i], kBytes), "cudaMalloc");
        if (!ok) break;
        for (std::size_t j = 0; j < kElements; ++j)
            pinned[i][j] = static_cast<int>(100 * i + j);
        srcs[i] = pinned[i];
        dsts[i] = device[i];
        sizes[i] = kBytes;
    }

    if (ok) {
        ok = check(cudaMalloc(&device[kPinnedCopies], kBytes), "cudaMalloc");
        for (std::size_t j = 0; ok && j < kElements; ++j)
            ephemeral[j] = static_cast<int>(900 + j);
        srcs[kPinnedCopies] = ephemeral.data();
        dsts[kPinnedCopies] = device[kPinnedCopies];
        sizes[kPinnedCopies] = kBytes;
    }

    if (ok) {
        cudaMemcpyAttributes attrs[2]{};
        attrs[0].srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        attrs[1].srcAccessOrder = cudaMemcpySrcAccessOrderDuringApiCall;
        std::size_t attrsIdxs[2] = {0, kPinnedCopies};

        // CUDA Runtime API 13.3.0: 8 arguments and const void** pointer arrays.
        ok = check(cudaMemcpyBatchAsync(
                       dsts.data(), srcs.data(), sizes.data(), kCopies,
                       attrs, attrsIdxs, 2, stream),
                   "cudaMemcpyBatchAsync");
    }

    if (stream != nullptr) {
        // Required before host reads device results or releases pinned sources.
        const bool syncOk = check(cudaStreamSynchronize(stream),
                                  "cudaStreamSynchronize");
        ok = ok && syncOk;
    }

    for (std::size_t i = 0; ok && i < kCopies; ++i) {
        std::array<int, kElements> actual{};
        ok = check(cudaMemcpy(actual.data(), device[i], kBytes,
                             cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H");
        for (std::size_t j = 0; ok && j < kElements; ++j) {
            const int expected = static_cast<int>(
                i < kPinnedCopies ? 100 * i + j : 900 + j);
            if (actual[j] != expected) {
                std::fprintf(stderr,
                             "mismatch at copy %zu element %zu: got %d, expected %d\n",
                             i, j, actual[j], expected);
                ok = false;
            }
        }
    }

    // Synchronization above makes these releases safe even after a failed check.
    for (void *ptr : device)
        if (ptr != nullptr) ok = check(cudaFree(ptr), "cudaFree") && ok;
    for (int *ptr : pinned)
        if (ptr != nullptr) ok = check(cudaFreeHost(ptr), "cudaFreeHost") && ok;
    if (stream != nullptr)
        ok = check(cudaStreamDestroy(stream), "cudaStreamDestroy") && ok;

    if (ok) std::puts("PASS: four independent H2D copies verified");
    return ok ? 0 : 1;
}
