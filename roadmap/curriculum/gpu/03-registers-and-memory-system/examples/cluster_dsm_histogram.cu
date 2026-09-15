#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace cg = cooperative_groups;

namespace {
constexpr int kClusterBlocks = 2;
constexpr int kThreads = 128;
constexpr int kBins = 32;
constexpr int kBinsPerBlock = kBins / kClusterBlocks;
constexpr int kSkipExitCode = 77;
constexpr unsigned long long kMaxElements = 1ULL << 24;

__global__ void cluster_histogram(const int* input, std::size_t n,
                                  unsigned int* histogram) {
    extern __shared__ unsigned int local_bins[];
    cg::cluster_group cluster = cg::this_cluster();
    const unsigned int block_rank = cluster.block_rank();

    // Shared memory belongs to this CTA. Each CTA initializes its own 16 bins.
    for (int i = threadIdx.x; i < kBinsPerBlock; i += blockDim.x) {
        local_bins[i] = 0;
    }

    // All cluster CTAs must exist and finish initialization before any remote
    // shared-memory address is formed or accessed.
    cluster.sync();

    const std::size_t cluster_rank = blockIdx.x / kClusterBlocks;
    const std::size_t cluster_count = gridDim.x / kClusterBlocks;
    const std::size_t first = cluster_rank * kClusterBlocks * blockDim.x +
                              block_rank * blockDim.x + threadIdx.x;
    const std::size_t stride = cluster_count * kClusterBlocks * blockDim.x;

    for (std::size_t i = first; i < n; i += stride) {
        const int bin = input[i];  // Host input is generated in [0, kBins).
        const unsigned int owner_rank = bin / kBinsPerBlock;
        const unsigned int owner_offset = bin % kBinsPerBlock;
        unsigned int* owner_bins =
            cluster.map_shared_rank(local_bins, owner_rank);
        atomicAdd(owner_bins + owner_offset, 1U);
    }

    // Do not read/retire local shared memory while a peer CTA may still update
    // it through DSM. This is a cluster-wide lifetime fence, not a CTA-local
    // __syncthreads().
    cluster.sync();

    // Every CTA exports only the bins physically owned by its own shared array.
    // Different clusters may export the same output bins, hence global atomics.
    for (int i = threadIdx.x; i < kBinsPerBlock; i += blockDim.x) {
        atomicAdd(histogram + block_rank * kBinsPerBlock + i, local_bins[i]);
    }
}

bool report_cuda(cudaError_t status, const char* expression, int line) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "CUDA error at line %d (%s): %s\n", line, expression,
                 cudaGetErrorString(status));
    return false;
}

#define CUDA_TRY(expression) \
    do { \
        if (!report_cuda((expression), #expression, __LINE__)) return false; \
    } while (0)

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;
    ~DeviceBuffer() {
        if (data != nullptr) cudaFree(data);
    }
    bool allocate(std::size_t count) {
        return report_cuda(cudaMalloc(reinterpret_cast<void**>(&data),
                                      count * sizeof(T)),
                           "cudaMalloc", __LINE__);
    }
};

bool run_case(std::size_t n, bool* unsupported) {
    if (n == 0) {
        std::fprintf(stderr, "n must be positive\n");
        return false;
    }

    std::vector<int> input(n);
    std::vector<unsigned int> reference(kBins, 0);
    for (std::size_t i = 0; i < n; ++i) {
        input[i] = static_cast<int>((i * 17 + (i / 7) * 3 + 1) % kBins);
        ++reference[input[i]];
    }

    DeviceBuffer<int> device_input;
    DeviceBuffer<unsigned int> device_histogram;
    if (!device_input.allocate(n) || !device_histogram.allocate(kBins)) {
        return false;
    }
    CUDA_TRY(cudaMemcpy(device_input.data, input.data(), n * sizeof(int),
                        cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemset(device_histogram.data, 0,
                        kBins * sizeof(unsigned int)));

    const std::size_t blocks_needed = (n + kThreads - 1) / kThreads;
    const std::size_t grid_blocks =
        ((blocks_needed + kClusterBlocks - 1) / kClusterBlocks) *
        kClusterBlocks;

    cudaLaunchAttribute cluster_attribute{};
    cluster_attribute.id = cudaLaunchAttributeClusterDimension;
    cluster_attribute.val.clusterDim.x = kClusterBlocks;
    cluster_attribute.val.clusterDim.y = 1;
    cluster_attribute.val.clusterDim.z = 1;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(static_cast<unsigned int>(grid_blocks), 1, 1);
    config.blockDim = dim3(kThreads, 1, 1);
    config.dynamicSmemBytes = kBinsPerBlock * sizeof(unsigned int);
    config.attrs = &cluster_attribute;
    config.numAttrs = 1;

    int max_cluster_size = 0;
    const cudaError_t cluster_query = cudaOccupancyMaxPotentialClusterSize(
        &max_cluster_size, cluster_histogram, &config);
    if (cluster_query == cudaErrorNotSupported) {
        std::fprintf(stderr,
                     "SKIP: runtime does not support cluster occupancy query\n");
        *unsupported = true;
        return false;
    }
    if (!report_cuda(cluster_query, "cudaOccupancyMaxPotentialClusterSize",
                     __LINE__)) {
        return false;
    }
    if (max_cluster_size < kClusterBlocks) {
        std::fprintf(stderr,
                     "SKIP: device/runtime supports maximum cluster size %d; "
                     "this example requires %d\n",
                     max_cluster_size, kClusterBlocks);
        *unsupported = true;
        return false;
    }

    const cudaError_t launch_status = cudaLaunchKernelEx(
        &config, cluster_histogram, device_input.data, n,
        device_histogram.data);
    if (launch_status == cudaErrorNotSupported) {
        std::fprintf(stderr,
                     "SKIP: runtime reports thread-block cluster launch is "
                     "not supported\n");
        *unsupported = true;
        return false;
    }
    if (!report_cuda(launch_status, "cudaLaunchKernelEx", __LINE__)) {
        return false;
    }
    CUDA_TRY(cudaDeviceSynchronize());

    std::vector<unsigned int> actual(kBins);
    CUDA_TRY(cudaMemcpy(actual.data(), device_histogram.data,
                        kBins * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
    if (actual != reference) {
        std::fprintf(stderr, "FAIL n=%zu: DSM histogram differs from CPU reference\n",
                     n);
        for (int bin = 0; bin < kBins; ++bin) {
            if (actual[bin] != reference[bin]) {
                std::fprintf(stderr, "  bin %d: GPU=%u CPU=%u\n", bin,
                             actual[bin], reference[bin]);
            }
        }
        return false;
    }

    std::printf("PASS n=%zu gridBlocks=%zu clusters=%zu bins=%d\n", n,
                grid_blocks, grid_blocks / kClusterBlocks, kBins);
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 &&
        (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::printf("usage: %s [positive-element-count]\n"
                    "default cases: 1 127 128 129 4099\n"
                    "accepted n range: 1..%llu\n",
                    argv[0], kMaxElements);
        return 0;
    }

    std::vector<std::size_t> sizes;
    if (argc == 1) {
        sizes = {1, 127, 128, 129, 4099};
    } else if (argc == 2) {
        if (argv[1][0] == '-') {
            std::fprintf(stderr, "n must be an unsigned decimal integer\n");
            return 2;
        }
        errno = 0;
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(argv[1], &end, 10);
        if (errno == ERANGE || end == argv[1] || *end != '\0' || parsed == 0 ||
            parsed > kMaxElements) {
            std::fprintf(stderr, "n must be in the range 1..%llu\n",
                         kMaxElements);
            return 2;
        }
        sizes = {static_cast<std::size_t>(parsed)};
    } else {
        std::fprintf(stderr, "usage: %s [positive-element-count]\n", argv[0]);
        return 2;
    }

    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice ||
        (status == cudaSuccess && device_count == 0)) {
        std::fprintf(stderr, "SKIP: no CUDA device is available\n");
        return kSkipExitCode;
    }
    if (!report_cuda(status, "cudaGetDeviceCount", __LINE__)) return 1;

    int device = 0;
    if (!report_cuda(cudaGetDevice(&device), "cudaGetDevice", __LINE__)) return 1;
    int major = 0;
    if (!report_cuda(cudaDeviceGetAttribute(
                         &major, cudaDevAttrComputeCapabilityMajor, device),
                     "cudaDeviceGetAttribute(ComputeCapabilityMajor)",
                     __LINE__)) {
        return 1;
    }
    if (major < 9) {
        std::fprintf(stderr,
                     "SKIP: thread-block clusters require compute capability "
                     "9.0 or newer; current device is %d.x\n",
                     major);
        return kSkipExitCode;
    }

    for (const std::size_t n : sizes) {
        bool unsupported = false;
        if (!run_case(n, &unsupported)) {
            return unsupported ? kSkipExitCode : 1;
        }
    }
    return 0;
}
