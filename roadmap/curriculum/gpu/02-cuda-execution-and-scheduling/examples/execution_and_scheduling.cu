#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t error__ = (call);                                        \
        if (error__ != cudaSuccess) {                                        \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n",                  \
                         __FILE__, __LINE__, cudaGetErrorString(error__));    \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (false)

__global__ void dependent_chain_kernel(const float* input, float* output,
                                       int n, int steps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = input[i];
    for (int step = 0; step < steps; ++step) {
        x = fmaf(x, 1.000001f, 0.00001f);
    }
    output[i] = x;
}

__global__ void independent_accumulators_kernel(const float* input,
                                                float* output, int n,
                                                int steps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x0 = input[i];
    float x1 = input[i] + 0.1f;
    float x2 = input[i] + 0.2f;
    float x3 = input[i] + 0.3f;
    for (int step = 0; step < steps; ++step) {
        x0 = fmaf(x0, 1.000001f, 0.00001f);
        x1 = fmaf(x1, 1.000001f, 0.00001f);
        x2 = fmaf(x2, 1.000001f, 0.00001f);
        x3 = fmaf(x3, 1.000001f, 0.00001f);
    }
    output[i] = 0.25f * (x0 + x1 + x2 + x3 - 0.6f);
}

__device__ __forceinline__ float even_path(float x) {
    return fmaf(x, 1.25f, 0.5f);
}

__device__ __forceinline__ float odd_path(float x) {
    return fmaf(x, 0.75f, -0.25f);
}

__global__ void divergent_branch_kernel(const float* input, float* output,
                                        int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if ((i & 1) == 0) {
        output[i] = even_path(input[i]);
    } else {
        output[i] = odd_path(input[i]);
    }
}

__global__ void predicated_select_kernel(const float* input, float* output,
                                         int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float even_value = even_path(input[i]);
    const float odd_value = odd_path(input[i]);
    output[i] = ((i & 1) == 0) ? even_value : odd_value;
}

__device__ __noinline__ float even_chain(float x, int steps) {
#pragma unroll 1
    for (int step = 0; step < steps; ++step) {
        x = fmaf(x, 1.000001f, 0.00001f);
    }
    return x;
}

__device__ __noinline__ float odd_chain(float x, int steps) {
#pragma unroll 1
    for (int step = 0; step < steps; ++step) {
        x = fmaf(x, 0.999999f, -0.00001f);
    }
    return x;
}

__global__ void branch_probe_kernel(const float* input,
                                    const unsigned char* flags,
                                    float* output, int n, int steps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    output[i] = flags[i] ? even_chain(input[i], steps)
                         : odd_chain(input[i], steps);
}

void reference_chain(const std::vector<float>& input,
                     std::vector<float>& output, int steps) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        for (int step = 0; step < steps; ++step) {
            x = std::fma(x, 1.000001f, 0.00001f);
        }
        output[i] = x;
    }
}

void reference_independent(const std::vector<float>& input,
                           std::vector<float>& output, int steps) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        float x0 = input[i];
        float x1 = input[i] + 0.1f;
        float x2 = input[i] + 0.2f;
        float x3 = input[i] + 0.3f;
        for (int step = 0; step < steps; ++step) {
            x0 = std::fma(x0, 1.000001f, 0.00001f);
            x1 = std::fma(x1, 1.000001f, 0.00001f);
            x2 = std::fma(x2, 1.000001f, 0.00001f);
            x3 = std::fma(x3, 1.000001f, 0.00001f);
        }
        output[i] = 0.25f * (x0 + x1 + x2 + x3 - 0.6f);
    }
}

void reference_branch(const std::vector<float>& input,
                      std::vector<float>& output) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = (i & 1) ? (input[i] * 0.75f - 0.25f)
                            : (input[i] * 1.25f + 0.5f);
    }
}

float max_abs_error(const std::vector<float>& actual,
                    const std::vector<float>& expected) {
    float error = 0.0f;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            return std::numeric_limits<float>::infinity();
        }
        error = std::max(error, std::fabs(actual[i] - expected[i]));
    }
    return error;
}

template <typename Launch>
float measure(Launch launch, float* device_output,
              std::vector<float>& host_output, int repeats) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaMemcpy(host_output.data(), device_output,
                          host_output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms / static_cast<float>(repeats);
}

constexpr int kProbeBlockThreads = 256;
constexpr int kProbeGuard = 32;
constexpr float kProbeGuardValue = 12345.0f;

struct ProbeOptions {
    int n = 1 << 20;
    int steps = 256;
    int repeats = 50;
    int rounds = 9;
};

bool parse_positive_int(const char* text, int* value) {
    if (!text || !*text) return false;
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno == ERANGE || *end != '\0' || parsed <= 0 ||
        parsed > static_cast<long>(std::numeric_limits<int>::max())) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}

bool valid_probe_options(const ProbeOptions& options) {
    return options.n > 0 &&
           options.n <= std::numeric_limits<int>::max() - 255 &&
           options.steps > 0 && options.steps <= 4096 &&
           options.repeats > 0 && options.repeats <= 10000 &&
           options.rounds > 0 && options.rounds <= 101;
}

bool parse_probe_options(int argc, char** argv, ProbeOptions* options,
                         bool* self_test) {
    *self_test = false;
    if (argc < 2 || std::strcmp(argv[1], "--branch-probe") != 0) return false;
    if (argc == 3 && std::strcmp(argv[2], "--self-test") == 0) {
        *self_test = true;
        return true;
    }
    if (argc > 6) return false;
    int* values[] = {&options->n, &options->steps, &options->repeats,
                     &options->rounds};
    for (int i = 2; i < argc; ++i) {
        if (!parse_positive_int(argv[i], values[i - 2])) return false;
    }
    return valid_probe_options(*options);
}

void make_probe_flags(int n, bool warp_uniform,
                      std::vector<unsigned char>& flags) {
    flags.resize(n);
    for (int i = 0; i < n; ++i) {
        flags[i] = warp_uniform ? (((i / 32) & 1) == 0)
                                : ((i & 1) == 0);
    }
}

int count_probe_even(const std::vector<unsigned char>& flags) {
    int count = 0;
    for (unsigned char flag : flags) count += flag != 0;
    return count;
}

float probe_reference_value(float input, bool even, int steps) {
    float x = input;
    for (int step = 0; step < steps; ++step) {
        x = std::fma(x, even ? 1.000001f : 0.999999f,
                     even ? 0.00001f : -0.00001f);
    }
    return x;
}

void make_probe_reference(const std::vector<float>& input,
                          const std::vector<unsigned char>& flags,
                          int steps, std::vector<float>& reference) {
    std::array<float, 1000> even_seed{};
    std::array<float, 1000> odd_seed{};
    for (int seed = 0; seed < 1000; ++seed) {
        const float value = 0.001f * static_cast<float>(seed);
        even_seed[seed] = probe_reference_value(value, true, steps);
        odd_seed[seed] = probe_reference_value(value, false, steps);
    }
    reference.resize(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        reference[i] = flags[i] ? even_seed[i % even_seed.size()]
                                : odd_seed[i % odd_seed.size()];
    }
}

float probe_max_abs_error(const std::vector<float>& actual,
                          const std::vector<float>& expected) {
    float error = 0.0f;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            return std::numeric_limits<float>::infinity();
        }
        error = std::max(error, std::fabs(actual[i] - expected[i]));
    }
    return error;
}

bool probe_guards_ok(const std::vector<float>& storage, int n) {
    if (static_cast<int>(storage.size()) != n + 2 * kProbeGuard) return false;
    for (int i = 0; i < kProbeGuard; ++i) {
        if (storage[i] != kProbeGuardValue ||
            storage[kProbeGuard + n + i] != kProbeGuardValue) return false;
    }
    return true;
}

struct ProbeStats {
    float min_ms;
    float median_ms;
    float max_ms;
};

ProbeStats probe_stats(std::vector<float> values);

bool probe_self_test() {
    bool ok = true;
    const int test_sizes[] = {32, 64, 33, 257};
    const int expected_uniform[] = {32, 32, 32, 129};
    const int expected_split[] = {16, 32, 17, 129};
    for (int j = 0; j < 4; ++j) {
        std::vector<unsigned char> uniform, split;
        make_probe_flags(test_sizes[j], true, uniform);
        make_probe_flags(test_sizes[j], false, split);
        ok = ok && count_probe_even(uniform) == expected_uniform[j];
        ok = ok && count_probe_even(split) == expected_split[j];
        std::vector<float> input(test_sizes[j]);
        for (int i = 0; i < test_sizes[j]; ++i) {
            input[i] = 0.001f * static_cast<float>(i % 1000);
        }
        std::vector<float> reference;
        make_probe_reference(input, split, 32, reference);
        for (float value : reference) ok = ok && std::isfinite(value);
    }

    std::vector<float> guarded(2 * kProbeGuard + 1, kProbeGuardValue);
    guarded[0] = 0.0f;
    ok = ok && !probe_guards_ok(guarded, 1);
    guarded[0] = kProbeGuardValue;
    guarded[2 * kProbeGuard] = 0.0f;
    ok = ok && !probe_guards_ok(guarded, 1);
    guarded[2 * kProbeGuard] = kProbeGuardValue;
    ok = ok && probe_guards_ok(guarded, 1);
    std::vector<float> nan_actual(1, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> zero_expected(1, 0.0f);
    ok = ok && !std::isfinite(probe_max_abs_error(nan_actual, zero_expected));
    ok = ok && !valid_probe_options(ProbeOptions{0, 1, 1, 1});
    ok = ok && !valid_probe_options(ProbeOptions{1, 0, 1, 1});
    ok = ok && !valid_probe_options(ProbeOptions{1, 1, 0, 1});
    ok = ok && !valid_probe_options(ProbeOptions{1, 1, 1, 0});
    ok = ok && !valid_probe_options(ProbeOptions{1, 4097, 50, 9});
    ok = ok && !valid_probe_options(ProbeOptions{1, 256, 10001, 9});
    ok = ok && !valid_probe_options(ProbeOptions{1, 256, 50, 102});
    int parse_value = 0;
    ok = ok && !parse_positive_int("abc", &parse_value);
    ok = ok && !parse_positive_int("2147483648", &parse_value);
    ok = ok && !parse_positive_int("999999999999999999999", &parse_value);
    ok = ok && !parse_positive_int("0", &parse_value);
    ok = ok && !parse_positive_int("-1", &parse_value);
    {
        char arg0[] = "probe";
        char arg1[] = "--branch-probe";
        char arg2[] = "1";
        char arg3[] = "256";
        char arg4[] = "50";
        char arg5[] = "9";
        char arg6[] = "extra";
        char* too_many[] = {arg0, arg1, arg2, arg3, arg4, arg5, arg6};
        ProbeOptions parsed;
        bool parsed_self_test = false;
        ok = ok && !parse_probe_options(7, too_many, &parsed,
                                        &parsed_self_test);
    }
    const std::vector<float> odd_samples{9.0f, 1.0f, 5.0f};
    const std::vector<float> even_samples{9.0f, 1.0f, 5.0f, 3.0f};
    const std::vector<float> repeated_samples{4.0f, 4.0f, 4.0f};
    const std::vector<float> single_sample{7.0f};
    const ProbeStats odd_stats = probe_stats(odd_samples);
    const ProbeStats even_stats = probe_stats(even_samples);
    const ProbeStats repeated_stats = probe_stats(repeated_samples);
    const ProbeStats single_stats = probe_stats(single_sample);
    ok = ok && odd_stats.min_ms == 1.0f && odd_stats.median_ms == 5.0f &&
         odd_stats.max_ms == 9.0f;
    ok = ok && even_stats.min_ms == 1.0f && even_stats.median_ms == 4.0f &&
         even_stats.max_ms == 9.0f;
    ok = ok && repeated_stats.min_ms == 4.0f &&
         repeated_stats.median_ms == 4.0f && repeated_stats.max_ms == 4.0f;
    ok = ok && single_stats.min_ms == 7.0f && single_stats.median_ms == 7.0f &&
         single_stats.max_ms == 7.0f;
    ok = ok && odd_samples == std::vector<float>({9.0f, 1.0f, 5.0f});
    ok = ok && even_samples == std::vector<float>({9.0f, 1.0f, 5.0f, 3.0f});
    std::puts(ok ? "branch_probe self-test PASS" : "branch_probe self-test FAIL");
    return ok;
}

template <typename Launch>
float measure_probe(Launch launch, int repeats) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms / static_cast<float>(repeats);
}

ProbeStats probe_stats(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    const float median = (values.size() & 1) != 0
                             ? values[values.size() / 2]
                             : 0.5f * (values[values.size() / 2 - 1] +
                                       values[values.size() / 2]);
    return ProbeStats{values.front(), median, values.back()};
}

void print_probe_stats(const char* name, const std::vector<float>& values) {
    const ProbeStats stats = probe_stats(values);
    std::printf("branch_probe %s min_ms=%.6f median_ms=%.6f max_ms=%.6f\n",
                name, stats.min_ms, stats.median_ms, stats.max_ms);
}

void reset_probe_output(float* device_storage, int n,
                        std::vector<float>& host_storage) {
    std::fill(host_storage.begin(), host_storage.end(),
              std::numeric_limits<float>::quiet_NaN());
    for (int i = 0; i < kProbeGuard; ++i) {
        host_storage[i] = kProbeGuardValue;
        host_storage[kProbeGuard + n + i] = kProbeGuardValue;
    }
    CUDA_CHECK(cudaMemcpy(device_storage, host_storage.data(),
                          host_storage.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
}

bool copy_and_check_probe_pattern(const char* name,
                                  const std::vector<float>& input,
                                  const std::vector<unsigned char>& flags,
                                  int steps, float* device_storage,
                                  std::vector<float>& host_storage) {
    const int n = static_cast<int>(input.size());
    CUDA_CHECK(cudaMemcpy(host_storage.data(), device_storage,
                          host_storage.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    std::vector<float> reference;
    make_probe_reference(input, flags, steps, reference);
    std::vector<float> actual(host_storage.begin() + kProbeGuard,
                              host_storage.begin() + kProbeGuard + n);
    const float error = probe_max_abs_error(actual, reference);
    const bool guards_ok = probe_guards_ok(host_storage, n);
    const bool pass = std::isfinite(error) && error <= 1.0e-4f && guards_ok;
    std::printf("branch_probe pattern=%s even=%d odd=%d max_abs_error=%.8g "
                "guard=%s %s\n",
                name, count_probe_even(flags), n - count_probe_even(flags),
                error, guards_ok ? "PASS" : "FAIL", pass ? "PASS" : "FAIL");
    return pass;
}

bool probe_guards_after_measure(float* device_storage, int n,
                                 std::vector<float>& host_storage) {
    CUDA_CHECK(cudaMemcpy(host_storage.data(), device_storage,
                          host_storage.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return probe_guards_ok(host_storage, n);
}

bool launch_and_check_probe_pattern(const char* name,
                                    const std::vector<float>& input,
                                    const std::vector<unsigned char>& flags,
                                    int steps, float* device_input,
                                    unsigned char* device_flags,
                                    float* device_output, float* device_storage,
                                    std::vector<float>& host_storage) {
    const int n = static_cast<int>(input.size());
    reset_probe_output(device_storage, n, host_storage);
    branch_probe_kernel<<<(n + kProbeBlockThreads - 1) / kProbeBlockThreads,
                          kProbeBlockThreads>>>(device_input, device_flags,
                                                device_output, n, steps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return copy_and_check_probe_pattern(name, input, flags, steps,
                                        device_storage, host_storage);
}

bool run_branch_probe(const ProbeOptions& options) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    int driver_version = 0;
    int runtime_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

    const int n = options.n;
    const int blocks = (n + kProbeBlockThreads - 1) / kProbeBlockThreads;
    std::vector<float> input(n);
    for (int i = 0; i < n; ++i) {
        input[i] = 0.001f * static_cast<float>(i % 1000);
    }
    std::vector<unsigned char> lane_split, warp_uniform;
    make_probe_flags(n, false, lane_split);
    make_probe_flags(n, true, warp_uniform);
    std::vector<float> host_storage(n + 2 * kProbeGuard);

    float* device_input = nullptr;
    unsigned char* device_lane_split = nullptr;
    unsigned char* device_warp_uniform = nullptr;
    float* device_storage = nullptr;
    CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_lane_split, n * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&device_warp_uniform, n * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&device_storage,
                          host_storage.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_lane_split, lane_split.data(),
                          n * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_warp_uniform, warp_uniform.data(),
                          n * sizeof(unsigned char), cudaMemcpyHostToDevice));
    float* device_output = device_storage + kProbeGuard;

    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, branch_probe_kernel));
    int theoretical_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &theoretical_blocks, branch_probe_kernel, kProbeBlockThreads, 0));
    std::printf("branch_probe GPU=%s cc=%d.%d driver=%d runtime=%d\n",
                properties.name, properties.major, properties.minor,
                driver_version, runtime_version);
    std::printf("branch_probe n=%d steps=%d repeats=%d rounds=%d block=%d "
                "grid=%d\n",
                n, options.steps, options.repeats, options.rounds,
                kProbeBlockThreads, blocks);
    std::printf("branch_probe kernel numRegs=%d sharedBytes=%zu localBytes=%zu "
                "theoretical_blocks_per_sm=%d\n",
                attributes.numRegs, attributes.sharedSizeBytes,
                attributes.localSizeBytes, theoretical_blocks);

    auto launch_uniform = [&] {
        branch_probe_kernel<<<blocks, kProbeBlockThreads>>>(
            device_input, device_warp_uniform, device_output, n, options.steps);
    };
    auto launch_split = [&] {
        branch_probe_kernel<<<blocks, kProbeBlockThreads>>>(
            device_input, device_lane_split, device_output, n, options.steps);
    };
    bool ok = launch_and_check_probe_pattern(
        "uniform_pre", input, warp_uniform, options.steps, device_input,
        device_warp_uniform, device_output, device_storage, host_storage);
    ok = launch_and_check_probe_pattern(
        "split_pre", input, lane_split, options.steps, device_input,
        device_lane_split, device_output, device_storage, host_storage) && ok;
    if (!ok) {
        CUDA_CHECK(cudaFree(device_input));
        CUDA_CHECK(cudaFree(device_lane_split));
        CUDA_CHECK(cudaFree(device_warp_uniform));
        CUDA_CHECK(cudaFree(device_storage));
        std::puts("branch_probe overall FAIL");
        return false;
    }
    for (int warmup = 0; warmup < 3; ++warmup) {
        launch_uniform();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        launch_split();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    reset_probe_output(device_storage, n, host_storage);

    std::vector<float> uniform_times, split_times;
    uniform_times.reserve(options.rounds);
    split_times.reserve(options.rounds);
    bool uniform_last = false;
    for (int round = 0; round < options.rounds; ++round) {
        float uniform_ms = 0.0f;
        float split_ms = 0.0f;
        if ((round & 1) == 0) {
            uniform_ms = measure_probe(launch_uniform, options.repeats);
            split_ms = measure_probe(launch_split, options.repeats);
        } else {
            split_ms = measure_probe(launch_split, options.repeats);
            uniform_ms = measure_probe(launch_uniform, options.repeats);
        }
        if (!std::isfinite(uniform_ms) || uniform_ms <= 0.0f ||
            !std::isfinite(split_ms) || split_ms <= 0.0f) {
            ok = false;
        }
        uniform_times.push_back(uniform_ms);
        split_times.push_back(split_ms);
        uniform_last = (round & 1) != 0;
        if (!probe_guards_after_measure(device_storage, n, host_storage)) {
            ok = false;
            std::fprintf(stderr,
                         "branch_probe guard corruption after round %d\n",
                         round);
        }
        std::printf("branch_probe round=%d order=%s uniform_ms=%.6f "
                    "split_ms=%.6f\n",
                    round, (round & 1) == 0 ? "uniform-split" : "split-uniform",
                    uniform_ms, split_ms);
    }
    if (ok) {
        print_probe_stats("uniform", uniform_times);
        print_probe_stats("split", split_times);
        const float uniform_median = probe_stats(uniform_times).median_ms;
        const float split_median = probe_stats(split_times).median_ms;
        if (std::isfinite(uniform_median) && uniform_median > 0.0f &&
            std::isfinite(split_median) && split_median > 0.0f) {
            std::printf("branch_probe split_median/uniform_median=%.6f\n",
                        split_median / uniform_median);
        } else {
            std::puts("branch_probe split_median/uniform_median=UNAVAILABLE");
            ok = false;
        }
    } else {
        std::puts("branch_probe timing stats=FAIL");
    }
    const char* final_name = uniform_last ? "uniform_final" : "split_final";
    const std::vector<unsigned char>& final_flags =
        uniform_last ? warp_uniform : lane_split;
    ok = copy_and_check_probe_pattern(final_name, input, final_flags,
                                      options.steps, device_storage,
                                      host_storage) && ok;
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_lane_split));
    CUDA_CHECK(cudaFree(device_warp_uniform));
    CUDA_CHECK(cudaFree(device_storage));
    std::puts(ok ? "branch_probe overall PASS" : "branch_probe overall FAIL");
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--branch-probe") == 0) {
        ProbeOptions options;
        bool self_test = false;
        if (!parse_probe_options(argc, argv, &options, &self_test)) {
            std::fprintf(stderr,
                         "usage: %s --branch-probe [n>0] [steps=1..4096] "
                         "[repeats=1..10000] [rounds=1..101]\n"
                         "       %s --branch-probe --self-test\n",
                         argv[0], argv[0]);
            return EXIT_FAILURE;
        }
        if (self_test) return probe_self_test() ? EXIT_SUCCESS : EXIT_FAILURE;
        return run_branch_probe(options) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    const int n = argc > 1 ? std::atoi(argv[1]) : 1 << 20;
    const int steps = argc > 2 ? std::atoi(argv[2]) : 200;
    const int repeats = argc > 3 ? std::atoi(argv[3]) : 50;
    if (n <= 0 || n > std::numeric_limits<int>::max() - 255 ||
        steps <= 0 || steps > std::numeric_limits<int>::max() / 4 ||
        repeats <= 0) return EXIT_FAILURE;

    std::vector<float> input(n), output(n), reference(n);
    for (int i = 0; i < n; ++i) input[i] = 0.001f * static_cast<float>(i % 1000);

    float* device_input = nullptr;
    float* device_output = nullptr;
    CUDA_CHECK(cudaMalloc(&device_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_output, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_input, input.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    bool all_ok = true;
    auto run = [&](const char* name, int source_fmas, auto launch, auto reference_fn) {
        reference_fn(input, reference);
        const float ms = measure(launch, device_output, output, repeats);
        const float error = max_abs_error(output, reference);
        const bool ok = std::isfinite(error) && error <= 1.0e-4f &&
                        std::isfinite(ms) && ms > 0.0f;
        all_ok = all_ok && ok;
        std::printf("%-14s %.4f ms max_abs_error=%.8g %s", name, ms,
                    error, ok ? "PASS" : "FAIL");
        if (source_fmas > 0 && ms > 0.0f) {
            // Source-level arithmetic count, not an instruction counter.
            const double gflops = 2.0 * n * source_fmas / (ms * 1.0e6);
            std::printf(" source_FMA_GFLOP_s=%.3f", gflops);
        }
        std::printf("\n");
    };

    run("dependent", steps, [&] {
        dependent_chain_kernel<<<blocks, threads>>>(device_input, device_output,
                                                     n, steps);
    }, [&](const auto& a, auto& b) { reference_chain(a, b, steps); });

    run("independent", 4 * steps, [&] {
        independent_accumulators_kernel<<<blocks, threads>>>(
            device_input, device_output, n, steps);
    }, [&](const auto& a, auto& b) { reference_independent(a, b, steps); });

    run("divergent", 0, [&] {
        divergent_branch_kernel<<<blocks, threads>>>(device_input, device_output,
                                                      n);
    }, [&](const auto& a, auto& b) { reference_branch(a, b); });

    run("predicated", 0, [&] {
        predicated_select_kernel<<<blocks, threads>>>(
            device_input, device_output, n);
    }, [&](const auto& a, auto& b) { reference_branch(a, b); });

    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
