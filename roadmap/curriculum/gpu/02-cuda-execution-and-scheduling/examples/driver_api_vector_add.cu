#include <cuda.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

const char* kVectorAddPtx = R"ptx(
.version 7.0
.target sm_70
.address_size 64

.visible .entry vector_add(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_c,
    .param .u32 param_n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<7>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [param_a];
    ld.param.u64 %rd2, [param_b];
    ld.param.u64 %rd3, [param_c];
    ld.param.u32 %r2, [param_n];
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r0, %r1, %r0, %r3;
    setp.ge.u32 %p1, %r0, %r2;
    @%p1 bra DONE;

    cvt.u64.u32 %rd4, %r0;
    shl.b64 %rd4, %rd4, 2;
    add.u64 %rd5, %rd1, %rd4;
    add.u64 %rd6, %rd2, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    add.u64 %rd5, %rd3, %rd4;
    st.global.f32 [%rd5], %f3;

DONE:
    ret;
}
)ptx";

bool check(CUresult result, const char* expression, int line) {
    if (result == CUDA_SUCCESS) return true;
    const char* name = nullptr;
    const char* description = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);
    std::fprintf(stderr, "CUDA Driver API %s:%d: %s (%s): %s\n",
                 __FILE__, line, expression,
                 name ? name : "unknown",
                 description ? description : "no description");
    return false;
}

#define CU_CHECK(call) \
    do { \
        if (!check((call), #call, __LINE__)) goto cleanup; \
    } while (false)

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--help") == 0) {
        std::printf("usage: %s [n: 1..16777216]\n", argv[0]);
        return EXIT_SUCCESS;
    }
    if (argc > 2) {
        std::fprintf(stderr, "usage: %s [n: 1..16777216]\n", argv[0]);
        return EXIT_FAILURE;
    }
    char* end = nullptr;
    const long parsed_n = argc > 1 ? std::strtol(argv[1], &end, 10) : 257;
    if ((argc > 1 && (end == argv[1] || *end != '\0')) ||
        parsed_n < 1 || parsed_n > 16777216) {
        std::fprintf(stderr, "n must be an integer in [1, 16777216]\n");
        return EXIT_FAILURE;
    }
    const int n = static_cast<int>(parsed_n);

    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    const unsigned int threads = 128;
    const unsigned int blocks = (static_cast<unsigned int>(n) + threads - 1) / threads;
    std::vector<float> host_a(n), host_b(n), host_c(n, -1.0f);
    for (int i = 0; i < n; ++i) {
        host_a[i] = static_cast<float>(i) * 0.5f;
        host_b[i] = static_cast<float>(i) * 0.25f;
    }

    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUstream stream = nullptr;
    CUdeviceptr d_a = 0, d_b = 0, d_c = 0;
    int exit_code = EXIT_FAILURE;
    int device_count = 0;
    char device_name[256] = {};
    void* kernel_params[] = {&d_a, &d_b, &d_c, const_cast<int*>(&n)};
    CUresult init_status = CUDA_SUCCESS;
    CUresult count_status = CUDA_SUCCESS;

    init_status = cuInit(0);
    if (init_status == CUDA_ERROR_NO_DEVICE) {
        std::fprintf(stderr, "SKIP: CUDA driver reports no device\n");
        exit_code = 77;
        goto cleanup;
    }
    if (!check(init_status, "cuInit(0)", __LINE__)) goto cleanup;
    count_status = cuDeviceGetCount(&device_count);
    if (count_status == CUDA_ERROR_NO_DEVICE) {
        std::fprintf(stderr, "SKIP: CUDA driver reports no device\n");
        exit_code = 77;
        goto cleanup;
    }
    if (!check(count_status, "cuDeviceGetCount(&device_count)", __LINE__)) goto cleanup;
    if (device_count == 0) {
        std::fprintf(stderr, "No CUDA device is visible to the Driver API\n");
        exit_code = 77;
        goto cleanup;
    }
    CU_CHECK(cuDeviceGet(&device, 0));
    CU_CHECK(cuDeviceGetName(device_name, sizeof(device_name), device));
    std::printf("device=%s n=%d\n", device_name, n);

    // Retain the device primary context so this Driver API work can interoperate
    // with Runtime API code that selects the same device.
    CU_CHECK(cuDevicePrimaryCtxRetain(&context, device));
    CU_CHECK(cuCtxSetCurrent(context));
    CU_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    CU_CHECK(cuModuleLoadDataEx(&module, kVectorAddPtx, 0, nullptr, nullptr));
    CU_CHECK(cuModuleGetFunction(&function, module, "vector_add"));

    CU_CHECK(cuMemAlloc(&d_a, bytes));
    CU_CHECK(cuMemAlloc(&d_b, bytes));
    CU_CHECK(cuMemAlloc(&d_c, bytes));
    // Synchronous copies accept these std::vector pageable host buffers.
    CU_CHECK(cuMemcpyHtoD(d_a, host_a.data(), bytes));
    CU_CHECK(cuMemcpyHtoD(d_b, host_b.data(), bytes));

    CU_CHECK(cuLaunchKernel(function,
                            blocks, 1, 1,
                            threads, 1, 1,
                            0, stream, kernel_params, nullptr));
    CU_CHECK(cuStreamSynchronize(stream));
    CU_CHECK(cuMemcpyDtoH(host_c.data(), d_c, bytes));

    for (int i = 0; i < n; ++i) {
        const float expected = host_a[i] + host_b[i];
        if (!std::isfinite(host_c[i]) || std::fabs(host_c[i] - expected) > 1.0e-6f) {
            std::fprintf(stderr, "mismatch at %d: got %.8f expected %.8f\n",
                         i, host_c[i], expected);
            goto cleanup;
        }
    }
    std::printf("PASS: all %d elements match the host reference\n", n);
    exit_code = EXIT_SUCCESS;

cleanup:
    if (context != nullptr) {
        (void)cuCtxSetCurrent(context);
        (void)cuCtxSynchronize();
    }
    if (stream != nullptr) (void)cuStreamDestroy(stream);
    if (d_c != 0) (void)cuMemFree(d_c);
    if (d_b != 0) (void)cuMemFree(d_b);
    if (d_a != 0) (void)cuMemFree(d_a);
    if (module != nullptr) (void)cuModuleUnload(module);
    if (context != nullptr) (void)cuDevicePrimaryCtxRelease(device);
    return exit_code;
}
