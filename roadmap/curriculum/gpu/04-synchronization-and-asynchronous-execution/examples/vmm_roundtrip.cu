#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

bool check(CUresult result, const char* expression, int line) {
    if (result == CUDA_SUCCESS) return true;
    const char* name = nullptr;
    const char* description = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);
    std::fprintf(stderr, "CUDA Driver API %s:%d: %s: %s (%s)\n",
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
    if (argc > 1 && std::string(argv[1]) == "--help") {
        std::printf("usage: %s\n", argv[0]);
        return EXIT_SUCCESS;
    }
    if (argc != 1) {
        std::fprintf(stderr, "usage: %s\n", argv[0]);
        return EXIT_FAILURE;
    }

    constexpr std::size_t kElements = 1024;
    constexpr std::size_t kLogicalBytes = kElements * sizeof(unsigned int);
    const std::vector<unsigned int> expected(kElements, 0xA5A5A5A5u);
    std::vector<unsigned int> host(kElements, 0u);

    CUdevice device = 0;
    CUcontext context = nullptr;
    CUdeviceptr address = 0;
    CUmemGenericAllocationHandle allocation = 0;
    CUmemAllocationProp property{};
    CUmemAccessDesc access{};
    std::size_t granularity = 0;
    std::size_t allocation_bytes = 0;
    int device_count = 0;
    int vmm_supported = 0;
    bool mapped = false;
    int exit_code = EXIT_FAILURE;

    CU_CHECK(cuInit(0));
    CU_CHECK(cuDeviceGetCount(&device_count));
    if (device_count == 0) {
        std::fprintf(stderr, "No CUDA device is visible\n");
        exit_code = 77;
        goto cleanup;
    }
    CU_CHECK(cuDeviceGet(&device, 0));
    CU_CHECK(cuDeviceGetAttribute(&vmm_supported,
                                 CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                                 device));
    if (!vmm_supported) {
        std::fprintf(stderr, "device does not support CUDA VMM\n");
        exit_code = 77;
        goto cleanup;
    }
    CU_CHECK(cuDevicePrimaryCtxRetain(&context, device));
    CU_CHECK(cuCtxSetCurrent(context));

    property.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    property.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    property.location.id = device;
    CU_CHECK(cuMemGetAllocationGranularity(&granularity, &property,
                                          CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    allocation_bytes = ((kLogicalBytes + granularity - 1) / granularity) * granularity;

    // VA reservation, physical allocation, mapping, and access permission are
    // separate operations.  A mapped range is not usable until access is set.
    CU_CHECK(cuMemAddressReserve(&address, allocation_bytes, 0, 0, 0));
    CU_CHECK(cuMemCreate(&allocation, allocation_bytes, &property, 0));
    CU_CHECK(cuMemMap(address, allocation_bytes, 0, allocation, 0));
    mapped = true;
    access.location = property.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(cuMemSetAccess(address, allocation_bytes, &access, 1));

    // Exercise the mapped device range through a Driver API device operation.
    CU_CHECK(cuMemsetD32(address, 0xA5A5A5A5u, kElements));
    CU_CHECK(cuCtxSynchronize());
    CU_CHECK(cuMemcpyDtoH(host.data(), address, kLogicalBytes));
    for (std::size_t i = 0; i < kElements; ++i) {
        if (host[i] != expected[i]) {
            std::fprintf(stderr, "mismatch at %zu: got 0x%08x expected 0x%08x\n",
                         i, host[i], expected[i]);
            goto cleanup;
        }
    }
    std::printf("PASS: logical=%zu bytes mapped=%zu bytes granularity=%zu bytes\n",
                kLogicalBytes, allocation_bytes, granularity);
    exit_code = EXIT_SUCCESS;

cleanup:
    if (context != nullptr) {
        (void)cuCtxSetCurrent(context);
        (void)cuCtxSynchronize();
    }
    if (mapped) (void)cuMemUnmap(address, allocation_bytes);
    if (allocation != 0) (void)cuMemRelease(allocation);
    if (address != 0) (void)cuMemAddressFree(address, allocation_bytes);
    if (context != nullptr) (void)cuDevicePrimaryCtxRelease(device);
    return exit_code;
}
