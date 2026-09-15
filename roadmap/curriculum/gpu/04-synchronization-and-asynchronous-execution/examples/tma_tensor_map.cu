#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

constexpr int TileRows = 8, TileCols = 32;
namespace ptx = cuda::ptx;
using Barrier = cuda::barrier<cuda::thread_scope_block>;

void runtime_check(cudaError_t code) {
    if (code != cudaSuccess) {
        std::fprintf(stderr, "runtime: %s\n", cudaGetErrorString(code));
        std::exit(1);
    }
}
void driver_check(CUresult code) {
    if (code != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(code, &message);
        std::fprintf(stderr, "driver: %s\n", message ? message : "unknown error");
        std::exit(1);
    }
}

CUtensorMap make_map(int* input, int rows, int cols, int pitch_elements) {
    alignas(64) CUtensorMap map{};
    const uint64_t dimensions[2] = {uint64_t(cols), uint64_t(rows)};
    const uint64_t strides[1] = {uint64_t(pitch_elements) * sizeof(int)};
    const uint32_t box[2] = {TileCols, TileRows};
    const uint32_t element_strides[2] = {1, 1};
    driver_check(cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_INT32, 2, input,
        dimensions, strides, box, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

__global__ void gather_tiles(const __grid_constant__ CUtensorMap map, int* tiles) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    __shared__ alignas(128) int tile[TileRows][TileCols];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ Barrier ready;
    if (threadIdx.x == 0) init(&ready, blockDim.x);
    __syncthreads();
    const unsigned warp = __shfl_sync(0xffffffffu, threadIdx.x / 32, 0);
    const bool leader = warp == 0 && ptx::elect_sync(0xffffffffu);
    Barrier::arrival_token token;
    if (leader) {
        const int32_t coordinates[2] = {
            int32_t(blockIdx.x * TileCols), int32_t(blockIdx.y * TileRows)};
        ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                                 &tile, &map, coordinates,
                                 cuda::device::barrier_native_handle(ready));
        token = cuda::device::barrier_arrive_tx(ready, 1, sizeof(tile));
    } else {
        token = ready.arrive();
    }
    ready.wait(std::move(token));
    // Export every shared element, including zero-filled OOB positions.
    const size_t tile_id = size_t(blockIdx.y) * gridDim.x + blockIdx.x;
    for (int i = threadIdx.x; i < TileRows * TileCols; i += blockDim.x) {
        tiles[tile_id * TileRows * TileCols + i] = tile[i / TileCols][i % TileCols];
    }
#else
    asm volatile("trap;");
#endif
}

bool run_case(int rows, int cols) {
    const int pitch = ((cols + 3) / 4) * 4;
    const int grid_x = (cols + TileCols - 1) / TileCols;
    const int grid_y = (rows + TileRows - 1) / TileRows;
    const size_t tile_elements = size_t(grid_x) * grid_y * TileRows * TileCols;
    std::vector<int> input(size_t(rows) * pitch, -77);
    std::vector<int> result(tile_elements + 16, -999);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) input[size_t(r) * pitch + c] = 1000 * r + c + 1;
    int *device_input = nullptr, *device_tiles = nullptr;
    runtime_check(cudaMalloc(reinterpret_cast<void**>(&device_input), input.size() * sizeof(int)));
    runtime_check(cudaMalloc(reinterpret_cast<void**>(&device_tiles), result.size() * sizeof(int)));
    runtime_check(cudaMemcpy(device_input, input.data(), input.size() * sizeof(int), cudaMemcpyHostToDevice));
    runtime_check(cudaMemcpy(device_tiles, result.data(), result.size() * sizeof(int), cudaMemcpyHostToDevice));
    const CUtensorMap map = make_map(device_input, rows, cols, pitch);
    gather_tiles<<<dim3(grid_x, grid_y), 128>>>(map, device_tiles);
    runtime_check(cudaGetLastError());
    runtime_check(cudaDeviceSynchronize());
    runtime_check(cudaMemcpy(result.data(), device_tiles, result.size() * sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int by = 0; by < grid_y; ++by) {
        for (int bx = 0; bx < grid_x; ++bx) {
            for (int y = 0; y < TileRows; ++y) {
                for (int x = 0; x < TileCols; ++x) {
                    const int r = by * TileRows + y, c = bx * TileCols + x;
                    const int expected = r < rows && c < cols ? 1000 * r + c + 1 : 0;
                    const size_t index = (size_t(by) * grid_x + bx) * TileRows * TileCols + y * TileCols + x;
                    if (result[index] != expected) {
                        std::fprintf(stderr, "FAIL %dx%d tile=(%d,%d) local=(%d,%d) got=%d expected=%d\n",
                                     rows, cols, by, bx, y, x, result[index], expected);
                        ok = false;
                    }
                }
            }
        }
    }
    for (size_t i = tile_elements; i < result.size(); ++i) {
        if (result[i] != -999) { std::fprintf(stderr, "FAIL output guard\n"); ok = false; }
    }
    runtime_check(cudaFree(device_tiles));
    runtime_check(cudaFree(device_input));
    std::printf("%dx%d pitch=%d grid=%dx%d: %s\n", rows, cols, pitch, grid_x, grid_y, ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::puts("Usage: tma_tensor_map [--help]\n"
                  "CUDA 13.3 headers; supported CC9.0+ target; link CUDA driver (-lcuda).\n"
                  "Checks 2-D tensor-map coordinates, pitched rows, logical OOB zero fill and guards.");
        return 0;
    }
    if (argc != 1) return 2;
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || (status == cudaSuccess && count == 0)) {
        std::puts("SKIP: no CUDA device"); return 77;
    }
    runtime_check(status);
    runtime_check(cudaSetDevice(0));
    driver_check(cuInit(0));
    cudaDeviceProp prop{};
    runtime_check(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 9) { std::puts("SKIP: tensor TMA needs supported CC9.0+"); return 77; }
    std::printf("GPU=%s CC=%d.%d tile=%dx%d\n", prop.name, prop.major, prop.minor, TileRows, TileCols);
    bool ok = true;
    for (auto shape : {std::pair<int,int>{1,1}, {8,32}, {9,33}, {17,65}})
        ok = run_case(shape.first, shape.second) && ok;
    return ok ? 0 : 1;
}
