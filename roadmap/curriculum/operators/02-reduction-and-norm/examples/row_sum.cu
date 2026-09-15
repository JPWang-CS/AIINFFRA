#include <cuda_runtime.h>

// One CTA owns one row. Every lane executes the same reduction path,
// including rows and columns that are out of range.
template <int BLOCK_SIZE>
__global__ void row_sum_fp32(const float* x, float* y, int rows, int cols) {
    static_assert(BLOCK_SIZE == 128 || BLOCK_SIZE == 256, "use four or eight warps");
    __shared__ float warp_sum[BLOCK_SIZE / 32];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = blockIdx.x;
    const long long rounded_cols =
        ((static_cast<long long>(cols) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    float local = 0.0f;

    // The conditional load is the mask: tail elements contribute zero.
    // Do not return before the barriers; all BLOCK_SIZE threads must arrive.
    for (long long col = tid; col < rounded_cols; col += BLOCK_SIZE) {
        const bool valid = row < rows && col < cols;
        local += valid ? x[row * cols + col] : 0.0f;
    }

    // All warps are complete for BLOCK_SIZE 128 or 256. The FULL mask is
    // therefore valid here; shuffle itself does not provide a memory barrier.
    for (int delta = 16; delta > 0; delta >>= 1)
        local += __shfl_down_sync(0xffffffffu, local, delta);

    if (lane == 0) warp_sum[warp] = local;
    __syncthreads();

    // Warp zero reduces one value from each participating warp. Lanes beyond
    // the warp count contribute zero but still execute every shuffle.
    if (warp == 0) {
        float block_total = lane < BLOCK_SIZE / 32 ? warp_sum[lane] : 0.0f;
        for (int delta = 16; delta > 0; delta >>= 1)
            block_total += __shfl_down_sync(0xffffffffu, block_total, delta);
        if (lane == 0 && row < rows) y[row] = block_total;
    }
}

extern "C" cudaError_t launch_row_sum(const float* x, float* y, int rows, int cols,
                                       int block_size) {
    if (rows <= 0 || cols <= 0 || x == nullptr || y == nullptr)
        return cudaErrorInvalidValue;
    if (static_cast<long long>(rows) * cols > 0x7fffffffLL)
        return cudaErrorInvalidValue;
    const dim3 grid(rows);
    if (block_size == 128) {
        row_sum_fp32<128><<<grid, 128>>>(x, y, rows, cols);
    } else if (block_size == 256) {
        row_sum_fp32<256><<<grid, 256>>>(x, y, rows, cols);
    } else {
        return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
