#include "cpp_device_ops.cuh"

extern __device__ int scale_in_other_tu(int value) {
    return value * 3;
}
