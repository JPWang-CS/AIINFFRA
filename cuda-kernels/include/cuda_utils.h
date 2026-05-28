#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                       \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

// Simple random float fill
void random_fill(float *data, int n);

// Compare two float arrays, return max absolute difference
float compare_arrays(const float *a, const float *b, int n);
