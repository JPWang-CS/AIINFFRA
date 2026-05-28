#include "cuda_utils.h"
#include <cmath>
#include <cstdlib>

void random_fill(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
  }
}

float compare_arrays(const float *a, const float *b, int n) {
  float max_diff = 0.0f;
  for (int i = 0; i < n; i++) {
    float diff = fabsf(a[i] - b[i]);
    if (diff > max_diff) max_diff = diff;
  }
  return max_diff;
}
