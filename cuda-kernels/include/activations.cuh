#pragma once

// Header-only device functions for common ML activations.
// These are meant to be called from within kernels (fused with other ops).

__device__ __forceinline__ float relu(float x) {
  return fmaxf(0.0f, x);
}

__device__ __forceinline__ float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu(float x) {
  return x * sigmoid(x);
}

// GELU (tanh approximation), same as PyTorch's default
__device__ __forceinline__ float gelu(float x) {
  const float sqrt_2_over_pi = 0.7978845608028654f;
  const float coeff = 0.044715f;
  float x3 = x * x * x;
  float inner = sqrt_2_over_pi * (x + coeff * x3);
  return 0.5f * x * (1.0f + tanhf(inner));
}

// SwiGLU: split input in half along last dim → silu(left) * right
// y = silu(x_left) * x_right, where left/right are two halves
__device__ __forceinline__ float swiglu(float x_left, float x_right) {
  return silu(x_left) * x_right;
}

// GeGLU: same split, but gelu(left) * right
__device__ __forceinline__ float geglu(float x_left, float x_right) {
  return gelu(x_left) * x_right;
}
