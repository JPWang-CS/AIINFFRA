#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

constexpr int kCapacity = 1025;
constexpr int kThreads = 256;
constexpr float kTailSentinel = -12345.25f;

bool check_cuda(cudaError_t status, const char* expression) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "CUDA error in %s: %s\n", expression,
               cudaGetErrorString(status));
  return false;
}

__global__ void affine_kernel(const float* input, float* output, int n,
                              float alpha, float bias) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    output[index] = alpha * input[index] + bias;
  }
}

__global__ void relu_kernel(float* values, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n && values[index] < 0.0f) {
    values[index] = 0.0f;
  }
}

void print_help(const char* program) {
  std::printf("Usage: %s [--help]\n"
              "Build one affine->ReLU CUDA graph and replay it for n=1,257,1025.\n"
              "Host-to-device and device-to-host copies remain outside the graph.\n",
              program);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      print_help(argv[0]);
      return EXIT_SUCCESS;
    }
    std::fprintf(stderr, "unknown argument\n");
    return EXIT_FAILURE;
  }

  int device_count = 0;
  cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status == cudaErrorNoDevice || (status == cudaSuccess && device_count == 0)) {
    std::puts("SKIP: no usable CUDA device");
    return 77;
  }
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceCount: %s\n", cudaGetErrorString(status));
    return EXIT_FAILURE;
  }

  int rc = EXIT_FAILURE;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaGraphNode_t affine_node = nullptr;
  cudaGraphNode_t relu_node = nullptr;
  float* host_input = nullptr;
  float* host_output = nullptr;
  float* device_input = nullptr;
  float* device_values = nullptr;
  int n = 1;
  float alpha = 1.0f;
  float bias = -0.25f;
  dim3 block(kThreads);
  dim3 grid(1);
  void* affine_args[5] = {&device_input, &device_values, &n, &alpha, &bias};
  void* relu_args[2] = {&device_values, &n};
  cudaKernelNodeParams affine_params{};
  cudaKernelNodeParams relu_params{};

#define CUDA_TRY(call) \
  do { \
    if (!check_cuda((call), #call)) { \
      goto cleanup; \
    } \
  } while (false)

  CUDA_TRY(cudaSetDevice(0));
  CUDA_TRY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&host_input),
                          kCapacity * sizeof(float)));
  CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&host_output),
                          kCapacity * sizeof(float)));
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&device_input),
                      kCapacity * sizeof(float)));
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&device_values),
                      kCapacity * sizeof(float)));

  affine_params.func = reinterpret_cast<void*>(affine_kernel);
  affine_params.gridDim = grid;
  affine_params.blockDim = block;
  affine_params.sharedMemBytes = 0;
  affine_params.kernelParams = affine_args;
  affine_params.extra = nullptr;

  relu_params.func = reinterpret_cast<void*>(relu_kernel);
  relu_params.gridDim = grid;
  relu_params.blockDim = block;
  relu_params.sharedMemBytes = 0;
  relu_params.kernelParams = relu_args;
  relu_params.extra = nullptr;

  CUDA_TRY(cudaGraphCreate(&graph, 0));
  // Keep the node handles: replay-time updates target these exact nodes.
  CUDA_TRY(cudaGraphAddKernelNode(&affine_node, graph, nullptr, 0,
                                  &affine_params));
  CUDA_TRY(cudaGraphAddKernelNode(&relu_node, graph, &affine_node, 1,
                                  &relu_params));
  CUDA_TRY(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  {
    const int sizes[] = {1, 257, kCapacity};
    for (int case_index = 0; case_index < 3; ++case_index) {
      // The previous D2H must finish before reusing the same pinned host arrays.
      CUDA_TRY(cudaStreamSynchronize(stream));

      n = sizes[case_index];
      alpha = 0.75f + static_cast<float>(case_index);
      bias = case_index == 1 ? -2.0f : 0.5f * static_cast<float>(case_index);
      for (int i = 0; i < n; ++i) {
        host_input[i] = static_cast<float>((i % 11) - 5) * 0.5f +
                        static_cast<float>(case_index);
      }
      for (int i = 0; i < kCapacity; ++i) {
        host_output[i] = kTailSentinel;
      }

      grid = dim3((n + kThreads - 1) / kThreads);
      affine_params.gridDim = grid;
      relu_params.gridDim = grid;
      // Changing a CPU scalar alone does not mutate an instantiated graph.
      // SetParams is required; both calls also carry the new grid and n.
      CUDA_TRY(cudaGraphExecKernelNodeSetParams(exec, affine_node,
                                                &affine_params));
      CUDA_TRY(cudaGraphExecKernelNodeSetParams(exec, relu_node, &relu_params));

      CUDA_TRY(cudaMemcpyAsync(device_input, host_input,
                               n * sizeof(float), cudaMemcpyHostToDevice,
                               stream));
      CUDA_TRY(cudaMemcpyAsync(device_values, host_output,
                               kCapacity * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
      CUDA_TRY(cudaGraphLaunch(exec, stream));
      CUDA_TRY(cudaMemcpyAsync(host_output, device_values,
                               kCapacity * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
      CUDA_TRY(cudaStreamSynchronize(stream));

      for (int i = 0; i < n; ++i) {
        const float input = host_input[i];
        const float affine = alpha * input + bias;
        const float expected = affine > 0.0f ? affine : 0.0f;
        if (!std::isfinite(host_output[i]) ||
            std::fabs(host_output[i] - expected) > 1.0e-6f) {
          std::fprintf(stderr,
                       "FAIL: case=%d index=%d actual=%g expected=%g\n",
                       case_index, i, host_output[i], expected);
          goto cleanup;
        }
      }
      for (int i = n; i < kCapacity; ++i) {
        if (host_output[i] != kTailSentinel) {
          std::fprintf(stderr,
                       "FAIL: case=%d tail index=%d was overwritten (%g)\n",
                       case_index, i, host_output[i]);
          goto cleanup;
        }
      }
    }
  }

  rc = EXIT_SUCCESS;
  std::puts("PASS: explicit affine/ReLU graph nodes updated and replayed for 3 sizes");

cleanup:
  bool cleanup_ok = true;
  // The stream is synchronized before releasing buffers that may be referenced by it.
  if (stream != nullptr) {
    cleanup_ok = check_cuda(cudaStreamSynchronize(stream),
                            "cudaStreamSynchronize(cleanup)") && cleanup_ok;
  }
  if (exec != nullptr) {
    cleanup_ok = check_cuda(cudaGraphExecDestroy(exec),
                            "cudaGraphExecDestroy(cleanup)") && cleanup_ok;
  }
  if (graph != nullptr) {
    cleanup_ok = check_cuda(cudaGraphDestroy(graph),
                            "cudaGraphDestroy(cleanup)") && cleanup_ok;
  }
  if (device_values != nullptr) {
    cleanup_ok = check_cuda(cudaFree(device_values),
                            "cudaFree(device_values)") && cleanup_ok;
  }
  if (device_input != nullptr) {
    cleanup_ok = check_cuda(cudaFree(device_input),
                            "cudaFree(device_input)") && cleanup_ok;
  }
  if (host_output != nullptr) {
    cleanup_ok = check_cuda(cudaFreeHost(host_output),
                            "cudaFreeHost(host_output)") && cleanup_ok;
  }
  if (host_input != nullptr) {
    cleanup_ok = check_cuda(cudaFreeHost(host_input),
                            "cudaFreeHost(host_input)") && cleanup_ok;
  }
  if (stream != nullptr) {
    cleanup_ok = check_cuda(cudaStreamDestroy(stream),
                            "cudaStreamDestroy(cleanup)") && cleanup_ok;
  }
  if (!cleanup_ok) {
    rc = EXIT_FAILURE;
  }

#undef CUDA_TRY
  return rc;
}
