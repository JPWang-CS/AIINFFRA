#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

constexpr int kElements = 1025;
constexpr int kThreads = 256;

bool check_cuda(cudaError_t status, const char* expression) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "CUDA error in %s: %s\n", expression,
               cudaGetErrorString(status));
  return false;
}

__global__ void fill_transform(int* values, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    values[index] = 3 * index + 7;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      std::puts("Usage: graph_memory [--help]\n"
                "Capture cudaMallocAsync/cudaFreeAsync and replay the graph twice.");
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
  int memory_pools_supported = 0;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  int* graph_scratch = nullptr;
  int* device_output = nullptr;
  int* host_output = nullptr;
  bool capturing = false;

#define CUDA_TRY(call) \
  do { \
    if (!check_cuda((call), #call)) { \
      goto cleanup; \
    } \
  } while (false)

  CUDA_TRY(cudaSetDevice(0));
  CUDA_TRY(cudaDeviceGetAttribute(&memory_pools_supported,
                                  cudaDevAttrMemoryPoolsSupported, 0));
  if (!memory_pools_supported) {
    std::puts("SKIP: stream-ordered memory pools are unsupported");
    rc = 77;
    goto cleanup;
  }

  CUDA_TRY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&device_output),
                      kElements * sizeof(int)));
  CUDA_TRY(cudaMallocHost(reinterpret_cast<void**>(&host_output),
                          kElements * sizeof(int)));

  CUDA_TRY(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  capturing = true;
  // This allocation and its matching free become graph nodes.  The pointer is
  // only used by captured operations; it is never dereferenced after capture.
  CUDA_TRY(cudaMallocAsync(reinterpret_cast<void**>(&graph_scratch),
                           kElements * sizeof(int), stream));
  fill_transform<<<(kElements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      graph_scratch, kElements);
  CUDA_TRY(cudaMemcpyAsync(device_output, graph_scratch,
                           kElements * sizeof(int), cudaMemcpyDeviceToDevice,
                           stream));
  CUDA_TRY(cudaFreeAsync(graph_scratch, stream));
  status = cudaStreamEndCapture(stream, &graph);
  capturing = false;
  if (!check_cuda(status, "cudaStreamEndCapture")) {
    goto cleanup;
  }
  CUDA_TRY(cudaGetLastError());

  // A graph replay may keep a fixed virtual address for scratch, but that does
  // not promise its old contents.  Its lifetime ends at the captured free.
  graph_scratch = nullptr;
  CUDA_TRY(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  for (int replay = 0; replay < 2; ++replay) {
    CUDA_TRY(cudaMemsetAsync(device_output, 0xA5,
                             kElements * sizeof(int), stream));
    CUDA_TRY(cudaGraphLaunch(exec, stream));
    CUDA_TRY(cudaMemcpyAsync(host_output, device_output,
                             kElements * sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
    CUDA_TRY(cudaStreamSynchronize(stream));

    // Only the persistent output is inspected.  Scratch was freed by the graph.
    for (int i = 0; i < kElements; ++i) {
      const int expected = 3 * i + 7;
      if (host_output[i] != expected) {
        std::fprintf(stderr,
                     "FAIL: replay=%d index=%d actual=%d expected=%d\n",
                     replay, i, host_output[i], expected);
        goto cleanup;
      }
    }
  }

  rc = EXIT_SUCCESS;
  std::puts("PASS: captured async allocation/free and replayed persistent output twice");

cleanup:
  bool cleanup_ok = true;
  if (capturing && stream != nullptr) {
    cudaGraph_t abandoned_graph = nullptr;
    status = cudaStreamEndCapture(stream, &abandoned_graph);
    capturing = false;
    cleanup_ok = check_cuda(status, "cudaStreamEndCapture(cleanup)") &&
                 cleanup_ok;
    if (abandoned_graph != nullptr) {
      cleanup_ok = check_cuda(cudaGraphDestroy(abandoned_graph),
                              "cudaGraphDestroy(abandoned)") && cleanup_ok;
    }
  }
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
  // Do not cudaFree(graph_scratch): its lifetime is owned by the captured free.
  if (device_output != nullptr) {
    cleanup_ok = check_cuda(cudaFree(device_output),
                            "cudaFree(device_output)") && cleanup_ok;
  }
  if (host_output != nullptr) {
    cleanup_ok = check_cuda(cudaFreeHost(host_output),
                            "cudaFreeHost(host_output)") && cleanup_ok;
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
