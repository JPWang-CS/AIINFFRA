#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int kStreams = 2;
constexpr int kChunkElements = 1 << 16;
constexpr int kChunks = 9;  // odd count exercises the final slot boundary

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                              \
    if (status != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(status));                               \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (false)

__global__ void scale(float* values, int n, float factor) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    values[index] *= factor;
  }
}

template <typename T>
T* allocate_host(std::size_t count, bool pinned) {
  if (!pinned) {
    return new T[count];
  }
  T* pointer = nullptr;
  CUDA_CHECK(cudaMallocHost(&pointer, count * sizeof(T)));
  return pointer;
}

template <typename T>
void release_host(T* pointer, bool pinned) {
  if (pinned) {
    CUDA_CHECK(cudaFreeHost(pointer));
  } else {
    delete[] pointer;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const bool pinned = argc <= 1 || std::string(argv[1]) != "--pageable";
  const std::size_t slot_elements = kChunkElements;
  const std::size_t host_elements = kStreams * slot_elements;
  float* host_input = allocate_host<float>(host_elements, pinned);
  float* host_output = allocate_host<float>(host_elements, pinned);
  std::vector<float> reference(kChunks * slot_elements);
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    for (int i = 0; i < kChunkElements; ++i) {
      const float value = 1.0f + static_cast<float>((chunk + i) % 97) * 0.01f;
      reference[chunk * slot_elements + i] = value * 3.0f;
    }
  }

  cudaStream_t streams[kStreams];
  cudaEvent_t completed[kStreams];
  float* device_slots[kStreams] = {};
  for (int slot = 0; slot < kStreams; ++slot) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[slot], cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&completed[slot], cudaEventDisableTiming));
    CUDA_CHECK(cudaMalloc(&device_slots[slot], slot_elements * sizeof(float)));
  }

  for (int chunk = 0; chunk < kChunks; ++chunk) {
    const int slot = chunk % kStreams;
    // The host must not overwrite a slot until its previous D2H has completed.
    // This is the double-buffer reuse boundary: chunk 2 waits for chunk 0,
    // chunk 3 waits for chunk 1, and so on.
    if (chunk >= kStreams) {
      CUDA_CHECK(cudaEventSynchronize(completed[slot]));
      const int previous_chunk = chunk - kStreams;
      for (int i = 0; i < kChunkElements; ++i) {
        const float actual = host_output[slot * slot_elements + i];
        const float expected = reference[previous_chunk * slot_elements + i];
        if (!std::isfinite(actual) || !std::isfinite(expected) ||
            !std::isfinite(actual - expected) ||
            std::abs(actual - expected) > 1e-5f) {
          std::fprintf(stderr, "chunk %d validation failed at %d\n",
                       previous_chunk, i);
          return EXIT_FAILURE;
        }
      }
    }
    for (int i = 0; i < kChunkElements; ++i) {
      host_input[slot * slot_elements + i] =
          1.0f + static_cast<float>((chunk + i) % 97) * 0.01f;
    }

    CUDA_CHECK(cudaMemcpyAsync(device_slots[slot], host_input + slot * slot_elements,
                               slot_elements * sizeof(float),
                               cudaMemcpyHostToDevice, streams[slot]));
    scale<<<(kChunkElements + 255) / 256, 256, 0, streams[slot]>>>(
        device_slots[slot], kChunkElements, 3.0f);
    CUDA_CHECK(cudaMemcpyAsync(host_output + slot * slot_elements,
                               device_slots[slot],
                               slot_elements * sizeof(float),
                               cudaMemcpyDeviceToHost, streams[slot]));
    CUDA_CHECK(cudaEventRecord(completed[slot], streams[slot]));
    CUDA_CHECK(cudaGetLastError());
  }

  for (int slot = 0; slot < kStreams; ++slot) {
    CUDA_CHECK(cudaEventSynchronize(completed[slot]));
  }

  float max_error = 0.0f;
  bool valid = true;
  for (int chunk = kChunks - kStreams; chunk < kChunks; ++chunk) {
    const int slot = chunk % kStreams;
    for (int i = 0; i < kChunkElements; ++i) {
      const float actual = host_output[slot * slot_elements + i];
      const float expected = reference[chunk * slot_elements + i];
      const float error = actual - expected;
      if (!std::isfinite(actual) || !std::isfinite(expected) ||
          !std::isfinite(error)) {
        valid = false;
      } else {
        max_error = std::max(max_error, std::abs(error));
      }
    }
  }
  std::printf("host_memory=%s streams=%d chunks=%d max_error=%.7g\n",
              pinned ? "pinned" : "pageable", kStreams, kChunks, max_error);

  for (int slot = 0; slot < kStreams; ++slot) {
    CUDA_CHECK(cudaFree(device_slots[slot]));
    CUDA_CHECK(cudaEventDestroy(completed[slot]));
    CUDA_CHECK(cudaStreamDestroy(streams[slot]));
  }
  release_host(host_output, pinned);
  release_host(host_input, pinned);
  return valid && std::isfinite(max_error) && max_error < 1e-5f
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
