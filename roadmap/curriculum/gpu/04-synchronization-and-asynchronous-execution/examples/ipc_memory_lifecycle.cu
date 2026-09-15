#include <cuda_runtime.h>

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

// The example intentionally shares a 2 MiB allocation: cudaMalloc may sub-
// allocate, and NVIDIA recommends sharing allocations with a 2 MiB-aligned size.
constexpr std::size_t kElements = 512 * 1024;
constexpr std::size_t kBytes = kElements * sizeof(float);

struct IpcMessage {
    cudaIpcMemHandle_t handle;
    std::size_t elements;
};

struct IpcAck {
    int status;
};

bool transfer_all(int fd, void* data, std::size_t bytes, bool write_mode) {
    auto* cursor = static_cast<unsigned char*>(data);
    while (bytes != 0) {
        const ssize_t moved = write_mode ? ::write(fd, cursor, bytes)
                                         : ::read(fd, cursor, bytes);
        if (moved < 0 && errno == EINTR) continue;
        if (moved <= 0) return false;
        cursor += moved;
        bytes -= static_cast<std::size_t>(moved);
    }
    return true;
}

bool cuda_ok(cudaError_t error, const char* expression, int line) {
    if (error == cudaSuccess) return true;
    std::fprintf(stderr, "CUDA %s:%d %s: %s\n", __FILE__, line,
                 expression, cudaGetErrorString(error));
    return false;
}

#define CUDA_CHECK(call) \
    do { \
        if (!cuda_ok((call), #call, __LINE__)) return EXIT_FAILURE; \
    } while (false)

int consumer_main(int read_fd, int write_fd) {
    IpcMessage message{};
    IpcAck ack{1};
    if (!transfer_all(read_fd, &message, sizeof(message), false)) {
        std::fprintf(stderr, "consumer: producer closed before sending the handle\n");
        return EXIT_FAILURE;
    }
    ::close(read_fd);

    CUDA_CHECK(cudaSetDevice(0));
    void* imported = nullptr;
    if (!cuda_ok(cudaIpcOpenMemHandle(&imported, message.handle,
                                     cudaIpcMemLazyEnablePeerAccess),
                 "cudaIpcOpenMemHandle", __LINE__)) {
        (void)transfer_all(write_fd, &ack, sizeof(ack), true);
        ::close(write_fd);
        return EXIT_FAILURE;
    }

    std::vector<float> host(message.elements);
    const cudaError_t copy_status = cudaMemcpy(
        host.data(), imported, message.elements * sizeof(float),
        cudaMemcpyDeviceToHost);
    bool values_ok = copy_status == cudaSuccess;
    if (!values_ok) {
        (void)cuda_ok(copy_status, "cudaMemcpy(device to host)", __LINE__);
    } else {
        for (std::size_t i = 0; i < message.elements; ++i) {
            const float expected = static_cast<float>(i % 1024) + 17.0f;
            if (!std::isfinite(host[i]) || host[i] != expected) {
                std::fprintf(stderr, "consumer mismatch at %zu: %.1f != %.1f\n",
                             i, host[i], expected);
                values_ok = false;
                break;
            }
        }
    }

    // The synchronous D2H copy is complete here.  Close the imported mapping
    // before acknowledging; the producer waits for this message before free.
    const cudaError_t close_status = cudaIpcCloseMemHandle(imported);
    if (!cuda_ok(close_status, "cudaIpcCloseMemHandle", __LINE__)) values_ok = false;
    ack.status = values_ok ? 0 : 1;
    const bool sent = transfer_all(write_fd, &ack, sizeof(ack), true);
    ::close(write_fd);
    if (!sent || !values_ok) return EXIT_FAILURE;
    std::printf("consumer: verified %zu values and closed imported mapping\n",
                message.elements);
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::printf("usage: %s  (Linux only; parent execs itself as consumer)\n", argv[0]);
        return EXIT_SUCCESS;
    }
    if (argc == 4 && std::strcmp(argv[1], "--consumer") == 0) {
        return consumer_main(std::atoi(argv[2]), std::atoi(argv[3]));
    }
    if (argc != 1) {
        std::fprintf(stderr, "usage: %s\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Do not initialize CUDA before fork.  The child immediately execs a fresh
    // process, so no CUDA context is inherited or reused across fork.
    int producer_to_consumer[2] = {-1, -1};
    int consumer_to_producer[2] = {-1, -1};
    if (::pipe(producer_to_consumer) != 0) {
        std::perror("pipe");
        return EXIT_FAILURE;
    }
    if (::pipe(consumer_to_producer) != 0) {
        std::perror("pipe");
        ::close(producer_to_consumer[0]);
        ::close(producer_to_consumer[1]);
        return EXIT_FAILURE;
    }
    std::signal(SIGPIPE, SIG_IGN);
    const pid_t child = ::fork();
    if (child < 0) {
        std::perror("fork");
        return EXIT_FAILURE;
    }
    if (child == 0) {
        ::close(producer_to_consumer[1]);
        ::close(consumer_to_producer[0]);
        char read_fd[32], write_fd[32];
        std::snprintf(read_fd, sizeof(read_fd), "%d", producer_to_consumer[0]);
        std::snprintf(write_fd, sizeof(write_fd), "%d", consumer_to_producer[1]);
        ::execl(argv[0], argv[0], "--consumer", read_fd, write_fd,
                static_cast<char*>(nullptr));
        std::perror("exec self (run this program as ./ipc_memory_lifecycle)");
        _exit(127);
    }

    ::close(producer_to_consumer[0]);
    producer_to_consumer[0] = -1;
    ::close(consumer_to_producer[1]);
    consumer_to_producer[1] = -1;
    std::vector<float> host(kElements);
    for (std::size_t i = 0; i < kElements; ++i) {
        host[i] = static_cast<float>(i % 1024) + 17.0f;
    }

    float* device_buffer = nullptr;
    bool allocated = false;
    bool child_waited = false;
    bool ack_received = false;
    int child_status = 0;
    int device_count = 0;
    int exit_code = EXIT_FAILURE;
    IpcMessage message{};
    IpcAck ack{1};
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status == cudaErrorNoDevice ||
        (cuda_status == cudaSuccess && device_count == 0)) {
        std::fprintf(stderr, "SKIP: CUDA runtime reports no device\n");
        exit_code = 77;
        goto parent_cleanup;
    }
    if (!cuda_ok(cuda_status, "cudaGetDeviceCount", __LINE__)) goto parent_cleanup;
    if (!cuda_ok(cudaSetDevice(0), "cudaSetDevice(0)", __LINE__)) goto parent_cleanup;
    cuda_status = cudaMalloc(&device_buffer, kBytes);
    if (!cuda_ok(cuda_status, "cudaMalloc", __LINE__)) goto parent_cleanup;
    allocated = true;
    if (!cuda_ok(cudaMemcpy(device_buffer, host.data(), kBytes,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(host to device)", __LINE__)) goto parent_cleanup;

    message.elements = kElements;
    if (!cuda_ok(cudaIpcGetMemHandle(&message.handle, device_buffer),
                 "cudaIpcGetMemHandle", __LINE__)) goto parent_cleanup;
    if (!transfer_all(producer_to_consumer[1], &message, sizeof(message), true)) {
        std::perror("send IPC handle");
        goto parent_cleanup;
    }
    ::close(producer_to_consumer[1]);
    producer_to_consumer[1] = -1;

    ack_received = transfer_all(consumer_to_producer[0], &ack, sizeof(ack), false);
    ::close(consumer_to_producer[0]);
    consumer_to_producer[0] = -1;
    {
        pid_t waited = -1;
        do {
            waited = ::waitpid(child, &child_status, 0);
        } while (waited < 0 && errno == EINTR);
        if (waited != child) {
            std::perror("waitpid");
            goto parent_cleanup;
        }
        child_waited = true;
    }

    if (!ack_received || ack.status != 0 || !WIFEXITED(child_status) ||
        WEXITSTATUS(child_status) != EXIT_SUCCESS) {
        std::fprintf(stderr, "producer: consumer did not complete cleanly\n");
        goto parent_cleanup;
    }
    exit_code = EXIT_SUCCESS;

parent_cleanup:
    if (producer_to_consumer[0] >= 0) ::close(producer_to_consumer[0]);
    if (producer_to_consumer[1] >= 0) ::close(producer_to_consumer[1]);
    if (consumer_to_producer[0] >= 0) ::close(consumer_to_producer[0]);
    if (consumer_to_producer[1] >= 0) ::close(consumer_to_producer[1]);
    if (!child_waited) {
        do {
            child_status = 0;
        } while (::waitpid(child, &child_status, 0) < 0 && errno == EINTR);
    }
    // Close-ack is the normal lifetime boundary; waitpid also makes every
    // error path safe before freeing the producer allocation.
    if (allocated) {
        const cudaError_t free_status = cudaFree(device_buffer);
        if (!cuda_ok(free_status, "cudaFree after consumer exit", __LINE__)) {
            exit_code = EXIT_FAILURE;
        }
    }
    if (exit_code == EXIT_SUCCESS) {
        std::printf("producer: consumer closed the import before cudaFree; PASS\n");
    }
    return exit_code;
}
