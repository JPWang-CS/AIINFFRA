#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

struct CudaFailure : std::runtime_error {
    cudaError_t code;
    CudaFailure(cudaError_t c, const char* call)
        : std::runtime_error(std::string(call) + ": " + cudaGetErrorString(c)), code(c) {}
};
void checked(cudaError_t status, const char* call) {
    if (status != cudaSuccess) throw CudaFailure(status, call);
}
#define CHECK(call) checked((call), #call)

__global__ void add_value(int* output, int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *output += value;
}
__global__ void launch_device_graph(cudaGraphExec_t child, int* status) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *status = int(cudaGraphLaunch(child, cudaStreamGraphFireAndForget));
}

cudaGraphNode_t add_kernel(cudaGraph_t graph, cudaGraphNode_t dependency,
                           int* output, int value) {
    void* args[] = {&output, &value};
    cudaKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(add_value);
    params.gridDim = dim3(1); params.blockDim = dim3(1);
    params.kernelParams = args;
    cudaGraphNode_t node{};
    CHECK(cudaGraphAddKernelNode(&node, graph, dependency ? &dependency : nullptr,
                                 dependency ? 1 : 0, &params));
    return node;
}

void verify(int* output, int expected, cudaStream_t stream, const char* label) {
    CHECK(cudaStreamSynchronize(stream));
    int actual = -1;
    CHECK(cudaMemcpy(&actual, output, sizeof(int), cudaMemcpyDeviceToHost));
    if (actual != expected) throw std::runtime_error(std::string(label) + " value mismatch");
    std::printf("PASS %s: %d\n", label, actual);
}

void run_child(int* output, cudaStream_t stream) {
    cudaGraph_t child{}, parent{};
    cudaGraphExec_t exec{};
    CHECK(cudaGraphCreate(&child, 0)); CHECK(cudaGraphCreate(&parent, 0));
    add_kernel(child, nullptr, output, 7);
    cudaGraphNode_t embedded{};
    CHECK(cudaGraphAddChildGraphNode(&embedded, parent, nullptr, 0, child));
    // The child node owns an embedded copy; the source graph may now be destroyed.
    CHECK(cudaGraphDestroy(child));
    add_kernel(parent, embedded, output, 2);
    CHECK(cudaGraphInstantiateWithFlags(&exec, parent, 0));
    CHECK(cudaMemsetAsync(output, 0, sizeof(int), stream));
    CHECK(cudaGraphLaunch(exec, stream));
    verify(output, 9, stream, "child then dependent kernel");
    CHECK(cudaGraphExecDestroy(exec)); CHECK(cudaGraphDestroy(parent));
}

void run_conditional(int* output, cudaStream_t stream) {
    for (unsigned condition : {0u, 1u}) {
        cudaGraph_t graph{}; cudaGraphExec_t exec{};
        CHECK(cudaGraphCreate(&graph, 0));
        cudaGraphConditionalHandle handle{};
        CHECK(cudaGraphConditionalHandleCreate(&handle, graph, condition, cudaGraphCondAssignDefault));
        cudaGraphNodeParams params{};
        params.type = cudaGraphNodeTypeConditional;
        params.conditional.handle = handle;
        params.conditional.type = cudaGraphCondTypeIf;
        params.conditional.size = 1;
        cudaGraphNode_t node{};
        CHECK(cudaGraphAddNode(&node, graph, nullptr, nullptr, 0, &params));
        add_kernel(params.conditional.phGraph_out[0], nullptr, output, 7);
        CHECK(cudaGraphInstantiateWithFlags(&exec, graph, 0));
        for (int replay = 0; replay < 2; ++replay) {
            CHECK(cudaMemsetAsync(output, 0, sizeof(int), stream));
            CHECK(cudaGraphLaunch(exec, stream));
            verify(output, condition ? 7 : 0, stream, "conditional default reset per replay");
        }
        CHECK(cudaGraphExecDestroy(exec)); CHECK(cudaGraphDestroy(graph));
    }
}

void run_device(int* output, cudaStream_t stream) {
    cudaGraph_t child{}, parent{};
    cudaGraphExec_t child_exec{}, parent_exec{};
    int* device_status = nullptr;
    CHECK(cudaMalloc(&device_status, sizeof(int)));
    CHECK(cudaGraphCreate(&child, 0));
    add_kernel(child, nullptr, output, 7);
    CHECK(cudaGraphInstantiateWithFlags(&child_exec, child, cudaGraphInstantiateFlagDeviceLaunch));
    CHECK(cudaGraphUpload(child_exec, stream));
    CHECK(cudaStreamSynchronize(stream));

    // Device launch must originate inside another graph, not an ordinary launch.
    CHECK(cudaGraphCreate(&parent, 0));
    void* args[] = {&child_exec, &device_status};
    cudaKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(launch_device_graph);
    params.gridDim = dim3(1); params.blockDim = dim3(1);
    params.kernelParams = args;
    cudaGraphNode_t node{};
    CHECK(cudaGraphAddKernelNode(&node, parent, nullptr, 0, &params));
    CHECK(cudaGraphInstantiateWithFlags(&parent_exec, parent, 0));
    CHECK(cudaMemsetAsync(output, 0, sizeof(int), stream));
    CHECK(cudaMemsetAsync(device_status, 0xff, sizeof(int), stream));
    CHECK(cudaGraphLaunch(parent_exec, stream));
    // Parent stream completion includes its fire-and-forget child environment.
    CHECK(cudaStreamSynchronize(stream));
    int launch_status = -1;
    CHECK(cudaMemcpy(&launch_status, device_status, sizeof(int), cudaMemcpyDeviceToHost));
    if (launch_status != int(cudaSuccess))
        throw CudaFailure(cudaError_t(launch_status), "device cudaGraphLaunch");
    verify(output, 7, stream, "device fire-and-forget child");
    CHECK(cudaGraphExecDestroy(parent_exec)); CHECK(cudaGraphDestroy(parent));
    CHECK(cudaGraphExecDestroy(child_exec)); CHECK(cudaGraphDestroy(child));
    CHECK(cudaFree(device_status));
}

int main(int argc, char** argv) {
    const std::string mode = argc == 2 ? argv[1] : "all";
    if (mode == "--help") {
        std::puts("Usage: graph_advanced [all|child|conditional|device]\n"
                  "CUDA13.3 headers, compile example for sm_80 with -rdc=true.\n"
                  "Each mode validates an integer result; no performance claim.");
        return 0;
    }
    if (argc > 2 || (mode != "all" && mode != "child" && mode != "conditional" && mode != "device")) return 2;
    try {
        int count = 0;
        const auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || (status == cudaSuccess && count == 0)) {
            std::puts("SKIP: no CUDA device"); return 77;
        }
        CHECK(status); CHECK(cudaSetDevice(0));
        cudaDeviceProp prop{}; CHECK(cudaGetDeviceProperties(&prop, 0));
        if (prop.major < 8 || !prop.unifiedAddressing) {
            std::puts("SKIP: this build targets CC8.0+ and requires UVA"); return 77;
        }
        std::printf("GPU=%s CC=%d.%d\n", prop.name, prop.major, prop.minor);
        int* output = nullptr; cudaStream_t stream{};
        CHECK(cudaMalloc(&output, sizeof(int)));
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        if (mode == "all" || mode == "child") run_child(output, stream);
        if (mode == "all" || mode == "conditional") run_conditional(output, stream);
        if (mode == "all" || mode == "device") run_device(output, stream);
        CHECK(cudaStreamDestroy(stream)); CHECK(cudaFree(output));
    } catch (const CudaFailure& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return e.code == cudaErrorNotSupported ? 77 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what()); return 1;
    }
    return 0;
}
