// ============================================================
// main.cu — softmax solve() 的本地测试 harness（服务器跑）
//
// 作用：把 LeetGPU 上跑的 solve(input, output, N) 搬到本地服务器跑，
//       对比 CPU 双精度参考实现验证精度，并测吞吐 / 有效带宽。
//       本文件【不含任何 kernel】—— kernel 在链接进来的 .cu 里
//       （softmax_naive.cu 或后续 online/warp 版本，签名都是 extern "C" solve）。
//
// 编译：nvcc -O2 main.cu softmax_naive.cu -o test_softmax   （见 run.sh）
// 运行：./test_softmax            # 跑默认几个 N
//       ./test_softmax 500000     # 只跑指定 N
//
// 【算子是什么】1D softmax over N（LeetGPU 5_softmax 题型）
// 【在模型里干嘛】Attention 的归一化层 —— 把 raw scores 变概率分布
//                S = Q@K^T/√d  →  P = softmax(S)  →  O = P@V
// 【什么模型用】所有 Transformer（LLaMA/GPT/BERT/DeepSeek/Qwen/Claude ...）
// ============================================================
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(e) do { cudaError_t _r=(e); if(_r){ fprintf(stderr, \
        "CUDA ERR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(_r)); exit(1);} } while(0)

// 被测入口 —— 由 softmax_naive.cu（或后续 online/warp 版本）提供。
// input/output 均为 device pointer，与 LeetGPU 题面一致，签名不可改。
extern "C" void solve(const float* input, float* output, int N);

// ---- CPU 双精度参考（带 max trick），作为精度 ground truth ----
static void softmax_cpu_ref(const float* in, float* out, int N) {
    double mx = in[0];
    for (int i = 1; i < N; i++) { double v = in[i]; if (v > mx) mx = v; }
    double s = 0.0;
    for (int i = 0; i < N; i++) { out[i] = (float)std::exp((double)in[i] - mx); s += out[i]; }
    double inv = 1.0 / s;
    for (int i = 0; i < N; i++) out[i] = (float)((double)out[i] * inv);
}

// ---- 精度对比：返回 max/mean abs err 与输出总和 ----
static bool compare(const float* gpu, const float* ref, int N,
                    double& max_abs, double& mean_abs, double& out_sum) {
    max_abs = 0.0; double sum_abs = 0.0; out_sum = 0.0;
    for (int i = 0; i < N; i++) {
        double d = std::fabs((double)gpu[i] - (double)ref[i]);
        if (d > max_abs) max_abs = d;
        sum_abs += d;
        out_sum += gpu[i];
    }
    mean_abs = sum_abs / N;
    return max_abs < 1e-4;   // 与 bench_softmax.cu 阈值一致
}

static void run_one(int N) {
    printf("\n=== N = %d  (%.2f MB) ===\n", N, N * 4.0 / 1e6);

    std::vector<float> h_in(N), h_gpu(N), h_ref(N);
    srand(42);   // 固定种子，可复现
    for (int i = 0; i < N; i++) h_in[i] = (float)rand() / RAND_MAX * 20.0f - 10.0f;  // [-10, 10]

    softmax_cpu_ref(h_in.data(), h_ref.data(), N);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // warmup（solve 内部有 cudaMalloc/同步，先排掉冷启动 & 编译预热）
    solve(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // timed：solve() 内部自带 cudaDeviceSynchronize + D2H，每调用一次即同步，
    //        所以 cudaEvent 围一圈循环取平均即可反映单次真实成本
    int iters = (N >= (1 << 17)) ? 20 : 50;   // 小 N 单次太短，多跑几轮降噪声
    cudaEvent_t st, en; cudaEventCreate(&st); cudaEventCreate(&en);
    cudaEventRecord(st);
    for (int i = 0; i < iters; i++) solve(d_in, d_out, N);
    cudaEventRecord(en); cudaEventSynchronize(en);
    float ms = 0; cudaEventElapsedTime(&ms, st, en); ms /= iters;
    cudaEventDestroy(st); cudaEventDestroy(en);

    // correctness
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    double max_abs = 0, mean_abs = 0, out_sum = 0;
    bool ok = compare(h_gpu.data(), h_ref.data(), N, max_abs, mean_abs, out_sum);

    // naive 3-pass 的【算法】HBM 流量：input 读 3 遍（findMax/countSum/softmax）+ output 写 1 遍
    //   = 4·N·4 bytes。partial_max/sum 数组很小、D2H 也很小，忽略。
    //   后续 online 1-pass 会降到 2·N·4 → 同时间下 BW 翻倍，就是优化收益。
    double bytes = 4.0 * N * sizeof(float);
    double gb_s  = bytes / (ms / 1000.0) / 1e9;

    printf("  max abs err : %.3e\n", max_abs);
    printf("  mean abs err: %.3e\n", mean_abs);
    printf("  output sum  : %.6f   (应 ≈ 1.0)\n", out_sum);
    printf("  time        : %.4f ms   (avg of %d)\n", ms, iters);
    printf("  HBM traffic : 3 read + 1 write = %.2f MB\n", bytes / 1e6);
    printf("  effective BW: %.1f GB/s\n", gb_s);
    printf("  %s\n", ok ? "[PASS]   (max abs err < 1e-4)" : "[FAIL]");

    cudaFree(d_in); cudaFree(d_out);
}

int main(int argc, char** argv) {
#ifdef KERNEL_FILE
    printf("softmax solve() harness  |  kernel: %s\n", KERNEL_FILE);
#else
    printf("softmax solve() harness  |  kernel: (link-time, see run.sh KERNEL)\n");
#endif

    std::vector<int> sizes;
    if (argc > 1) {
        for (int i = 1; i < argc; i++) sizes.push_back(atoi(argv[i]));
    } else {
        sizes = { 1024, 65536, 500000, 1048576 };   // 500000 = LeetGPU 题面 max
    }

    for (int N : sizes) {
        if (N <= 0) { fprintf(stderr, "[skip] bad N=%d\n", N); continue; }
        run_one(N);
    }

    printf("\n=== DONE ===\n");
    return 0;
}
