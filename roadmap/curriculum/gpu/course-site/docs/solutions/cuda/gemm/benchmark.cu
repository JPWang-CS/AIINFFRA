#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#define CUDA_CHECK(e) do{cudaError_t r=e;if(r){printf("ERR %s\n",cudaGetErrorString(r));exit(1);}}while(0)

// ============================================================
// GEMM benchmark — Naive vs Tiled tile-size sweep (K=2048/8192/32768)
//
// 【算子是什么】矩阵乘法 GEMM (FP16)
// 【在模型里干嘛】所有 Linear/FFN 层。K 维度对应模型 hidden_dim/intermediate_dim
//   - QKV projection: K = d_model (如 4096)
//   - FFN up/gate: K = d_model (如 4096)
//   - FFN down: K = d_ff = d_model × inter_size (如 4096×8/3≈11008)
// 【什么模型用】所有 Transformer 的推理/训练 GEMM kernel
// ============================================================
__global__ void naive_k(const half* A, const half* B, half* C, int M, int N, int K) {
    int m = blockIdx.y * 16 + threadIdx.y, n = blockIdx.x * 16 + threadIdx.x;
    if (n >= N || m >= M) return;
    float s = 0;
    for (int k = 0; k < K; k++) s += __half2float(A[m * K + k]) * __half2float(B[k * N + n]);
    C[m * N + n] = __float2half_rn(s);
}

template<int TILE>
__global__ void tiled_k(const half* A, const half* B, half* C, int M, int N, int K) {
    int m = blockIdx.x * TILE + threadIdx.x, n = blockIdx.y * TILE + threadIdx.y;
    float s = 0;
    __shared__ half As[TILE][TILE+1], Bs[TILE][TILE+1];
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aK = t * TILE + threadIdx.y;
        As[threadIdx.x][threadIdx.y] = (m < M && aK < K) ? A[m * K + aK] : __float2half_rn(0.0f);
        int bK = t * TILE + threadIdx.x;
        Bs[threadIdx.x][threadIdx.y] = (n < N && bK < K) ? B[bK * N + n] : __float2half_rn(0.0f);
        __syncthreads();
        for (int k = 0; k < TILE; k++) s += __half2float(As[threadIdx.x][k]) * __half2float(Bs[k][threadIdx.y]);
        __syncthreads();
    }
    if (m < M && n < N) C[m * N + n] = __float2half_rn(s);
}

// ====== Test runner ======
template<int TILE>
void run_tiled(const half* dA, const half* dB, half* dC, int M, int N, int K, int iters, float* ms_out) {
    dim3 tb(TILE, TILE), tg((M+TILE-1)/TILE, (N+TILE-1)/TILE);
    cudaEvent_t st, en; cudaEventCreate(&st); cudaEventCreate(&en);
    cudaEventRecord(st);
    for (int i = 0; i < iters; i++) tiled_k<TILE><<<tg, tb>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(en); cudaEventSynchronize(en);
    cudaEventElapsedTime(ms_out, st, en); *ms_out /= iters;
    cudaEventDestroy(st); cudaEventDestroy(en);
}

int main() {
    // ---- correctness (256x256x256) ----
    int Mc = 256, Nc = 256, Kc = 256;
    size_t sc = Mc * Nc * 2;
    half *A = (half*)malloc(Mc*Kc*2), *B = (half*)malloc(Kc*Nc*2);
    half *Cn = (half*)malloc(sc), *Ct = (half*)malloc(sc), *Cc = (half*)malloc(sc);
    srand(42);
    for (int i = 0; i < Mc*Kc; i++) A[i] = __float2half_rn((float)rand() / RAND_MAX - 0.5f);
    for (int i = 0; i < Kc*Nc; i++) B[i] = __float2half_rn((float)rand() / RAND_MAX - 0.5f);
    memset(Cn, 0, sc); memset(Ct, 0, sc); memset(Cc, 0, sc);
    for (int m = 0; m < Mc; m++) for (int n = 0; n < Nc; n++) {
        float ss = 0; for (int k = 0; k < Kc; k++) ss += __half2float(A[m*Kc+k]) * __half2float(B[k*Nc+n]);
        Cc[m*Nc+n] = __float2half_rn(ss);
    }

    half *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, Mc*Kc*2)); CUDA_CHECK(cudaMalloc(&dB, Kc*Nc*2)); CUDA_CHECK(cudaMalloc(&dC, sc));
    CUDA_CHECK(cudaMemcpy(dA, A, Mc*Kc*2, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dB, B, Kc*Nc*2, cudaMemcpyHostToDevice));

    dim3 nb(16, 16), ng((Nc+15)/16, (Mc+15)/16);
    CUDA_CHECK(cudaMemset(dC, 0, sc)); naive_k<<<ng, nb>>>(dA, dB, dC, Mc, Nc, Kc);
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaMemcpy(Cn, dC, sc, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(dC, 0, sc)); tiled_k<32><<<dim3((Mc+31)/32,(Nc+31)/32),dim3(32,32)>>>(dA, dB, dC, Mc, Nc, Kc);
    CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaMemcpy(Ct, dC, sc, cudaMemcpyDeviceToHost));

    float en=0, et=0;
    for (int i = 0; i < Mc*Nc; i++) { float e = fabsf(__half2float(Cn[i])-__half2float(Cc[i])); if(e>en) en=e; }
    for (int i = 0; i < Mc*Nc; i++) { float e = fabsf(__half2float(Ct[i])-__half2float(Cc[i])); if(e>et) et=e; }
    printf("Correctness (256): naive_err=%.6f tiled_err=%.6f %s\n", en, et, (en<1e-3&&et<1e-3)?"PASS":"FAIL");
    cudaFree(dA); cudaFree(dB); cudaFree(dC); free(A); free(B); free(Cn); free(Ct); free(Cc);

    // ---- K=2048 baseline ----
    {
        int M = 2048, N = 2048, K = 2048;
        size_t sA = M * K * 2, sB = K * N * 2, sC = M * N * 2;
        printf("\n=== GEMM K=%d ===\n", K);
        half *hA = (half*)malloc(sA), *hB = (half*)malloc(sB);
        srand(42);
        for (int i = 0; i < M*K; i++) hA[i] = __float2half_rn((float)rand()/RAND_MAX-.5f);
        for (int i = 0; i < K*N; i++) hB[i] = __float2half_rn((float)rand()/RAND_MAX-.5f);
        half *dA2, *dB2, *dC2;
        CUDA_CHECK(cudaMalloc(&dA2, sA)); CUDA_CHECK(cudaMalloc(&dB2, sB)); CUDA_CHECK(cudaMalloc(&dC2, sC));
        CUDA_CHECK(cudaMemcpy(dA2, hA, sA, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dB2, hB, sB, cudaMemcpyHostToDevice));

        dim3 ng2((N+15)/16, (M+15)/16);
        cudaEvent_t st, en; cudaEventCreate(&st); cudaEventCreate(&en);

        CUDA_CHECK(cudaMemset(dC2, 0, sC)); CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(st); for(int i=0;i<10;i++) naive_k<<<ng2,nb>>>(dA2,dB2,dC2,M,N,K);
        cudaEventRecord(en); cudaEventSynchronize(en);
        float mn; cudaEventElapsedTime(&mn, st, en); mn/=10;
        float gn = 2.0f * M * N * K / (mn / 1000) / 1e9;
        printf("Naive(256):   %.2f ms  %.0f GFLOPS\n", mn, gn);

        float mt; run_tiled<32>(dA2,dB2,dC2,M,N,K,10,&mt);
        float gt = 2.0f * M * N * K / (mt / 1000) / 1e9;
        printf("Tiled(32):   %.2f ms  %.0f GFLOPS\n", mt, gt);
        printf("Speedup: %.1fx\n", gt/gn);

        cudaFree(dA2); cudaFree(dB2); cudaFree(dC2); free(hA); free(hB);
    }

    // ---- Tile size comparison K=8192 ----
    {
        int M=2048,N=2048,K=8192;
        size_t sA=M*K*2,sB=K*N*2,sC=M*N*2;
        printf("\n=== GEMM K=%d (A+B=%dMB) Tile size sweep ===\n",K,(int)((M*K+K*N)*2/1e6));
        half *hA=(half*)malloc(sA),*hB=(half*)malloc(sB);
        srand(42);for(int i=0;i<M*K;i++)hA[i]=__float2half_rn((float)rand()/RAND_MAX-.5f);
        for(int i=0;i<K*N;i++)hB[i]=__float2half_rn((float)rand()/RAND_MAX-.5f);
        half *dA,*dB,*dC;
        CUDA_CHECK(cudaMalloc(&dA,sA));CUDA_CHECK(cudaMalloc(&dB,sB));CUDA_CHECK(cudaMalloc(&dC,sC));
        CUDA_CHECK(cudaMemcpy(dA,hA,sA,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dB,hB,sB,cudaMemcpyHostToDevice));
        dim3 ng2((N+15)/16,(M+15)/16);

        cudaEvent_t st,en;cudaEventCreate(&st);cudaEventCreate(&en);
        CUDA_CHECK(cudaMemset(dC,0,sC));CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(st);for(int i=0;i<5;i++)naive_k<<<ng2,nb>>>(dA,dB,dC,M,N,K);cudaEventRecord(en);cudaEventSynchronize(en);
        float mn;cudaEventElapsedTime(&mn,st,en);mn/=5;
        printf("Naive(256):     %.1f ms\n",mn);

        float m16,m32;
        CUDA_CHECK(cudaMemset(dC,0,sC));run_tiled<16>(dA,dB,dC,M,N,K,5,&m16);
        printf("Tiled(16,256th): %.1f ms\n",m16);
        CUDA_CHECK(cudaMemset(dC,0,sC));run_tiled<32>(dA,dB,dC,M,N,K,5,&m32);
        printf("Tiled(32,1024th): %.1f ms\n",m32);
        printf("T16 vs naive: %.1fx  T32 vs naive: %.1fx\n",mn/m16,mn/m32);

        cudaFree(dA);cudaFree(dB);cudaFree(dC);free(hA);free(hB);
    }

    // ---- K=32768: truly overflow L2 ----
    {
        int M=1024,N=1024,K=32768;
        size_t sA=M*K*2,sB=K*N*2,sC=M*N*2;
        printf("\n=== GEMM K=%d (A+B=%dMB > 3x L2) ===\n",K,(int)((M*K+K*N)*2/1e6));
        half *hA=(half*)malloc(sA),*hB=(half*)malloc(sB);
        srand(42);for(int i=0;i<M*K;i++)hA[i]=__float2half_rn((float)rand()/RAND_MAX-.5f);
        for(int i=0;i<K*N;i++)hB[i]=__float2half_rn((float)rand()/RAND_MAX-.5f);
        half *dA,*dB,*dC;
        CUDA_CHECK(cudaMalloc(&dA,sA));CUDA_CHECK(cudaMalloc(&dB,sB));CUDA_CHECK(cudaMalloc(&dC,sC));
        CUDA_CHECK(cudaMemcpy(dA,hA,sA,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dB,hB,sB,cudaMemcpyHostToDevice));
        dim3 ng2((N+15)/16,(M+15)/16);

        cudaEvent_t st,en;cudaEventCreate(&st);cudaEventCreate(&en);
        CUDA_CHECK(cudaMemset(dC,0,sC));CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(st);for(int i=0;i<3;i++)naive_k<<<ng2,nb>>>(dA,dB,dC,M,N,K);cudaEventRecord(en);cudaEventSynchronize(en);
        float mn;cudaEventElapsedTime(&mn,st,en);mn/=3;
        float gn=2.0f*M*N*K/(mn/1000)/1e9;
        printf("Naive(256):     %.1f ms  %.0f GFLOPS\n",mn,gn);

        float m16,m32; run_tiled<16>(dA,dB,dC,M,N,K,3,&m16); run_tiled<32>(dA,dB,dC,M,N,K,3,&m32);
        float gt16=2.0f*M*N*K/(m16/1000)/1e9, gt32=2.0f*M*N*K/(m32/1000)/1e9;
        printf("Tiled(16):      %.1f ms  %.0f GFLOPS  Speedup: %.1fx\n",m16,gt16,mn/m16);
        printf("Tiled(32):      %.1f ms  %.0f GFLOPS  Speedup: %.1fx\n",m32,gt32,mn/m32);

        dim3 tb16(16,16),tg16((M+15)/16,(N+15)/16);
        printf("\nOccupancy analysis:\n");
        printf("  Naive: 256 threads/block, max 6 blocks/SM = 1536 threads\n");
        printf("  T16:   256 threads/block, max 6 blocks/SM = 1536 threads (same)\n");
        printf("  T32:   1024 threads/block, max 1 block/SM = 1024 threads (%.0f%% of naive)\n",100.0*1024/1536);

        cudaFree(dA);cudaFree(dB);cudaFree(dC);free(hA);free(hB);
    }

    printf("\n=== DONE ===\n");
    return 0;
}
