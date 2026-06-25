#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(e) do { cudaError_t r=e; if(r){printf("ERR %s\n",cudaGetErrorString(r));exit(1);} }while(0)

constexpr int TILE = 32;

__global__ void naive_k(const half* A, const half* B, half* C, int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || m >= M) return;
    float s = 0;
    for (int k = 0; k < K; k++)
        s += __half2float(A[m * K + k]) * __half2float(B[k * N + n]);
    C[m * N + n] = __float2half_rn(s);
}

__global__ void tiled_k(const half* A, const half* B, half* C, int M, int N, int K) {
    int m = blockIdx.x * TILE + threadIdx.x;
    int n = blockIdx.y * TILE + threadIdx.y;
    float s = 0;
    __shared__ half As[TILE][TILE], Bs[TILE][TILE];
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aK = t * TILE + threadIdx.y;
        As[threadIdx.x][threadIdx.y] = (m < M && aK < K) ? A[m * K + aK] : __float2half_rn(0.0f);
        int bK = t * TILE + threadIdx.x;
        Bs[threadIdx.x][threadIdx.y] = (n < N && bK < K) ? B[bK * N + n] : __float2half_rn(0.0f);
        __syncthreads();
        for (int k = 0; k < TILE; k++)
            s += __half2float(As[threadIdx.x][k]) * __half2float(Bs[k][threadIdx.y]);
        __syncthreads();
    }
    if (m < M && n < N)
        C[m * N + n] = __float2half_rn(s);
}

// 只用小矩阵跑 CPU 参考（太大单核太慢）
int main() {
    // === 正确性验证：小矩阵 ===
    int MC = 256, NC = 256, KC = 256;
    size_t sa = MC*KC*2, sb = KC*NC*2, sc = MC*NC*2;
    half *A=(half*)malloc(sa),*B=(half*)malloc(sb);
    half *Cn=(half*)malloc(sc),*Ct=(half*)malloc(sc),*Cc=(half*)malloc(sc);
    srand(42);
    for(int i=0;i<MC*KC;i++)A[i]=__float2half_rn((float)rand()/RAND_MAX-.5f);
    for(int i=0;i<KC*NC;i++)B[i]=__float2half_rn((float)rand()/RAND_MAX-.5f);
    memset(Cn,0,sc);memset(Ct,0,sc);memset(Cc,0,sc);

    // CPU
    for(int m=0;m<MC;m++) for(int n=0;n<NC;n++) {
        float ss=0;
        for(int k=0;k<KC;k++) ss+=__half2float(A[m*KC+k])*__half2float(B[k*NC+n]);
        Cc[m*NC+n]=__float2half_rn(ss);
    }

    // GPU (correctness, small)
    half *dA,*dB,*dC;
    CUDA_CHECK(cudaMalloc(&dA,sa));CUDA_CHECK(cudaMalloc(&dB,sb));CUDA_CHECK(cudaMalloc(&dC,sc));
    CUDA_CHECK(cudaMemcpy(dA,A,sa,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dB,B,sb,cudaMemcpyHostToDevice));

    dim3 nb(16,16), ng((NC+15)/16,(MC+15)/16), tb(TILE,TILE), tg((MC+TILE-1)/TILE,(NC+TILE-1)/TILE);

    CUDA_CHECK(cudaMemset(dC,0,sc)); naive_k<<<ng,nb>>>(dA,dB,dC,MC,NC,KC);
    CUDA_CHECK(cudaDeviceSynchronize());CUDA_CHECK(cudaMemcpy(Cn,dC,sc,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(dC,0,sc)); tiled_k<<<tg,tb>>>(dA,dB,dC,MC,NC,KC);
    CUDA_CHECK(cudaDeviceSynchronize());CUDA_CHECK(cudaMemcpy(Ct,dC,sc,cudaMemcpyDeviceToHost));

    float en=0; for(int i=0;i<MC*NC;i++){float e=fabsf(__half2float(Cn[i])-__half2float(Cc[i]));if(e>en)en=e;}
    float et=0; for(int i=0;i<MC*NC;i++){float e=fabsf(__half2float(Ct[i])-__half2float(Cc[i]));if(e>et)et=e;}
    printf("Correctness (256x256x256): naive_err=%.6f tiled_err=%.6f %s\n",en,et,(en<1e-3&&et<1e-3)?"PASS":"FAIL");

    cudaFree(dA);cudaFree(dB);cudaFree(dC);
    free(A);free(B);free(Cn);free(Ct);free(Cc);

    // === 性能基准：大矩阵 ===
    const int M=2048,N=2048,K=2048;
    size_t sA=M*K*2,sB=K*N*2,sC=M*N*2;
    printf("\n=== GEMM fp16 M=N=K=%d (perf test) ===\n",M);

    half *hA=(half*)malloc(sA),*hB=(half*)malloc(sB),*hCn=(half*)malloc(sC),*hCt=(half*)malloc(sC);
    srand(42);
    for(int i=0;i<M*K;i++)hA[i]=__float2half_rn((float)rand()/RAND_MAX-.5f);
    for(int i=0;i<K*N;i++)hB[i]=__float2half_rn((float)rand()/RAND_MAX-.5f);
    memset(hCn,0,sC);memset(hCt,0,sC);

    half *dA2,*dB2,*dC2;
    CUDA_CHECK(cudaMalloc(&dA2,sA));CUDA_CHECK(cudaMalloc(&dB2,sB));CUDA_CHECK(cudaMalloc(&dC2,sC));
    CUDA_CHECK(cudaMemcpy(dA2,hA,sA,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dB2,hB,sB,cudaMemcpyHostToDevice));

    dim3 ng2((N+15)/16,(M+15)/16), tg2((M+TILE-1)/TILE,(N+TILE-1)/TILE);

    // warmup
    CUDA_CHECK(cudaMemset(dC2,0,sC));naive_k<<<ng2,nb>>>(dA2,dB2,dC2,M,N,K);
    tiled_k<<<tg2,tb>>>(dA2,dB2,dC2,M,N,K);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t st,en;cudaEventCreate(&st);cudaEventCreate(&en);

    CUDA_CHECK(cudaMemset(dC2,0,sC));
    cudaEventRecord(st);
    for(int i=0;i<10;i++) naive_k<<<ng2,nb>>>(dA2,dB2,dC2,M,N,K);
    cudaEventRecord(en);cudaEventSynchronize(en);
    float mn;cudaEventElapsedTime(&mn,st,en);mn/=10;
    float gn=2.0f*M*N*K/(mn/1000)/1e9;
    printf("Naive: %.2f ms  %.0f GFLOPS\n",mn,gn);

    CUDA_CHECK(cudaMemset(dC2,0,sC));
    cudaEventRecord(st);
    for(int i=0;i<10;i++) tiled_k<<<tg2,tb>>>(dA2,dB2,dC2,M,N,K);
    cudaEventRecord(en);cudaEventSynchronize(en);
    float mt;cudaEventElapsedTime(&mt,st,en);mt/=10;
    float gt=2.0f*M*N*K/(mt/1000)/1e9;
    printf("Tiled: %.2f ms  %.0f GFLOPS\n",mt,gt);

    printf("\n=== RESULT ===\nSpeedup: %.1fx (Naive %.0f -> Tiled %.0f GFLOPS)\n",gt/gn,gn,gt);
    printf("4090 FP16 peak: ~330 TFLOPS (Tensor Core)\n");

    cudaFree(dA2);cudaFree(dB2);cudaFree(dC2);
    free(hA);free(hB);free(hCn);free(hCt);
    return 0;
}
